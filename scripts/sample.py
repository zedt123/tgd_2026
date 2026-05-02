from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import time

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.io import save_json, save_text, save_image_batch, save_particle_from_uint8
from src.cfg import sync, set_device, compute_nfe
from src.eval.evaluator import Evaluator, EvaluatorOptions
from src.eval.metrics import build_metrics
from src.models import get_model
from src.models.utils import count_parameters
from src.datasets import get_dataset
from src.seed import make_rng
from src.factory import build_sampler, build_operator_and_y

def _to_display_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    if x.ndim == 2:
        return x

    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (H,W) or (C,H,W), got {tuple(x.shape)}")

    x = x.permute(1, 2, 0)  # CHW -> HWC
    return x[..., 0] if x.shape[-1] == 1 else x


def _summary_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().cpu().float().flatten()
    mean = x.mean()
    std = x.std(unbiased=False) if x.numel() > 1 else torch.zeros_like(mean)
    median = torch.quantile(x, 0.5)
    return {
        "mean": float(mean.item()),
        "median": float(median.item()),
        "std": float(std.item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


@torch.no_grad()
def _compute_metric_arrays(
    *,
    cfg: DictConfig,
    device: str,
    samples_bk: torch.Tensor,
    data: torch.Tensor,
    y: Optional[torch.Tensor],
    operator: Any,
) -> Dict[str, torch.Tensor]:
    paired_metrics, _, _ = build_metrics(cfg.eval)
    evaluator = Evaluator(
        device=device,
        paired_metrics=paired_metrics,
        options=EvaluatorOptions(reduce_k="mean", store_per_condition=False),
    )
    evaluator.create_batch(
        B=int(samples_bk.shape[0]),
        K=int(samples_bk.shape[1]),
        gen_bk=samples_bk,
        gt_b=data,
        y_b=y,
        operator=operator,
        log_w_tbk=None,
        ess_tb=None,
    )

    metric_arrays: Dict[str, torch.Tensor] = {}
    try:
        assert evaluator.batch is not None
        for metric in paired_metrics:
            values_bk = metric.compute(evaluator.batch, device=device).detach().cpu()
            metric_arrays[str(metric.name)] = values_bk
    finally:
        evaluator.reset_batch()

    return metric_arrays


def _save_sample_panels(
    *,
    panels_dir: Path,
    samples_bk: torch.Tensor,
    data: torch.Tensor,
    y: Optional[torch.Tensor],
    metric_arrays: Dict[str, torch.Tensor],
    show_metrics: bool,
    panel_dpi: int,
) -> None:
    panels_dir.mkdir(parents=True, exist_ok=True)

    B, K = int(samples_bk.shape[0]), int(samples_bk.shape[1])

    for b in range(B):
        has_y = y is not None
        cols = K + 1 + int(has_y)  # gt + samples + optional measurement
        fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.8), squeeze=False)
        axes = axes[0]

        col = 0
        if has_y:
            axes[col].imshow(_to_display_image(y[b]), cmap="gray" if y[b].ndim == 2 or y[b].shape[0] == 1 else None)
            axes[col].set_title("Measurement")
            axes[col].axis("off")
            col += 1

        axes[col].imshow(_to_display_image(data[b]), cmap="gray" if data[b].ndim == 2 or data[b].shape[0] == 1 else None)
        axes[col].set_title("Ground truth")
        axes[col].axis("off")
        col += 1

        for k in range(K):
            ax = axes[col + k]
            ax.imshow(
                _to_display_image(samples_bk[b, k]),
                cmap="gray" if samples_bk[b, k].ndim == 2 or samples_bk[b, k].shape[0] == 1 else None,
            )
            ax.set_title(f"Sample {k + 1}")
            ax.axis("off")

            if show_metrics and len(metric_arrays) > 0:
                lines: List[str] = []
                if "lpips" in metric_arrays:
                    lines.append(f"LPIPS: {float(metric_arrays['lpips'][b, k]):.4f}")
                if "psnr" in metric_arrays:
                    lines.append(f"PSNR: {float(metric_arrays['psnr'][b, k]):.2f}")
                for name, arr in metric_arrays.items():
                    if name in {"lpips", "psnr"}:
                        continue
                    lines.append(f"{name}: {float(arr[b, k]):.4f}")

                if lines:
                    ax.text(
                        0.5,
                        -0.12,
                        "\n".join(lines),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=9,
                    )

        fig.tight_layout()
        fig.savefig(panels_dir / f"image_{b:03d}.png", dpi=panel_dpi, bbox_inches="tight")
        plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    run_cfg = cfg.run

    device = set_device(cfg=run_cfg)

    model = get_model(cfg.model, device=device)
    total_params = count_parameters(model)

    rng = make_rng(int(run_cfg.seed), device=device)

    requested_indices = cfg.get("image_indices")
    if requested_indices is not None:
        image_indices = [int(i) for i in requested_indices]
        if len(image_indices) == 0:
            raise ValueError("image_indices was provided but is empty")
        max_idx = max(image_indices)
        data = get_dataset(cfg.data).get_data(size=max_idx + 1)
        data = data[image_indices]
        total_data = len(image_indices)
    else:
        total_data = int(run_cfg.get("total_data"))
        data = get_dataset(cfg.data).get_data(size=total_data)
        image_indices = list(range(total_data))

    operator, y = build_operator_and_y(cfg.operator, x_ref=data, generator=rng.operator, device=device)

    sampler = build_sampler(cfg, model=model, operator=operator, device=device)

    batch_size = int(run_cfg.get("batch_size", 1000))
    K_requested = int(run_cfg.get("num_particles", 1))

    traj_steps = int(run_cfg.get("traj_steps", 1))
    particles_to_save = int(run_cfg.get("particles_to_save", 1))

    sync(device)
    t0 = time.perf_counter()
    if y is not None and total_data > batch_size:
        out = sampler.batch_sample(
            N=K_requested,
            y=y,
            B_chunk=batch_size,
            rng=rng,
            traj_steps=traj_steps,
            particles_for_traj=particles_to_save,
        )
    else:
        out = sampler.sample(
            N=K_requested,
            y=y,
            rng=rng,
            traj_steps=traj_steps,
            particles_for_traj=particles_to_save,
        )
    sync(device)
    t1 = time.perf_counter()

    sample_seconds = float(t1 - t0)
    samples_bk = out.x0  # (B,K,C,H,W) in [-1,1]
    B = int(samples_bk.shape[0])
    K_actual = int(samples_bk.shape[1])
    nfe = compute_nfe(cfg)

    metric_arrays = _compute_metric_arrays(
        cfg=cfg,
        device=device,
        samples_bk=samples_bk,
        data=data,
        y=y,
        operator=sampler.ctx.operator,
    )

    out_root = Path(to_absolute_path(str(cfg.get("out_dir", "results_samples"))))
    run_dir = out_root
    run_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = run_dir / f"samples_k{K_actual}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("save_images", True)):
        save_image_batch(samples_dir, samples_bk)

    if bool(cfg.get("save_traj", True)) and traj_steps >= 2:
        smc_info = out.info["smc"]
        traj = smc_info.traj
        times = smc_info.traj_times
        traj_dir = run_dir / "traj"
        save_particle_from_uint8(traj_dir, traj, times)

    vis_cfg = cfg.get("visualization")
    if vis_cfg is not None and bool(vis_cfg.get("save_panels", True)):
        panels_dir = run_dir / "panels"
        _save_sample_panels(
            panels_dir=panels_dir,
            samples_bk=samples_bk,
            data=data,
            y=y,
            metric_arrays=metric_arrays,
            show_metrics=bool(vis_cfg.get("show_metrics", True)),
            panel_dpi=int(vis_cfg.get("panel_dpi", 160)),
        )

    metric_payload: Dict[str, Any] = {"per_metric": {}, "per_image": []}
    for name, values_bk in metric_arrays.items():
        metric_payload["per_metric"][name] = {
            "values_bk": values_bk.tolist(),
            "summary": _summary_stats(values_bk),
        }

    for b in range(B):
        item: Dict[str, Any] = {
            "local_index": int(b),
            "dataset_index": int(image_indices[b]),
            "samples": [],
        }
        for k in range(K_actual):
            sample_item: Dict[str, Any] = {"sample_index": int(k)}
            for name, values_bk in metric_arrays.items():
                sample_item[name] = float(values_bk[b, k])
            item["samples"].append(sample_item)
        metric_payload["per_image"].append(item)

    meta: Dict[str, Any] = {
        "device": device,
        "total_params": int(total_params),
        "B": B,
        "K_requested": K_requested,
        "K_generated": K_actual,
        "sample_seconds": sample_seconds,
        "seconds_per_image": sample_seconds / float(B),
        "nfe": int(nfe),
        "dataset_indices": image_indices,
    }

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "meta": meta,
        "metrics": metric_payload,
        "outputs": {
            "run_dir": str(run_dir),
            "samples_dir": str(samples_dir),
            "panels_dir": str(run_dir / "panels"),
        },
    }
    save_text(run_dir / "sampler.yaml", OmegaConf.to_yaml(cfg.sampler, resolve=True))
    save_json(run_dir / "run.json", payload)
    save_text(run_dir / "run.txt", OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()