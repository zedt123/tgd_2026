from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

import time
import torch
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.factory import build_sampler
from src.io import save_json, save_text
from src.cfg import sync, compute_nfe
from src.eval.evaluator import EvaluatorOptions

def _extract_sampler_override(overrides: List[str]) -> tuple[str, List[str]]:
    sampler_path: Optional[str] = None
    remaining: List[str] = []

    for override in overrides:
        if override.startswith("sampler="):
            if sampler_path is not None:
                raise ValueError(f"Multiple sampler overrides found: {overrides}")
            sampler_path = override.split("=", 1)[1]
        else:
            remaining.append(override)

    if sampler_path is None:
        raise ValueError(
            f"Expected one sampler override like 'sampler=ffhq/daps/mcmc_daps', "
            f"got: {overrides}"
        )

    return sampler_path, remaining


def build_ablation_point_cfg(
    *,
    root_cfg: DictConfig,
    sampler_spec: DictConfig,
    target_nfe: int,
) -> DictConfig:
    """
    Build the config for one sampler / NFE point using root_cfg as the base.
    """
    target_nfe = int(target_nfe)
    if target_nfe < 1:
        raise ValueError(f"target_nfe must be >= 1, got {target_nfe}")

    # Start from the already-loaded table_results config.
    point_cfg = deepcopy(root_cfg)

    overrides = list(sampler_spec.overrides)
    sampler_path, remaining_overrides = _extract_sampler_override(overrides)

    # Load only the sampler config requested by this algorithm.
    sampler_cfg = hydra.compose(
        config_name=None,
        overrides=[f"+sampler={sampler_path}"],
    )

    point_cfg.sampler = sampler_cfg.sampler

    if remaining_overrides:
        point_cfg = OmegaConf.merge(
            point_cfg,
            OmegaConf.from_dotlist(remaining_overrides),
        )

    # Algorithm-specific runtime setting.
    if sampler_spec.get("num_particles") is not None:
        point_cfg.run.num_particles = int(sampler_spec.num_particles)

    sweep_kind = str(sampler_spec.sweep_kind)

    if sweep_kind == "mpgd":
        point_cfg.sampler.process_schedule.kwargs.num_steps = 1
        point_cfg.sampler.annealing.kwargs.num_steps = 1
        point_cfg.sampler.x0_solver.schedule.kwargs.num_steps = target_nfe

    elif sweep_kind in {"outer_inner", "atgd"}:
        inner_steps = int(point_cfg.sampler.x0_solver.schedule.kwargs.num_steps)
        if inner_steps < 1:
            raise ValueError(f"inner_steps must be >= 1, got {inner_steps}")

        if target_nfe % inner_steps != 0:
            raise ValueError(
                f"target_nfe={target_nfe} is not divisible by inner_steps={inner_steps} "
                f"for sampler '{sampler_spec.name}'."
            )

        outer_steps = target_nfe // inner_steps
        point_cfg.sampler.process_schedule.kwargs.num_steps = outer_steps
        point_cfg.sampler.annealing.kwargs.num_steps = outer_steps

        if sweep_kind == "atgd":
            if point_cfg.sampler.get("kwargs") is None or point_cfg.sampler.kwargs.get("stop") is None:
                raise ValueError(
                    f"Sampler '{sampler_spec.name}' uses sweep_kind='atgd' "
                    f"but sampler.kwargs.stop is missing."
                )

    else:
        raise ValueError(
            f"Unknown sweep_kind='{sweep_kind}' for sampler '{sampler_spec.name}'."
        )

    return point_cfg

def _slugify_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")

@torch.no_grad()
def _evaluate_time_ablation_variant(
    *,
    point_cfg: DictConfig,
    sampler_name: str,
    variant_cfg: DictConfig,
    sweep_value: int,
    save_curve_data: bool,
    samples_bk: torch.Tensor,
    evaluator,
    data: torch.Tensor,
    y: torch.Tensor | None,
    operator: Any,
    device: str,
    total_params: int,
    point_out_dir: Path,
    nfe: int,
    runtime_seconds_base: float,
    seconds_per_image_base: float,
    num_particles_sampled: int,
) -> Dict[str, Any]:
    """
    Evaluate one logical variant from a single completed base sampling run.

    Example variants:
      - k1_first
      - k4_best
      - k4_measurement_best
    """
    variant_name = str(variant_cfg.name)
    method_name = f"{sampler_name} | {variant_name}"

    use_particles = int(variant_cfg.use_particles)
    reduce_k = str(variant_cfg.reduce_k)
    time_mode = str(variant_cfg.time_mode)

    if use_particles < 1:
        raise ValueError(f"Variant '{variant_name}' requires use_particles >= 1, got {use_particles}")

    k_available = int(samples_bk.shape[1])
    if use_particles > k_available:
        raise ValueError(
            f"Variant '{variant_name}' for sampler '{sampler_name}' requires "
            f"use_particles={use_particles}, but only k_available={k_available} "
            f"particles were returned by the sampler."
        )

    samples_variant_bk = samples_bk[:, :use_particles]

    if time_mode == "full":
        runtime_seconds = runtime_seconds_base
        seconds_per_image = seconds_per_image_base
    elif time_mode == "proportional":
        if num_particles_sampled < 1:
            raise ValueError(f"num_particles_sampled must be >= 1, got {num_particles_sampled}")
        factor = float(use_particles) / float(num_particles_sampled)
        runtime_seconds = runtime_seconds_base * factor
        seconds_per_image = seconds_per_image_base * factor
    else:
        raise ValueError(
            f"Unknown time_mode='{time_mode}' for sampler '{sampler_name}', variant '{variant_name}'."
        )

    old_options = evaluator.options
    evaluator.options = EvaluatorOptions(
        reduce_k=reduce_k,
        store_per_condition=bool(save_curve_data),
    )

    try:
        evaluator.create_batch(
            B=int(data.shape[0]),
            K=int(samples_variant_bk.shape[1]),
            gen_bk=samples_variant_bk,
            gt_b=data,
            y_b=y,
            operator=operator,
            log_w_tbk=None,
            ess_tb=None,
        )

        result = evaluator.evaluate(
            timings={
                "sample_seconds": runtime_seconds,
                "seconds_per_image": seconds_per_image,
            },
            meta={
                "device": device,
                "total_params": int(total_params),
                "B": int(data.shape[0]),
                "sampler_name": sampler_name,
                "variant_name": variant_name,
                "method": method_name,
                "sweep_value": int(sweep_value),
                "num_particles_sampled": int(num_particles_sampled),
                "num_particles_used": int(use_particles),
                "reduce_k": reduce_k,
                "time_mode": time_mode,
            },
            run_paired=True,
            run_dist=False,
        )

    except Exception as e:
        print(
            f"Error in sampler={sampler_name}, variant={variant_name}, sweep_value={sweep_value}: {e}",
            flush=True,
        )
        raise

    finally:
        evaluator.reset_batch()
        evaluator.options = old_options

    metrics = result.metrics

    row: Dict[str, Any] = {
        "sampler": str(sampler_name),
        "variant": variant_name,
        "method": method_name,
        "sweep_value": int(sweep_value),
        "curve_source": bool(save_curve_data),
        "nfe": int(nfe),
        "num_particles": int(use_particles),
        "runtime_seconds": float(runtime_seconds),
        "seconds_per_image": float(seconds_per_image),
        "lpips_mean": float(metrics["lpips/mean"]),
        "lpips_median": float(metrics["lpips/median"]),
        "lpips_std": float(metrics["lpips/std"]),
        "lpips_min": float(metrics["lpips/min"]),
        "lpips_max": float(metrics["lpips/max"]),
        "psnr_mean": float(metrics["psnr/mean"]),
        "psnr_median": float(metrics["psnr/median"]),
        "psnr_std": float(metrics["psnr/std"]),
        "psnr_min": float(metrics["psnr/min"]),
        "psnr_max": float(metrics["psnr/max"]),
    }

    if "consistency_mse/mean" in metrics:
        row.update(
            {
                "consistency_mse_mean": float(metrics["consistency_mse/mean"]),
                "consistency_mse_median": float(metrics["consistency_mse/median"]),
                "consistency_mse_std": float(metrics["consistency_mse/std"]),
                "consistency_mse_min": float(metrics["consistency_mse/min"]),
                "consistency_mse_max": float(metrics["consistency_mse/max"]),
            }
        )

    variant_out_dir = point_out_dir / _slugify_name(variant_name)
    variant_out_dir.mkdir(parents=True, exist_ok=True)
    row["out_dir"] = str(variant_out_dir)

    payload = {
        "row": row,
        "metrics": result.metrics,
        "extra": result.extra,
    }
    save_json(variant_out_dir / "result.json", payload)

    summary_lines = [
        f"sampler: {row['sampler']}",
        f"variant: {row['variant']}",
        f"method: {row['method']}",
        f"sweep_value: {row['sweep_value']}",
        f"curve_source: {row['curve_source']}",
        f"nfe: {row['nfe']}",
        f"num_particles: {row['num_particles']}",
        f"runtime_seconds: {row['runtime_seconds']:.6f}",
        f"seconds_per_image: {row['seconds_per_image']:.6f}",
        f"lpips_mean: {row['lpips_mean']:.6g}",
        f"lpips_median: {row['lpips_median']:.6g}",
        f"lpips_std: {row['lpips_std']:.6g}",
        f"lpips_min: {row['lpips_min']:.6g}",
        f"lpips_max: {row['lpips_max']:.6g}",
        f"psnr_mean: {row['psnr_mean']:.6g}",
        f"psnr_median: {row['psnr_median']:.6g}",
        f"psnr_std: {row['psnr_std']:.6g}",
        f"psnr_min: {row['psnr_min']:.6g}",
        f"psnr_max: {row['psnr_max']:.6g}",
    ]

    if "consistency_mse_mean" in row:
        summary_lines.extend(
            [
                f"consistency_mse_mean: {row['consistency_mse_mean']:.6g}",
                f"consistency_mse_median: {row['consistency_mse_median']:.6g}",
                f"consistency_mse_std: {row['consistency_mse_std']:.6g}",
                f"consistency_mse_min: {row['consistency_mse_min']:.6g}",
                f"consistency_mse_max: {row['consistency_mse_max']:.6g}",
            ]
        )

    save_text(variant_out_dir / "result.txt", "\n".join(summary_lines))

    if save_curve_data:
        per_condition = {}
        if result.extra is not None:
            per_condition = dict(result.extra.get("per_condition", {}))

        curve_payload = {
            "sampler": str(sampler_name),
            "variant": variant_name,
            "method": method_name,
            "sweep_value": int(sweep_value),
            "nfe": int(nfe),
            "num_particles": int(use_particles),
            "runtime_seconds": float(runtime_seconds),
            "seconds_per_image": float(seconds_per_image),
            "reduce_k": reduce_k,
            "per_condition": per_condition,
        }
        save_json(variant_out_dir / "curve_metrics.json", curve_payload)

    return row

@torch.no_grad()
def run_image_sampler_time_ablation_point(
    *,
    point_cfg: DictConfig,
    sampler_spec: DictConfig,
    sampler_name: str,
    sweep_value: int,
    save_curve_data: bool,
    evaluator,
    model: torch.nn.Module,
    operator: Any,
    y: torch.Tensor | None,
    data: torch.Tensor,
    device: str,
    total_params: int,
    out_dir: Path,
    rng: Any,
) -> List[Dict[str, Any]]:
    """
    Run one sampler / sweep-value point for the time-based image ablation.

    One physical sampling run may produce multiple logical rows, depending on
    sampler_spec.eval_variants.

    Saves
    -----
    {out_dir}/{sampler_slug}/sweep_{sweep_value}/
        - config.yaml
        - {variant_slug}/result.json
        - {variant_slug}/result.txt
        - {variant_slug}/curve_metrics.json   (only if save_curve_data=True)
    """
    run_cfg = point_cfg.run

    sampler = build_sampler(point_cfg, model=model, operator=operator, device=device)

    batch_size = int(run_cfg.get("batch_size", 1000))
    total_data = int(run_cfg.get("total_data", data.shape[0]))
    num_particles_sampled = int(run_cfg.get("num_particles", 1))

    sync(device)
    t0 = time.perf_counter()
    if y is not None and total_data > batch_size:
        out = sampler.batch_sample(
            N=num_particles_sampled,
            y=y,
            B_chunk=batch_size,
            rng=rng,
            traj_steps=1,
            particles_for_traj=1,
        )
    else:
        out = sampler.sample(
            N=num_particles_sampled,
            y=y,
            rng=rng,
            traj_steps=1,
            particles_for_traj=1,
        )
    sync(device)
    t1 = time.perf_counter()

    runtime_seconds_base = float(t1 - t0)
    seconds_per_image_base = runtime_seconds_base / float(total_data)

    samples_bk = out.x0
    nfe = compute_nfe(point_cfg)

    sampler_slug = _slugify_name(sampler_name)
    point_out_dir = out_dir / sampler_slug / f"sweep_{int(sweep_value)}"
    point_out_dir.mkdir(parents=True, exist_ok=True)

    save_text(point_out_dir / "config.yaml", OmegaConf.to_yaml(point_cfg, resolve=True))

    rows: List[Dict[str, Any]] = []

    for variant_cfg in sampler_spec.eval_variants:
        row = _evaluate_time_ablation_variant(
            point_cfg=point_cfg,
            sampler_name=sampler_name,
            variant_cfg=variant_cfg,
            sweep_value=int(sweep_value),
            save_curve_data=bool(save_curve_data),
            samples_bk=samples_bk,
            evaluator=evaluator,
            data=data,
            y=y,
            operator=sampler.ctx.operator,
            device=device,
            total_params=total_params,
            point_out_dir=point_out_dir,
            nfe=nfe,
            runtime_seconds_base=runtime_seconds_base,
            seconds_per_image_base=seconds_per_image_base,
            num_particles_sampled=num_particles_sampled,
        )
        rows.append(row)

    return rows

def print_time_ablation_row(row: Dict[str, Any]) -> None:
    """
    Print one concise progress line for a completed time-ablation row.
    """
    msg = (
        f"[{row['method']}] "
        f"sweep={row['sweep_value']} "
        f"nfe={row['nfe']} "
        f"num_particles={row['num_particles']} "
        f"time/img={row['seconds_per_image']:.6f}s "
        f"total={row['runtime_seconds']:.2f}s | "
        f"LPIPS mean={row['lpips_mean']:.6g} "
        f"median={row['lpips_median']:.6g} "
        f"std={row['lpips_std']:.6g} "
        f"min={row['lpips_min']:.6g} "
        f"max={row['lpips_max']:.6g} | "
        f"PSNR mean={row['psnr_mean']:.6g} "
        f"median={row['psnr_median']:.6g} "
        f"std={row['psnr_std']:.6g} "
        f"min={row['psnr_min']:.6g} "
        f"max={row['psnr_max']:.6g}"
    )

    if "consistency_mse_mean" in row:
        msg += (
            f" | ConsistencyMSE mean={row['consistency_mse_mean']:.6g} "
            f"median={row['consistency_mse_median']:.6g} "
            f"std={row['consistency_mse_std']:.6g} "
            f"min={row['consistency_mse_min']:.6g} "
            f"max={row['consistency_mse_max']:.6g}"
        )

    print(msg, flush=True)