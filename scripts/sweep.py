from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.cfg import cleanup_cuda, set_device
from src.datasets import get_dataset
from src.eval.evaluator import Evaluator, EvaluatorOptions
from src.eval.metrics import build_metrics
from src.experiments.image_ablations import build_ablation_point_cfg
from src.experiments.image_sweeps import (
    rank_images,
    run_image_sampler_point,
)
from src.factory import build_operator_and_y
from src.io import save_examples, save_json, save_text
from src.models import get_model
from src.models.utils import count_parameters
from src.seed import make_rng

def _print_algorithm_result(result: Dict[str, Any]) -> None:
    metrics = result["summary_metrics"]

    msg = (
        f"[{result['name']}] "
        f"selection_rule={result['selection_rule']} "
        f"nfe={result['nfe']} "
        f"K={result['num_particles_sampled']} "
        f"time={result['runtime_seconds']:.2f}s "
        f"time/img={result['seconds_per_image']:.6f}s | "
        f"LPIPS mean={metrics['lpips/mean']:.6g} "
        f"median={metrics['lpips/median']:.6g} "
        f"std={metrics['lpips/std']:.6g} | "
        f"PSNR mean={metrics['psnr/mean']:.6g} "
        f"median={metrics['psnr/median']:.6g} "
        f"std={metrics['psnr/std']:.6g}"
    )

    if "ssim/mean" in metrics:
        msg += (
            f" | SSIM mean={metrics['ssim/mean']:.6g} "
            f"median={metrics['ssim/median']:.6g} "
            f"std={metrics['ssim/std']:.6g}"
        )

    if "consistency_mse/mean" in metrics:
        msg += (
            f" | ConsistencyMSE mean={metrics['consistency_mse/mean']:.6g} "
            f"median={metrics['consistency_mse/median']:.6g} "
            f"std={metrics['consistency_mse/std']:.6g}"
        )

    print(msg, flush=True)

@hydra.main(version_base=None, config_path="../configs", config_name="image_configs/ffhq_pr/table_results")
def main(cfg: DictConfig) -> None:
    out_dir = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    root_cfg_name = str(cfg.outputs.get("root_config_yaml"))
    save_text(out_dir / root_cfg_name, OmegaConf.to_yaml(cfg, resolve=True))

    device = set_device(cfg=cfg)

    paired_metrics, dist_metrics = build_metrics(cfg.eval)
    evaluator = Evaluator(
        device=device,
        paired_metrics=paired_metrics,
        dist_metrics=dist_metrics,
        options=EvaluatorOptions(
            reduce_k="mean",
            store_per_condition=False,
        ),
    )

    model = get_model(cfg.model, device=device)
    total_params = count_parameters(model)

    total_data = int(cfg.run.total_data)
    data = get_dataset(cfg.data).get_data(size=total_data)

    operator_rng = make_rng(int(cfg.run.seed), device=device)
    operator, y = build_operator_and_y(
        cfg.operator,
        x_ref=data,
        generator=operator_rng.operator,
        device=device,
    )

    algorithm_results: List[Dict[str, Any]] = []

    for alg_cfg in cfg.algorithms:
        point_cfg = build_ablation_point_cfg(
            root_cfg=cfg,
            sampler_spec=alg_cfg,
            target_nfe=int(alg_cfg.sweep_value)
        )

        rng = make_rng(int(cfg.run.seed), device=device)

        result = run_image_sampler_point(
            point_cfg=point_cfg,
            algorithm_name=str(alg_cfg.name),
            selection_rule=str(alg_cfg.selection_rule),
            evaluator=evaluator,
            model=model,
            operator=operator,
            y=y,
            data=data,
            device=device,
            total_params=total_params,
            rng=rng,
        )

        algorithm_results.append(result)
        _print_algorithm_result(result)
        cleanup_cuda(device=device)

    ranked = rank_images(
        algorithm_results=algorithm_results,
        rank_pair=[str(x) for x in cfg.pick.rank_pair],
        rank_metric=str(cfg.pick.rank_metric),
        top_k=int(cfg.pick.top_k),
    )

    summary_payload = {
        "rank_metric": str(cfg.pick.rank_metric),
        "rank_pair": [str(x) for x in cfg.pick.rank_pair],
        "top_k": int(cfg.pick.top_k),
        "ranked_examples": ranked,
        "algorithms": [
            {
                "name": r["name"],
                "selection_rule": r["selection_rule"],
                "nfe": r["nfe"],
                "num_particles_sampled": r["num_particles_sampled"],
                "runtime_seconds": r["runtime_seconds"],
                "seconds_per_image": r["seconds_per_image"],
                "summary_metrics": r["summary_metrics"],
            }
            for r in algorithm_results
        ],
    }
    save_json(out_dir / str(cfg.outputs.summary_json), summary_payload)

    save_examples(
        out_dir=out_dir,
        ranked_examples=ranked,
        algorithm_results=algorithm_results,
        gt_b=data,
        y_b=y,
        rank_metric=str(cfg.pick.rank_metric),
        rank_pair=[str(x) for x in cfg.pick.rank_pair],
    )


if __name__ == "__main__":
    main()