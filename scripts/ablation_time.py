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
from src.experiments.image_ablations import (
    build_ablation_point_cfg,
    print_time_ablation_row,
    run_image_sampler_time_ablation_point,
)
from src.factory import build_operator_and_y
from src.io import (
    format_sampler_time_ablation_table,
    plot_sampler_time_ablation,
    save_json,
    save_text,
)
from src.models import get_model
from src.models.utils import count_parameters
from src.seed import make_rng


@hydra.main(version_base=None, config_path="../configs", config_name="image_configs/ffhq_pr/time_sweep")
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
            reduce_k=str(cfg.eval.reduce_k),
            store_per_condition=bool(cfg.eval.store_per_condition),
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

    rows: List[Dict[str, Any]] = []

    for sampler_spec in cfg.samplers:
        sweep_values = [int(x) for x in sampler_spec.sweep_values]
        curve_sweep_value = int(sampler_spec.curve_sweep_value)

        for sweep_value in sweep_values:
            point_cfg = build_ablation_point_cfg(
                root_cfg=cfg,
                sampler_spec=sampler_spec,
                target_nfe=sweep_value,
            )

            rng = make_rng(int(cfg.run.seed), device=device)
            save_curve_data = (sweep_value == curve_sweep_value)

            point_rows = run_image_sampler_time_ablation_point(
                point_cfg=point_cfg,
                sampler_spec=sampler_spec,
                sampler_name=str(sampler_spec.name),
                sweep_value=sweep_value,
                save_curve_data=save_curve_data,
                evaluator=evaluator,
                model=model,
                operator=operator,
                y=y,
                data=data,
                device=device,
                total_params=total_params,
                out_dir=out_dir,
                rng=rng,
            )

            rows.extend(point_rows)
            for row in point_rows:
                print_time_ablation_row(row)

            save_json(out_dir / str(cfg.outputs.partial_json), {"results": rows})
            cleanup_cuda(device=device)

    save_json(out_dir / str(cfg.outputs.summary_json), {"results": rows})
    save_text(out_dir / str(cfg.outputs.table_txt), format_sampler_time_ablation_table(rows))

    plot_sampler_time_ablation(
        rows=rows,
        metric="lpips",
        out_path=out_dir / str(cfg.outputs.lpips_time_svg),
        xscale=str(cfg.plot.xscale),
        summary_stat=str(cfg.report.summary_stat),
        error_bars=str(cfg.plot.error_bars),
        x_jitter=float(cfg.plot.x_jitter),
    )
    plot_sampler_time_ablation(
        rows=rows,
        metric="psnr",
        out_path=out_dir / str(cfg.outputs.psnr_time_svg),
        xscale=str(cfg.plot.xscale),
        summary_stat=str(cfg.report.summary_stat),
        error_bars=str(cfg.plot.error_bars),
        x_jitter=float(cfg.plot.x_jitter),
    )

if __name__ == "__main__":
    main()