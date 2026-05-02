from __future__ import annotations

from pathlib import Path

import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.cfg import set_device
from src.datasets import get_dataset
from src.eval.evaluator import Evaluator, EvaluatorOptions
from src.eval.metrics import build_metrics
from src.factory import build_operator_and_y
from src.io import (save_json, 
                    save_text, 
                    format_particle_sweep_table,
                    format_fixed_particle_table
)
from src.models import get_model
from src.seed import make_rng
from src.experiments.toy2d import (
    run_particle_sweep_experiment,
    build_particle_sweep_table,
    build_fixed_particle_table,
)
from src.viz.toy2d import (
    plot_metric_vs_particles,
    plot_prior_context_posterior_zoom,
)

BASE_CONFIG_NAME = "2d_configs/sweeps"

@hydra.main(version_base=None, config_path="../configs", config_name=BASE_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    out_dir = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shared setup
    # ------------------------------------------------------------------
    device = set_device(cfg=cfg.run)
    rng = make_rng(cfg.run.seed, device=device)

    model = get_model(cfg.model, device=device, generator=rng.model)

    torch.manual_seed(cfg.run.seed)

    total_data = int(cfg.run.total_data)
    dataset = get_dataset(cfg.data, model=model, N=total_data)
    data = dataset.get_data(total_data)

    operator, y = build_operator_and_y(
        cfg.operator,
        x_ref=data,
        generator=rng.operator,
        device=device,
    )

    gt_b = dataset.build_ground_truth_conditionals(
        y=y,
        operator=operator,
        rng_cpu=rng.cpu,
        N=int(cfg.study.gt_reference_samples),
    )

    paired_metrics, dist_metrics = build_metrics(cfg.eval)
    evaluator = Evaluator(
        device=device,
        paired_metrics=paired_metrics,
        dist_metrics=dist_metrics,
        options=EvaluatorOptions(
            reduce_k=str(cfg.eval.get("reduce_k", "mean")),
            store_per_condition=bool(cfg.eval.get("store_per_condition", False)),
        ),
    )

    # Save resolved config for reproducibility.
    save_json(out_dir / "resolved_config.json", {"cfg": cfg})

    # ------------------------------------------------------------------
    # 1) Particle sweep across algorithms
    # ------------------------------------------------------------------
    particle_results = []

    for override in cfg.sampler_overrides:
        print(override)
        result = run_particle_sweep_experiment(
            base_config_name=BASE_CONFIG_NAME,
            override=str(override),
            sweep_cfg=cfg,
            model=model,
            gt_b=gt_b,
            evaluator=evaluator,
            operator=operator,
            y=y,
            device=device,
            out_dir=out_dir,
        )
        particle_results.append(result)

    particle_sweep_table = build_particle_sweep_table(particle_results)
    save_text(out_dir / "particle_sweep.txt", format_particle_sweep_table(particle_sweep_table))

    fixed_particle_table = build_fixed_particle_table(
        particle_results,
        num_particles=int(cfg.study.table_num_particles),
    )
    save_text(out_dir / "fixed_particle_table.txt", format_fixed_particle_table(fixed_particle_table))

    plot_metric_vs_particles(
        particle_results,
        metric="swd",
        particle_counts=cfg.study.particle_counts,
        savepath=out_dir / "swd_vs_particles.png",
    )
    # ------------------------------------------------------------------
    # 2) Prior context + posterior zoom figures
    # ------------------------------------------------------------------
    posterior_overlay_cfg = cfg.study.posterior_overlay

    if bool(posterior_overlay_cfg.enabled):
        prior_reference_samples = int(posterior_overlay_cfg.prior_reference_samples)

        prior_dataset = get_dataset(
            cfg.data,
            model=model,
            N=prior_reference_samples,
        )

        prior_reference = prior_dataset.get_data(prior_reference_samples).detach().cpu()

        plot_prior_context_posterior_zoom(
            particle_results=particle_results,
            gt_b=gt_b,
            prior_samples=prior_reference,
            true_x_b=data.detach().cpu(),
            algorithm=str(posterior_overlay_cfg.algorithm),
            num_particles=int(posterior_overlay_cfg.num_particles),
            max_prior_samples=int(posterior_overlay_cfg.max_prior_samples),
            max_posterior_samples=int(posterior_overlay_cfg.max_posterior_samples),
            grid_size_global=int(posterior_overlay_cfg.grid_size_global),
            grid_size_zoom=int(posterior_overlay_cfg.grid_size_zoom),
            prior_kde_bandwidth=float(posterior_overlay_cfg.prior_kde_bandwidth),
            posterior_kde_bandwidth=float(posterior_overlay_cfg.posterior_kde_bandwidth),
            global_padding_frac=float(posterior_overlay_cfg.global_padding_frac),
            zoom_padding_frac=float(posterior_overlay_cfg.zoom_padding_frac),
            output_dir=out_dir / "prior_context_posterior_zoom",
            output_formats=list(posterior_overlay_cfg.output_formats),
        )

if __name__ == "__main__":
    main()