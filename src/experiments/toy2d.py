from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig

from src.factory import build_sampler
from src.io import save_json
from src.seed import make_rng

def run_particle_sweep_experiment(
    *,
    base_config_name: str,
    override: str,
    sweep_cfg: DictConfig,
    model: torch.nn.Module,
    gt_b: torch.Tensor,
    evaluator: Any,
    operator: Any,
    y: torch.Tensor,
    device: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Run the particle-count sweep for a single algorithm override.
    """
    algorithm = Path(override.split("=", 1)[1].strip()).name

    particle_counts = list(sweep_cfg.study.particle_counts)
    target_total_particles = int(sweep_cfg.study.target_total_particles)

    posterior_overlay_cfg = sweep_cfg.study.get("posterior_overlay", None)

    should_use_posterior_overlay = (
        posterior_overlay_cfg is not None
        and bool(posterior_overlay_cfg.get("enabled", False))
    )

    posterior_overlay_algorithm = (
        str(posterior_overlay_cfg.get("algorithm", "tgd"))
        if should_use_posterior_overlay
        else None
    )
    posterior_overlay_num_particles = (
        int(posterior_overlay_cfg.get("num_particles", 64))
        if should_use_posterior_overlay
        else None
    )

    algorithm_out_dir = out_dir / "particle_sweep" / algorithm
    algorithm_out_dir.mkdir(parents=True, exist_ok=True)

    algorithm_results: List[Dict[str, Any]] = []

    for num_particles in particle_counts:
        print(f"Num particles:{num_particles}")
        logical_k = int(num_particles)

        # Optional K=1 speed hack
        actual_k = logical_k
        extra_overrides = [override]
        if logical_k == 1 and bool(sweep_cfg.study.k1_hack.enabled):
            actual_k = int(sweep_cfg.study.k1_hack.actual_particles)
            extra_overrides.append(f"run.num_particles={actual_k}")
            extra_overrides.append(
                f"sampler.kwargs.ess_threshold={float(sweep_cfg.study.k1_hack.ess_threshold)}"
            )
        else:
            extra_overrides.append(f"run.num_particles={actual_k}")

        exp_cfg = hydra.compose(
            config_name=base_config_name,
            overrides=extra_overrides,
        )

        rng = make_rng(exp_cfg.run.seed, device=device)
        sampler = build_sampler(exp_cfg, model=model, operator=operator, device=device)

        n_runs = (target_total_particles + actual_k - 1) // actual_k
        pooled_samples = []

        traj_chunks = []
        traj_times = None
        collected_traj_particles = 0

        should_collect_posterior_overlay = (
            should_use_posterior_overlay
            and algorithm == posterior_overlay_algorithm
            and logical_k == posterior_overlay_num_particles
        )
        posterior_overlay_particles_bk = None

        for run_idx in tqdm(
            range(n_runs),
            desc=f"[{algorithm}] K={logical_k} (actual={actual_k}) pooling",
            leave=False,
        ):

            particles_for_traj = 0

            if y is not None and int(exp_cfg.run.total_data) > int(exp_cfg.run.batch_size):
                out = sampler.batch_sample(
                    N=actual_k,
                    y=y,
                    B_chunk=int(exp_cfg.run.batch_size),
                    rng=rng,
                    traj_steps=0,
                    particles_for_traj=0,
                )
            else:
                out = sampler.sample(
                    N=actual_k,
                    y=y,
                    rng=rng,
                    traj_steps=0,
                    particles_for_traj=0,
                )

            x0_bk = out.x0.detach().cpu()

            if should_collect_posterior_overlay and posterior_overlay_particles_bk is None:
                # Keep one actual final N=64 particle cloud for every condition y.
                # Expected shape: (B, 64, D)
                posterior_overlay_particles_bk = x0_bk

            pooled_samples.append(x0_bk)

        samples_bk = torch.cat(pooled_samples, dim=1)[:, :target_total_particles]

        trajectory_payload = None

        posterior_overlay_payload = None
        if posterior_overlay_particles_bk is not None:
            posterior_overlay_path = (
                algorithm_out_dir
                / f"posterior_overlay_k_{logical_k}_particles_bk.pt"
            )

            torch.save(
                {
                    "algorithm": algorithm,
                    "num_particles": int(logical_k),
                    "particles_bk": posterior_overlay_particles_bk,
                },
                posterior_overlay_path,
            )

            posterior_overlay_payload = {
                "algorithm": algorithm,
                "num_particles": int(logical_k),
                "particles_bk": posterior_overlay_particles_bk,
                "particles_path": str(posterior_overlay_path),
                "num_conditions": int(posterior_overlay_particles_bk.shape[0]),
            }

        evaluator.create_batch(
            B=samples_bk.shape[0],
            K=samples_bk.shape[1],
            gen_bk=samples_bk,
            gt_b=gt_b,
            y_b=y,
            operator=sampler.ctx.operator,
            log_w_tbk=None,
            ess_tb=None,
        )

        result = evaluator.evaluate(
            ref=gt_b,
            gen_for_dist=samples_bk,
            timings={},
            meta={
                "device": device,
                "B": int(samples_bk.shape[0]),
                "K": int(samples_bk.shape[1]),
                "algorithm": algorithm,
                "exp_config_path": override,
            },
            run_paired=True,
            run_dist=True,
        )
        evaluator.reset_batch()

        result_entry: Dict[str, Any] = {
            "num_particles": num_particles,
            "metrics": dict(result.metrics),
            "extra": dict(result.extra),
            "trajectory": trajectory_payload,
            "posterior_overlay": posterior_overlay_payload,
            "meta": {
                "K_pooled": samples_bk.shape[1],
                "K_logical": logical_k,
                "K_actual": actual_k,
            },
        }

        save_json(
            algorithm_out_dir / f"k_{num_particles}.json",
            {
                "num_particles": result_entry["num_particles"],
                "metrics": result_entry["metrics"],
                "extra": result_entry["extra"],
                "meta": result_entry["meta"],
            },
        )

        algorithm_results.append(result_entry)

    summary = {
        "algorithm": algorithm,
        "override": override,
        "results": algorithm_results,
    }

    save_json(
        algorithm_out_dir / "summary.json",
        {
            "algorithm": algorithm,
            "override": override,
            "results": [
                {
                    "num_particles": r["num_particles"],
                    "metrics": r["metrics"],
                    "extra": r["extra"],
                    "meta": r["meta"],
                }
                for r in algorithm_results
            ],
        },
    )

    return summary

def build_particle_sweep_table(
    particle_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten particle sweep outputs into a CSV-friendly table.

    Input:
        particle_results: output list from repeated calls to
            run_particle_sweep_experiment(...)

    Output:
        List of rows, one per (algorithm, num_particles), e.g.
        [
            {
                "algorithm": "daps",
                "num_particles": 1,
                "swd_mean": ...,
                "swd_std": ...,
                "consistency_mse_mean": ...,
                "consistency_mse_std": ...,
            },
            ...
        ]
    """
    rows: List[Dict[str, Any]] = []

    for algorithm_result in particle_results:
        algorithm = algorithm_result["algorithm"]

        for run_result in algorithm_result["results"]:
            metrics = run_result["metrics"]

            row = {
                "algorithm": algorithm,
                "num_particles": int(run_result["num_particles"]),
                "swd_mean": float(metrics["swd/mean"]),
                "swd_std": float(metrics["swd/std"]),
                "consistency_mse_mean": float(metrics["consistency_mse/mean"]),
                "consistency_mse_std": float(metrics["consistency_mse/std"]),
            }

            rows.append(row)

    rows.sort(key=lambda r: (r["algorithm"], r["num_particles"]))
    return rows

def build_fixed_particle_table(
    particle_results: List[Dict[str, Any]],
    *,
    num_particles: int,
) -> List[Dict[str, Any]]:
    """
    Build a flat table for a single fixed particle count across all algorithms.

    Input:
        particle_results: output list from repeated calls to
            run_particle_sweep_experiment(...)

    Output:
        [
            {
                "algorithm": str,
                "num_particles": int,
                "swd_mean": float,
                "swd_std": float,
                "consistency_mse_mean": float,
                "consistency_mse_std": float,
            },
            ...
        ]
    """
    rows: List[Dict[str, Any]] = []

    for algorithm_result in particle_results:
        algorithm = str(algorithm_result["algorithm"])

        matched = None
        for run_result in algorithm_result["results"]:
            if int(run_result["num_particles"]) == int(num_particles):
                matched = run_result
                break

        if matched is None:
            raise ValueError(
                f"Could not find num_particles={num_particles} for algorithm '{algorithm}'."
            )

        metrics = matched["metrics"]
        rows.append(
            {
                "algorithm": algorithm,
                "num_particles": int(num_particles),
                "swd_mean": float(metrics["swd/mean"]),
                "swd_std": float(metrics["swd/std"]),
                "consistency_mse_mean": float(metrics["consistency_mse/mean"]),
                "consistency_mse_std": float(metrics["consistency_mse/std"]),
            }
        )

    rows.sort(key=lambda r: r["algorithm"])
    return rows