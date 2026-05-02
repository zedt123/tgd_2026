from __future__ import annotations

from typing import List, Dict, Any, Sequence, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

import logging
import matplotlib as mpl

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
mpl.set_loglevel("warning")

Array = np.ndarray

def plot_metric_vs_particles(
    particle_results: List[Dict[str, Any]],
    *,
    metric: str,
    particle_counts: Sequence[int],
    savepath: Path,
) -> None:
    """
    Plot a metric against number of particles for all algorithms.

    The plot uses:
      - y = metrics[f"{metric}/mean"]
      - shaded band = [metrics[f"{metric}/min"], metrics[f"{metric}/max"]]  

    Args:
        particle_results: output of repeated calls to
            run_particle_sweep_experiment(...)
        metric: metric name, e.g. "swd" or "consistency_mse"
        savepath: output PNG path
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    ylabel_map = {
        "swd": "SWD",
        "consistency_mse": "Consistency MSE",
    }
    ylabel = ylabel_map.get(metric, metric)

    particle_counts = [int(x) for x in particle_counts]

    plt.figure(figsize=(6, 4))

    for algorithm_result in particle_results:
        algorithm = str(algorithm_result["algorithm"])
        run_results = sorted(
            algorithm_result["results"],
            key=lambda r: int(r["num_particles"]),
        )

        xs, ys, lower_err, upper_err = [], [], [], []

        for run_result in run_results:
            m = run_result["metrics"]
            x = int(run_result["num_particles"])
            median = float(m[f"{metric}/mean"])
            vmin = float(m[f"{metric}/min"])
            vmax = float(m[f"{metric}/max"])

            xs.append(x)
            ys.append(median)
            lower_err.append(max(0.0, median - vmin))
            upper_err.append(max(0.0, vmax - median))

        plt.errorbar(
            xs,
            ys,
            yerr=[lower_err, upper_err],  # asymmetric error bars
            marker="o",
            capsize=3,
            label=algorithm,
        )

    plt.xscale("log", base=2)
    plt.xticks(particle_counts, [str(x) for x in particle_counts])

    plt.xlabel("Number of particles")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def _find_particles_bk_for_overlay(
    *,
    particle_results: List[Dict[str, Any]],
    algorithm: str,
    num_particles: int,
) -> torch.Tensor:
    matched_algorithm = None
    for algorithm_result in particle_results:
        if str(algorithm_result["algorithm"]) == str(algorithm):
            matched_algorithm = algorithm_result
            break

    if matched_algorithm is None:
        raise ValueError(f"Could not find algorithm='{algorithm}' in particle_results.")

    matched_run = None
    for run_result in matched_algorithm["results"]:
        if int(run_result["num_particles"]) == int(num_particles):
            matched_run = run_result
            break

    if matched_run is None:
        raise ValueError(
            f"Could not find num_particles={num_particles} for algorithm='{algorithm}'."
        )

    posterior_overlay = matched_run.get("posterior_overlay", None)
    if posterior_overlay is None:
        raise ValueError(
            f"No posterior_overlay payload found for algorithm='{algorithm}', "
            f"num_particles={num_particles}. Make sure study.posterior_overlay is enabled."
        )

    if "particles_bk" in posterior_overlay and posterior_overlay["particles_bk"] is not None:
        return posterior_overlay["particles_bk"].detach().cpu()

    particles_path = posterior_overlay.get("particles_path", None)
    if particles_path is None:
        raise ValueError("posterior_overlay has neither particles_bk nor particles_path.")

    payload = torch.load(particles_path, map_location="cpu")
    return payload["particles_bk"].detach().cpu()


def _limit_samples(samples: torch.Tensor, max_samples: int) -> torch.Tensor:
    samples = samples.detach().cpu()
    if samples.shape[0] <= int(max_samples):
        return samples
    return samples[: int(max_samples)]


def _sample_limits(
    samples: torch.Tensor,
    *,
    padding_frac: float,
) -> Tuple[float, float, float, float]:
    mins = samples.min(dim=0).values
    maxs = samples.max(dim=0).values

    x_min = float(mins[0])
    y_min = float(mins[1])
    x_max = float(maxs[0])
    y_max = float(maxs[1])

    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)

    x_pad = float(padding_frac) * x_range
    y_pad = float(padding_frac) * y_range

    return (
        x_min - x_pad,
        x_max + x_pad,
        y_min - y_pad,
        y_max + y_pad,
    )


def _kde_density_grid(
    *,
    samples: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int,
    bandwidth: float,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = samples.detach().cpu()

    x_grid = torch.linspace(float(x_min), float(x_max), int(grid_size))
    y_grid = torch.linspace(float(y_min), float(y_max), int(grid_size))

    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    density_chunks = []
    normalizer = 2.0 * float(bandwidth) * float(bandwidth)

    for start in range(0, grid.shape[0], int(chunk_size)):
        grid_chunk = grid[start : start + int(chunk_size)]
        dist2 = torch.cdist(grid_chunk, samples).pow(2)
        density_chunk = torch.exp(-dist2 / normalizer).mean(dim=1)
        density_chunks.append(density_chunk)

    density = torch.cat(density_chunks, dim=0).reshape(
        int(grid_size),
        int(grid_size),
    )

    return xx, yy, density


def _positive_contour_levels(
    density: torch.Tensor,
    *,
    num_levels: int,
    min_frac_of_max: float,
) -> Optional[torch.Tensor]:
    density_max = float(density.max())
    if density_max <= 0.0:
        return None

    low = float(min_frac_of_max) * density_max
    high = density_max

    if low >= high:
        return None

    return torch.linspace(low, high, int(num_levels))

def _abs_clean_sign_flip_points(
    true_x: torch.Tensor,
) -> torch.Tensor:
    true_x = true_x.detach().cpu().flatten()

    if true_x.numel() != 2:
        raise ValueError(
            f"Expected a 2D clean point, got shape {tuple(true_x.shape)}."
        )

    magnitude = true_x.abs()

    signs = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=magnitude.dtype,
    )

    return signs * magnitude.view(1, 2)

def plot_prior_context_posterior_zoom(
    *,
    particle_results: List[Dict[str, Any]],
    gt_b: torch.Tensor,
    prior_samples: torch.Tensor,
    true_x_b: torch.Tensor,
    algorithm: str,
    num_particles: int,
    max_prior_samples: int,
    max_posterior_samples: int,
    grid_size_global: int,
    grid_size_zoom: int,
    prior_kde_bandwidth: float,
    posterior_kde_bandwidth: float,
    global_padding_frac: float,
    zoom_padding_frac: float,
    output_dir: Path,
    output_formats: Sequence[str] = ("svg", "pdf"),
) -> List[Path]:
    """
    Save one two-panel paper-quality figure per condition y.

    Left:
        Global prior context, zoom box, true x, and final TGD particles.

    Right:
        Local posterior contours from gt_b[y], true x, and final TGD particles.
    """
    particles_bk = _find_particles_bk_for_overlay(
        particle_results=particle_results,
        algorithm=algorithm,
        num_particles=num_particles,
    )

    reference_bk = gt_b.detach().cpu()
    prior_samples = prior_samples.detach().cpu()
    true_x_b = true_x_b.detach().cpu()

    if reference_bk.ndim != 3 or particles_bk.ndim != 3:
        raise ValueError("Expected gt_b and particles_bk to have shape (B, K, D).")

    if prior_samples.ndim != 2:
        raise ValueError("Expected prior_samples to have shape (K, D).")

    if true_x_b.ndim != 2:
        raise ValueError("Expected true_x_b to have shape (B, D).")

    if reference_bk.shape[0] != particles_bk.shape[0]:
        raise ValueError(
            f"Mismatched number of conditions: gt_b has B={reference_bk.shape[0]}, "
            f"particles_bk has B={particles_bk.shape[0]}."
        )

    if true_x_b.shape[0] != reference_bk.shape[0]:
        raise ValueError(
            f"Mismatched number of conditions: true_x_b has B={true_x_b.shape[0]}, "
            f"gt_b has B={reference_bk.shape[0]}."
        )

    if reference_bk.shape[-1] != 2 or particles_bk.shape[-1] != 2 or prior_samples.shape[-1] != 2:
        raise ValueError(
            "plot_prior_context_posterior_zoom expects 2D samples with final dimension D=2."
        )

    prior_for_kde = _limit_samples(prior_samples, int(max_prior_samples))

    global_x_min, global_x_max, global_y_min, global_y_max = _sample_limits(
        prior_for_kde,
        padding_frac=float(global_padding_frac),
    )

    prior_xx, prior_yy, prior_density = _kde_density_grid(
        samples=prior_for_kde,
        x_min=global_x_min,
        x_max=global_x_max,
        y_min=global_y_min,
        y_max=global_y_max,
        grid_size=int(grid_size_global),
        bandwidth=float(prior_kde_bandwidth),
    )

    prior_levels = _positive_contour_levels(
        prior_density,
        num_levels=8,
        min_frac_of_max=0.05,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    num_conditions = int(reference_bk.shape[0])

    rc = {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "contour.negative_linestyle": "solid",
    }

    with plt.rc_context(rc):
        for y_index in range(num_conditions):
            posterior = _limit_samples(reference_bk[y_index], int(max_posterior_samples))
            particles = particles_bk[y_index].detach().cpu()
            true_x = true_x_b[y_index].detach().cpu()

            sign_flip_points = _abs_clean_sign_flip_points(true_x)

            local_samples = torch.cat([posterior, particles, true_x.view(1, 2)], dim=0)
            zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = _sample_limits(
                local_samples,
                padding_frac=float(zoom_padding_frac),
            )

            post_xx_global, post_yy_global, post_density_global = _kde_density_grid(
                samples=posterior,
                x_min=global_x_min,
                x_max=global_x_max,
                y_min=global_y_min,
                y_max=global_y_max,
                grid_size=int(grid_size_global),
                bandwidth=float(posterior_kde_bandwidth),
            )

            post_global_levels = _positive_contour_levels(
                post_density_global,
                num_levels=4,
                min_frac_of_max=0.20,
            )

            post_xx_zoom, post_yy_zoom, post_density_zoom = _kde_density_grid(
                samples=posterior,
                x_min=zoom_x_min,
                x_max=zoom_x_max,
                y_min=zoom_y_min,
                y_max=zoom_y_max,
                grid_size=int(grid_size_zoom),
                bandwidth=float(posterior_kde_bandwidth),
            )

            post_zoom_levels = _positive_contour_levels(
                post_density_zoom,
                num_levels=8,
                min_frac_of_max=0.05,
            )

            fig, axes = plt.subplots(
                ncols=2,
                figsize=(6.3, 2.75),
                constrained_layout=False,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.995,
                bottom=0.27,
                top=0.88,
                wspace=0.34,
            )
            ax_global, ax_zoom = axes

            # ----------------------------------------------------------
            # Left: global prior context
            # ----------------------------------------------------------
            if prior_levels is not None:
                ax_global.contour(
                    prior_xx.numpy(),
                    prior_yy.numpy(),
                    prior_density.numpy(),
                    levels=prior_levels.numpy(),
                    colors="0.45",
                    linewidths=0.8,
                    alpha=0.8,
                )

            if post_global_levels is not None:
                ax_global.contour(
                    post_xx_global.numpy(),
                    post_yy_global.numpy(),
                    post_density_global.numpy(),
                    levels=post_global_levels.numpy(),
                    colors="C0",
                    linewidths=1.1,
                )

            ax_global.scatter(
                particles[:, 0].numpy(),
                particles[:, 1].numpy(),
                s=13,
                marker="x",
                linewidths=0.75,
                color="C1",
                zorder=4,
            )

            ax_global.scatter(
                sign_flip_points[:, 0].numpy(),
                sign_flip_points[:, 1].numpy(),
                s=54,
                marker="*",
                color="black",
                edgecolors="white",
                linewidths=0.45,
                zorder=6,
            )

            zoom_rect = patches.Rectangle(
                (zoom_x_min, zoom_y_min),
                zoom_x_max - zoom_x_min,
                zoom_y_max - zoom_y_min,
                fill=False,
                linewidth=0.9,
                linestyle="--",
                edgecolor="black",
                alpha=0.8,
            )
            ax_global.add_patch(zoom_rect)

            ax_global.set_title(r"(a) Prior context", pad=3)
            ax_global.set_xlabel("$x_1$")
            ax_global.set_ylabel("$x_2$")
            ax_global.set_xlim(global_x_min, global_x_max)
            ax_global.set_ylim(global_y_min, global_y_max)
            ax_global.set_aspect("equal", adjustable="box")

            # ----------------------------------------------------------
            # Right: local posterior zoom
            # ----------------------------------------------------------
            if post_zoom_levels is not None:
                ax_zoom.contour(
                    post_xx_zoom.numpy(),
                    post_yy_zoom.numpy(),
                    post_density_zoom.numpy(),
                    levels=post_zoom_levels.numpy(),
                    colors="C0",
                    linewidths=1.0,
                )

            ax_zoom.scatter(
                particles[:, 0].numpy(),
                particles[:, 1].numpy(),
                s=15,
                marker="x",
                linewidths=0.75,
                color="C1",
                zorder=4,
            )

            ax_zoom.scatter(
                [float(true_x[0])],
                [float(true_x[1])],
                s=62,
                marker="*",
                color="black",
                edgecolors="white",
                linewidths=0.45,
                zorder=6,
            )

            ax_zoom.set_title(r"(b) Posterior zoom", pad=3)
            ax_zoom.set_xlabel("$x_1$")
            ax_zoom.set_ylabel("$x_2$")
            ax_zoom.set_xlim(zoom_x_min, zoom_x_max)
            ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
            ax_zoom.set_aspect("equal", adjustable="box")

            # Shared, clean legend.
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color="0.45",
                    lw=0.9,
                    label="Prior contours",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="C0",
                    lw=1.1,
                    label="Posterior contours",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="C1",
                    linestyle="None",
                    markersize=4.5,
                    markeredgewidth=0.9,
                    label=f"{algorithm.upper()} particles (N={num_particles})",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="black",
                    linestyle="None",
                    markersize=7,
                    label="Sign-flip candidates / true x",
                ),
            ]

            fig.legend(
                handles=handles,
                loc="lower center",
                ncols=4,
                frameon=False,
                bbox_to_anchor=(0.5, 0.045),
                handlelength=1.6,
                columnspacing=1.4,
                borderaxespad=0.0,
                fontsize=8
            )

            for ext in output_formats:
                ext = str(ext).lower().lstrip(".")
                savepath = (
                    output_dir
                    / f"prior_context_posterior_zoom_{algorithm}_N{num_particles}_y{y_index:03d}.{ext}"
                )
                fig.savefig(savepath, bbox_inches="tight", pad_inches=0.03)
                saved_paths.append(savepath)

            plt.close(fig)

    return saved_paths