"""
Input/output utilities for experiments.

Defines the canonical on-disk layout for a run directory and provides helpers to:
- create run folders and write metadata,
- save sample artifacts (image grids, NPZ),
- read/write metric summaries.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
from PIL import Image

import re
import prettytable
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter

Json = Dict[str, Any]

def save_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """
    Save an object as JSON.

    - Creates parent directories if needed.
    - Uses UTF-8.
    - Falls back to str() for non-JSON-serializable objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def save_text(path: Union[str, Path], text: str) -> None:
    """
    Save plain text.

    - Creates parent directories if needed.
    - Uses UTF-8.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

def save_image(path: Union[str, Path], img: torch.Tensor) -> None:
    """
    Save a single image tensor from save_image_batch.

    Assumes:
      - img is a CPU torch.Tensor
      - shape (C,H,W)
      - float in [0,1]
      - C in {1,3}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.is_tensor(img):
        raise TypeError(f"Expected torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img.shape)}")

    img = (img * 255.0 + 0.5).to(torch.uint8)             # uint8 CHW
    img = img.permute(1, 2, 0).contiguous().numpy()       # uint8 HWC

    if img.shape[2] == 1:
        img = img[:, :, 0]                                # uint8 HW for grayscale

    Image.fromarray(img).save(path)

def save_image_batch(
    out_dir: Union[str, Path],
    images: torch.Tensor, # (B,K,C,H,W) float on cpu
    prefix: str = "sample",
    start_idx: int = 0,
    ext: str = "png",
    norm: Optional[Tuple[float, float]] = (0.5, 0.5),
) -> None:
    """
    Save a batch of images as individual files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if images.ndim != 5:
        raise ValueError(f"Expected (B,K,C,H,W), got {tuple(images.shape)}")

    x = images.to(torch.float32)
    if norm is not None:
        a, b = norm
        x = x * float(a) + float(b)
    x = x.clamp(0.0, 1.0)

    B, K = x.shape[0], x.shape[1]
    for i in range(B):
        for k in range(K):
            # Sorted order: i-major, then k-minor => matches loader reshape(B,K,...)
            path = out_dir / f"{prefix}_{start_idx + i:06d}_k{k:03d}.{ext}"
            save_image(path, x[i, k]) 

def save_particle_from_uint8(
    out_dir: str,
    images: torch.Tensor,
    times: torch.Tensor,
    *,
    filename_prefix: str = "traj",
    show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Save multiple trajectories as PNGs from a tensor of shape (T, B, N, C, H, W).

    Saves ALL batches. For each batch index b and each particle index n this saves one PNG:
        {out_dir}/{filename_prefix}_b{b:03d}_n{n:03d}.png

    The PNG shows frames from the trajectory. Each frame is annotated with its
    corresponding `times[t]` using 2 decimal places.

    Args:
        out_dir: Directory to write PNGs into (created if missing).
        images: uint8 torch.Tensor on CPU, shape (T, B, N, C, H, W).
        times: torch.Tensor on CPU, shape (T,).
        filename_prefix: Prefix for saved filenames.
        show: If True, displays figures (still saves them).
        dpi: PNG resolution.
    """
    # Validate shapes
    if images.ndim != 6:
        raise ValueError(f"Expected images_T shape (T,B,N,C,H,W), got {tuple(images.shape)}")
    if times.ndim != 1:
        raise ValueError(f"Expected times shape (T,), got {tuple(times.shape)}")
    if images.dtype != torch.uint8:
        raise ValueError(f"Expected images dtype uint8, got {images.dtype}")

    T, B, N, C, H, W = images.shape
    if times.shape[0] != T:
        raise ValueError(f"times length {times.shape[0]} must match T={T}")

    t_idxs = list(range(T))
    os.makedirs(out_dir, exist_ok=True)

    # Saving
    # One PNG per (b, n): a single row of frames (len(t_idxs) columns).
    for b in range(B):
        for n in range(N):
            fig_w = 2.2 * len(t_idxs)
            fig_h = 2.6
            fig, axes = plt.subplots(1, len(t_idxs), figsize=(fig_w, fig_h), squeeze=False)
            axes = axes[0]

            for j, t in enumerate(t_idxs):
                ax = axes[j]
                frame = images[t, b, n]  # (C,H,W)

                if C == 1:
                    ax.imshow(frame, cmap="gray")
                else:
                    ax.imshow(frame.permute(1, 2, 0))

                ax.axis("off")
                ax.set_title(f"t={float(times[t].item()):.2f}", fontsize=9)

            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{filename_prefix}_b{b:03d}_n{n:03d}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

def pack_for_traj(x: torch.Tensor, B: int, N: int, particles_to_save: int) -> torch.Tensor:
    x = x.reshape(B, N, *x.shape[1:]).detach()

    x = x[:, :particles_to_save]

    x = x.float().cpu()

    if x.ndim == 5:
        x = (x * 0.5 + 0.5).clamp(0, 1)
        x = (x * 255.0).to(torch.uint8)

    return x  # (B, particles_to_save, ...)

def format_particle_sweep_table(rows: List[Dict[str, Any]]) -> str:
    """
    Format particle sweep rows into a readable PrettyTable string.

    Expected row keys:
      - algorithm
      - num_particles
      - swd_mean
      - swd_std
      - consistency_mse_mean
      - consistency_mse_std
    """
    table = prettytable.PrettyTable()
    table.field_names = [
        "algorithm",
        "num_particles",
        "swd_mean",
        "swd_std",
        "consistency_mse_mean",
        "consistency_mse_std",
    ]

    for row in rows:
        table.add_row(
            [
                row["algorithm"],
                row["num_particles"],
                f'{row["swd_mean"]:.6g}',
                f'{row["swd_std"]:.6g}',
                f'{row["consistency_mse_mean"]:.6g}',
                f'{row["consistency_mse_std"]:.6g}',
            ]
        )

    return table.get_string()

def format_fixed_particle_table(rows: List[Dict[str, Any]]) -> str:
    """
    Format the fixed-particle comparison table into a readable PrettyTable string.

    Expected row keys:
      - algorithm
      - num_particles
      - swd_mean
      - swd_std
      - consistency_mse_mean
      - consistency_mse_std
    """
    table = prettytable.PrettyTable()
    table.field_names = [
        "algorithm",
        "num_particles",
        "swd_mean",
        "swd_std",
        "consistency_mse_mean",
        "consistency_mse_std",
    ]

    for row in rows:
        table.add_row(
            [
                row["algorithm"],
                row["num_particles"],
                f'{row["swd_mean"]:.6g}',
                f'{row["swd_std"]:.6g}',
                f'{row["consistency_mse_mean"]:.6g}',
                f'{row["consistency_mse_std"]:.6g}',
            ]
        )

    return table.get_string()

def format_sampler_time_ablation_table(rows: List[Dict[str, Any]]) -> str:
    """
    Format the time-based sampler ablation results into a readable PrettyTable string.

    Expected row keys
    -----------------
    Always:
    - sampler
    - variant
    - method
    - sweep_value
    - curve_source
    - nfe
    - num_particles
    - runtime_seconds
    - seconds_per_image
    - lpips_mean / median / std / min / max
    - psnr_mean / median / std / min / max

    Optional:
    - consistency_mse_mean / median / std / min / max
    """
    has_consistency = any("consistency_mse_mean" in row for row in rows)

    field_names = [
        "sampler",
        "variant",
        "method",
        "sweep_value",
        "curve_source",
        "nfe",
        "num_particles",
        "runtime_seconds",
        "seconds_per_image",
        "lpips_mean",
        "lpips_median",
        "lpips_std",
        "lpips_min",
        "lpips_max",
        "psnr_mean",
        "psnr_median",
        "psnr_std",
        "psnr_min",
        "psnr_max",
    ]

    if has_consistency:
        field_names.extend(
            [
                "consistency_mse_mean",
                "consistency_mse_median",
                "consistency_mse_std",
                "consistency_mse_min",
                "consistency_mse_max",
            ]
        )

    table = prettytable.PrettyTable()
    table.field_names = field_names

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r["method"]),
            float(r["seconds_per_image"]),
            int(r["sweep_value"]),
        ),
    )

    for row in rows_sorted:
        values = [
            row["sampler"],
            row["variant"],
            row["method"],
            int(row["sweep_value"]),
            bool(row["curve_source"]),
            int(row["nfe"]),
            int(row["num_particles"]),
            f'{row["runtime_seconds"]:.6g}',
            f'{row["seconds_per_image"]:.6g}',
            f'{row["lpips_mean"]:.6g}',
            f'{row["lpips_median"]:.6g}',
            f'{row["lpips_std"]:.6g}',
            f'{row["lpips_min"]:.6g}',
            f'{row["lpips_max"]:.6g}',
            f'{row["psnr_mean"]:.6g}',
            f'{row["psnr_median"]:.6g}',
            f'{row["psnr_std"]:.6g}',
            f'{row["psnr_min"]:.6g}',
            f'{row["psnr_max"]:.6g}',
        ]

        if has_consistency:
            values.extend(
                [
                    f'{row["consistency_mse_mean"]:.6g}',
                    f'{row["consistency_mse_median"]:.6g}',
                    f'{row["consistency_mse_std"]:.6g}',
                    f'{row["consistency_mse_min"]:.6g}',
                    f'{row["consistency_mse_max"]:.6g}',
                ]
            )

        table.add_row(values)

    return table.get_string()

def _format_method_label(raw_name: str) -> str:
    """
    Convert internal method names into paper-friendly legend labels.

    Examples:
      - 'A-TGD | k4' -> 'A-TGD (N=4)'
      - 'DAPS | k1' -> 'DAPS (N=1)'
      - 'DAPS | k4_measurement_best' -> 'DAPS (N=4, meas.)'
      - 'DAPS | k4_best' -> 'DAPS (N=4, best)'
    """
    parts = [p.strip() for p in raw_name.split("|")]
    if len(parts) == 1:
        return raw_name.strip()

    method = parts[0]
    variant = parts[1]

    match = re.match(r"^k(\d+)(?:_(.*))?$", variant)
    if match is None:
        return raw_name.strip()

    n_particles = match.group(1)
    suffix = match.group(2)

    extras = [f"N={n_particles}"]

    if suffix == "measurement_best":
        extras.append("meas.")
    elif suffix == "best":
        extras.append("best")
    elif suffix:
        extras.append(suffix.replace("_", " "))

    return f"{method} ({', '.join(extras)})"

def plot_sampler_time_ablation(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
    xscale: str = "log",
    summary_stat: str = "mean",
    error_bars: str = "none",
    x_jitter: float = 0.0,
) -> None:
    """
    Plot seconds-per-image vs metric across methods and save as SVG.

    Style defaults:
      - log-scaled x-axis
      - mean center statistic
      - scatter-first design with light connecting lines
      - legend outside the axes
      - optional uncertainty:
          * 'none'   : no uncertainty overlay
          * '2std'   : faint dashed +/- 2 std envelope lines
          * 'minmax' : light asymmetric error bars
    """
    valid_metrics = {"lpips", "psnr", "consistency_mse"}
    if metric not in valid_metrics:
        raise ValueError(
            f"metric must be one of {sorted(valid_metrics)}, got '{metric}'"
        )
    if xscale not in {"linear", "log"}:
        raise ValueError(f"xscale must be 'linear' or 'log', got '{xscale}'")
    if summary_stat not in {"mean", "median"}:
        raise ValueError(
            f"summary_stat must be 'mean' or 'median', got '{summary_stat}'"
        )
    if error_bars not in {"none", "minmax", "2std"}:
        raise ValueError(
            f"error_bars must be 'none', 'minmax', or '2std', got '{error_bars}'"
        )
    if x_jitter < 0:
        raise ValueError(f"x_jitter must be >= 0, got '{x_jitter}'")

    center_key = f"{metric}_{summary_stat}"
    min_key = f"{metric}_min"
    max_key = f"{metric}_max"
    std_key = f"{metric}_std"

    if any(center_key not in row for row in rows):
        raise ValueError(
            f"Some rows are missing required key '{center_key}' for metric='{metric}'"
        )

    if error_bars == "minmax":
        if any(min_key not in row or max_key not in row for row in rows):
            raise ValueError(
                f"Some rows are missing required keys '{min_key}'/'{max_key}'"
            )
    if error_bars == "2std":
        if any(std_key not in row for row in rows):
            raise ValueError(
                f"Some rows are missing required key '{std_key}' for metric='{metric}'"
            )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if center_key not in row:
            continue
        method_name = str(row["method"])
        grouped.setdefault(method_name, []).append(row)

    method_names = list(grouped.keys())
    if not method_names:
        raise ValueError("No valid rows to plot.")

    if len(method_names) <= 1 or x_jitter == 0.0:
        mult_factors = np.ones(len(method_names), dtype=float)
        add_offsets = np.zeros(len(method_names), dtype=float)
    else:
        if xscale == "log":
            mult_factors = np.linspace(
                1.0 - x_jitter, 1.0 + x_jitter, len(method_names)
            )
            add_offsets = np.zeros(len(method_names), dtype=float)
        else:
            mult_factors = np.ones(len(method_names), dtype=float)
            all_x = sorted(set(float(r["seconds_per_image"]) for r in rows))
            min_gap = float(np.min(np.diff(all_x))) if len(all_x) >= 2 else 1.0
            span = x_jitter * min_gap
            add_offsets = np.linspace(-0.5 * span, 0.5 * span, len(method_names))

    metric_ylabel = {
        "lpips": "LPIPS ↓",
        "psnr": "PSNR ↑",
        "consistency_mse": "Consistency MSE ↓",
    }[metric]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, method_name in enumerate(method_names):
        method_rows = sorted(
            grouped[method_name],
            key=lambda r: (float(r["seconds_per_image"]), int(r["sweep_value"])),
        )

        x = np.array([float(r["seconds_per_image"]) for r in method_rows], dtype=float)
        y = np.array([float(r[center_key]) for r in method_rows], dtype=float)

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(
                f"Non-finite x/y values encountered for method '{method_name}'"
            )

        if xscale == "log" and np.any(x <= 0):
            raise ValueError(
                f"All x values must be > 0 for log scale, but found {x} "
                f"for method '{method_name}'"
            )

        if xscale == "log":
            x_plot = x * mult_factors[i]
        else:
            x_plot = x + add_offsets[i]

        label = _format_method_label(method_name)
        is_highlight = "A-TGD" in label

        line_width = 1.8 if is_highlight else 1.4
        line_alpha = 0.78 if is_highlight else 0.60
        marker_size = 52 if is_highlight else 44

        (line,) = ax.plot(
            x_plot,
            y,
            linestyle="-",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=2,
            label=label,
        )
        color = line.get_color()

        if error_bars == "2std":
            y_std2 = np.array(
                [2.0 * float(r[std_key]) for r in method_rows],
                dtype=float,
            )
            lower = np.maximum(y - y_std2, 0.0)
            upper = y + y_std2

            ax.plot(
                x_plot,
                lower,
                color=color,
                linewidth=0.85,
                alpha=0.24,
                linestyle="--",
                zorder=1,
            )
            ax.plot(
                x_plot,
                upper,
                color=color,
                linewidth=0.85,
                alpha=0.24,
                linestyle="--",
                zorder=1,
            )

        elif error_bars == "minmax":
            y_lower = np.array(
                [max(0.0, float(r[center_key]) - float(r[min_key])) for r in method_rows],
                dtype=float,
            )
            y_upper = np.array(
                [max(0.0, float(r[max_key]) - float(r[center_key])) for r in method_rows],
                dtype=float,
            )
            ax.errorbar(
                x_plot,
                y,
                yerr=np.vstack([y_lower, y_upper]),
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.28,
                zorder=1,
            )

        ax.scatter(
            x_plot,
            y,
            s=marker_size,
            alpha=0.98,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xscale(xscale)
    ax.set_xlabel("Seconds per image", fontsize=12)
    ax.set_ylabel(metric_ylabel, fontsize=12)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)

    if xscale == "log":
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    if metric == "consistency_mse":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if metric in {"lpips", "consistency_mse"}:
        ax.set_ylim(bottom=0.0)

    ncol = 2 if len(method_names) >= 4 else 1
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        fontsize=10.0,
        columnspacing=1.2,
        handlelength=2.0,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

def _tensor_to_uint8_hwc(x: torch.Tensor, norm=(0.5, 0.5)) -> np.ndarray:
    """
    Convert image tensor to uint8 HWC numpy array.

    Parameters
    ----------
    x:
        Tensor with shape (H, W) or (C, H, W).
    norm:
        Tuple (std, mean) used as in plot_images_gt:
            x = x * std + mean
        Pass None to disable.
    """
    x = x.detach().cpu().float()

    if norm is not None:
        std, mean = norm
        x = x * std + mean

    x = x.clamp(0.0, 1.0)

    if x.ndim == 2:
        x = x.unsqueeze(0)  # 1,H,W

    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (H,W) or (C,H,W), got {tuple(x.shape)}")

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(x.shape)}")

    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()  # HWC
    return x


def _save_tensor_image(path: Path, x: torch.Tensor, norm=(0.5, 0.5)) -> None:
    """
    Save one image tensor to disk as PNG.
    """
    arr = _tensor_to_uint8_hwc(x, norm=norm)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

def _format_example_metrics_text(
    *,
    rank_info: Dict[str, Any],
    algorithm_results: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    lines.append(f"image_index: {rank_info['image_index']}")
    lines.append(f"rank_metric: {rank_info['rank_metric']}")
    lines.append(f"algorithm_a: {rank_info['algorithm_a']}")
    lines.append(f"algorithm_b: {rank_info['algorithm_b']}")
    lines.append(f"value_a: {rank_info['value_a']:.6g}")
    lines.append(f"value_b: {rank_info['value_b']:.6g}")
    lines.append(f"gap: {rank_info['gap']:.6g}")
    lines.append("")

    idx = int(rank_info["image_index"])

    for result in algorithm_results:
        lines.append(f"[{result['name']}]")
        lines.append(f"selection_rule: {result['selection_rule']}")
        lines.append(f"nfe: {result['nfe']}")
        lines.append(f"num_particles_sampled: {result['num_particles_sampled']}")
        lines.append(f"runtime_seconds: {result['runtime_seconds']:.6f}")
        lines.append(f"seconds_per_image: {result['seconds_per_image']:.6f}")

        per_image_metrics = result["per_image_metrics"]

        for metric_name, values in per_image_metrics.items():
            lines.append(f"{metric_name}: {float(values[idx]):.6g}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_examples(
    *,
    out_dir: Path,
    ranked_examples: List[Dict[str, Any]],
    algorithm_results: List[Dict[str, Any]],
    gt_b: torch.Tensor,
    y_b: torch.Tensor | None,
    rank_metric: str,
    rank_pair: List[str],
    image_norm=(0.5, 0.5),
) -> None:
    """
    Save top-k examples.

    For each selected image, saves:
      - gt.png
      - observation.png (or fallback files)
      - one final output image per algorithm
      - metrics.json
      - metrics.txt

    Parameters
    ----------
    ranked_examples:
        Output of rank_images(...).
    algorithm_results:
        Outputs of run_image_sampler_point(...).
    gt_b:
        Ground truth images, shape (B, C, H, W) or similar.
    y_b:
        Observations in measurement space.
    observation_plotter:
        Optional custom function:
            (out_path, gt_x, y_x, operator) -> None

        This is the best place to plug in your plot_image_gt-based observation visualization.
        If omitted, a generic fallback is used.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "rank_metric": str(rank_metric),
        "rank_pair": [str(x) for x in rank_pair],
        "examples": ranked_examples,
    }
    save_json(out_dir / "examples.json", summary_payload)

    for rank_pos, rank_info in enumerate(ranked_examples):
        image_index = int(rank_info["image_index"])
        example_dir = out_dir / f"example_{rank_pos:03d}_idx_{image_index:03d}"
        example_dir.mkdir(parents=True, exist_ok=True)

        # Save GT
        _save_tensor_image(example_dir / "gt.png", gt_b[image_index], norm=image_norm)

        # Save observation as its own image
        if y_b is not None:
            _save_tensor_image(example_dir / "observation.png", y_b[image_index], norm=image_norm)
        else:
            save_text(example_dir / "observation.txt", "No observation available (y is None).")

        # Save one final selected image per algorithm
        metrics_payload: Dict[str, Any] = {
            "image_index": image_index,
            "rank_metric": str(rank_metric),
            "rank_pair": [str(x) for x in rank_pair],
            "ranking": rank_info,
            "algorithms": {},
        }

        for result in algorithm_results:
            alg_name = str(result["name"])
            safe_name = alg_name.lower().replace(" ", "_").replace("-", "_")

            selected_images_b = result["selected_images_b"]
            _save_tensor_image(
                example_dir / f"{safe_name}.png",
                selected_images_b[image_index],
                norm=image_norm,
            )

            alg_metrics = {}
            for metric_name, values in result["per_image_metrics"].items():
                alg_metrics[metric_name] = float(values[image_index])

            metrics_payload["algorithms"][alg_name] = {
                "selection_rule": str(result["selection_rule"]),
                "nfe": int(result["nfe"]),
                "num_particles_sampled": int(result["num_particles_sampled"]),
                "runtime_seconds": float(result["runtime_seconds"]),
                "seconds_per_image": float(result["seconds_per_image"]),
                "metrics": alg_metrics,
            }
        save_json(example_dir / "metrics.json", metrics_payload)
        save_text(
            example_dir / "metrics.txt",
            _format_example_metrics_text(
                rank_info=rank_info,
                algorithm_results=algorithm_results,
            ),
        )