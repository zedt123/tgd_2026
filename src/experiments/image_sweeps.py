from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig

from src.cfg import sync, compute_nfe
from src.eval.evaluator import EvaluatorOptions
from src.factory import build_sampler

def _find_paired_metric(evaluator, metric_name: str):
    for m in evaluator.paired_metrics:
        if getattr(m, "name", None) == metric_name:
            return m
    raise ValueError(f"Paired metric '{metric_name}' is not configured in evaluator.paired_metrics")


def _compute_measurement_error_bk(
    *,
    samples_bk: torch.Tensor,
    y_b: torch.Tensor | None,
    operator: Any,
    device: str,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute measurement error mse(y, operator.forward(x)) for all particles.

    Returns
    -------
    torch.Tensor
        Shape (B, K) on CPU.
    """
    if y_b is None:
        raise ValueError("selection_rule='measurement_best' requires y_b")
    if operator is None:
        raise ValueError("selection_rule='measurement_best' requires operator")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

    B, K = samples_bk.shape[:2]

    gen_bk = samples_bk.to(device)
    gen_flat = gen_bk.flatten(0, 1)

    y_b = y_b.to(device)
    y_bk = y_b.unsqueeze(1).expand(B, K, *y_b.shape[1:])
    y_flat = y_bk.flatten(0, 1)

    y_hat = operator.forward(gen_flat)
    diff2 = (y_hat - y_flat) ** 2

    if diff2.ndim == 1:
        v = diff2
    else:
        dims = tuple(range(1, diff2.ndim))
        v = diff2.mean(dim=dims) if reduction == "mean" else diff2.sum(dim=dims)

    return v.reshape(B, K).detach().cpu()


@torch.no_grad()
def _compute_metric_values_bk(
    *,
    samples_bk: torch.Tensor,
    metric_name: str,
    evaluator,
    gt_b: torch.Tensor | None,
    y_b: torch.Tensor | None,
    operator: Any,
    device: str,
) -> Tuple[torch.Tensor, str]:
    """
    Compute per-particle metric values of shape (B, K) for one paired metric.

    Returns
    -------
    values_bk:
        Tensor of shape (B, K) on CPU.
    cmp:
        The metric comparison direction, "min" or "max".
    """
    metric = _find_paired_metric(evaluator, metric_name)

    old_options = evaluator.options
    evaluator.options = EvaluatorOptions(
        reduce_k="mean",
        store_per_condition=False,
    )

    try:
        evaluator.create_batch(
            B=int(samples_bk.shape[0]),
            K=int(samples_bk.shape[1]),
            gen_bk=samples_bk,
            gt_b=gt_b,
            y_b=y_b,
            operator=operator,
            log_w_tbk=None,
            ess_tb=None,
        )
        values_bk = metric.compute(evaluator.batch, device=device).detach().cpu()
    finally:
        evaluator.reset_batch()
        evaluator.options = old_options

    cmp = str(getattr(metric, "cmp", "min"))
    return values_bk, cmp


@torch.no_grad()
def _select_particles(
    *,
    samples_bk: torch.Tensor,
    selection_rule: str,
    evaluator,
    gt_b: torch.Tensor | None,
    y_b: torch.Tensor | None,
    operator: Any,
    device: str,
    measurement_reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select one final particle per image according to selection_rule.

    Supported selection_rule values
    -------------------------------
    - "first"
    - "measurement_best"
    - "best_lpips"
    - "best_psnr"
    - "best_consistency_mse"

    Returns
    -------
    selected_b:
        Tensor of shape (B, ...) on CPU.
    idx_b:
        Selected particle index for each image, shape (B,) on CPU.
    """
    B, K = samples_bk.shape[:2]
    if K < 1:
        raise ValueError(f"samples_bk must have K >= 1, got shape {tuple(samples_bk.shape)}")

    if selection_rule == "first" or K == 1:
        idx_b = torch.zeros(B, dtype=torch.long)

    elif selection_rule == "measurement_best":
        values_bk = _compute_measurement_error_bk(
            samples_bk=samples_bk,
            y_b=y_b,
            operator=operator,
            device=device,
            reduction=measurement_reduction,
        )
        idx_b = values_bk.argmin(dim=1)

    elif selection_rule.startswith("best_"):
        metric_name = selection_rule[len("best_"):]
        values_bk, cmp = _compute_metric_values_bk(
            samples_bk=samples_bk,
            metric_name=metric_name,
            evaluator=evaluator,
            gt_b=gt_b,
            y_b=y_b,
            operator=operator,
            device=device,
        )

        if cmp == "min":
            idx_b = values_bk.argmin(dim=1)
        elif cmp == "max":
            idx_b = values_bk.argmax(dim=1)
        else:
            raise ValueError(f"Unsupported cmp='{cmp}' for metric '{metric_name}'")

    else:
        raise ValueError(
            f"Unknown selection_rule='{selection_rule}'. "
            f"Supported: first, measurement_best, best_lpips, best_psnr, best_consistency_mse"
        )

    b_idx = torch.arange(B, dtype=torch.long)
    selected_b = samples_bk[b_idx, idx_b]
    return selected_b, idx_b


@torch.no_grad()
def _evaluate_selected_outputs(
    *,
    selected_b: torch.Tensor,
    evaluator,
    gt_b: torch.Tensor,
    y_b: torch.Tensor | None,
    operator: Any,
    device: str,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Compute per-image paired metrics and summary metrics for selected outputs.

    Returns
    -------
    per_image_metrics:
        Dict mapping metric name -> list[float] of length B.
    summary_metrics:
        Flattened metrics dict from Evaluator.
    """
    gen_bk = selected_b.unsqueeze(1)  # (B, 1, ...)

    old_options = evaluator.options
    evaluator.options = EvaluatorOptions(
        reduce_k="mean",
        store_per_condition=True,
    )

    try:
        evaluator.create_batch(
            B=int(selected_b.shape[0]),
            K=1,
            gen_bk=gen_bk,
            gt_b=gt_b,
            y_b=y_b,
            operator=operator,
            log_w_tbk=None,
            ess_tb=None,
        )

        result = evaluator.evaluate(
            timings=None,
            meta=None,
            run_paired=True,
            run_dist=False,
            run_smc=False,
        )
    finally:
        evaluator.reset_batch()
        evaluator.options = old_options

    per_image_metrics = {}
    if result.extra is not None:
        per_image_metrics = dict(result.extra.get("per_condition", {}))

    return per_image_metrics, result.metrics


@torch.no_grad()
def run_image_sampler_point(
    *,
    point_cfg: DictConfig,
    algorithm_name: str,
    selection_rule: str,
    evaluator,
    model: torch.nn.Module,
    operator: Any,
    y: torch.Tensor | None,
    data: torch.Tensor,
    device: str,
    total_params: int,
    rng: Any,
) -> Dict[str, Any]:
    """
    Run one algorithm once for.

    The sampler may generate multiple particles per image, but this function
    reduces them to one final selected output image per input according to
    `selection_rule`.

    Returns
    -------
    Dict[str, Any]
        Contains:
        - name
        - selection_rule
        - nfe
        - num_particles_sampled
        - runtime_seconds
        - seconds_per_image
        - selected_indices
        - selected_images_b
        - per_image_metrics
        - summary_metrics
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

    runtime_seconds = float(t1 - t0)
    seconds_per_image = runtime_seconds / float(total_data)

    samples_bk = out.x0  # expected on CPU
    nfe = compute_nfe(point_cfg)

    selected_b, idx_b = _select_particles(
        samples_bk=samples_bk,
        selection_rule=selection_rule,
        evaluator=evaluator,
        gt_b=data,
        y_b=y,
        operator=sampler.ctx.operator,
        device=device,
    )

    per_image_metrics, summary_metrics = _evaluate_selected_outputs(
        selected_b=selected_b,
        evaluator=evaluator,
        gt_b=data,
        y_b=y,
        operator=sampler.ctx.operator,
        device=device,
    )

    return {
        "name": str(algorithm_name),
        "selection_rule": str(selection_rule),
        "nfe": int(nfe),
        "num_particles_sampled": int(num_particles_sampled),
        "runtime_seconds": float(runtime_seconds),
        "seconds_per_image": float(seconds_per_image),
        "selected_indices": idx_b.tolist(),
        "selected_images_b": selected_b.cpu(),
        "per_image_metrics": per_image_metrics,
        "summary_metrics": summary_metrics,
        "meta": {
            "device": device,
            "total_params": int(total_params),
            "B": int(total_data),
        },
    }


def rank_images(
    *,
    algorithm_results: List[Dict[str, Any]],
    rank_pair: Tuple[str, str] | List[str],
    rank_metric: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Rank dataset images by absolute gap between two chosen algorithms on one metric.

    Parameters
    ----------
    algorithm_results:
        Results returned by run_image_sampler_point(...).
    rank_pair:
        Names of the two algorithms used to rank images.
    rank_metric:
        Metric name, e.g. "lpips", "psnr", "consistency_mse".
    top_k:
        Number of top images to return.

    Returns
    -------
    List[Dict[str, Any]]
        Sorted from largest gap to smallest, each entry containing:
        - image_index
        - gap
        - algorithm_a
        - algorithm_b
        - value_a
        - value_b
    """
    if len(rank_pair) != 2:
        raise ValueError(f"rank_pair must contain exactly 2 algorithm names, got {rank_pair}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    by_name = {str(r["name"]): r for r in algorithm_results}

    alg_a, alg_b = str(rank_pair[0]), str(rank_pair[1])
    if alg_a not in by_name:
        raise ValueError(f"Algorithm '{alg_a}' from rank_pair was not run")
    if alg_b not in by_name:
        raise ValueError(f"Algorithm '{alg_b}' from rank_pair was not run")

    values_a = by_name[alg_a]["per_image_metrics"].get(rank_metric)
    values_b = by_name[alg_b]["per_image_metrics"].get(rank_metric)

    if values_a is None:
        raise ValueError(f"Algorithm '{alg_a}' is missing per-image metric '{rank_metric}'")
    if values_b is None:
        raise ValueError(f"Algorithm '{alg_b}' is missing per-image metric '{rank_metric}'")

    a = torch.tensor(values_a, dtype=torch.float32)
    b = torch.tensor(values_b, dtype=torch.float32)

    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Metric arrays for ranking must both be 1D and same length, got {tuple(a.shape)} and {tuple(b.shape)}"
        )

    gaps = (a - b).abs()
    k = min(int(top_k), int(gaps.numel()))
    top_vals, top_idx = torch.topk(gaps, k=k, largest=True, sorted=True)

    ranked = []
    for gap, idx in zip(top_vals.tolist(), top_idx.tolist()):
        ranked.append(
            {
                "image_index": int(idx),
                "gap": float(gap),
                "algorithm_a": alg_a,
                "algorithm_b": alg_b,
                "value_a": float(a[idx].item()),
                "value_b": float(b[idx].item()),
                "rank_metric": str(rank_metric),
            }
        )

    return ranked