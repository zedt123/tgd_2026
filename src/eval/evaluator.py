"""
Evaluator orchestration.

Goals
-----
- Modality-agnostic: works for images, toy2d, etc.
- Handles sampler outputs shaped (B, K, ...).
- Provides consistent reduction over K (per-condition) and over B (dataset/batch).
- Produces a flat metrics dict (JSON/CSV friendly) plus optional structured extras.

Notes
-----
- PairedMetric operates on EvalBatch (B,K,...).
- DistributionMetric operates on (N,...) reference and (M,...) generated sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .types import (
    EvalBatch,
    EvalResult,
    PairedMetric,
    DistributionMetric,
    ReduceK,
)


def _expand_b_to_bk(x_b: torch.Tensor, K: int) -> torch.Tensor:
    """(B, ...) -> (B, K, ...) via expand."""
    if x_b.ndim < 1:
        raise ValueError(f"Expected (B, ...), got {tuple(x_b.shape)}")
    return x_b.unsqueeze(1).expand(x_b.shape[0], K, *x_b.shape[1:])


def _reduce_k(values_bk: torch.Tensor, *, reduce_k: ReduceK, cmp: str) -> torch.Tensor:
    """
    values_bk: (B, K) -> (B,)

    Note:
      - "measurement_best" is handled separately in Evaluator.evaluate_paired(),
        because it requires a shared particle selection based on measurement error.
    """
    if values_bk.ndim != 2:
        raise ValueError(f"Metric must return (B,K). Got {tuple(values_bk.shape)}")

    if reduce_k == "mean":
        return values_bk.mean(dim=1)
    if reduce_k == "min":
        return values_bk.min(dim=1).values
    if reduce_k == "max":
        return values_bk.max(dim=1).values
    if reduce_k == "best":
        if cmp not in ("min", "max"):
            raise ValueError(f"cmp must be 'min' or 'max', got {cmp}")
        return values_bk.min(dim=1).values if cmp == "min" else values_bk.max(dim=1).values
    if reduce_k == "measurement_best":
        raise ValueError("reduce_k='measurement_best' must be handled outside _reduce_k().")

    raise ValueError(f"Unknown reduce_k='{reduce_k}'")


def _select_k(values_bk: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """values_bk: (B, K), idx_b: (B,) -> selected values: (B,)"""
    if values_bk.ndim != 2:
        raise ValueError(f"Expected values_bk with shape (B,K), got {tuple(values_bk.shape)}")
    if idx_b.ndim != 1:
        raise ValueError(f"Expected idx_b with shape (B,), got {tuple(idx_b.shape)}")
    if values_bk.shape[0] != idx_b.shape[0]:
        raise ValueError(
            f"Mismatched batch sizes: values_bk.shape[0]={values_bk.shape[0]} vs idx_b.shape[0]={idx_b.shape[0]}"
        )

    b_idx = torch.arange(values_bk.shape[0], device=values_bk.device)
    return values_bk[b_idx, idx_b]


def _measurement_error_bk(batch: EvalBatch, *, reduction: str = "mean") -> torch.Tensor:
    """
    Compute measurement error for each generated particle, returning shape (B, K).

    This matches the consistency / faithfulness metric in observation space:
        mse(y, operator.forward(x))
    """
    if batch.operator is None:
        raise ValueError("measurement_best requires batch.operator")
    if batch.y_flat is None:
        raise ValueError("measurement_best requires batch.y_flat")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

    y = batch.y_flat
    y_hat = batch.operator.forward(batch.gen_flat)

    diff2 = (y_hat - y) ** 2
    if diff2.ndim == 1:
        v = diff2
    else:
        dims = tuple(range(1, diff2.ndim))
        v = diff2.mean(dim=dims) if reduction == "mean" else diff2.sum(dim=dims)

    return v.reshape(batch.B, batch.K)


def _measurement_reduction_from_metrics(metrics: Sequence[PairedMetric]) -> str:
    """
    Try to match the reduction used by the configured consistency_mse metric.
    Falls back to 'mean' if that metric is not present.
    """
    for m in metrics:
        if getattr(m, "name", None) == "consistency_mse":
            reduction = getattr(m, "reduction", "mean")
            if reduction in ("mean", "sum"):
                return reduction
    return "mean"


def _summary_stats(x_b: torch.Tensor) -> Dict[str, float]:
    """
    x_b: (B,) -> summary stats.

    We report both mean and median:
      - mean/std for comparability with prior work
      - median/IQR (q25,q75) for robustness to outliers
    """
    if x_b.ndim != 1:
        raise ValueError(f"Expected (B,), got {tuple(x_b.shape)}")
    if x_b.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    x = x_b.detach()
    mean = x.mean()
    std = x.std(unbiased=False) if x.numel() > 1 else torch.zeros_like(mean)

    q25 = torch.quantile(x, 0.25)
    median = torch.quantile(x, 0.50)
    q75 = torch.quantile(x, 0.75)
    return {
        "mean": float(mean.cpu().item()),
        "std": float(std.cpu().item()),
        "median": float(median.cpu().item()),
        "q25": float(q25.cpu().item()),
        "q75": float(q75.cpu().item()),
        "min": float(x.min().cpu().item()),
        "max": float(x.max().cpu().item()),
    }

def _to_device(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(device, non_blocking=True)


@dataclass(frozen=True)
class EvaluatorOptions:
    reduce_k: ReduceK = "mean"
    store_per_condition: bool = False

class Evaluator:
    """
    Evaluator that can compute:
      - paired metrics: must return torch.Tensor of shape (B, K)
      - distribution metrics: must return a single float

    Timing/meta are attached once in evaluate().
    """

    def __init__(
        self,
        *,
        device: str,
        paired_metrics: Sequence[PairedMetric] = (),
        dist_metrics: Sequence[DistributionMetric] = (),
        options: Optional[EvaluatorOptions] = None,
    ) -> None:

        self.device = device
        self.paired_metrics = list(paired_metrics)
        self.dist_metrics = list(dist_metrics)
        self.options = options or EvaluatorOptions()

        self.batch: Optional[EvalBatch] = None

    @torch.no_grad()
    def create_batch(
        self,
        *,
        B: int,
        K: int,
        gen_bk: torch.Tensor,
        gt_b: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
        operator: Optional[Any] = None,
        log_w_tbk: Optional[torch.Tensor] = None,
        ess_tb: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EvalBatch:
        device = self.device
        gen_bk = _to_device(gen_bk, device)
        gen_flat = gen_bk.flatten(0, 1)

        gt_bk = None
        gt_flat = None
        if gt_b is not None:
            gt_b = _to_device(gt_b, device)          # (B, ...)
            gt_bk = _expand_b_to_bk(gt_b, K)         # (B, K, ...)
            gt_flat = gt_bk.flatten(0, 1)            # (B*K, ...)

        y_bk = None
        y_flat = None
        if y_b is not None:
            y_b = _to_device(y_b, device)            # (B, ...)
            y_bk = _expand_b_to_bk(y_b, K)           # (B, K, ...)
            y_flat = y_bk.flatten(0, 1)              # (B*K, ...)

        batch = EvalBatch(
            B=B,
            K=K,
            gen_bk=gen_bk,
            gen_flat=gen_flat,
            gt_bk=gt_bk,
            gt_flat=gt_flat,
            y_bk=y_bk,
            y_flat=y_flat,
            operator=operator,
            log_w_tbk=_to_device(log_w_tbk, device),
            ess_tb=_to_device(ess_tb, device),
            meta=meta,
        )

        self.batch = batch
        return batch

    def reset_batch(self) -> None:
        self.batch = None

    # -----------------------
    # Metric runners
    # -----------------------
    @torch.no_grad()
    def evaluate_paired(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if self.batch is None:
            raise ValueError("No batch set. Call create_batch(...) first.")

        rk = self.options.reduce_k
        metrics_out: Dict[str, float] = {}
        extra: Dict[str, Any] = {}

        measurement_best_idx = None
        measurement_best_vals = None

        if rk == "measurement_best":
            reduction = _measurement_reduction_from_metrics(self.paired_metrics)
            measurement_vals_bk = _measurement_error_bk(self.batch, reduction=reduction)   # (B, K)
            measurement_best_idx = measurement_vals_bk.argmin(dim=1)                        # (B,)
            measurement_best_vals = _select_k(measurement_vals_bk, measurement_best_idx)    # (B,)

            if self.options.store_per_condition:
                extra.setdefault("per_condition", {})["measurement_error"] = (
                    measurement_best_vals.detach().cpu().tolist()
                )
            extra.setdefault("selection", {})["reduce_k"] = "measurement_best"

        for m in self.paired_metrics:
            values_bk = m.compute(self.batch, device=self.device)

            if not isinstance(values_bk, torch.Tensor):
                raise TypeError(f"PairedMetric '{m.name}' must return torch.Tensor (B,K).")
            if values_bk.ndim != 2:
                raise ValueError(f"PairedMetric '{m.name}' must return (B,K), got {tuple(values_bk.shape)}.")

            if rk == "measurement_best":
                assert measurement_best_idx is not None
                per_cond_b = _select_k(values_bk, measurement_best_idx)
            else:
                per_cond_b = _reduce_k(values_bk, reduce_k=rk, cmp=getattr(m, "cmp", "min"))

            stats = _summary_stats(per_cond_b)
            metrics_out.update({f"{m.name}/{k}": v for k, v in stats.items()})

            if self.options.store_per_condition:
                extra.setdefault("per_condition", {})[m.name] = per_cond_b.detach().cpu().tolist()

        return metrics_out, extra

    @torch.no_grad()
    def evaluate_distribution(self, *, ref: torch.Tensor, gen: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Distribution metrics:
        - if ref/gen are unbatched -> metric returns float
        - if ref/gen are batched (B,...) and metric implements compute_batched -> summarize over B
        """
        device = self.device
        ref = _to_device(ref, device)
        gen = _to_device(gen, device)

        metrics_out: Dict[str, float] = {}
        extra: Dict[str, Any] = {}

        batched = (ref.ndim >= 3) and (gen.ndim >= 3) and (ref.shape[0] == gen.shape[0])

        for m in self.dist_metrics:
            if batched and hasattr(m, "compute_batched"):
                try:
                    vals_b = m.compute_batched(ref, gen, device=device)
                except NotImplementedError:
                    vals_b = None

                if vals_b is not None:
                    if not isinstance(vals_b, torch.Tensor) or vals_b.ndim != 1:
                        raise TypeError(f"DistMetric '{m.name}.compute_batched' must return torch.Tensor (B,).")
                    stats = _summary_stats(vals_b)
                    metrics_out.update({f"{m.name}/{k}": v for k, v in stats.items()})
                    if self.options.store_per_condition:
                        extra.setdefault("per_condition", {})[m.name] = vals_b.detach().cpu().tolist()
                    continue  # done with this metric

            # Fallback: scalar metric
            metrics_out[m.name] = float(m.compute(ref, gen, device=device))

        return metrics_out, extra

    # -----------------------
    # One-shot evaluation
    # -----------------------
    @torch.no_grad()
    def evaluate(
        self,
        *,
        ref: Optional[torch.Tensor] = None,
        gen_for_dist: Optional[torch.Tensor] = None,
        timings: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        run_paired: bool = True,
        run_dist: bool = True,
    ) -> EvalResult:
        """
        Run enabled evaluations and attach timing/meta once.

        Expected workflow:
          1) create_batch(...)
          2) evaluate(ref=..., gen_for_dist=..., timings=..., meta=...)
        """
        all_metrics: Dict[str, float] = {}
        extra: Dict[str, Any] = {}

        if run_paired and self.paired_metrics:
            paired_metrics, paired_extra = self.evaluate_paired()
            all_metrics.update(paired_metrics)
            extra.update(paired_extra)

        if run_dist and self.dist_metrics:
            if ref is None or gen_for_dist is None:
                raise ValueError("ref and gen_for_dist must be provided to run distribution metrics")
            dist_metrics, dist_extra = self.evaluate_distribution(ref=ref, gen=gen_for_dist)
            all_metrics.update(dist_metrics)
            extra.update(dist_extra)

        if timings:
            extra["timing"] = dict(timings)
        if meta:
            extra["meta"] = dict(meta)

        return EvalResult(metrics=all_metrics, extra=extra or None)