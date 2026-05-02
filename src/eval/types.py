"""
Shared types for evaluation.

This module is intentionally modality-agnostic:
- It does not assume images vs toy2d vs audio.
- It only defines the interfaces and containers used by Evaluator and metrics.
- Evaluator creates EvalBatch once and passes it in the metrics.

Conventions
-----------
Paired evaluation:
  gt:  (B, ...)           # one "ground truth" per condition
  gen: (B, K, ...)        # K generated samples per condition

Distribution evaluation:
  ref: (N, ...)           # reference set
  gen: (M, ...)           # generated set
Reduction over K
----------------
reduce_k controls how per-particle paired metric values of shape (B, K) are reduced
to one value per condition of shape (B,):

- "mean":
    average over particles
- "min":
    take the minimum over particles
- "max":
    take the maximum over particles
- "best":
    use the metric's own cmp ("min" or "max") to select the best particle
    independently for each metric
- "measurement_best":
    select one particle per condition using the lowest measurement error in
    observation space, then use that same selected particle for all paired metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from abc import ABC, abstractmethod

import torch

ReduceK = Literal["mean", "min", "max", "best", "measurement_best"]
Cmp = Literal["min", "max"]

@dataclass(frozen=True)
class EvalBatch:
    """
    Batch container for paired evaluation.

    Required
    --------
    B:
        Number of conditions.
    K:
        Generates samples per condition.
    gen_bk:
        Generated samples of shape (B, K, ...).
    gen_flat:
        Generated samples of shape (B*K, ...).

    Optional
    --------
    gt_bk/gt_flat:
        Ground truth, expanded to (B, K, ...) / (B*K, ...), for GT-based metrics (e.g. PSNR).
    y_bk/y_flat:
        Observation / condition in measurement space, expanded to (B, K, ...) / (B*K, ...),
        for consistency / faithfulness metrics.
    operator:
        Optional object used by faithfulness metrics (e.g. forward operator).
    log_w_tbk:
        Log weights across diffusion timesteps, shape (T, B, K).
    ess_tb:
        Effective sample size across timesteps, shape (T, B).
    meta:
        Optional dict for extra sampler/run information (acceptance rates, step counts, etc.).
    """
    B: int
    K: int

    gen_bk: torch.Tensor
    gen_flat: torch.Tensor

    gt_bk: Optional[torch.Tensor] = None
    gt_flat: Optional[torch.Tensor] = None

    y_bk: Optional[torch.Tensor] = None
    y_flat: Optional[torch.Tensor] = None

    operator: Optional[Any] = None

    log_w_tbk: Optional[torch.Tensor] = None
    ess_tb: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class EvalResult:
    """
    Standard output container for evaluation.

    metrics:
        Flattened metric dictionary, suitable for CSV/JSONL writing.
    extra:
        Optional diagnostics (timings, intermediate stats, etc.).
    """
    metrics: Dict[str, float]
    extra: Optional[Dict[str, Any]] = None

class PairedMetric(ABC):
    """
    Base class for paired metrics over (B, K, ...).

    Subclasses should set:
      name: str
      cmp:  "min" or "max" (used for reduce_k="best" by the Evaluator/metric)
    """
    name: str
    cmp: Cmp = "min"

    @abstractmethod
    def compute(
        self,
        batch: EvalBatch,
        *,
        device: str,
    ) -> torch.Tensor:
        raise NotImplementedError


class DistributionMetric(ABC):
    """
    Base class for distribution-level metrics.

    Contract:
      compute(ref, gen) where
        ref: (N, ...)
        gen: (M, ...)
    """
    name: str

    @abstractmethod
    def compute(
        self,
        ref: torch.Tensor,
        gen: torch.Tensor,
        *,
        device: str,
    ) -> float:
        raise NotImplementedError
    
    def compute_batched(
        self,
        ref: torch.Tensor,
        gen: torch.Tensor,
        *,
        device: str,
    ) -> torch.Tensor:
        """
        Optional contract for conditional evaluation:
          ref: (B, N, ...)
          gen: (B, M, ...)
        returns:
          (B,) tensor (one value per condition)
        """
        raise NotImplementedError