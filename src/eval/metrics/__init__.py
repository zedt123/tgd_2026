from __future__ import annotations
from typing import Any, Dict, Sequence, Optional
from omegaconf import DictConfig

from . import image_metrics  
from . import toy2d_metrics  

from src.registry import PAIRED_METRICS, DIST_METRICS


def build_paired_metrics(names: Sequence[str], kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None):
    kwargs_by_name = kwargs_by_name or {}
    return [PAIRED_METRICS.create(n, **kwargs_by_name.get(n, {})) for n in names]


def build_dist_metrics(names: Sequence[str], kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None):
    kwargs_by_name = kwargs_by_name or {}
    return [DIST_METRICS.create(n, **kwargs_by_name.get(n, {})) for n in names]

def build_metrics(cfg: DictConfig):
    paired_names = cfg.get("paired_metrics") or []
    dist_names   = cfg.get("dist_metrics") or []

    pm_kwargs  = cfg.get("paired_metric_kwargs") or {}
    dm_kwargs  = cfg.get("dist_metric_kwargs") or {}

    paired = [PAIRED_METRICS.create(name, **pm_kwargs.get(name, {})) for name in paired_names]
    dist   = [DIST_METRICS.create(name,   **dm_kwargs.get(name, {})) for name in dist_names]
    return paired, dist
