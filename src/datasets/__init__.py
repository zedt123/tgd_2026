from __future__ import annotations
from omegaconf import DictConfig, OmegaConf

from src.registry import DATASET

from . import images, null_data, toy2d
from .base import DiffusionData


def get_dataset(cfg: DictConfig, **overrides) -> DiffusionData:
    name = str(cfg.name)
    base_kwargs = {}
    if cfg.get("kwargs") is not None:
        base_kwargs = OmegaConf.to_container(cfg.kwargs, resolve=True)
        if base_kwargs is None:
            base_kwargs = {}

    # Merge runtime-only objects here (model is fine in a normal dict)
    base_kwargs.update(overrides)
    return DATASET.create(name, **base_kwargs)