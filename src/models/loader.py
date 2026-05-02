from __future__ import annotations

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, device: str | None = None, generator: torch.Generator | None = None,):
    """
    Dispatch construction of a model from config.
    Expected config format:
      model:
        name: edm | analytic | ...
        kwargs: {...}
    """
    name = str(cfg.name)

    if name == "edm_image":
        # Wrapper that builds backbone + applies VPPrecond/EDMPrecond internally
        from .wrappers import ImageDiffusion
        return ImageDiffusion(cfg=cfg.model_config, device=device)

    if name == "analytic":
        # Your analytic model should subclass DiffusionModel directly
        from .analytic.score_from_distributions import AnalyticDiffusionModel
        return AnalyticDiffusionModel(cfg=cfg.model_config, device=device, generator=generator)

    raise KeyError(
        f"Unknown model backend '{name}'. "
        f"Expected one of: ['edm_image', 'analytic']"
    )
