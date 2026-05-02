from __future__ import annotations

from typing import Any, Dict, Union

import torch

from .base import DiffusionModel

class ImageDiffusion(DiffusionModel):
    """
    Wrapper for pretrained *image* diffusion models that conforms to DiffusionModel.

    This wrapper:
      - Constructs a base architecture via create_model(...)
      - Applies an EDM-style preconditioner (VPPrecond or EDMPrecond) if requested
      - Exposes predict_x0(x, sigma) as the standard API

    Note: Conditioning is handled elsewhere (samplers/operators). Models here are unconditional.
    """

    domain = "image"  # class attribute metadata (subclasses can override)

    def __init__(self, cfg: Union[Dict[str, Any], Any], device: str = "cuda", requires_grad: bool = False):
        super().__init__()

        precond = cfg.precond.type
        # Import backend pieces here to avoid heavy imports at module import time
        from .pretrained.precond import VPPrecond, EDMPrecond
        from .pretrained.create import create_model

        base_model = create_model(**cfg.net, device=device)
        precond_config = {k: v for k, v in cfg.precond.items() if k != "type"}
        if precond == "VP":
            self.model = VPPrecond(model=base_model, **precond_config).to(device)
        elif precond == "EDM":
            self.model = EDMPrecond(model=base_model, **precond_config).to(device)
        elif precond in ("None", "none", "NONE"):
            self.model = base_model.to(device)
        else:
            raise ValueError(f"Unknown precond='{precond}'. Expected 'VP', 'EDM', or None.")
        
        self.model.eval()
        self.model.requires_grad_(requires_grad)
        self.img_resolution = getattr(base_model, "img_resolution", None)
        self.img_channels = getattr(base_model, "in_channels", None)
        self.sigma_min = getattr(self.model, "sigma_min", None)
        self.sigma_max = getattr(self.model, "sigma_max", None)

    def predict_x0(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        # Ensure sigma is a tensor on the correct device/dtype
        sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        return self.model(x, sigma_t)

    def get_in_shape(self):
        if self.img_channels is None or self.img_resolution is None:
            raise NotImplementedError("Underlying model does not expose img_channels/img_resolution.")
        return (int(self.img_channels), int(self.img_resolution), int(self.img_resolution))

