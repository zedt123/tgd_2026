from .base import Operator
from src.registry import OPERATORS
import torch

from typing import Any, Tuple

@OPERATORS.register("inpainting")    
class Inpainting(Operator):
    def __init__(self, std_schedule, mask_type, box_shape=None, mask_prob=None, margin=None, sigma=0.05, **kwargs):
        super().__init__(std_schedule, sigma, **kwargs)
        self.mask_gen = MaskGenerator(mask_type, box_shape, mask_prob, margin)
        self.mask = None 

    def forward(self, x, generator: torch.Generator | None = None):
        if self.mask is None:
            self.mask = self.mask_gen(x, generator=generator)
        return x * self.mask
    
    @torch.no_grad()
    def mpgd(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes gradient without autograd for efficiency.
        y needs to be already masked (0 where we don't have info).
        """
        x0 = model(x, t)
        grad_x0 = (y - x0) * self.mask / self.std_t(t) ** 2
        return grad_x0, x0
    

# modified from https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/util/img_utils.py
class MaskGenerator:
    def __init__(
        self,
        mask_type,
        box_shape=None,
        mask_prob=None,
        margin=(32, 32),
        mask_dims=None, 
    ):
        """
        Produces boolean masks where:
          True  = keep
          False = mask out

        For images (B,C,H,W): returns shape (1,1,H,W)
        For vectors (B,d):    returns shape (1,d)

        mask_type: "box", "random", "extreme"
          - "box": mask out a contiguous region (rectangle for images, segment for vectors)
          - "random": Bernoulli keep/drop
          - "extreme": invert of "box" (keep only the box region)
          - "dims" for vectors: mask out specified feature dimensions
        """
        valid = ("box", "random", "extreme", "dims")
        if mask_type not in valid:
            raise ValueError(f"mask_type must be one of {valid}")

        if mask_type in ("box", "extreme") and box_shape is None:
            raise ValueError("box_shape must be provided for box/extreme masks")

        if mask_type == "random" and mask_prob is None:
            raise ValueError("mask_prob must be provided for random masks")

        if mask_type == "dims" and mask_dims is None:
            raise ValueError("mask_dims must be provided for dims masks")

        if mask_prob is not None and not (0.0 <= float(mask_prob) <= 1.0):
            raise ValueError("mask_prob must be in [0, 1]")

        self.mask_type = mask_type
        self.box_shape = None if box_shape is None else tuple(box_shape)
        self.mask_prob = None if mask_prob is None else float(mask_prob)
        self.margin = None if margin is None else tuple(margin)
        self.mask_dims = None if mask_dims is None else list(mask_dims)

    # ---------- image masks ----------
    def _box_image(self, img: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        _, _, H, W = img.shape
        h, w = self.box_shape
        mh, mw = self.margin   

        max_t = H - mh - h
        max_l = W - mw - w
        if max_t < mh or max_l < mw:
            raise ValueError(
                f"Box {self.box_shape} with margin {self.margin} doesn't fit image {(H, W)}"
            )

        t = torch.randint(mh, H - mh - h + 1, (1,), device=img.device, generator=generator).item()
        l = torch.randint(mw, W - mw - w + 1, (1,), device=img.device, generator=generator).item()

        # make mask
        mask = torch.ones([1, 1, H, W], device=img.device, dtype=torch.bool)
        mask[:, :, t:t + h, l:l + w] = False
        return mask
    
    def _random_image(self, img: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        _, _, H, W = img.shape
        keep_prob = 1.0 - self.mask_prob
        r = torch.rand((1, 1, H, W), device=img.device, generator=generator)
        return r < keep_prob

    def _retrieve_dims(self, x: torch.Tensor) -> torch.Tensor:
        _, d = x.shape
        mask = torch.ones((1, d), device=x.device, dtype=torch.bool)
        for idx in self.mask_dims:
            if idx < 0 or idx >= d:
                raise ValueError(f"mask dim {idx} out of range for d={d}")
            mask[:, idx] = False
        return mask
    
    def _random_vector(self, x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        _, d = x.shape
        keep_prob = 1.0 - self.mask_prob
        r = torch.rand((1, d), device=x.device, generator=generator)
        return r < keep_prob

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None):
        if x.ndim == 4:
            if self.mask_type == "random":
                return self._random_image(x, generator)
            else:
                mask = self._box_image(x, generator)
            return mask if self.mask_type != "extreme" else ~mask
        
        if x.ndim == 2:
            if self.mask_type == "random":
                mask = self._random_vector(x, generator)
            else:
                mask = self._retrieve_dims(x)
            return mask
        
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}; expected (B,C,H,W) or (B,d).")