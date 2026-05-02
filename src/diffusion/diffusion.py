from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import torch

from src.registry import DIFFUSIONS

if TYPE_CHECKING:
    from .types import SDEContext 

class Diffusion(ABC):
    """
    Abstract base class representing a diffusion coefficient g(t) of an SDE.
    Usually g(t) is scalar (or broadcastable) and independent of x.
    """
    def __init__(self, ctx: SDEContext):
        self.ctx = ctx

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return diffusion coefficient g(t) broadcastable to x.

        Args:
            t: scalar tensor (0-dim) or 1-element tensor.
        Returns:
            Tensor broadcastable to x representing diffusion magnitude.
        """
        raise NotImplementedError

@DIFFUSIONS.register("edm")
class EDMDiffusion(Diffusion):
    def __init__(self, ctx: SDEContext):
        super().__init__(ctx)
    def __call__(self, t: torch.Tensor) -> torch.Tensor:  
        return self.ctx.lam_tensor * torch.sqrt(2*t)
