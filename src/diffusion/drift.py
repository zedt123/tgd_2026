from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import torch

from src.registry import DRIFTS

if TYPE_CHECKING:
    from .types import SDEContext

class Drift(ABC):
    """
    Base class for drifts f(x, t) used by ODE/SDE solvers.
    """

    def __init__(self, ctx: SDEContext, cond_t_min: float = 0.0):
        self.ctx = ctx
        self.cond_t_min = cond_t_min

    @abstractmethod
    def __call__(self,
                 x: torch.Tensor,
                 t: torch.Tensor, 
                 return_aux: bool = False
                 ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Return the drift f(x, t).

        Args:
            x: Tensor of shape (batch, *data_dims) holding particle states.
            t: Tensor of current time (1 element not a batch).
            condition: Optional conditioning information (e.g. class label, text
                embedding) to be consumed by the drift implementation.

        Returns:
            Tensor of same shape as x with the drift vector for each
            particle. Might contain extra conditional tensors.
        """
        raise NotImplementedError
    
@DRIFTS.register("edmdrift")        
class EDMDrift(Drift):
    def __init__(self, ctx: SDEContext, cond_t_min: float = 0.0):
        super().__init__(ctx, cond_t_min)

    def __call__(self,
                 x: torch.Tensor,
                 t: torch.Tensor, 
                 return_aux: bool = False
                 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        -t (1 + lam^2) * score = -t (1 + lam^2) * (x0-x)/t^2 = (1 + lam^2)(x-x0)/t
        """
        ctx = self.ctx
        x0 = ctx.model(x,t)
        if return_aux:
            return ctx.one_plus_lam2 * (x - x0) / t, x0
        else:
            return ctx.one_plus_lam2 * (x - x0) / t

@DRIFTS.register("dps")        
class DPS(Drift):
    def __init__(self, ctx: SDEContext, cond_t_min: float = 0.0):
        super().__init__(ctx, cond_t_min)

    def __call__(self,
                 x: torch.Tensor,
                 t: torch.Tensor, 
                 return_aux: bool = False
                 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        -t (1 + lam^2) * (score + cond_score) = (1 + lam^2) [(x-x0)/t - t * cond_score]
        """
        ctx = self.ctx
        if t.item() < self.cond_t_min:
               x0 = ctx.model(x,t)
               total_drift = (x - x0) / t
        else:
            cond_score, x0 = ctx.operator.dps(ctx.model, x, t, ctx.y_flat)
            total_drift = (x - x0) / t - t * cond_score * ctx.beta

        if return_aux:
            return ctx.one_plus_lam2 * total_drift, ctx.one_plus_lam2 * (x - x0) / t, x0
        else:
            return ctx.one_plus_lam2 * total_drift

@DRIFTS.register("mpgd")    
class MPGD(Drift):
    def __init__(self, ctx: SDEContext, cond_t_min: float = 0.0):
        super().__init__(ctx, cond_t_min)

    def __call__(self,
                 x: torch.Tensor,
                 t: torch.Tensor, 
                 return_aux: bool = False
                 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        -t (1 + lam^2) * (score + cond_score) = (1 + lam^2) [(x-x0)/t - t * cond_score]
        """    
        ctx = self.ctx
        if t.item() < self.cond_t_min:
               x0 = ctx.model(x,t)
               total_drift = (x - x0) / t
        else:
            cond_score, x0 = ctx.operator.mpgd(ctx.model, x, t, ctx.y_flat)
            total_drift = (x - x0) / t - t * cond_score * ctx.beta

        if return_aux:
            return ctx.one_plus_lam2 * total_drift, ctx.one_plus_lam2 * (x - x0) / t, x0
        else:
            return ctx.one_plus_lam2 * total_drift