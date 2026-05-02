"""
x0 modifiers.

An x0 modifier is an (optional) step applied after the x0 solver
produces an x0 estimate. It can implement training-free conditioning (e.g. DAPS-like
updates), or be a no-op.

Design in this repo:
- Conditioning is bound at init time via a shared SDEContext (operator, y, beta, etc.).
- At runtime, modifiers are called with x0 of shape (B*N, ...) and may use RNG.
- Modifiers may optionally use 'outer_ratio' / 'rt' to adapt their strength across
  outer annealing steps.

Implementations should be batch-vectorized on the leading dimension (B*N).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from src.registry import MODIFIERS


class Modifier(ABC):
    """
    Base class for x0 modifiers.

    The modifier is constructed with a context that may include:
      - operator, y (conditioning)
      - beta (guidance/annealing strength)
      - device, model, etc.

    Subclasses implement __call__ and return modified x0.
    """

    def __init__(self, ctx: Any):
        self.ctx = ctx

    @abstractmethod
    def __call__(
        self,
        x0: torch.Tensor,
        *,
        rt: Optional[torch.Tensor] = None,
        outer_ratio: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Modify x0.

        Args:
            x0: tensor of shape (B*N, ...)
            rt: optional current diffusion time associated with this x0 prediction
            outer_ratio: optional progress ratio in [0,1] across outer steps
            rng: torch.Generator for any stochasticity (MCMC proposals etc.)
            kwargs: modifier-specific extras

        Returns:
            x0_new
        """
        raise NotImplementedError

@MODIFIERS.register("identity")
class IdentityModifier(Modifier):
    """No-op modifier."""
    def __init__(self, ctx: Any | None = None):
        self.ctx = ctx

    def __call__(
        self,
        x0: torch.Tensor,
        *,
        rt: Optional[torch.Tensor] = None,
        outer_ratio: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        return x0

@MODIFIERS.register("daps")    
class DAPS(Modifier):
    """
    DAPS-like x0 modification using MCMC.
    Supported MCMC algorithms: Langevin Dynamics,
    Hamiltionian Monte Carlo.
    """
    def __init__(
        self,
        ctx: Any,
        num_steps: int = 100,
        eta0: float = 5e-5,            
        lr_min_ratio: float = 0.01,   
        mc_algo: str = 'langevin',
        momentum: float | None = None,
        tau: float | None = 0.01, 
        ):
        super().__init__(ctx)
        self.num_steps = num_steps
        self.eta0 = eta0
        self.lr_min_ratio = lr_min_ratio
        self.mc_algo = mc_algo
        self.momentum = momentum
        self.tau2 = None if tau is None else tau ** 2

    def _step_size(self, outer_ratio: float) -> float:
        r = float(outer_ratio)
        return self.eta0 * (1.0 + r * (self.lr_min_ratio - 1.0))

    def mcmc_step(self, x, total_grad, eta, noise):
        if self.mc_algo == 'langevin':
            return x + eta * total_grad + (2.0 * eta) ** 0.5 * noise
        
        elif self.mc_algo == 'hmc':
            step_size = eta ** 0.5
            self.velocity = self.momentum * self.velocity + step_size * total_grad + ((2 * (1 - self.momentum)) ** 0.5) * noise
            return x + self.velocity * step_size
        
        else: 
            raise NotImplementedError

    def __call__(
        self,
        x0: torch.Tensor,
        *,
        rt: torch.Tensor,
        outer_ratio: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
        ):
        ctx = self.ctx
        # Initialize chain at x0_hat
        x0 = x0.to(dtype)
        x = x0.detach().clone()

        rt2 = rt.to(dtype).square()
        eta = self._step_size(outer_ratio)  

        if self.mc_algo == 'hmc':
            self.velocity = torch.randn(x.shape, generator=generator, device=x.device, dtype=dtype)

        y_flat = ctx.y_flat.to(dtype)
        beta = ctx.beta
        for _ in range(self.num_steps):
            prior_grad = (x0 - x) / rt2
            y_grad = ctx.operator.gradient(x, y_flat, tau2 = self.tau2)
            total_grad = prior_grad + beta * y_grad
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=dtype)
            x = self.mcmc_step(x, total_grad, eta, noise)
        return x

