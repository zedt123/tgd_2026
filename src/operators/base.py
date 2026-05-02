from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Tuple
from .proposal_schedules import ProposalSchedule

class Operator(ABC, nn.Module):
    """
    Abstract base class for operators in diffusion processes.
    """
    def __init__(self, std_schedule: type[ProposalSchedule], sigma: float = 0.05, **kwargs):
        """
        Initializes the operator with a noise standard deviation.

        Args:
            sigma (float, optional): Measurement noise level. Defaults to 0.05.
            std_schedule: a class (or factory) such that std_schedule(sigma) returns
                          a callable schedule std(t). 
        """
        super().__init__()
        self.sigma = float(sigma)
        self.sigma2 = float(sigma * sigma)
        self.std_t = std_schedule(self.sigma, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor,  generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply operator A to x. For images, x is expected to be in [-1,1]."""
        raise NotImplementedError

    @torch.no_grad()
    def measure(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        y = A(x) + sigma * N(0, I)
        """
        y0 = self(x, generator=generator)
        noise = torch.randn(y0.shape, generator=generator, device=y0.device, dtype=y0.dtype)
        return y0 + noise * y0.new_tensor(self.sigma)

    @torch.enable_grad()
    def dps(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DPS: ∇_{xt} log p(y | x0(xt))

        We detach xt to make it a leaf (avoids backprop into sampler history),
        but we DO allow gradient to flow xt -> x0_fn(xt) -> loglik.
        """
        xt_leaf = x.detach().requires_grad_(True)

        x0 = model(xt_leaf, t).to(dtype)  # must be differentiable wrt xt_leaf for DPS
        ll = self.log_likelihood(x0, y, t, generator=generator).sum()

        (grad_xt,) = torch.autograd.grad(ll, xt_leaf)
        return grad_xt, x0.detach()

    @torch.enable_grad()
    def mpgd(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MPGD approximation:
          approximate ∇_{xt} log p(y|xt) with ∇_{x0} log p(y|x0)

        Implementation detail:
          - compute x0 = x0(xt, t) but DETACH it so it is treated as an independent variable
          - then take gradient wrt x0 (not xt)
        """
        with torch.no_grad():
            x0 = model(x, t).to(dtype)

        x0_leaf = x0.detach().requires_grad_(True)
        ll = self.log_likelihood(x0_leaf, y, t, generator=generator).sum()

        (grad_x0,) = torch.autograd.grad(ll, x0_leaf)
        return grad_x0, x0.detach()  

    def loss(self, x: torch.Tensor, y: torch.Tensor, reshape=None,  generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Computes squared-error loss.
        Returns:
        - shape [B, N] if reshape=(B, N) is provided
        - shape [x.shape[0]] otherwise
        """
        out = self(x, generator)        
        if reshape is not None:
            B, N = reshape
            out_shape = y.shape[1:]

            out_bn = out.reshape(B, N, *out_shape)
            y_bn = y.reshape(B, N, *out_shape)

            return ((out_bn - y_bn) ** 2).flatten(start_dim=2).sum(dim=-1)
        else:
            return ((out - y) ** 2).flatten(start_dim=1).sum(-1)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reshape=None, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Log likelihood (for grad calculation purposes) of the proposal of p(y|xt)
        For DPS and MPGD: p(y|xt) \approx N(y; A(x0), std_t(t)^2)
        """
        return - 0.5 * self.loss(x, y, reshape=reshape, generator=generator) / self.std_t(t) ** 2

    def weight(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reshape=None, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Weight calculation.
        """
        return - 0.5 * self.loss(x, y, reshape=reshape, generator=generator) / self.sigma2

    @torch.enable_grad()
    def gradient(self, x: torch.Tensor, y: torch.Tensor, tau2: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Computes gradient of the loss with respect to input x.
        Used in MCMC in modifiers.
        We do the scaling (same as in DAPS) here instead of in the modifier.

        Args:
            x : Input tensor requiring gradient.
            y : Observed measurements.
        """
        x_tmp = x.clone().detach().requires_grad_(True)
        loss = self.loss(x_tmp, y, generator=generator).sum()
        x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if tau2 is not None:
            x_grad = - x_grad / tau2
        else:
            x_grad = - x_grad / self.sigma2
        return x_grad