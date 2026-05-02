from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from src.registry import SOLVERS

if TYPE_CHECKING:
    from .types import Process
    

@dataclass
class SolverInfo:
    n_steps: int
    ts: torch.Tensor
    traj: Optional[torch.Tensor] = None  # (T+1, B, ...) if save_traj=True


class Solver(ABC):
    """
    Base solver for dynamics of the form:

        dX = f(X, t) dt + g(t) dW

    Uses a bundled `Process`:
      - process.drift is the drift object (ctx bound at init)
      - process.diffusion is optional (None => ODE-style run)

    Notes:
      - ODE solving is supported by setting diffusion=None (or having g(t)=0).
      - TimeDiscretisation appends t=0 for integration, but the *model/drift*
        must never be evaluated at t=0.
    """

    def __init__(self, process: Process):
        self.process = process

        self.schedule = process.schedule
        self.drift = process.drift
        self.diffusion = process.diffusion        


    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        dt: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Advance state from time t to t_next.

        Args:
            x: current state, shape (B, ...)
            t: scalar tensor
            t_next: scalar tensor (may be 0 at final step)
            dt: scalar tensor, dt = t - t_next (positive for descending schedules)
            generator: generator used for stochastic increments (if needed)

        Returns:
            x_next: next state, same shape as x
        """
        raise NotImplementedError

    def solve(
        self,
        x: torch.Tensor,
        t_max: float,
        generator: Optional[torch.Generator] = None,
        save_traj: bool = False,
    ) -> Tuple[torch.Tensor, SolverInfo]:
        """
        Integrate along schedule-defined timesteps.

        Args:
            x: initial noise, shape (B, ...)
            generator: generator used for stochastic solvers
            save_traj: if True, returns full trajectory stacked as (T+1, B, ...)

        Returns:
            x0: clean samples
            info: SolverInfo
        """
        ts = self.schedule.get_timesteps(t_max=t_max, device=x.device, dtype=x.dtype)  # (num_steps+1,)
        dts = self.schedule.get_step_sizes(ts)  # (num_steps,)
        self.num_steps = dts.numel()

        traj = [x.detach().cpu()] if save_traj else None

        # i runs over num_steps; final t_next is 0.0 by convention.
        for i in range(dts.numel()):
            t = ts[i]
            t_next = ts[i + 1]
            dt = dts[i]
            x = self.step(x, t, t_next, dt, generator=generator)
            if save_traj:
                traj.append(x.detach().cpu())

        info = SolverInfo(
            n_steps=dts.numel(),
            ts=ts,
            traj=torch.stack(traj, dim=0) if save_traj else None,
        )
        return x, info

    def is_stochastic(self) -> bool:
        return self.diffusion is not None

@SOLVERS.register("euler_maruyama")    
class EulerMaruyama(Solver):
    """
    Euler Maruyama solver.
    """
    def __init__(self, process: Process):
        super().__init__(process)

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        dt: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        x_next = x + self.drift(x,t) * dt
        if self.is_stochastic():
            sqrt_dt = torch.sqrt(torch.abs(dt))
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x_next = x_next + self.diffusion(t) * noise * sqrt_dt
        return x_next

@SOLVERS.register("heun")    
class Heun(Solver):
    """
    Algorithm 2 from Elucidating Diffusion Models.
    """
    def __init__(
        self,
        process: Process,
        S_churn: float = 30.0,
        S_min: float = 0.01,
        S_max: float = 1.0,
        S_noise: float = 1.007
    ):
        super().__init__(process)
        if S_churn < 0:
            raise ValueError("S_churn must be >= 0")
        if not (S_min >= 0 and S_max >= S_min):
            raise ValueError("Require 0 <= S_min <= S_max")
        if S_noise < 0:
            raise ValueError("S_noise must be >= 0")

        self.S_churn = float(S_churn)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.S_noise = float(S_noise)

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        dt: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t <= self.S_max else 0  
        t_hat = t + gamma * t

        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x_hat =  x + (t_hat ** 2 - t ** 2).sqrt() * self.S_noise * noise  
        d_cur = self.drift(x_hat, t_hat)
        x_next = x_hat + (t_next - t_hat) * d_cur

        if t_next > 0:
            d_prime = self.drift(x_next, t_next)
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next