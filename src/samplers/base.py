"""
Sampler interface.

A Sampler in this repo is an algorithm that produces samples x0 from an
(unconditional) pretrained diffusion model, optionally using conditioning
information via an operator and/or x0 modification.

Samplers may internally:
- run an x0 solver (ODE/SDE integrator, possibly SMC),
- apply an x0_modifier,
- forward diffuse,
- perform annealed SMC across pipeline steps.

This base class defines a small, consistent API for scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from src.diffusion.types import SDEContext

import torch


@dataclass
class SamplerOutput:
    """
    Standard output container for samplers.

    Attributes:
        x0: Final samples at time 0 (or "clean" domain), shape (B, ...)
        info: Optional diagnostics (ESS history, acceptance rates, timing, etc.)
    """
    x0: torch.Tensor
    info: Optional[Dict[str, Any]] = None

@dataclass
class SMCInfo:
    ess_history:   Optional[torch.Tensor] = None          # (T+1,B)
    logw_hist:     Optional[torch.Tensor] = None          # (T+1,B,N)
    resample_hist: Optional[torch.Tensor] = None          # (T+1,B)
    timesteps:     Optional[torch.Tensor] = None          # (T)
    traj: Optional[torch.Tensor] = None  # (T+1,B,N,...) if save_traj=True
    traj_times: Optional[torch.Tensor] = None
    ancestor_hist: Optional[torch.Tensor] = None


class Sampler(ABC):
    """
    Abstract base sampler.

    Conditioning is not passed to `sample`. If needed, it is stored in the
    sampler's components at init time (e.g., in drift/modifier contexts).
    """
    def __init__(self, ctx: SDEContext):
        self.ctx = ctx

    def create_xT(
        self,
        *,
        N: int,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create initial terminal noise xT of shape (B*N, *model.get_in_shape()).

        Args:
            N: number of samples/particles per conditioning item
            generator: torch.Generator for reproducible sampling
            dtype: dtype for xT (defaults to torch.float32)

        Returns:
            xT: tensor of shape (B*N, *in_shape)
        """
        B = int(getattr(self.ctx, "B"))
        if B < 1:
            raise ValueError(f"ctx.B must be >= 1, got {B}")
        N = int(N)
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")

        in_shape = self.ctx.model.get_in_shape()
        self.in_shape = in_shape
        if in_shape is None:
            raise ValueError("model.get_in_shape() returned None")
        if not isinstance(in_shape, (tuple, list)) or len(in_shape) == 0:
            raise ValueError(f"model.get_in_shape() must be a non-empty tuple/list, got {in_shape}")

        xT = torch.randn((B * N, *in_shape), device=self.ctx.device, dtype=dtype, generator=generator)
        return xT

    @abstractmethod
    def sample(
        self,
        *,
        N: int,
        y: torch.Tensor | None,
        rng: Optional[Any] = None,
        traj_steps: int = 1,
        particles_for_traj: int = 1,
    ) -> SamplerOutput:
        """
        Generate samples x0.

        Args:
            N: number of samples/particles per conditioning item (or output batch 
            size if no conditioning).
            y (optional): conditioning tensor of shape (B_total, ...)
            rng: RNG bundle or torch.Generator(s) used by solver/SMC/modifiers.

        Returns:
            SamplerOutput containing x0 and optional diagnostics.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def batch_sample(
        self,
        *,
        N: int,
        y: torch.Tensor,
        B_chunk: int,
        rng: Optional[Any] = None,
        traj_steps: int = 1,
        particles_for_traj: int = 1,
    ) -> SamplerOutput:
        """
        Generate samples x0 by splitting y into batches.
        Args:
            N: number of samples per condition
            y: conditioning tensor of shape (B_total, ...)
            B_chunk: number of conditions per batch call
            rng: optional RNG bundle

        Returns:
            SamplerOutput with:
            - x0: (B_total, N, ...)
            - info: {"chunks": [info_chunk0, info_chunk1, ...]}
        """ 
        B_total = int(y.shape[0]) 

        all_x0 = []
        smc_infos = []

        for s in range(0, B_total, B_chunk):
            y_chunk = y[s : s + B_chunk]
            out = self.sample(N=N, y=y_chunk, rng=rng, traj_steps=traj_steps, particles_for_traj=particles_for_traj)
            
            all_x0.append(out.x0)       # (B_chunk, N, ...)
            smc_infos.append(out.info["smc"])
        x0 = torch.cat(all_x0, dim=0)   # (B_total, N, ...)

        timesteps = smc_infos[0].timesteps

        def _cat_optional(xs, dim: int):
            # Return None if all entries are None.
            if all(x is None for x in xs):
                return None
            # Guard against inconsistent chunk outputs.
            if any(x is None for x in xs):
                raise ValueError("Inconsistent optional history across chunks: some are None and some are not.")
            return torch.cat(xs, dim=dim)

        # dim=0 is time, we want to cat across batch
        ess_history = _cat_optional([i.ess_history for i in smc_infos], dim=1)
        logw_hist   = _cat_optional([i.logw_hist for i in smc_infos],   dim=1)
        res_hist    = _cat_optional([i.resample_hist for i in smc_infos], dim=1)

        if traj_steps >= 2:
            traj = torch.cat([i.traj for i in smc_infos], dim=1)
            traj_times = smc_infos[0].traj_times
            ancestor_hist = _cat_optional([i.ancestor_hist for i in smc_infos], dim=1)
        else:
            traj = None
            traj_times = None
            ancestor_hist = None

        info = SMCInfo(
            ess_history=ess_history,
            logw_hist=logw_hist,
            resample_hist=res_hist,
            timesteps=timesteps,
            traj=traj,
            traj_times=traj_times,
            ancestor_hist=ancestor_hist
        )

        return SamplerOutput(x0=x0, info={"smc": info})
    
    def _rng(self, rng: Optional[Any], name: str):
        return None if rng is None else getattr(rng, name, None)
    
    def _prepare_ctx(self, N: int, y: torch.Tensor | None, dtype: torch.dtype):
        if y is not None:
            self.ctx.set_y(y, dtype=dtype)
        self.ctx.set_N(N)
