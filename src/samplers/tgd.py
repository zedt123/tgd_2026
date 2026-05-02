"""
Annealed SMC sampler over a pipeline kernel:

    xt  --(x0_solver)-->  x0  --(x0_modifier)-->  x0_tilde  --(forward_process)-->  x_{t-1}

At each annealing step k, we update particle weights using an annealed likelihood
increment, typically:

    logw += (beta_k - beta_{k-1}) * log p(y | x0_tilde)

This single sampler can recover special cases like DAPS / RePS by choosing an
annealing schedule with constant beta=1 and/or specific components.

Design choices in this repo:
- Conditioning (operator, y, masks, etc.) is bound into the components at init time
  (usually via a shared context), not passed through `sample(...)`.
- The pretrained diffusion model is unconditional; at runtime we bind `model`
  into contexts that need it (drift, x0_modifier, etc.).
"""

from __future__ import annotations

from typing import Any, Optional, List

import math
import torch

from src.registry import SAMPLERS
from .base import Sampler, SamplerOutput, SMCInfo

from src.diffusion import Solver, SDEContext, TimeDiscretisation
from .forward_process import ForwardDiffusionSampler
from .annealing import AnnealingSchedule
from .x0_modifiers import Modifier

from src.smc.smc_utils import log_ess_from_logw         
from src.smc.resampling import systematic_resample_batched   
from src.io import pack_for_traj


@SAMPLERS.register("tgd")
class TGD(Sampler):
    """
    Tempered/Annealed guided diffusion.

    Parameters
    ----------
    x0_solver:
        An object that can map xT -> x0. Typically a diffusion Solver instance
        (from src/diffusion) already initialised with drift/diffusion/schedule.
        Must implement: solve(xT, rng=..., save_traj=False) -> (x0, info)

    x0_modifier:
        Callable or object that modifies x0 conditionally (or no-op).
        Must implement: __call__(x0, rng=...) -> x0_tilde

    forward_process:
        Object implementing the forward diffusion / propagation step.
        Must implement: forward(x0_tilde, rng=...) -> xT_next
                      log_likelihood(x0_tilde) -> (B,) tensor  (or scalar if unconditional)

    annealing_schedule:
        Provides a sequence of betas in [0,1] (or generally nondecreasing).
        Must implement: betas() -> 1D tensor/list of length K (usually ending at 1.0)

    process_schedule:
        Time schedule for where to forward diffusion after each x0 prediction.        

    ess_threshold:
        Resample when ESS / N < ess_threshold. Typical values: 0.5.

    resampler:
        Callable that returns ancestor indices given normalized log weights.
        Default: systematic resampling.

    Notes
    -----
    - This is the *outer* SMC. If your x0_solver itself uses SMC internally, that's fine;
      treat it as part of the kernel.
    """

    def __init__(
        self,
        ctx: SDEContext,
        *,
        x0_solver: Solver,
        x0_modifier: Modifier,
        forward_process: ForwardDiffusionSampler,
        annealing_schedule: AnnealingSchedule,
        process_schedule: TimeDiscretisation,
        ess_threshold: float = 0.5,
        resampler=systematic_resample_batched,
    ):
        self.x0_solver = x0_solver
        self.x0_modifier = x0_modifier
        self.forward_process = forward_process

        self.annealing_schedule = annealing_schedule
        self.process_schedule = process_schedule

        self.ess_threshold = float(ess_threshold)
        self.resampler = resampler
        super().__init__(ctx)
        
    @torch.no_grad()
    def sample(
        self,
        *,
        N: int,
        y: torch.Tensor | None,
        rng: Optional[Any] = None,
        dtype: torch.dtype = torch.float32,
        modifier_dtype: torch.dtype = torch.float32,
        traj_steps: int = 1, # number of trajectory steps to save
        particles_for_traj: int = 1, # particles of trajectory to save
    ) -> SamplerOutput:
        """
        Run outer annealed SMC and return final x0 samples (after last modify step).
        Net is internally float32 (so is modifer) but everything else uses float64 (similar to EDM).

        Args:
            model: unconditional diffusion model (x0 predictor). Bound into contexts at runtime.
            rng: RNG bundle (recommended to have attributes: sde, mcmc, smc).

        Returns:
            SamplerOutput(x0=<final samples>, info=<diagnostics>).
            Final samples are of shape (B, N, ...) where B is the number of conditions and
            N is the number of particles.
        """
        g_init = self._rng(rng, "init")
        g_sde = self._rng(rng, "sde")
        g_mcmc = self._rng(rng, "mcmc")
        g_smc = self._rng(rng, "smc")        
        g_forward = self._rng(rng, "forward")

        self._prepare_ctx(N=N, y=y, dtype=dtype)
        B = self.ctx.B
        device  = self.ctx.device 
        x = self.create_xT(N=N, generator=g_init)

        t_grid = self.process_schedule.get_timesteps(device=device, dtype=dtype)
        beta_grid = self.annealing_schedule.get_betas(device=device, dtype=dtype, sigmas=t_grid)
        assert t_grid.shape == beta_grid.shape, "Time grid and betas grid have different shapes."

        steps = len(t_grid) - 1

        x = x * t_grid[0]

        # Log-weights for outer SMC
        logw = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
        log_ess_threshold = torch.log(torch.tensor(self.ess_threshold, device=device, dtype=dtype))
        logN = torch.log(torch.tensor(N, device=device, dtype=dtype))

        ess_hist: List[torch.Tensor] = []
        logw_hist: List[torch.Tensor] = []
        resample_hist: List[torch.Tensor] = []

        save_traj = traj_steps >= 2
        if save_traj:
            # pick indices to store (evenly spaced)
            target = min(traj_steps, steps + 1)
            save_idx = torch.linspace(0, steps, steps=target).round().long().tolist()
            save_idx = set(save_idx)

            traj = []
            traj_times = []
            ancestor_hist = []

            # Save initial frame (step 0)
            traj.append(pack_for_traj(x, B, N, particles_to_save=particles_for_traj))
            traj_times.append(t_grid[0].detach().cpu())

        for i in range(steps):
            t_prev = t_grid[i]
            t_curr = t_grid[i + 1]
            beta_prev = beta_grid[i]
            beta_curr = beta_grid[i + 1]

            self.ctx.set_beta(beta_prev.item())

            # Propagate: xt -> x0 (output is dtype) # (B*N,)
            x0, _ = self.x0_solver.solve(x, t_max=t_prev, generator=g_sde)

            # Modify x0 (internally uses modifier_dtype)
            outer_ratio = 1.0 if steps == 1 else i / float(steps - 1)
            x0_tilde = self.x0_modifier(x0, rt=t_prev, outer_ratio=outer_ratio, generator=g_mcmc, dtype=modifier_dtype)
            x0_tilde = x0_tilde.to(dtype)

            if save_traj and (i+1) in save_idx:
                traj.append(pack_for_traj(x0_tilde, B, N, particles_to_save=particles_for_traj)) 
                traj_times.append(t_curr.detach().cpu())
            # Update weights and resample
            if beta_curr > beta_prev:
                logw_new = self.ctx.operator.weight(x=x0_tilde, y=self.ctx.y_flat, t=t_prev, reshape=(B,N)) # [B,N]
                #logw_new = self.ctx.operator.log_likelihood(x=x0_tilde, y=self.ctx.y_flat, t=t_prev, reshape=(B,N)) # [B,N]
                logw_new = (beta_curr - beta_prev) * logw_new
                logw = logw + logw_new
                # Normalise weights.
                logw = torch.log_softmax(logw, dim=-1)
                logw_hist.append(logw.cpu()) # weights before resampling
                log_ess = log_ess_from_logw(logw)
                do_resample = log_ess <= log_ess_threshold + logN   # (B,) bool
                (x0_tilde,), logw, ancestors = self.resampler(
                                    x0_tilde,
                                    log_w=logw,
                                    do_resample=do_resample,
                                    reshape=(B, N),
                                    generator=g_smc,
                                    return_ancestor=save_traj
                                )
                if save_traj:
                    ancestor_hist.append(ancestors.detach().cpu())

                x0_tilde, logw = x0_tilde.to(dtype), logw.to(dtype)
                resample_hist.append(do_resample.detach().cpu())
            else:
                if save_traj:
                    ancestor_hist.append(torch.arange(N, device="cpu")[None, :].expand(B, N))
                log_ess = log_ess_from_logw(logw)
                resample_hist.append(torch.zeros(B, dtype=torch.bool, device="cpu"))
                logw_hist.append(logw.cpu())

            ess_hist.append(log_ess.cpu())               

            # Forward diffuse to get next xt state
            if t_curr > 0.0:
                x = self.forward_process(x0_tilde, t_to=t_curr, generator=g_forward)
            else:
                x = x0_tilde

        info = SMCInfo(
            ess_history=   torch.stack(ess_hist, dim=0),
            logw_hist=     torch.stack(logw_hist, dim=0),
            resample_hist= torch.stack(resample_hist, dim=0),
            timesteps=     t_grid.cpu(),
            traj=          torch.stack(traj, dim=0) if save_traj else None,
            traj_times=    torch.stack(traj_times, dim=0) if save_traj else None,
            ancestor_hist= torch.stack(ancestor_hist, dim=0) if save_traj else None,
        )
        x = x.reshape(B, N, *x.shape[1:]).cpu()

        return SamplerOutput(x0=x, info={"smc": info})
