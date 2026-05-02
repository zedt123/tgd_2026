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

from typing import Any, Optional

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

def _pack_for_traj_padded(
    x: torch.Tensor,
    *,
    B: int,
    N: int,
    particles_to_save: int,
) -> torch.Tensor:
    """
    Save up to `particles_to_save` particles and pad with zeros if N is smaller,
    so trajectory stacking keeps a fixed shape across time.

    Assumes pack_for_traj returns shape:
        (B, K_saved, C, H, W)
    """
    take = min(int(N), int(particles_to_save))
    frame = pack_for_traj(x, B, N, particles_to_save=take)

    current = int(frame.shape[1])   # particle dimension
    target = int(particles_to_save)

    if current < target:
        pad_shape = list(frame.shape)
        pad_shape[1] = target - current
        pad = torch.zeros(pad_shape, dtype=frame.dtype, device=frame.device)
        frame = torch.cat([frame, pad], dim=1)

    return frame

@SAMPLERS.register("atgd")
class ATGD(Sampler):
    def __init__(
        self,
        ctx: SDEContext,
        *,
        x0_solver: Solver,
        x0_modifier: Modifier,
        forward_process: ForwardDiffusionSampler,
        annealing_schedule: AnnealingSchedule,
        process_schedule: TimeDiscretisation,
        stop: float = 0.35,
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

        if not (0.0 <= stop <= 1.0):
            raise ValueError(f"stop must be in [0,1], got {stop}")
        self.stop = float(stop)
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
        traj_steps: int = 1,
        particles_for_traj: int = 1,
    ) -> SamplerOutput:
        stop = self.stop
        assert 0 <= stop <= 1, "Stop must be between 0 and 1."

        g_init = self._rng(rng, "init")
        g_sde = self._rng(rng, "sde")
        g_mcmc = self._rng(rng, "mcmc")
        g_smc = self._rng(rng, "smc")
        g_forward = self._rng(rng, "forward")

        self._prepare_ctx(N=N, y=y, dtype=dtype)
        B = self.ctx.B
        device = self.ctx.device

        x = self.create_xT(N=N, generator=g_init)
        t_grid = self.process_schedule.get_timesteps(device=device, dtype=dtype)
        beta_grid = self.annealing_schedule.get_betas(device=device, dtype=dtype, sigmas=t_grid)
        assert t_grid.shape == beta_grid.shape, "Time grid and betas grid have different shapes."

        steps = len(t_grid) - 1
        x = x * t_grid[0]

        # Outer SMC state before pruning.
        N_current = int(N)
        pruned = False

        logw = torch.full((B, N_current), -math.log(N_current), device=device, dtype=dtype)
        log_ess_threshold = torch.log(torch.tensor(self.ess_threshold, device=device, dtype=dtype))
        logN = torch.log(torch.tensor(N_current, device=device, dtype=dtype))

        save_traj = traj_steps >= 2
        if save_traj:
            target = min(traj_steps, steps + 1)
            save_idx = torch.linspace(0, steps, steps=target).round().long().tolist()
            save_idx = set(save_idx)

            traj = []
            traj_times = []

            fixed_particles_to_save = min(int(particles_for_traj), int(N))

            traj.append(
                _pack_for_traj_padded(
                    x,
                    B=B,
                    N=N_current,
                    particles_to_save=fixed_particles_to_save,
                )
            )
            traj_times.append(t_grid[0].detach().cpu())

        for i in range(steps):
            t_prev = t_grid[i]
            t_curr = t_grid[i + 1]
            beta_prev = beta_grid[i]
            beta_curr = beta_grid[i + 1]

            self.ctx.set_beta(beta_prev.item())

            # Propagate: xt -> x0
            x0, _ = self.x0_solver.solve(x, t_max=t_prev, generator=g_sde)

            # Modify x0
            outer_ratio = 1.0 if steps == 1 else i / float(steps - 1)
            x0_tilde = self.x0_modifier(
                x0,
                rt=t_prev,
                outer_ratio=outer_ratio,
                generator=g_mcmc,
                dtype=modifier_dtype,
            )
            x0_tilde = x0_tilde.to(dtype)

            if save_traj and (i + 1) in save_idx:
                traj.append(
                    _pack_for_traj_padded(
                        x0_tilde,
                        B=B,
                        N=N_current,
                        particles_to_save=fixed_particles_to_save,
                    )
                )
                traj_times.append(t_curr.detach().cpu())

            # Outer SMC update only before pruning.
            if (not pruned) and (beta_curr > beta_prev):
                logw_new = self.ctx.operator.weight(
                    x=x0_tilde,
                    y=self.ctx.y_flat,
                    t=t_prev,
                    reshape=(B, N_current),
                )
                logw_new = (beta_curr - beta_prev) * logw_new
                logw = logw + logw_new

                logw = torch.log_softmax(logw, dim=-1)
                log_ess = log_ess_from_logw(logw)
                do_resample = log_ess <= log_ess_threshold + logN

                (x0_tilde,), logw, _ = self.resampler(
                    x0_tilde,
                    log_w=logw,
                    do_resample=do_resample,
                    reshape=(B, N_current),
                    generator=g_smc,
                    return_ancestor=save_traj,
                )

                x0_tilde = x0_tilde.to(dtype)
                logw = logw.to(dtype)

            # Forward diffuse to get next xt state.
            if t_curr > 0.0:
                x = self.forward_process(x0_tilde, t_to=t_curr, generator=g_forward)
            else:
                x = x0_tilde

            # Prune once after stop, using the same loss rule, but keep the already-forwarded x.
            if (not pruned) and (i >= (stop * steps)):
                loss = self.ctx.operator.loss(x=x0_tilde, y=self.ctx.y_flat, reshape=(B, N_current))  # [B, N]
                best_idx = loss.argmin(dim=1)  # [B]

                x = x.reshape(B, N_current, *x.shape[1:])
                x = x[torch.arange(B, device=x.device), best_idx]  # [B, ...]

                N_current = 1
                self.ctx.set_N(1)
                pruned = True

                # Outer SMC state no longer needed after pruning.
                logw = torch.zeros((B, 1), device=device, dtype=dtype)
                logN = torch.log(torch.tensor(1.0, device=device, dtype=dtype))

        info = SMCInfo(
            timesteps=t_grid.cpu(),
            traj=torch.stack(traj, dim=0) if save_traj else None,
            traj_times=torch.stack(traj_times, dim=0) if save_traj else None,
        )

        x = x.reshape(B, N_current, *x.shape[1:]).cpu()
        return SamplerOutput(x0=x, info={"smc": info, "stop": self.stop})