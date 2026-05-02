# src/factory.py
from __future__ import annotations

from typing import Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from src.diffusion import SDEContext, Process, Solver, get_schedule, get_drift, get_diffusion, get_solver
from src.samplers import get_annealing, get_modifier, get_sampler, get_forward, Sampler
from src.operators import get_operator, get_prop_schedule

import torch


def _to_kwargs(node) -> dict:
    """DictConfig/dict -> plain dict (resolved)."""
    if node is None:
        return {}
    return dict(OmegaConf.to_container(node, resolve=True))


def build_context(
    cfg: DictConfig,
    *,
    model: Any,
    device: str,
    operator: Optional[Any] = None,
) -> SDEContext:
    """
    Build the shared SDEContext.

    Expects lam/lam_2 under:
      cfg.diffusion_params.lam
      cfg.diffusion_params.lam_2 (optional)

    Required:
      - model, device, lam

    Optional:
      - lam_2, operator
    """
    if "diffusion_params" not in cfg or cfg.diffusion_params is None:
        raise ValueError("Sampler cfg must contain diffusion_params (with at least lam).")

    lam = float(cfg.diffusion_params.lam)
    lam_2 = cfg.diffusion_params.get('lam_2')
    lam_2 = float(lam_2) if lam_2 is not None else None

    ctx = SDEContext(
        model=model,
        lam=lam,
        operator=operator,
        lam_2=lam_2,
        device=device,
    )
    return ctx

def build_operator_and_y(
    op_cfg: DictConfig,
    *,
    x_ref: Optional[Any] = None,
    generator: torch.Generator | None = None,
    device: str = 'cuda'
) -> Tuple[Any, Optional[Any]]:
    """
    Build operator and measurements y from a clean reference x_ref.
    """
    op_name = op_cfg.name

    proposal_name = op_cfg.get("proposal", None)  
    ProposalCls = get_prop_schedule(proposal_name) if proposal_name is not None else None
    operator = get_operator(op_name, std_schedule=ProposalCls, **_to_kwargs(op_cfg.kwargs))

    # must have x_ref
    if x_ref is None:
        raise ValueError(f"Operator '{op_name}' requires x_ref to build y.")

    with torch.no_grad():
        y = operator.measure(x_ref.to(device), generator=generator).to('cpu')
    return operator, y

def build_x0_solver(cfg: DictConfig, *, ctx: SDEContext) -> Solver:
    """
    Build the diffusion x0 solver stack:
      schedule + drift + diffusion -> process -> solver
    """
    sched_cfg = cfg.schedule
    schedule = get_schedule(sched_cfg.name, **_to_kwargs(sched_cfg.kwargs))

    drift_cfg = cfg.drift
    drift = get_drift(drift_cfg.name, ctx=ctx, **_to_kwargs(drift_cfg.kwargs))

    diff_cfg = cfg.get("diffusion", None)
    if diff_cfg is None:
        diffusion = None
    else:
        diffusion = get_diffusion(diff_cfg.name, ctx=ctx, **_to_kwargs(diff_cfg.kwargs))

    process = Process(schedule=schedule, drift=drift, diffusion=diffusion)

    solver_cfg = cfg.solver
    solver = get_solver(solver_cfg.name, process=process, **_to_kwargs(solver_cfg.kwargs))

    return solver

def build_sampler(
    cfg: DictConfig,
    *,
    model: Any,
    operator: Any,
    device: str,
) -> Tuple[Sampler]:
    """
    Build a fully-wired sampler (e.g. TGD) from config.

    Inputs
    ------
    model:
      Unconditional pretrained diffusion model.
    operator:
      Forward operator.
    device:
      "cuda" or "cpu".

    Returns
    -------
    sampler:
      Configured sampler instance ready to call sampler.sample()
    """

    # Sampler main components
    cfg = cfg.sampler
    ctx = build_context(cfg, model=model, device=device, operator=operator)
    solver = build_x0_solver(cfg.x0_solver, ctx=ctx)
    mod_cfg = cfg.x0_modifier
    modifier = get_modifier(mod_cfg.name, ctx=ctx, **_to_kwargs(mod_cfg.kwargs))
    fwd_cfg = cfg.forward_process
    forward_process = get_forward(fwd_cfg.name, **_to_kwargs(fwd_cfg.kwargs))

    # Annealing schedule (betas)
    ann_cfg = cfg.annealing
    annealing_schedule = get_annealing(ann_cfg.name, **_to_kwargs(ann_cfg.kwargs))

    # Outer schedule (restarts)
    proc_sched_cfg = cfg.process_schedule
    process_schedule = get_schedule(proc_sched_cfg.name, **_to_kwargs(proc_sched_cfg.kwargs))

    sampler_kwargs = _to_kwargs(cfg.get("kwargs", {}))

    # Build sampler
    sampler = get_sampler(
        cfg.name,
        ctx=ctx,
        x0_solver=solver,
        x0_modifier=modifier,
        forward_process=forward_process,
        annealing_schedule=annealing_schedule,
        process_schedule=process_schedule,
        **sampler_kwargs,
    )
    return sampler