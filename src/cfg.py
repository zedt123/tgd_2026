from __future__ import annotations

import torch
from omegaconf import DictConfig


def set_device(cfg: DictConfig) -> str:
    device = cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def cleanup_cuda(device: str) -> None:
    import gc
    import torch

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def compute_nfe(cfg: DictConfig) -> int:
    process_steps = int(cfg.sampler.process_schedule.kwargs.num_steps)
    inner_steps = int(cfg.sampler.x0_solver.schedule.kwargs.num_steps)
    effective_nfe = process_steps * inner_steps

    return int(round(float(effective_nfe)))