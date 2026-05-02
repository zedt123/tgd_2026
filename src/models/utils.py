"""
Small utilities.

Keep this file lightweight: helpers here should not import heavy model backends.
"""

from __future__ import annotations

import torch

def strip_prefix(k: str) -> str:
    for p in ("model.", "module."):
        if k.startswith(p):
            return k[len(p):]
    return k

def count_parameters(module: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in module.parameters())