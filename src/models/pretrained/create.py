from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from . import nets
from ..utils import strip_prefix

def create_model(
    *,
    net_name: str,
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = 'cuda',
    **net_kwargs: Any,
) -> torch.nn.Module:
    """
    Create an EDM-style backbone network and optionally load pretrained weights.

    Args:
        net_name: Name of architecture in nets.py (e.g., "UNet64", "UNet128", ...)
        ckpt_path: Optional path to checkpoint (.pt/.pth). If None, returns randomly initialized net.
        net_kwargs: Keyword args forwarded to the architecture constructor.

    Returns:
        torch.nn.Module: the raw backbone network (not preconditioned).
    """
    if not hasattr(nets, net_name):
        available = [n for n in dir(nets) if not n.startswith("_")]
        raise KeyError(f"Unknown arch '{net_name}'. Available (subset): {available}")

    NetCls = getattr(nets, net_name)
    net = NetCls(**net_kwargs)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = {strip_prefix(k): v for k, v in ckpt.items()}
        net.load_state_dict(ckpt)
    except Exception as e:
        print(f"Got exception: {e} / Randomly initialize")

    return net