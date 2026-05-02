from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from src.operators import Operator
    from .drift import Drift
    from .diffusion import Diffusion
    from .schedules import TimeDiscretisation

@dataclass
class SDEContext:
    """
    Context shared across drift/diffusion/operator/modifier components.

    Conventions:
    - Conditioning y is provided as shape (B, ...).
    - Computation runs on a flattened particle batch of shape (B*N, ...).
    - When computing SMC diagnostics (ESS, resampling), we reshape to (B, N, ...),
      operate per-row, then flatten back.
    """
    model: Any
    lam: float
    operator: Operator
    beta: float = 1.0  # guidance strength
    lam_2: float | None = None
    device: Union[str, torch.device] = "cuda"

    def __post_init__(self) -> None:
        # Automatically precompute lambda fields at construction time.
        self.init_lambda_fields()
        # Set B to 1 for no conditioning. Change it when we add y.
        self.B = 1

    def init_lambda_fields(self) -> None:
        lam_a = torch.tensor(float(self.lam), device=self.device, dtype=torch.float32)
        if self.lam_2 is None:
            lam = lam_a
        else:
            mask = self.operator.mask # boolean mask of shape (1,1,C,H) or (1,d)
            lam_b = torch.tensor(float(self.lam_2), device=self.device, dtype=torch.float32)
            lam = torch.where(mask, lam_a, lam_b)

        self.lam_tensor = lam
        self.one_plus_lam2 = 1.0 + lam**2

    def set_N(self, N: int) -> None:
        self.N = N
        if getattr(self, "y", None) is not None:
            self.y_flat = self.y.repeat_interleave(self.N, dim=0).to(self.device)

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def set_y(self, y: torch.Tensor, dtype: torch.dtype = torch.float32,) -> None:
        self.y = y.to(dtype)
        self.B = self.y.shape[0]

@dataclass
class Process:
    schedule: TimeDiscretisation
    drift: Drift
    diffusion: Optional[Diffusion] = None
