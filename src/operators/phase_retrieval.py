from .base import Operator
from src.registry import OPERATORS
import torch

@OPERATORS.register('phase_retrieval')    
class PhaseRetrieval(Operator):
    def __init__(self, std_schedule, oversample=2.0, resolution=256, sigma=0.05, norm="ortho", **kwargs):
        super().__init__(std_schedule, sigma, **kwargs)
        self.pad = int((oversample / 8.0) * resolution)
        self.norm = norm

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        # Expected x range [-1,1]
        x = x * 0.5 + 0.5
        x = torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad))

        # Same as in DPS and DAPS in 3 lines.
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        X = torch.fft.fft2(x, dim=(-2, -1), norm=self.norm)
        X = torch.fft.fftshift(X, dim=(-2, -1))

        return X.abs()