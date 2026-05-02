"""
Absolute Value Operator.
"""

from .base import Operator
from src.registry import OPERATORS
import torch

@OPERATORS.register('abs')    
class AbsVal(Operator):
    def __init__(self, std_schedule, sigma=0.05, **kwargs):
        super().__init__(std_schedule, sigma, **kwargs)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.abs(x)
        