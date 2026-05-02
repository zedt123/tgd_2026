from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
import math

from src.registry import FORWARDS

class ForwardDiffusionSampler(ABC):
    """
    Abstract base class for forward diffusion samplers using PyTorch tensors.
    """
    def sample(
        self,
        x: Tensor,
        *,
        t_to: float,
        t_from: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if t_from < 0.0:
            raise ValueError("Time must be non-negative!")
        if t_to <= t_from:
            raise ValueError(f"t_to {t_to} must be > t_from {t_from} for forward diffusion.")

        return self._sample_impl(x, t_from, t_to, generator=generator)

    def __call__(
        self,
        x: Tensor,
        *,
        t_to: float,
        t_from: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        return self.sample(x, t_to=t_to, t_from=t_from, generator=generator)

    @abstractmethod
    def _sample_impl(
        self,
        x: Tensor,
        t_from: float,
        t_to: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Implement the transition x(t_from) -> x(t_to).
        """
        raise NotImplementedError

@FORWARDS.register('edm')    
class EDMForwardDiffusionSampler(ForwardDiffusionSampler):
    def _sample_impl(
        self,
        x: Tensor,
        t_from: float,
        t_to: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if t_from == 0.0:
            std_increment = t_to
        else:
            var_increment = t_to**2 - t_from**2
            std_increment = math.sqrt(var_increment)

        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + std_increment * noise