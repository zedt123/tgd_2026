from abc import ABC, abstractmethod
import torch

from src.registry import ANNEALING

class AnnealingSchedule(ABC):
    """
    Base class for annealing schedules for tempered diffusion.

    Convention:
    - 0 <= beta_min <= beta_max <= 1
    """

    def __init__(self, num_steps: int, beta_min: float, beta_max: float):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not (0.0 <= beta_min <= beta_max <= 1.0):
            raise ValueError("Require 0 <= beta_min <= beta_max <= 1.0")

        self.num_steps = int(num_steps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    @abstractmethod
    def get_betas(self, device=None, dtype=None, *, sigmas) -> torch.Tensor:
        """
        Returns:
            betas: shape (num_steps,)
        """
        raise NotImplementedError

@ANNEALING.register('uniform')
class UniformAnnealing(AnnealingSchedule):
    """
    Uniform spacing for betas.
    If initialised with beta_min=beta_max=1 we get no annealing.
    """
    def __init__(
        self,
        num_steps: int,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
    ):
        super().__init__(num_steps, beta_min, beta_max)

    def get_betas(self, device=None, dtype=None, *, sigmas) -> torch.Tensor:
        return torch.linspace(
            self.beta_min,
            self.beta_max,
            self.num_steps + 1,
            device=device,
            dtype=dtype,
        )
    
@ANNEALING.register('sigma')    
class SigmaDependedAnnealing(AnnealingSchedule):
    """
    Beta schedule that depends on the current noise level in the diffusion.
    """
    def __init__(
        self,
        num_steps: int,
        beta_min: float = 0.0,    
        beta_max: float = 1.0,   
        alpha: float = 1.0,
        sigma_on: float = 100,
        sigma_off: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.sigma_on = sigma_on
        self.sigma_off = sigma_off
        super().__init__(num_steps, beta_min, beta_max)

    def get_betas(self, device=None, dtype=None, *, sigmas) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor(
                [self.beta_max, self.beta_max], device=device, dtype=dtype
            )
        
        betas = torch.full_like(sigmas, fill_value=self.beta_min, device=device, dtype=dtype)

        # beta_max if larger than sigma_off
        betas = torch.where(sigmas <= self.sigma_off,
                            torch.full_like(sigmas, self.beta_max),
                            betas)
        mask = (sigmas > self.sigma_off) & (sigmas < self.sigma_on)
        t = (self.sigma_on - sigmas) / (self.sigma_on - self.sigma_off)   # t in (0,1)
        t = t.clamp(0.0, 1.0) ** self.alpha

        mid = self.beta_min + (self.beta_max - self.beta_min) * t
        betas = torch.where(mask, mid, betas)

        return betas     



