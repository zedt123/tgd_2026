from abc import ABC, abstractmethod
import torch
from src.registry import SCHEDULES

class TimeDiscretisation(ABC):
    """
    Base class for timestep schedules.

    Convention:
    - self.t_max is a GLOBAL maximum allowed t_max.
    - You can request a smaller t_max per call via get_timesteps(t_max=...).
    - t_max > t_min > 0
    - 0.0 is appended at the end for numerical integration
    - model is NEVER evaluated at t = 0
    """

    def __init__(self, num_steps: int, t_max: float, t_min: float):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not (t_max > t_min > 0.0):
            raise ValueError("Require t_max > t_min > 0")

        self.num_steps = int(num_steps)
        self.t_max = float(t_max)
        self.t_min = float(t_min)

    @abstractmethod
    def _positive_timesteps(self, t_max: float, device=None, dtype=None) -> torch.Tensor:
        """
        Args:
            t_max: per-call maximum time (<= self.t_max)
        Returns:
            t_pos: shape (num_steps,), strictly positive, descending
        """
        raise NotImplementedError

    def get_timesteps(self, t_max: float | None = None, device=None, dtype=None) -> torch.Tensor:
        """
        Args:
            t_max: optional per-call override for maximum time.
                  If None, uses self.t_max.
        Returns:
            t: shape (num_steps + 1,)
               [t_max, ..., t_min, 0.0]
        """
        if t_max is None:
            t_max = self.t_max
        else:
            t_max = float(t_max)
            if t_max > self.t_max:
                raise ValueError(f"Requested t_max={t_max} exceeds global self.t_max={self.t_max}")
            if t_max < self.t_min:
                raise ValueError(f"Require requested t_max >= t_min. Got t_max={t_max}, t_min={self.t_min}")
            
        zero = torch.zeros(1, device=device, dtype=dtype)

        # Edge case: exactly equal -> return [t_max, 0]
        if t_max == self.t_min:
            t0 = torch.tensor([t_max], device=device, dtype=dtype)
            return torch.cat([t0, zero], dim=0)

        t_pos = self._positive_timesteps(t_max=t_max, device=device, dtype=dtype)
        return torch.cat([t_pos, zero], dim=0)

    def get_step_sizes(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Return per-step Δt_i = t_{i+1} - t_i (signed; negative for descending schedules).
        """
        return ts[1:] - ts[:-1]

@SCHEDULES.register("uniform")    
class UniformTimeDiscretisation(TimeDiscretisation):
    """
    Uniform spacing in time.
    """

    def __init__(
        self,
        num_steps: int,
        t_max: float = 1.0,
        t_min: float = 0.002,
    ):
        super().__init__(num_steps, t_max, t_min)

    def _positive_timesteps(self, t_max: float, device=None, dtype=None) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor(
                [t_max], device=device, dtype=dtype
            )

        return torch.linspace(
            t_max,
            self.t_min,
            self.num_steps,
            device=device,
            dtype=dtype,
        )

@SCHEDULES.register("edm")
class EDMTimeDiscretisation(TimeDiscretisation):
    """
    EDM / Karras-style sigma discretisation.
    """
    def __init__(
        self,
        num_steps: int,
        t_max: float = 80.0,    
        t_min: float = 0.002,   
        rho: float = 7.0,
    ):
        self.rho = float(rho)
        super().__init__(num_steps, t_max, t_min)

    def _positive_timesteps(self, t_max: float, device=None, dtype=None) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor(
                [t_max], device=device, dtype=dtype
            )

        i = torch.arange(self.num_steps, device=device, dtype=dtype)
        ramp = i / (self.num_steps - 1)

        inv_rho = 1.0 / self.rho
        t_pow = (t_max ** inv_rho) + ramp * (
            (self.t_min ** inv_rho) - (t_max ** inv_rho)
        )
        t = t_pow ** self.rho

        # Set first/last times explicitly to avoid rounding errors.
        t[0] = t_max
        t[-1] = self.t_min       
        return t
