"""
Simple registries and factory helpers for pluggable components.

Used to map string names from config (e.g., "inpaint", "smc_daps_like") to the
corresponding Python classes/functions via a decorator-based registration.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple name -> object registry with decorator-based registration."""
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator to register an item under a given name."""
        def wrapper(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                warnings.warn(
                    f"{self.kind} '{name}' already registered with {self._items[name]}; "
                    f"overwriting with {obj}.",
                    UserWarning,
                )
            self._items[name] = obj
            # Optional: attach name attribute for debugging
            try:
                setattr(obj, "name", name)
            except Exception:
                pass
            return obj
        return wrapper

    def get(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._items[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class/callable with kwargs."""
        obj = self.get(name)
        # If it's a class, calling it constructs; if function, calling executes factory
        return obj(**kwargs)

    def names(self) -> Iterable[str]:
        return self._items.keys()


# Create shared registries
OPERATORS: Registry[Any] = Registry("operator")
PROPOSAL_SCHEDULES: Registry[Any] = Registry("proposal_schedule")
SAMPLERS: Registry[Any] = Registry("sampler")
DRIFTS: Registry[Any] = Registry("drift")
DIFFUSIONS: Registry[Any] = Registry("diffusion")
SCHEDULES: Registry[Any] = Registry("schedule")
MODIFIERS: Registry[Any] = Registry("modifier")
FORWARDS: Registry[Any] = Registry("forward")
ANNEALING: Registry[Any] = Registry("annealing")
SOLVERS: Registry[Any] = Registry("solver")
DATASET: Registry[Any] = Registry("dataset")
PAIRED_METRICS: Registry[Any] = Registry("paired_metric")
DIST_METRICS: Registry[Any] = Registry("dist_metric")