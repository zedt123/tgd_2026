from src.registry import SAMPLERS, ANNEALING, FORWARDS, MODIFIERS

from . import tgd, annealing, forward_process, x0_modifiers, a_tgd
from .base import Sampler

def get_modifier(name: str, **kwargs):
    return MODIFIERS.create(name, **kwargs)

def get_forward(name: str, **kwargs):
    return FORWARDS.create(name, **kwargs)

def get_annealing(name: str, **kwargs):
    return ANNEALING.create(name, **kwargs)

def get_sampler(name: str, **kwargs):
    return SAMPLERS.create(name, **kwargs)

__all__ = ["Sampler", "get_modifier", "get_forward", "get_annealing", "get_sampler"]