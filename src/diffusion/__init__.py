from src.registry import DRIFTS, DIFFUSIONS, SCHEDULES, SOLVERS

from .types import SDEContext, Process
from .solvers import Solver
from .schedules import TimeDiscretisation

# Import modules so classes register
from . import drift 
from . import diffusion 
from . import schedules
from . import solvers 

def get_drift(name: str, **kwargs):
    return DRIFTS.create(name, **kwargs)

def get_diffusion(name: str, **kwargs):
    return DIFFUSIONS.create(name, **kwargs)

def get_schedule(name: str, **kwargs):
    return SCHEDULES.create(name, **kwargs)

def get_solver(name: str, **kwargs):
    return SOLVERS.create(name, **kwargs)


__all__ = [
    "SDEContext",
    "Process",
    "Solver",
    "TimeDiscretisation",
    "get_schedule",
    "get_drift",
    "get_diffusion",
    "get_solver",
]