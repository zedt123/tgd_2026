# Import registry API
from src.registry import OPERATORS, PROPOSAL_SCHEDULES
from .base import Operator

# Import operator modules so they register
from . import abs_val, inpaint, phase_retrieval, proposal_schedules

def get_operator(name: str, **kwargs):
    return OPERATORS.create(name, **kwargs)

def get_prop_schedule(name: str):
    return PROPOSAL_SCHEDULES.get(name)

__all__ = ["get_operator", "get_prop_schedule", "Operator"]
