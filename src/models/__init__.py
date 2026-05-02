"""
Models package public API.

Models are constructed via loader/dispatch logic (no registry). This keeps imports
light and avoids requiring model backends to self-register at import time.
"""

from .base import DiffusionModel
from .loader import get_model

__all__ = ["DiffusionModel", "get_model"]
