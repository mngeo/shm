"""Top-level package for shgeomag geomagnetic field modeling."""

from .data.container import InputData
from .io.reader import load_model
from .model.model import GeomagModel

__all__ = ["GeomagModel", "InputData", "load_model"]
