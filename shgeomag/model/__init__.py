"""Model abstractions and interpolation for geomagnetic coefficients."""

from .coefficients import GaussCoefficients
from .model import GeomagModel

__all__ = ["GaussCoefficients", "GeomagModel"]
