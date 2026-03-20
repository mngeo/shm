"""High-level geomagnetic model object wrapping coefficients and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from .coefficients import GaussCoefficients
from .interpolation import interpolate_coefficients


@dataclass(slots=True)
class GeomagModel:
    """Geomagnetic model with metadata and interpolation behavior."""

    source_file: str
    model_name: str
    format: str
    valid_from: float
    valid_to: float
    epochs: list[float]
    coeffs: dict[float, GaussCoefficients]
    sv_coeffs: GaussCoefficients | None = None
    n_max: int = 0
    n_truncate: int = field(default=0)

    def __post_init__(self) -> None:
        if self.n_max <= 0:
            self.n_max = max(c.n_max for c in self.coeffs.values())
        if self.n_truncate <= 0:
            self.n_truncate = self.n_max

    def info(self) -> str:
        """Return a one-line summary for this model."""
        n_coeff = self.n_truncate * (self.n_truncate + 2)
        return (
            f"{self.model_name} ({self.format}), epochs={self.epochs}, "
            f"n_max={self.n_max}, n_truncate={self.n_truncate}, coeffs={n_coeff}"
        )

    def set_truncation(self, n: int) -> None:
        """Set active truncation degree."""
        if n <= 0:
            raise ValueError("Truncation degree must be positive")
        if n > self.n_max:
            raise ValueError("Truncation cannot exceed model n_max")
        self.n_truncate = n

    def get_coefficients(self, year: float) -> GaussCoefficients:
        """Return interpolated/truncated coefficients for the requested year."""
        coeffs = interpolate_coefficients(year, self.coeffs, self.valid_from, self.valid_to, self.sv_coeffs)
        if self.n_truncate < coeffs.n_max:
            return coeffs.truncate(self.n_truncate)
        return coeffs


__all__ = ["GeomagModel"]
