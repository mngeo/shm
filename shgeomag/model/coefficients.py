"""Gauss coefficient container used by geomagnetic field models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GaussCoefficients:
    """Container for one coefficient epoch.

    Parameters
    ----------
    epoch : float
        Coefficient epoch in decimal year.
    n_max : int
        Maximum spherical harmonic degree.
    n_array, m_array : ndarray
        Degree/order arrays for each coefficient pair entry.
    g, h : ndarray
        Main field Gauss coefficients in nT.
    dg_dt, dh_dt : ndarray, optional
        Secular variation coefficients in nT/year.
    """

    epoch: float
    n_max: int
    n_array: np.ndarray
    m_array: np.ndarray
    g: np.ndarray
    h: np.ndarray
    dg_dt: np.ndarray | None = None
    dh_dt: np.ndarray | None = None

    def to_gh_dict(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Return dictionary mapping ``(n, m)`` to ``(g, h)`` values."""
        out: dict[tuple[int, int], tuple[float, float]] = {}
        for n, m, gv, hv in zip(self.n_array, self.m_array, self.g, self.h, strict=True):
            out[(int(n), int(m))] = (float(gv), float(hv))
        return out

    def truncate(self, n_max_new: int) -> "GaussCoefficients":
        """Return a new coefficient object truncated to degree ``n_max_new``."""
        if n_max_new <= 0:
            raise ValueError("n_max_new must be positive")
        if n_max_new > self.n_max:
            raise ValueError("n_max_new cannot exceed existing n_max")

        mask = self.n_array <= n_max_new
        return GaussCoefficients(
            epoch=self.epoch,
            n_max=n_max_new,
            n_array=self.n_array[mask].copy(),
            m_array=self.m_array[mask].copy(),
            g=self.g[mask].copy(),
            h=self.h[mask].copy(),
            dg_dt=None if self.dg_dt is None else self.dg_dt[mask].copy(),
            dh_dt=None if self.dh_dt is None else self.dh_dt[mask].copy(),
        )

    def coefficient_vector(self) -> np.ndarray:
        """Return flattened coefficient vector ``[g10, g11, h11, g20, ...]``."""
        vals: list[float] = []
        gh = self.to_gh_dict()
        for n in range(1, self.n_max + 1):
            for m in range(0, n + 1):
                g, h = gh[(n, m)]
                vals.append(g)
                if m > 0:
                    vals.append(h)
        return np.asarray(vals, dtype=float)


__all__ = ["GaussCoefficients"]
