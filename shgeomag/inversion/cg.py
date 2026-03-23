"""Conjugate-gradient inversion of geodetic NED observations to Gauss coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shgeomag.core.design_matrix import build_ned_jacobian
from shgeomag.model.coefficients import GaussCoefficients
from shgeomag.utils.time_utils import mjd2000_to_decimal_year


@dataclass(slots=True)
class CGInversionResult:
    """Container for CG inversion outputs."""

    coefficient_vector: np.ndarray
    coefficients: GaussCoefficients
    predicted_xyz: np.ndarray
    residual_xyz: np.ndarray
    iterations: int
    converged: bool
    final_relative_residual: float


def _coeff_arrays_from_vector(c: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_vals: list[int] = []
    m_vals: list[int] = []
    g_vals: list[float] = []
    h_vals: list[float] = []
    idx = 0
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            n_vals.append(n)
            m_vals.append(m)
            g_vals.append(float(c[idx]))
            idx += 1
            if m > 0:
                h_vals.append(float(c[idx]))
                idx += 1
            else:
                h_vals.append(0.0)
    if idx != c.size:
        raise ValueError("coefficient vector length does not match n_max")
    return (
        np.asarray(n_vals, dtype=int),
        np.asarray(m_vals, dtype=int),
        np.asarray(g_vals, dtype=float),
        np.asarray(h_vals, dtype=float),
    )


class _CachedNEDJacobianOperator:
    """Linear operator for NED Jacobian with optional unique-geometry caching."""

    def __init__(
        self,
        gc_lat_rad: np.ndarray,
        lon_rad: np.ndarray,
        r_km: np.ndarray,
        n_max: int,
        use_cache: bool = True,
    ) -> None:
        lat = np.asarray(gc_lat_rad, dtype=float).ravel()
        lon = np.asarray(lon_rad, dtype=float).ravel()
        r = np.asarray(r_km, dtype=float).ravel()
        if not (lat.size == lon.size == r.size):
            raise ValueError("gc_lat_rad, lon_rad, r_km must have same length")

        self.n_pts = lat.size
        self.n_coeff = n_max * (n_max + 2)
        self.use_cache = bool(use_cache)

        if self.use_cache:
            geom = np.column_stack([lat, lon, r])
            self.unique_geom, self.inverse = np.unique(geom, axis=0, return_inverse=True)
            self.j_unique = build_ned_jacobian(
                self.unique_geom[:, 0],
                self.unique_geom[:, 1],
                self.unique_geom[:, 2],
                n_max=n_max,
                use_cache=False,
            )
            self.n_unique = self.unique_geom.shape[0]
            self.j_full = None
        else:
            self.j_full = build_ned_jacobian(lat, lon, r, n_max=n_max, use_cache=False)
            self.unique_geom = None
            self.inverse = None
            self.j_unique = None
            self.n_unique = 0

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``J @ x`` in interleaved ``[X0,Y0,Z0,...]`` order."""
        vec = np.asarray(x, dtype=float).ravel()
        if vec.size != self.n_coeff:
            raise ValueError("input vector length must equal n_coeff")

        if self.j_full is not None:
            return self.j_full @ vec

        assert self.j_unique is not None and self.inverse is not None
        pred_unique = (self.j_unique @ vec).reshape(self.n_unique, 3)
        return pred_unique[self.inverse].reshape(-1)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """Return ``J.T @ y`` for interleaved ``[X0,Y0,Z0,...]`` residual vectors."""
        vec = np.asarray(y, dtype=float).ravel()
        if vec.size != 3 * self.n_pts:
            raise ValueError("input residual length must equal 3*n_points")

        if self.j_full is not None:
            return self.j_full.T @ vec

        assert self.j_unique is not None and self.inverse is not None
        y_pts = vec.reshape(self.n_pts, 3)
        y_unique = np.zeros((self.n_unique, 3), dtype=float)
        np.add.at(y_unique, self.inverse, y_pts)
        return self.j_unique.T @ y_unique.reshape(-1)


def forward_ned_from_coefficients(
    coefficient_vector: np.ndarray,
    gc_lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    r_km: np.ndarray,
    n_max: int,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute predicted geodetic ``[X, Y, Z]`` from Gauss coefficients.

    Parameters
    ----------
    coefficient_vector : ndarray
        Coefficient vector in ``[g10, g11, h11, g20, ...]`` order.
    gc_lat_rad : ndarray
        Geocentric latitude in radians.
    lon_rad : ndarray
        Geocentric longitude in radians.
    r_km : ndarray
        Geocentric radius in km.
    n_max : int
        Maximum spherical harmonic degree represented in ``coefficient_vector``.
    use_cache : bool, default=True
        Enable unique-geometry cache when computing Jacobian products.

    Returns
    -------
    ndarray
        Predicted geodetic NED components as shape ``(N, 3)``.
    """
    op = _CachedNEDJacobianOperator(gc_lat_rad, lon_rad, r_km, n_max=n_max, use_cache=use_cache)
    pred = op.matvec(np.asarray(coefficient_vector, dtype=float))
    return pred.reshape(-1, 3)


def invert_gauss_coefficients_cg(
    data: np.ndarray,
    n_max: int,
    max_iter: int = 200,
    tol: float = 1e-8,
    damping: float = 0.0,
    x0: np.ndarray | None = None,
    use_cache: bool = True,
    epoch_year: float | None = None,
) -> CGInversionResult:
    """Invert NED observations for Gauss coefficients with conjugate gradient.

    Parameters
    ----------
    data : ndarray
        Observation matrix with columns
        ``[MJD2000, gc_lat_deg, lon_deg, r_km, X_nT, Y_nT, Z_nT]``.
    n_max : int
        Target maximum spherical harmonic degree.
    max_iter : int, default=200
        Maximum CG iterations on normal equations.
    tol : float, default=1e-8
        Relative residual stopping threshold.
    damping : float, default=0.0
        Tikhonov damping parameter ``lambda`` for
        ``(J^T J + lambda I) c = J^T d``.
    x0 : ndarray, optional
        Initial coefficient vector. Defaults to zeros.
    use_cache : bool, default=True
        Enable unique-geometry cache for Jacobian products.
    epoch_year : float, optional
        Epoch assigned to the output ``GaussCoefficients`` object. If omitted,
        the median input MJD2000 is converted to decimal year.

    Returns
    -------
    CGInversionResult
        Inverted coefficients, diagnostics, and fitted NED values.

    Notes
    -----
    Solves the linear least-squares problem in coefficient space with CG on
    the normal equations, where ``J = d[X,Y,Z]/d[g,h]``.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("data must have shape (N, 7): [MJD2000,gc_lat,lon,r,X,Y,Z]")
    if n_max <= 0:
        raise ValueError("n_max must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")

    gc_lat_rad = np.deg2rad(arr[:, 1])
    lon_rad = np.deg2rad(arr[:, 2])
    r_km = arr[:, 3]
    d_obs = arr[:, 4:7].reshape(-1)

    op = _CachedNEDJacobianOperator(gc_lat_rad, lon_rad, r_km, n_max=n_max, use_cache=use_cache)
    n_coeff = op.n_coeff

    c = np.zeros(n_coeff, dtype=float) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    if c.size != n_coeff:
        raise ValueError(f"x0 must have length {n_coeff}")

    def normal_matvec(v: np.ndarray) -> np.ndarray:
        return op.rmatvec(op.matvec(v)) + damping * v

    b = op.rmatvec(d_obs)
    r = b - normal_matvec(c)
    p = r.copy()
    rr = float(r @ r)
    b_norm = float(np.linalg.norm(b))
    target = tol * (b_norm if b_norm > 0.0 else 1.0)
    converged = False

    for it in range(1, max_iter + 1):
        ap = normal_matvec(p)
        denom = float(p @ ap)
        if np.isclose(denom, 0.0):
            iterations = it - 1
            break
        alpha = rr / denom
        c += alpha * p
        r -= alpha * ap
        rr_new = float(r @ r)
        if np.sqrt(rr_new) <= target:
            converged = True
            rr = rr_new
            iterations = it
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    else:
        iterations = max_iter

    predicted = op.matvec(c).reshape(-1, 3)
    residual = arr[:, 4:7] - predicted
    rel_res = float(np.sqrt(rr) / (b_norm if b_norm > 0.0 else 1.0))

    if epoch_year is None:
        epoch_year = float(np.median(mjd2000_to_decimal_year(arr[:, 0])))
    n_array, m_array, g_vals, h_vals = _coeff_arrays_from_vector(c, n_max)
    coeff = GaussCoefficients(
        epoch=float(epoch_year),
        n_max=n_max,
        n_array=n_array,
        m_array=m_array,
        g=g_vals,
        h=h_vals,
    )

    return CGInversionResult(
        coefficient_vector=c,
        coefficients=coeff,
        predicted_xyz=predicted,
        residual_xyz=residual,
        iterations=iterations,
        converged=converged,
        final_relative_residual=rel_res,
    )


__all__ = ["CGInversionResult", "invert_gauss_coefficients_cg", "forward_ned_from_coefficients"]

