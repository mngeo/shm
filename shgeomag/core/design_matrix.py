"""Design-matrix and spherical harmonic basis routines for geomagnetism."""

from __future__ import annotations

import numpy as np

from shgeomag._constants import A_REF_KM


def _prepare_angles(gc_lat_rad: np.ndarray, lon_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = 0.5 * np.pi - gc_lat_rad
    phi = np.mod(lon_rad, 2.0 * np.pi)
    sin_theta = np.sin(theta)
    return theta, phi, sin_theta


def _compute_trig(phi: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    n_pts = phi.size
    cos_mphi = np.empty((n_max + 1, n_pts), dtype=float)
    sin_mphi = np.empty((n_max + 1, n_pts), dtype=float)
    cos_mphi[0] = 1.0
    sin_mphi[0] = 0.0

    z = np.cos(phi) + 1j * np.sin(phi)
    p = np.ones_like(z, dtype=complex)
    for m in range(1, n_max + 1):
        p *= z
        cos_mphi[m] = p.real
        sin_mphi[m] = p.imag

    return cos_mphi, sin_mphi


def _compute_schmidt_p_and_dp(theta: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n_pts = theta.size

    p = np.zeros((n_max + 1, n_max + 2, n_pts), dtype=float)
    p[0, 0] = 1.0

    if n_max >= 1:
        p[1, 0] = cos_t
        p[1, 1] = sin_t

    for n in range(2, n_max + 1):
        m = n
        p[n, m] = sin_t * np.sqrt((2.0 * n - 1.0) / (2.0 * n)) * p[n - 1, m - 1]

        m = n - 1
        p[n, m] = cos_t * np.sqrt(2.0 * n - 1.0) * p[n - 1, m]

        for m in range(0, n - 1):
            a = np.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
            b = np.sqrt(
                ((2.0 * n + 1.0) * (n - 1.0 - m) * (n - 1.0 + m))
                / ((2.0 * n - 3.0) * (n * n - m * m))
            )
            p[n, m] = cos_t * a * p[n - 1, m] - b * p[n - 2, m]

    dp = np.zeros((n_max + 1, n_max + 1, n_pts), dtype=float)
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            term_up = 0.0
            if m + 1 <= n:
                term_up = 0.5 * np.sqrt((n - m) * (n + m + 1.0)) * p[n, m + 1]
            term_dn = 0.0
            if m > 0:
                term_dn = 0.5 * np.sqrt((n + m) * (n - m + 1.0)) * p[n, m - 1]
            dp[n, m] = term_up - term_dn

    return p, dp


def build_design_matrix(
    gc_lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    r_km: np.ndarray,
    n_max: int,
    n_truncate: int | None = None,
    a_ref: float = A_REF_KM,
) -> np.ndarray:
    """Build vectorized design matrix mapping coefficients to ``[Br, Btheta, Bphi]``.

    Parameters
    ----------
    gc_lat_rad : ndarray
        Geocentric latitude in radians.
    lon_rad : ndarray
        Geocentric longitude in radians.
    r_km : ndarray
        Geocentric radius in km.
    n_max : int
        Model maximum degree.
    n_truncate : int, optional
        Active truncation degree, default ``n_max``.
    a_ref : float, default=6371.2
        Reference radius in km.

    Returns
    -------
    ndarray
        Matrix with shape ``(3*N, n_truncate*(n_truncate+2))``.

    Notes
    -----
    Implements partial derivatives of spherical harmonic field components from
    V(r, theta, phi) = a * sum_n (a/r)^(n+1) * sum_m (g cos(m phi) + h sin(m phi)) P_n^m.
    """
    lat = np.asarray(gc_lat_rad, dtype=float).ravel()
    lon = np.asarray(lon_rad, dtype=float).ravel()
    r = np.asarray(r_km, dtype=float).ravel()

    if lat.size != lon.size or lat.size != r.size:
        raise ValueError("gc_lat_rad, lon_rad, r_km must have same length")

    n_use = n_max if n_truncate is None else min(n_truncate, n_max)
    n_pts = lat.size
    n_coeff = n_use * (n_use + 2)
    g = np.zeros((3 * n_pts, n_coeff), dtype=float)

    theta, phi, sin_theta = _prepare_angles(lat, lon)
    cos_mphi, sin_mphi = _compute_trig(phi, n_use)
    p, dp = _compute_schmidt_p_and_dp(theta, n_use)

    ar = a_ref / r
    pow_n = np.empty((n_use + 1, n_pts), dtype=float)
    pow_n[0] = 1.0
    for n in range(1, n_use + 1):
        pow_n[n] = pow_n[n - 1] * ar

    col = 0
    for n in range(1, n_use + 1):
        common = pow_n[n + 1] if n + 1 <= n_use else pow_n[n] * ar
        common_br = (n + 1.0) * common
        for m in range(0, n + 1):
            pm = p[n, m]
            dpm = dp[n, m]
            c = cos_mphi[m]
            s = sin_mphi[m]

            br_g = common_br * pm * c
            bt_g = common * dpm * c
            bp_g = np.where(np.abs(sin_theta) < 1e-10, 0.0, common * m * pm * s / sin_theta)

            g[0::3, col] = br_g
            g[1::3, col] = bt_g
            g[2::3, col] = bp_g
            col += 1

            if m > 0:
                br_h = common_br * pm * s
                bt_h = common * dpm * s
                bp_h = np.where(np.abs(sin_theta) < 1e-10, 0.0, -common * m * pm * c / sin_theta)
                g[0::3, col] = br_h
                g[1::3, col] = bt_h
                g[2::3, col] = bp_h
                col += 1

    return g


__all__ = ["build_design_matrix", "_compute_schmidt_p_and_dp", "_compute_trig"]
