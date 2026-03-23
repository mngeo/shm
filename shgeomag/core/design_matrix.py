"""Design-matrix and spherical harmonic basis routines for geomagnetism."""

from __future__ import annotations

import numpy as np

from shgeomag._constants import A_REF_KM
from shgeomag.utils.coord_utils import geocentric_to_geodetic


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

    p = np.zeros((n_max + 1, n_max + 1, n_pts), dtype=float)
    dp = np.zeros((n_max + 1, n_max + 1, n_pts), dtype=float)
    p[0, 0] = 1.0

    s = np.zeros((n_max + 1, n_max + 1), dtype=float)
    s[0, 0] = 1.0

    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            if n == m:
                p[n, n] = sin_t * p[n - 1, n - 1]
                dp[n, n] = sin_t * dp[n - 1, n - 1] + cos_t * p[n - 1, n - 1]
            else:
                if n == 1:
                    p[n, m] = cos_t * p[n - 1, m]
                    dp[n, m] = cos_t * dp[n - 1, m] - sin_t * p[n - 1, m]
                elif n > 1:
                    knm = ((n - 1.0)**2 - m**2) / ((2.0 * n - 1.0) * (2.0 * n - 3.0))
                    p[n, m] = cos_t * p[n - 1, m] - knm * p[n - 2, m]
                    dp[n, m] = cos_t * dp[n - 1, m] - sin_t * p[n - 1, m] - knm * dp[n - 2, m]

            if m == 0:
                s[n, 0] = s[n - 1, 0] * (2.0 * n - 1.0) / n
            else:
                s[n, m] = s[n, m - 1] * np.sqrt((n - m + 1.0) * (2.0 if m == 1 else 1.0) / (n + m))

    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            p[n, m] *= s[n, m]
            dp[n, m] *= s[n, m]

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
    # Align radial power with `compute_sph` implementation.
    pow_n = np.empty((n_use + 3, n_pts), dtype=float)
    pow_n[0] = 1.0
    for n in range(1, n_use + 3):
        pow_n[n] = pow_n[n - 1] * ar

    col = 0
    for n in range(1, n_use + 1):
        common = pow_n[n + 2]
        common_br = (n + 1.0) * common
        for m in range(0, n + 1):
            pm = p[n, m]
            dpm = dp[n, m]
            c = cos_mphi[m]
            s = sin_mphi[m]

            br_g = common_br * pm * c
            bt_g = -common * dpm * c
            bp_g = np.where(np.abs(sin_theta) < 1e-10, 0.0, common * m * pm * s / sin_theta)

            g[0::3, col] = br_g
            g[1::3, col] = bt_g
            g[2::3, col] = bp_g
            col += 1

            if m > 0:
                br_h = common_br * pm * s
                bt_h = -common * dpm * s
                bp_h = np.where(np.abs(sin_theta) < 1e-10, 0.0, -common * m * pm * c / sin_theta)
                g[0::3, col] = br_h
                g[1::3, col] = bt_h
                g[2::3, col] = bp_h
                col += 1

    return g


def build_ned_jacobian(
    gc_lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    r_km: np.ndarray,
    n_max: int,
    n_truncate: int | None = None,
    a_ref: float = A_REF_KM,
    use_cache: bool = True,
) -> np.ndarray:
    """Build Jacobian ``∂[X, Y, Z]/∂c`` with respect to Gauss ``g/h`` coefficients.

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
    use_cache : bool, default=True
        If ``True``, reuse Jacobian rows for repeated geometry tuples
        ``(gc_lat, lon, r)``.

    Returns
    -------
    ndarray
        Jacobian with shape ``(3*N, n_truncate*(n_truncate+2))`` where each
        column is the partial derivative of geodetic NED components with
        respect to one coefficient in
        ``[g10, g11, h11, g20, g21, h21, g22, h22, ...]`` ordering.
        Rows are in point-major order
        ``[X_0, Y_0, Z_0, X_1, Y_1, Z_1, ...]``.

    Notes
    -----
    This function returns coefficient Jacobians, not a 3x3 rotation matrix.
    For each coefficient ``c_j``:
    ``dX/dc_j = -cos(psi) * dBtheta/dc_j - sin(psi) * dBr/dc_j``,
    ``dY/dc_j = dBphi/dc_j``,
    ``dZ/dc_j = sin(psi) * dBtheta/dc_j - cos(psi) * dBr/dc_j``,
    where ``psi = geodetic_lat - geocentric_lat``.
    """
    lat = np.asarray(gc_lat_rad, dtype=float).ravel()
    lon = np.asarray(lon_rad, dtype=float).ravel()
    r = np.asarray(r_km, dtype=float).ravel()
    if lat.size != lon.size or lat.size != r.size:
        raise ValueError("gc_lat_rad, lon_rad, r_km must have same length")

    if use_cache:
        geom = np.column_stack([lat, lon, r])
        unique_geom, inverse = np.unique(geom, axis=0, return_inverse=True)
        g_sph_unique = build_design_matrix(
            unique_geom[:, 0],
            unique_geom[:, 1],
            unique_geom[:, 2],
            n_max=n_max,
            n_truncate=n_truncate,
            a_ref=a_ref,
        )
        gc_lat_deg = np.rad2deg(unique_geom[:, 0])
        lon_deg = np.rad2deg(unique_geom[:, 1])
        gd_lat_deg, _, _ = geocentric_to_geodetic(gc_lat_deg, lon_deg, unique_geom[:, 2])
        psi = np.deg2rad(gd_lat_deg - gc_lat_deg)

        br = g_sph_unique[0::3]
        bt = g_sph_unique[1::3]
        bp = g_sph_unique[2::3]

        g_ned_unique = np.empty_like(g_sph_unique)
        g_ned_unique[0::3] = -bt * np.cos(psi)[:, None] - br * np.sin(psi)[:, None]
        g_ned_unique[1::3] = bp
        g_ned_unique[2::3] = bt * np.sin(psi)[:, None] - br * np.cos(psi)[:, None]

        n_coeff = g_ned_unique.shape[1]
        n_pts = lat.size
        g_ned = np.empty((3 * n_pts, n_coeff), dtype=float)
        g_ned[0::3] = g_ned_unique[0::3][inverse]
        g_ned[1::3] = g_ned_unique[1::3][inverse]
        g_ned[2::3] = g_ned_unique[2::3][inverse]
        return g_ned

    g_sph = build_design_matrix(lat, lon, r, n_max=n_max, n_truncate=n_truncate, a_ref=a_ref)
    gc_lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    gd_lat_deg, _, _ = geocentric_to_geodetic(gc_lat_deg, lon_deg, r)
    psi = np.deg2rad(gd_lat_deg - gc_lat_deg)

    br = g_sph[0::3]
    bt = g_sph[1::3]
    bp = g_sph[2::3]
    g_ned = np.empty_like(g_sph)
    g_ned[0::3] = -bt * np.cos(psi)[:, None] - br * np.sin(psi)[:, None]
    g_ned[1::3] = bp
    g_ned[2::3] = bt * np.sin(psi)[:, None] - br * np.cos(psi)[:, None]
    return g_ned


__all__ = ["build_design_matrix", "build_ned_jacobian", "_compute_schmidt_p_and_dp", "_compute_trig"]
