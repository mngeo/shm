"""Geomagnetic field evaluation routines for spherical and geodetic outputs."""

from __future__ import annotations

import numpy as np

from shgeomag._constants import A_REF_KM
from shgeomag.core.design_matrix import _compute_schmidt_p_and_dp, _compute_trig
from shgeomag.core.rotation import rotate_sph_to_geodetic_ned
from shgeomag.data.container import InputData
from shgeomag.utils.coord_utils import geocentric_to_geodetic
from shgeomag.utils.time_utils import mjd2000_to_decimal_year


def _accumulate_field(gc_lat_deg: np.ndarray, lon_deg: np.ndarray, r_km: np.ndarray, coeffs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gc_lat_rad = np.deg2rad(np.asarray(gc_lat_deg, dtype=float).ravel())
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=float).ravel())
    r = np.asarray(r_km, dtype=float).ravel()
    n_pts = r.size

    theta = 0.5 * np.pi - gc_lat_rad
    sin_theta = np.sin(theta)
    phi = np.mod(lon_rad, 2.0 * np.pi)

    p, dp = _compute_schmidt_p_and_dp(theta, coeffs.n_max)
    cos_mphi, sin_mphi = _compute_trig(phi, coeffs.n_max)

    ar = A_REF_KM / r
    ar_pow = np.ones((coeffs.n_max + 2, n_pts), dtype=float)
    for n in range(1, coeffs.n_max + 2):
        ar_pow[n] = ar_pow[n - 1] * ar

    gh = coeffs.to_gh_dict()
    br = np.zeros(n_pts, dtype=float)
    bt = np.zeros(n_pts, dtype=float)
    bp = np.zeros(n_pts, dtype=float)

    for n in range(1, coeffs.n_max + 1):
        common = ar_pow[n + 1]
        for m in range(0, n + 1):
            g, h = gh[(n, m)]
            c = cos_mphi[m]
            s = sin_mphi[m]
            pm = p[n, m]
            dpm = dp[n, m]

            gh_term = g * c + h * s
            br += (n + 1.0) * common * pm * gh_term
            bt += common * dpm * gh_term

            if m > 0:
                bp += np.where(
                    np.abs(sin_theta) < 1e-10,
                    0.0,
                    common * m * pm * (g * s - h * c) / sin_theta,
                )

    return br, bt, bp


def compute_sph(model, gc_lat_deg, lon_deg, r_km, year: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spherical field components ``(Br, Btheta, Bphi)`` in nT.

    Parameters
    ----------
    model : GeomagModel
        Geomagnetic model object.
    gc_lat_deg : float or ndarray
        Geocentric latitude in degrees.
    lon_deg : float or ndarray
        Geocentric longitude in degrees.
    r_km : float or ndarray
        Geocentric radius in km.
    year : float
        Decimal year.

    Returns
    -------
    tuple of ndarray
        ``(Br, Btheta, Bphi)`` where ``Br`` is outward, ``Btheta`` southward,
        ``Bphi`` eastward.
    """
    coeffs = model.get_coefficients(float(year))
    return _accumulate_field(np.asarray(gc_lat_deg), np.asarray(lon_deg), np.asarray(r_km), coeffs)


def compute_geo(model, gc_lat_deg, lon_deg, r_km, year: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute geodetic NED components ``(X, Y, Z)`` in nT."""
    br, bt, bp = compute_sph(model, gc_lat_deg, lon_deg, r_km, year)
    gd_lat, _, _ = geocentric_to_geodetic(gc_lat_deg, lon_deg, r_km)
    return rotate_sph_to_geodetic_ned(br, bt, bp, np.asarray(gc_lat_deg, dtype=float), gd_lat)


def compute_fdi(model, gc_lat_deg, lon_deg, r_km, year: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total intensity, declination, and inclination.

    Parameters
    ----------
    model : GeomagModel
        Geomagnetic model object.
    gc_lat_deg, lon_deg, r_km : float or ndarray
        Geocentric coordinates in degrees/degrees/km.
    year : float
        Decimal year.

    Returns
    -------
    tuple of ndarray
        ``(F, D, I)`` where F is nT and D/I are degrees.
    """
    x, y, z = compute_geo(model, gc_lat_deg, lon_deg, r_km, year)
    h = np.sqrt(x * x + y * y)
    f = np.sqrt(h * h + z * z)
    d = np.rad2deg(np.arctan2(y, x))
    i = np.rad2deg(np.arctan2(z, h))
    return f, d, i


def compute_multi_epoch(model, data: InputData) -> dict[str, np.ndarray]:
    """Compute ``X/Y/Z/F/D/I`` for vectorized ``InputData`` rows.

    Parameters
    ----------
    model : GeomagModel
        Geomagnetic model object.
    data : InputData
        Input rows in ``[MJD2000, gc_lat, lon, r]``.

    Returns
    -------
    dict of ndarray
        Component arrays in original row order.
    """
    arr = data.data
    years = mjd2000_to_decimal_year(arr[:, 0])
    order = np.argsort(years)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    years_sorted = years[order]
    xyz_out = np.zeros((arr.shape[0], 3), dtype=float)

    start = 0
    while start < years_sorted.size:
        year = float(years_sorted[start])
        stop = start + 1
        while stop < years_sorted.size and np.isclose(years_sorted[stop], year):
            stop += 1

        idx = order[start:stop]
        x, y, z = compute_geo(model, arr[idx, 1], arr[idx, 2], arr[idx, 3], year)
        xyz_out[idx, 0] = x
        xyz_out[idx, 1] = y
        xyz_out[idx, 2] = z
        start = stop

    h = np.sqrt(xyz_out[:, 0] ** 2 + xyz_out[:, 1] ** 2)
    f = np.sqrt(h**2 + xyz_out[:, 2] ** 2)
    d = np.rad2deg(np.arctan2(xyz_out[:, 1], xyz_out[:, 0]))
    i = np.rad2deg(np.arctan2(xyz_out[:, 2], h))

    _ = inv
    return {
        "X": xyz_out[:, 0],
        "Y": xyz_out[:, 1],
        "Z": xyz_out[:, 2],
        "F": f,
        "D": d,
        "I": i,
    }


__all__ = ["compute_sph", "compute_geo", "compute_fdi", "compute_multi_epoch"]
