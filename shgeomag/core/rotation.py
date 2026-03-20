"""Rotation utilities between geocentric spherical and geodetic NED frames."""

from __future__ import annotations

import numpy as np


def rotate_sph_to_geodetic_ned(
    br: np.ndarray,
    btheta: np.ndarray,
    bphi: np.ndarray,
    gc_lat_deg: np.ndarray,
    gd_lat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate spherical field components to geodetic NED components.

    Parameters
    ----------
    br, btheta, bphi : ndarray
        Spherical components in nT, with ``br`` outward, ``btheta`` southward,
        and ``bphi`` eastward.
    gc_lat_deg : ndarray
        Geocentric latitude in degrees.
    gd_lat_deg : ndarray
        Geodetic latitude in degrees.

    Returns
    -------
    tuple of ndarray
        ``(X, Y, Z)`` where X is North, Y is East, Z is Down in nT.

    Notes
    -----
    Uses psi = phi_d - phi_c and
    X = -Btheta * cos(psi) - Br * sin(psi), Y = Bphi,
    Z = Btheta * sin(psi) - Br * cos(psi).
    """
    psi = np.deg2rad(np.asarray(gd_lat_deg, dtype=float) - np.asarray(gc_lat_deg, dtype=float))
    br_arr = np.asarray(br, dtype=float)
    bt_arr = np.asarray(btheta, dtype=float)
    bp_arr = np.asarray(bphi, dtype=float)

    x = -bt_arr * np.cos(psi) - br_arr * np.sin(psi)
    y = bp_arr
    z = bt_arr * np.sin(psi) - br_arr * np.cos(psi)
    return x, y, z


__all__ = ["rotate_sph_to_geodetic_ned"]
