"""Coordinate conversion utilities between geodetic and geocentric systems."""

from __future__ import annotations

import warnings
from typing import Any, Tuple

import numpy as np

from shgeomag._constants import WGS84_A_KM, WGS84_B_KM, WGS84_E2

try:
    from pyproj import Transformer as _Transformer
    Transformer: Any = _Transformer
except Exception:  # pragma: no cover
    Transformer = None


def geodetic_to_geocentric(
    lat_gd_deg: float | np.ndarray,
    lon_deg: float | np.ndarray,
    h_km: float | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates to geocentric coordinates.

    Parameters
    ----------
    lat_gd_deg : float or ndarray
        Geodetic latitude in degrees.
    lon_deg : float or ndarray
        Longitude in degrees.
    h_km : float or ndarray
        Height above WGS-84 ellipsoid in km.

    Returns
    -------
    tuple of ndarray
        Geocentric latitude in degrees, longitude in degrees, and radius in km.
    """
    lat = np.asarray(lat_gd_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    h = np.asarray(h_km, dtype=float)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    n_phi = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n_phi + h) * cos_lat * np.cos(lon_rad)
    y = (n_phi + h) * cos_lat * np.sin(lon_rad)
    z = (n_phi * (1.0 - WGS84_E2) + h) * sin_lat

    r = np.sqrt(x * x + y * y + z * z)
    lat_gc_rad = np.arcsin(np.clip(z / r, -1.0, 1.0))
    lon_wrap = np.mod(lon, 360.0)
    return np.rad2deg(lat_gc_rad), lon_wrap, r


def geocentric_to_geodetic(
    lat_gc_deg: float | np.ndarray,
    lon_deg: float | np.ndarray,
    r_km: float | np.ndarray,
    max_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geocentric coordinates to geodetic coordinates.

    Parameters
    ----------
    lat_gc_deg : float or ndarray
        Geocentric latitude in degrees.
    lon_deg : float or ndarray
        Longitude in degrees.
    r_km : float or ndarray
        Geocentric radius in km.
    max_iter : int, default=5
        Iterations for Bowring-style geodetic latitude refinement.

    Returns
    -------
    tuple of ndarray
        Geodetic latitude in degrees, longitude in degrees, and ellipsoidal height in km.
    """
    lat_gc = np.deg2rad(np.asarray(lat_gc_deg, dtype=float))
    lon = np.asarray(lon_deg, dtype=float)
    r = np.asarray(r_km, dtype=float)

    x = r * np.cos(lat_gc)
    z = r * np.sin(lat_gc)
    ep2 = (WGS84_A_KM**2 - WGS84_B_KM**2) / (WGS84_B_KM**2)

    beta = np.arctan2(WGS84_B_KM * z, WGS84_A_KM * x)
    lat_gd = np.arctan2(z + ep2 * WGS84_B_KM * np.sin(beta) ** 3, x - WGS84_E2 * WGS84_A_KM * np.cos(beta) ** 3)
    for _ in range(max_iter - 1):
        beta = np.arctan2((1.0 - (1.0 - WGS84_B_KM / WGS84_A_KM)) * np.sin(lat_gd), np.cos(lat_gd))
        lat_gd = np.arctan2(z + ep2 * WGS84_B_KM * np.sin(beta) ** 3, x - WGS84_E2 * WGS84_A_KM * np.cos(beta) ** 3)

    sin_lat = np.sin(lat_gd)
    n_phi = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = x / np.cos(lat_gd) - n_phi
    lon_wrap = np.mod(lon, 360.0)
    return np.rad2deg(lat_gd), lon_wrap, h


def gps_height_to_radius(lat_gd_deg: float | np.ndarray, h_gps_km: float | np.ndarray) -> np.ndarray:
    """Convert geodetic (GPS) height to geocentric radius.

    Parameters
    ----------
    lat_gd_deg : float or ndarray
        Geodetic latitude in degrees.
    h_gps_km : float or ndarray
        Height above WGS-84 ellipsoid in km.

    Returns
    -------
    ndarray
        Geocentric radius in km.
    """
    _, _, r = geodetic_to_geocentric(lat_gd_deg, 0.0, h_gps_km)
    return r


def _geoid_undulation(lat_gd_deg: np.ndarray, lon_deg: np.ndarray, geoid_model: str) -> np.ndarray:
    if Transformer is None:
        raise ImportError("pyproj is required for sea-level conversions with a geoid model")

    if geoid_model.lower() != "egm96":
        warnings.warn("Only EGM96 lookup is currently attempted; falling back to EGM96 pipeline.")

    try:
        transformer = Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)
        lon, lat, h_orth = transformer.transform(lon_deg, lat_gd_deg, np.zeros_like(lat_gd_deg))
        return -np.asarray(h_orth, dtype=float)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Unable to resolve EGM96 geoid model via pyproj. Install PROJ datum grids.") from exc


def sea_level_height_to_radius(
    lat_gd_deg: float | np.ndarray,
    lon_deg: float | np.ndarray,
    h_msl_km: float | np.ndarray,
    geoid_model: str = "EGM96",
) -> np.ndarray:
    """Convert mean-sea-level height to geocentric radius.

    Parameters
    ----------
    lat_gd_deg : float or ndarray
        Geodetic latitude in degrees.
    lon_deg : float or ndarray
        Longitude in degrees.
    h_msl_km : float or ndarray
        Height above mean sea level in km.
    geoid_model : str, default="EGM96"
        Geoid model name passed to pyproj-based undulation resolution.

    Returns
    -------
    ndarray
        Geocentric radius in km.
    """
    lat = np.asarray(lat_gd_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    h_msl = np.asarray(h_msl_km, dtype=float)
    und_m = _geoid_undulation(lat, lon, geoid_model) / 1000.0
    return gps_height_to_radius(lat, h_msl + und_m)


def radius_to_gps_height(lat_gd_deg: float | np.ndarray, r_km: float | np.ndarray) -> np.ndarray:
    """Convert geocentric radius to GPS height above ellipsoid.

    Parameters
    ----------
    lat_gd_deg : float or ndarray
        Geodetic latitude in degrees.
    r_km : float or ndarray
        Geocentric radius in km.

    Returns
    -------
    ndarray
        Height above WGS-84 ellipsoid in km.
    """
    lat = np.asarray(lat_gd_deg, dtype=float)
    r = np.asarray(r_km, dtype=float)
    lat_gc, _, h = geocentric_to_geodetic(lat, 0.0, r)
    _ = lat_gc
    return h


def radius_to_sea_level_height(
    lat_gd_deg: float | np.ndarray,
    lon_deg: float | np.ndarray,
    r_km: float | np.ndarray,
    geoid_model: str = "EGM96",
) -> np.ndarray:
    """Convert geocentric radius to mean-sea-level height.

    Parameters
    ----------
    lat_gd_deg : float or ndarray
        Geodetic latitude in degrees.
    lon_deg : float or ndarray
        Longitude in degrees.
    r_km : float or ndarray
        Geocentric radius in km.
    geoid_model : str, default="EGM96"
        Geoid model name passed to pyproj-based undulation resolution.

    Returns
    -------
    ndarray
        Height above mean sea level in km.
    """
    lat = np.asarray(lat_gd_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    h_gps = radius_to_gps_height(lat, r_km)
    und_m = _geoid_undulation(lat, lon, geoid_model) / 1000.0
    return h_gps - und_m


__all__ = [
    "geodetic_to_geocentric",
    "geocentric_to_geodetic",
    "gps_height_to_radius",
    "sea_level_height_to_radius",
    "radius_to_gps_height",
    "radius_to_sea_level_height",
]
