"""Grid generation utilities for geomagnetic field maps."""

from __future__ import annotations

from typing import Literal

import numpy as np

from shgeomag.core.field import compute_fdi, compute_geo
from shgeomag.utils.coord_utils import geodetic_to_geocentric


def global_grid(
    model,
    year: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    dx_deg: float,
    dy_deg: float,
    h_gps_km: float,
    output: Literal["fdi", "xyz"] = "fdi",
) -> dict[str, np.ndarray]:
    """Compute a regular latitude/longitude grid of geomagnetic outputs.

    Parameters
    ----------
    model : GeomagModel
        Loaded geomagnetic model.
    year : float
        Decimal year.
    lon_min, lon_max, lat_min, lat_max : float
        Grid bounds in degrees.
    dx_deg, dy_deg : float
        Grid spacing in degrees.
    h_gps_km : float
        Height above ellipsoid in km.
    output : {"fdi", "xyz"}, default="fdi"
        Output component set.

    Returns
    -------
    dict of ndarray
        Grid arrays and component arrays.
    """
    lons = np.arange(lon_min, lon_max + 1e-12, dx_deg, dtype=float)
    lats = np.arange(lat_min, lat_max + 1e-12, dy_deg, dtype=float)
    lon2d, lat2d = np.meshgrid(lons, lats)

    gc_lat, gc_lon, r_km = geodetic_to_geocentric(lat2d.ravel(), lon2d.ravel(), h_gps_km)

    if output == "fdi":
        f, d, i = compute_fdi(model, gc_lat, gc_lon, r_km, year)
        return {
            "lon": lon2d,
            "lat": lat2d,
            "F": f.reshape(lat2d.shape),
            "D": d.reshape(lat2d.shape),
            "I": i.reshape(lat2d.shape),
        }

    x, y, z = compute_geo(model, gc_lat, gc_lon, r_km, year)
    return {
        "lon": lon2d,
        "lat": lat2d,
        "X": x.reshape(lat2d.shape),
        "Y": y.reshape(lat2d.shape),
        "Z": z.reshape(lat2d.shape),
    }


__all__ = ["global_grid"]
