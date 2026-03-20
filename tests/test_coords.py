import numpy as np

from shgeomag.utils.coord_utils import geocentric_to_geodetic, geodetic_to_geocentric, gps_height_to_radius


def test_geodetic_to_geocentric_vectorized():
    lat = np.array([0.0, 45.0, -30.0])
    lon = np.array([0.0, 120.0, 240.0])
    h = np.array([0.0, 0.1, 0.2])
    gc_lat, gc_lon, r = geodetic_to_geocentric(lat, lon, h)
    assert gc_lat.shape == gc_lon.shape == r.shape == (3,)


def test_roundtrip_approx():
    lat = np.array([0.0, 45.0])
    lon = np.array([10.0, 200.0])
    h = np.array([0.0, 1.0])
    gc_lat, gc_lon, r = geodetic_to_geocentric(lat, lon, h)
    gd_lat, gd_lon, _ = geocentric_to_geodetic(gc_lat, gc_lon, r)
    assert np.allclose(gd_lat, lat, atol=1e-5)
    assert np.allclose(gd_lon, np.mod(lon, 360.0), atol=1e-8)


def test_gps_height_to_radius_equator():
    r = gps_height_to_radius(0.0, 0.0)
    assert np.isclose(r, 6378.137, atol=1e-6)
