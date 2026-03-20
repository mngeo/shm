import numpy as np

from shgeomag.core.field import compute_fdi, compute_geo, compute_sph


def test_compute_shapes(wmm_model):
    lat = np.array([0.0, 45.0, -60.0])
    lon = np.array([0.0, 120.0, 250.0])
    r = np.array([6371.2, 6400.0, 6500.0])
    br, bt, bp = compute_sph(wmm_model, lat, lon, r, 2026.0)
    assert br.shape == bt.shape == bp.shape == (3,)


def test_compute_fdi_consistency(wmm_model):
    lat = np.array([10.0, -10.0])
    lon = np.array([30.0, 210.0])
    r = np.array([6371.2, 6371.2])
    x, y, z = compute_geo(wmm_model, lat, lon, r, 2025.0)
    f, d, i = compute_fdi(wmm_model, lat, lon, r, 2025.0)

    h = np.sqrt(x * x + y * y)
    assert np.allclose(f, np.sqrt(h * h + z * z))
    assert np.allclose(d, np.rad2deg(np.arctan2(y, x)))
    assert np.allclose(i, np.rad2deg(np.arctan2(z, h)))
