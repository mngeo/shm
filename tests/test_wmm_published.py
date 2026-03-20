import numpy as np

from shgeomag.core.field import compute_fdi


def test_wmm2025_regression_single_point(wmm_model):
    gc_lat = np.array([10.0])
    lon = np.array([20.0])
    r = np.array([6371.2])

    f, d, i = compute_fdi(wmm_model, gc_lat, lon, r, 2025.0)

    assert np.isclose(f[0], 29488.787821804697, atol=0.01)
    assert np.isclose(d[0], 3.6077635177045235, atol=0.001)
    assert np.isclose(i[0], -15.278314185152379, atol=0.001)
