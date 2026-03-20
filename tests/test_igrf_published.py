import numpy as np

from shgeomag.core.field import compute_fdi


def test_igrf14_regression_single_point(igrf_model):
    gc_lat = np.array([10.0])
    lon = np.array([20.0])
    r = np.array([6371.2])

    f, d, i = compute_fdi(igrf_model, gc_lat, lon, r, 2020.0)

    assert np.isclose(f[0], 29458.930739377098, atol=1.0)
    assert np.isclose(d[0], 3.1833029549132603, atol=0.01)
    assert np.isclose(i[0], -15.50560930636658, atol=0.01)
