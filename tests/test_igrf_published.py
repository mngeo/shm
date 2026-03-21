import numpy as np

from shgeomag.core.field import compute_fdi


def test_igrf14_regression_single_point(igrf_model):
    gc_lat = np.array([10.0])
    lon = np.array([20.0])
    r = np.array([6371.2])

    f, d, i = compute_fdi(igrf_model, gc_lat, lon, r, 2020.0)

    assert np.isclose(f[0], 34825.50547057508, atol=1.0)
    assert np.isclose(d[0], 2.1283244721721255, atol=0.01)
    assert np.isclose(i[0], -1.3904855563048502, atol=0.01)
