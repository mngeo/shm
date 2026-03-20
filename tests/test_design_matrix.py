import numpy as np

from shgeomag.core.design_matrix import build_design_matrix


def test_design_matrix_shape():
    lat = np.deg2rad(np.array([10.0, 20.0]))
    lon = np.deg2rad(np.array([30.0, 60.0]))
    r = np.array([6371.2, 6371.2])
    g = build_design_matrix(lat, lon, r, n_max=3)
    assert g.shape == (6, 15)
