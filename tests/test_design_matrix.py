import numpy as np

from shgeomag.core.design_matrix import build_design_matrix
from shgeomag.core.field import compute_sph
from shgeomag.io.reader import load_model


def test_design_matrix_shape():
    lat = np.deg2rad(np.array([10.0, 20.0]))
    lon = np.deg2rad(np.array([30.0, 60.0]))
    r = np.array([6371.2, 6371.2])
    g = build_design_matrix(lat, lon, r, n_max=3)
    assert g.shape == (6, 15)


def test_design_matrix_matches_compute_sph_regression():
    rng = np.random.default_rng(42)

    cases = [
        ("models/WMM2025.COF", 2026.0),
        ("models/igrf14coeffs.txt", 2020.0),
    ]
    for model_path, year in cases:
        model = load_model(model_path)
        gc_lat_deg = rng.uniform(-89.0, 89.0, 128)
        lon_deg = rng.uniform(0.0, 360.0, 128)
        r_km = rng.uniform(6200.0, 7000.0, 128)

        br_ref, bt_ref, bp_ref = compute_sph(model, gc_lat_deg, lon_deg, r_km, year)

        coeffs = model.get_coefficients(year)
        g_mat = build_design_matrix(
            np.deg2rad(gc_lat_deg),
            np.deg2rad(lon_deg),
            r_km,
            n_max=coeffs.n_max,
            n_truncate=coeffs.n_max,
        )
        b_vec = g_mat @ coeffs.coefficient_vector()
        br_dm = b_vec[0::3]
        bt_dm = b_vec[1::3]
        bp_dm = b_vec[2::3]

        assert np.allclose(br_ref, br_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(bt_ref, bt_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(bp_ref, bp_dm, rtol=1e-11, atol=1e-8)
