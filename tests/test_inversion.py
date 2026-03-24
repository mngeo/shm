import numpy as np
import pytest

from shgeomag.core.field import compute_geo
from shgeomag.inversion.cg import (
    forward_ned_from_coefficients,
    invert_gauss_coefficients_cg,
    invert_gauss_coefficients_cg_tikhonov,
)
from shgeomag.io.reader import load_model


def test_forward_ned_from_coefficients_matches_compute_geo():
    model = load_model("models/WMM2025.COF")
    model.set_truncation(3)
    year = 2026.0
    coeff = model.get_coefficients(year)

    gc_lat_deg = np.array([10.0, -20.0, 35.0, 10.0])
    lon_deg = np.array([30.0, 120.0, 250.0, 30.0])
    r_km = np.array([6371.2, 6400.0, 6700.0, 6371.2])

    pred = forward_ned_from_coefficients(
        coeff.coefficient_vector(),
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=coeff.n_max,
        use_cache=True,
    )
    x, y, z = compute_geo(model, gc_lat_deg, lon_deg, r_km, year)
    ref = np.column_stack([x, y, z])
    assert np.allclose(pred, ref, rtol=1e-11, atol=1e-8)


def test_invert_gauss_coefficients_cg_recovers_synthetic_coefficients():
    rng = np.random.default_rng(1234)
    model = load_model("models/WMM2025.COF")
    model.set_truncation(3)
    year = 2026.0
    coeff_true = model.get_coefficients(year).coefficient_vector()

    n = 160
    gc_lat_deg = rng.uniform(-80.0, 80.0, n)
    lon_deg = rng.uniform(0.0, 360.0, n)
    r_km = rng.uniform(6300.0, 6800.0, n)
    # Duplicate subsets to exercise cache path.
    gc_lat_deg[40:80] = gc_lat_deg[:40]
    lon_deg[40:80] = lon_deg[:40]
    r_km[40:80] = r_km[:40]

    pred = forward_ned_from_coefficients(
        coeff_true,
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=3,
        use_cache=True,
    )
    mjd2000 = np.full(n, -1000.0)
    data = np.column_stack([mjd2000, gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])

    result = invert_gauss_coefficients_cg(data, n_max=3, max_iter=200, tol=1e-12, damping=0.0, use_cache=True)
    assert result.converged
    assert np.allclose(result.coefficient_vector, coeff_true, rtol=1e-7, atol=1e-4)
    assert np.allclose(result.predicted_xyz, pred, rtol=1e-11, atol=1e-8)


def test_invert_gauss_coefficients_cache_matches_no_cache():
    rng = np.random.default_rng(77)
    c_true = rng.normal(0.0, 100.0, size=15)  # n_max=3 -> 15 coefficients

    n = 120
    gc_lat_deg = rng.uniform(-70.0, 70.0, n)
    lon_deg = rng.uniform(0.0, 360.0, n)
    r_km = np.full(n, 6371.2)
    gc_lat_deg[60:] = gc_lat_deg[:60]
    lon_deg[60:] = lon_deg[:60]

    pred = forward_ned_from_coefficients(
        c_true,
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=3,
        use_cache=True,
    )
    data = np.column_stack([np.full(n, 50.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])

    out_cache = invert_gauss_coefficients_cg(data, n_max=3, max_iter=300, tol=1e-12, use_cache=True)
    out_nocache = invert_gauss_coefficients_cg(data, n_max=3, max_iter=300, tol=1e-12, use_cache=False)
    assert np.allclose(out_cache.coefficient_vector, out_nocache.coefficient_vector, rtol=1e-10, atol=1e-7)


def test_invert_gauss_coefficients_tikhonov_matches_damping_path():
    rng = np.random.default_rng(2024)
    c_true = rng.normal(0.0, 100.0, size=15)  # n_max=3 -> 15 coefficients

    n = 90
    gc_lat_deg = rng.uniform(-70.0, 70.0, n)
    lon_deg = rng.uniform(0.0, 360.0, n)
    r_km = rng.uniform(6350.0, 6800.0, n)
    pred = forward_ned_from_coefficients(
        c_true,
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=3,
        use_cache=True,
    )
    data = np.column_stack([np.full(n, 100.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])

    lambda_reg = 2.5
    out_damping = invert_gauss_coefficients_cg(
        data,
        n_max=3,
        max_iter=300,
        tol=1e-12,
        damping=lambda_reg,
        use_cache=True,
    )
    out_reg = invert_gauss_coefficients_cg_tikhonov(
        data,
        n_max=3,
        lambda_reg=lambda_reg,
        max_iter=300,
        tol=1e-12,
        use_cache=True,
    )
    assert np.allclose(out_reg.coefficient_vector, out_damping.coefficient_vector, rtol=1e-11, atol=1e-9)
    assert np.allclose(out_reg.predicted_xyz, out_damping.predicted_xyz, rtol=1e-12, atol=1e-10)


def test_invert_gauss_coefficients_tikhonov_rejects_negative_lambda():
    data = np.array([[0.0, 0.0, 0.0, 6371.2, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="lambda_reg must be non-negative"):
        invert_gauss_coefficients_cg_tikhonov(data, n_max=1, lambda_reg=-1.0)
