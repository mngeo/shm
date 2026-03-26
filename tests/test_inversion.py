import numpy as np
import pytest
import warnings

from shgeomag.core.design_matrix import build_ned_jacobian
from shgeomag.core.field import compute_geo
from shgeomag.inversion.cg import (
    compute_l_curve_tikhonov,
    forward_ned_from_coefficients,
    invert_gauss_coefficients_cg,
    invert_gauss_coefficients_cg_tikhonov,
)
from shgeomag.io.reader import load_model
from shgeomag.utils.coord_utils import geodetic_to_geocentric
from shgeomag.utils.grid import global_grid
from shgeomag.utils.time_utils import decimal_year_to_mjd2000


def _synthetic_coeff_rms(model_path: str, year: float, noise_fraction: float) -> float:
    """Return RMS coefficient error from synthetic inversion at one epoch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = load_model(model_path)
        n_max = int(model.n_max)
        c_true = model.get_coefficients(year).coefficient_vector()
        grid = global_grid(
            model=model,
            year=year,
            lon_min=0.0,
            lon_max=340.0,
            lat_min=-90.0,
            lat_max=90.0,
            dx_deg=20.0,
            dy_deg=20.0,
            h_gps_km=0.0,
            output="xyz",
        )

    lon = grid["lon"].ravel()
    lat_gd = grid["lat"].ravel()
    x = grid["X"].ravel()
    y = grid["Y"].ravel()
    z = grid["Z"].ravel()
    gc_lat, gc_lon, r_km = geodetic_to_geocentric(lat_gd, lon, 0.0)

    if noise_fraction > 0.0:
        rng = np.random.default_rng(20260326)
        x = x + rng.normal(0.0, noise_fraction * np.abs(x))
        y = y + rng.normal(0.0, noise_fraction * np.abs(y))
        z = z + rng.normal(0.0, noise_fraction * np.abs(z))

    mjd = np.full_like(gc_lat, float(decimal_year_to_mjd2000(year)))
    data = np.column_stack([mjd, gc_lat, gc_lon, r_km, x, y, z])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = invert_gauss_coefficients_cg(
            data=data,
            n_max=n_max,
            max_iter=3000,
            tol=1e-10,
            damping=0.0,
            x0=c_true,
            use_cache=True,
            epoch_year=year,
        )
    return float(np.sqrt(np.mean((out.coefficient_vector - c_true) ** 2)))


@pytest.mark.parametrize("model_path", ["models/igrf14coeffs.txt", "models/WMM2025.COF"])
def test_synthetic_epoch2020_inversion_recovers_coefficients_rms_metric(model_path: str):
    """Generate synthetic epoch-2020 data and recover original coefficients."""
    rms = _synthetic_coeff_rms(model_path=model_path, year=2020.0, noise_fraction=0.0)
    assert np.isfinite(rms)
    assert rms <= 1e-8


@pytest.mark.parametrize("model_path", ["models/igrf14coeffs.txt", "models/WMM2025.COF"])
def test_synthetic_epoch2020_noise_levels_5pct_10pct_rms_metric(model_path: str):
    """Compare coefficient RMS error for 5% and 10% noise synthetic inversions."""
    rms_clean = _synthetic_coeff_rms(model_path=model_path, year=2020.0, noise_fraction=0.0)
    rms_5 = _synthetic_coeff_rms(model_path=model_path, year=2020.0, noise_fraction=0.05)
    rms_10 = _synthetic_coeff_rms(model_path=model_path, year=2020.0, noise_fraction=0.10)

    assert np.isfinite(rms_clean)
    assert np.isfinite(rms_5)
    assert np.isfinite(rms_10)
    assert rms_clean <= 1e-8
    assert rms_5 > rms_clean
    assert rms_10 > rms_5


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


def test_invert_gauss_coefficients_tikhonov_rejects_invalid_regularization_name():
    data = np.array([[0.0, 0.0, 0.0, 6371.2, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="regularization must be one of"):
        invert_gauss_coefficients_cg_tikhonov(
            data,
            n_max=1,
            lambda_reg=1.0,
            regularization="bad_scheme",  # type: ignore[arg-type]
        )


def test_invert_gauss_coefficients_tikhonov_Manojs_scheme_matches_dense_solution():
    rng = np.random.default_rng(991)
    c_true = rng.normal(0.0, 100.0, size=15)  # n_max=3 -> 15 coefficients

    n = 120
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

    lam = 5.0
    out = invert_gauss_coefficients_cg_tikhonov(
        data,
        n_max=3,
        lambda_reg=lam,
        regularization="Manojs_scheme",
        max_iter=500,
        tol=1e-12,
        use_cache=True,
    )

    j = build_ned_jacobian(
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=3,
        use_cache=False,
    )
    d = pred.reshape(-1)
    idx = np.arange(1, 16, dtype=float)
    reg_diag = idx**4
    a = j.T @ j + lam * np.diag(reg_diag)
    b = j.T @ d
    ref = np.linalg.solve(a, b)

    assert np.allclose(out.coefficient_vector, ref, rtol=1e-9, atol=1e-7)


def test_compute_l_curve_tikhonov_returns_norm_arrays():
    rng = np.random.default_rng(45)
    c_true = rng.normal(0.0, 100.0, size=15)  # n_max=3 -> 15 coefficients

    n = 100
    gc_lat_deg = rng.uniform(-75.0, 75.0, n)
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
    data = np.column_stack([np.full(n, 200.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])
    x0 = rng.normal(0.0, 20.0, size=15)
    lambdas = np.array([0.0, 1.0, 100.0], dtype=float)

    solution_norm, residual_norm = compute_l_curve_tikhonov(
        data=data,
        n_max=3,
        lambda_values=lambdas,
        max_iter=300,
        tol=1e-12,
        x0=x0,
        use_cache=True,
        plot=False,
    )

    assert solution_norm.shape == lambdas.shape
    assert residual_norm.shape == lambdas.shape
    assert np.all(np.isfinite(solution_norm))
    assert np.all(np.isfinite(residual_norm))
    assert np.all(solution_norm >= 0.0)
    assert np.all(residual_norm >= 0.0)

    ref = invert_gauss_coefficients_cg_tikhonov(
        data=data,
        n_max=3,
        lambda_reg=0.0,
        max_iter=300,
        tol=1e-12,
        x0=x0,
        use_cache=True,
    )
    assert np.isclose(solution_norm[0], np.linalg.norm(ref.coefficient_vector), rtol=1e-10, atol=1e-10)
    assert np.isclose(residual_norm[0], np.linalg.norm(ref.residual_xyz.reshape(-1)), rtol=1e-10, atol=1e-10)


def test_compute_l_curve_tikhonov_accepts_Manojs_scheme():
    rng = np.random.default_rng(21)
    c_true = rng.normal(0.0, 100.0, size=15)

    n = 80
    gc_lat_deg = rng.uniform(-75.0, 75.0, n)
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
    data = np.column_stack([np.full(n, 200.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])
    lambdas = np.array([0.0, 2.0, 20.0], dtype=float)

    solution_norm, residual_norm = compute_l_curve_tikhonov(
        data=data,
        n_max=3,
        lambda_values=lambdas,
        regularization="Manojs_scheme",
        max_iter=300,
        tol=1e-12,
        use_cache=True,
        plot=False,
    )
    assert solution_norm.shape == lambdas.shape
    assert residual_norm.shape == lambdas.shape
    assert np.all(np.isfinite(solution_norm))
    assert np.all(np.isfinite(residual_norm))


def test_compute_l_curve_tikhonov_rejects_invalid_lambda_array():
    data = np.array([[0.0, 0.0, 0.0, 6371.2, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-empty"):
        compute_l_curve_tikhonov(data, n_max=1, lambda_values=np.array([]), plot=False)
    with pytest.raises(ValueError, match="finite"):
        compute_l_curve_tikhonov(data, n_max=1, lambda_values=np.array([0.0, np.nan]), plot=False)
    with pytest.raises(ValueError, match="non-negative"):
        compute_l_curve_tikhonov(data, n_max=1, lambda_values=np.array([-1.0, 0.0]), plot=False)
