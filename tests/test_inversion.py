import numpy as np
import pytest

from shgeomag.core.field import compute_geo
from shgeomag.inversion.cg import (
    compute_l_curve_tikhonov,
    forward_ned_from_coefficients,
    invert_gauss_coefficients_cg,
    invert_gauss_coefficients_cg_tikhonov,
)
from shgeomag.io.reader import load_model


def _linear_system_from_model(model_path: str, year: float, noise_fraction: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic A, B system using model coefficients at one epoch."""
    model = load_model(model_path)
    n_max = int(model.n_max)
    c_true = model.get_coefficients(year).coefficient_vector()
    n_coeff = c_true.size

    rng = np.random.default_rng(20260327 + n_coeff)
    n_rows = max(4 * n_coeff, 64)
    a = rng.normal(0.0, 1.0, size=(n_rows, n_coeff))
    a[:n_coeff, :] = np.eye(n_coeff)  # ensure full-rank and stable inversion

    q_true = np.column_stack([c_true, c_true, c_true])
    b = a @ q_true
    if noise_fraction > 0.0:
        rng_noise = np.random.default_rng(20260326)
        sigma = noise_fraction * np.maximum(np.abs(b), 1e-12)
        b = b + rng_noise.normal(0.0, sigma)
    return n_max, c_true, a, b


@pytest.mark.parametrize("model_path", ["models/igrf14coeffs.txt", "models/WMM2025.COF"])
def test_synthetic_epoch2020_inversion_recovers_coefficients_rms_metric(model_path: str):
    """Generate synthetic A,B from model coefficients and recover coefficients."""
    n_max, c_true, a, b = _linear_system_from_model(model_path, year=2020.0, noise_fraction=0.0)
    out = invert_gauss_coefficients_cg(
        data=(a, b),
        n_max=n_max,
        max_iter=3000,
        tol=1e-12,
        damping=0.0,
        x0=None,
        use_cache=True,
        epoch_year=2020.0,
    )
    rms = float(np.sqrt(np.mean((out.coefficient_vector - c_true) ** 2)))
    assert np.isfinite(rms)
    assert rms <= 1e-8


@pytest.mark.parametrize("model_path", ["models/igrf14coeffs.txt", "models/WMM2025.COF"])
def test_synthetic_epoch2020_noise_levels_5pct_10pct_rms_metric(model_path: str):
    """5% and 10% noisy synthetic inversions compared by RMS coefficient error."""
    n_max, c_true, a, b_clean = _linear_system_from_model(model_path, year=2020.0, noise_fraction=0.0)
    _, _, _, b_5 = _linear_system_from_model(model_path, year=2020.0, noise_fraction=0.05)
    _, _, _, b_10 = _linear_system_from_model(model_path, year=2020.0, noise_fraction=0.10)

    out_clean = invert_gauss_coefficients_cg((a, b_clean), n_max=n_max, max_iter=3000, tol=1e-12, x0=None)
    out_5 = invert_gauss_coefficients_cg((a, b_5), n_max=n_max, max_iter=3000, tol=1e-12, x0=None)
    out_10 = invert_gauss_coefficients_cg((a, b_10), n_max=n_max, max_iter=3000, tol=1e-12, x0=None)

    rms_clean = float(np.sqrt(np.mean((out_clean.coefficient_vector - c_true) ** 2)))
    rms_5 = float(np.sqrt(np.mean((out_5.coefficient_vector - c_true) ** 2)))
    rms_10 = float(np.sqrt(np.mean((out_10.coefficient_vector - c_true) ** 2)))

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


def test_invert_gauss_coefficients_cg_solves_linear_system():
    rng = np.random.default_rng(1234)
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)
    q_true = np.column_stack([c_true, c_true, c_true])
    a = rng.normal(0.0, 1.0, size=(5 * n_coeff, n_coeff))
    a[:n_coeff, :] = np.eye(n_coeff)
    b = a @ q_true

    out = invert_gauss_coefficients_cg((a, b), n_max=n_max, max_iter=500, tol=1e-12, x0=None)
    assert np.allclose(out.coefficient_vector, c_true, rtol=1e-9, atol=1e-7)
    assert np.allclose(out.predicted_xyz, b, rtol=1e-10, atol=1e-8)


def test_invert_gauss_coefficients_cache_matches_no_cache():
    rng = np.random.default_rng(77)
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)
    q_true = np.column_stack([c_true, c_true, c_true])
    a = rng.normal(0.0, 1.0, size=(4 * n_coeff, n_coeff))
    a[:n_coeff, :] = np.eye(n_coeff)
    b = a @ q_true

    out_cache = invert_gauss_coefficients_cg((a, b), n_max=n_max, max_iter=300, tol=1e-12, use_cache=True)
    out_nocache = invert_gauss_coefficients_cg((a, b), n_max=n_max, max_iter=300, tol=1e-12, use_cache=False)
    assert np.allclose(out_cache.coefficient_vector, out_nocache.coefficient_vector, rtol=1e-12, atol=1e-10)


def test_invert_gauss_coefficients_tikhonov_matches_damping_path():
    rng = np.random.default_rng(2024)
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)
    q_true = np.column_stack([c_true, c_true, c_true])
    a = rng.normal(0.0, 1.0, size=(6 * n_coeff, n_coeff))
    a[:n_coeff, :] = np.eye(n_coeff)
    b = a @ q_true

    lambda_reg = 2.5
    out_damping = invert_gauss_coefficients_cg(
        (a, b),
        n_max=n_max,
        max_iter=600,
        tol=1e-12,
        damping=lambda_reg,
        use_cache=True,
    )
    out_reg = invert_gauss_coefficients_cg_tikhonov(
        (a, b),
        n_max=n_max,
        lambda_reg=lambda_reg,
        max_iter=600,
        tol=1e-12,
        use_cache=True,
    )
    assert np.allclose(out_reg.coefficient_vector, out_damping.coefficient_vector, rtol=1e-11, atol=1e-9)
    assert np.allclose(out_reg.predicted_xyz, out_damping.predicted_xyz, rtol=1e-12, atol=1e-10)


def test_invert_gauss_coefficients_tikhonov_rejects_negative_lambda():
    a = np.eye(4)
    b = np.ones((4, 3))
    with pytest.raises(ValueError, match="lambda_reg must be non-negative"):
        invert_gauss_coefficients_cg_tikhonov((a, b), n_max=1, lambda_reg=-1.0)


def test_invert_gauss_coefficients_tikhonov_rejects_invalid_regularization_name():
    data = np.array([[0.0, 0.0, 0.0, 6371.2, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="regularization must be one of"):
        invert_gauss_coefficients_cg_tikhonov(
            data,
            n_max=1,
            lambda_reg=1.0,
            regularization="bad_scheme",  # type: ignore[arg-type]
        )


def test_invert_gauss_coefficients_tikhonov_Manojs_scheme_runs():
    rng = np.random.default_rng(991)
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)

    n = 100
    gc_lat_deg = rng.uniform(-70.0, 70.0, n)
    lon_deg = rng.uniform(0.0, 360.0, n)
    r_km = rng.uniform(6350.0, 6800.0, n)
    pred = forward_ned_from_coefficients(
        c_true,
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=n_max,
        use_cache=True,
    )
    data = np.column_stack([np.full(n, 100.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])

    out = invert_gauss_coefficients_cg_tikhonov(
        data,
        n_max=n_max,
        lambda_reg=5.0,
        regularization="Manojs_scheme",
        max_iter=1000,
        tol=1e-12,
        use_cache=True,
    )
    assert out.coefficient_vector.shape == (n_coeff,)
    assert np.all(np.isfinite(out.coefficient_vector))
    assert np.all(np.isfinite(out.residual_xyz))


def test_compute_l_curve_tikhonov_returns_norm_arrays():
    rng = np.random.default_rng(45)
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)
    q_true = np.column_stack([c_true, c_true, c_true])
    a = rng.normal(0.0, 1.0, size=(6 * n_coeff, n_coeff))
    a[:n_coeff, :] = np.eye(n_coeff)
    b = a @ q_true

    x0 = rng.normal(0.0, 20.0, size=n_coeff)
    lambdas = np.array([0.0, 1.0, 100.0], dtype=float)

    solution_norm, residual_norm = compute_l_curve_tikhonov(
        data=(a, b),
        n_max=n_max,
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
        data=(a, b),
        n_max=n_max,
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
    n_max = 3
    n_coeff = n_max * (n_max + 2)
    c_true = rng.normal(0.0, 100.0, size=n_coeff)
    n = 80
    gc_lat_deg = rng.uniform(-75.0, 75.0, n)
    lon_deg = rng.uniform(0.0, 360.0, n)
    r_km = rng.uniform(6350.0, 6800.0, n)
    pred = forward_ned_from_coefficients(
        c_true,
        np.deg2rad(gc_lat_deg),
        np.deg2rad(lon_deg),
        r_km,
        n_max=n_max,
        use_cache=True,
    )
    data = np.column_stack([np.full(n, 200.0), gc_lat_deg, lon_deg, r_km, pred[:, 0], pred[:, 1], pred[:, 2]])
    lambdas = np.array([0.0, 2.0, 20.0], dtype=float)

    solution_norm, residual_norm = compute_l_curve_tikhonov(
        data=data,
        n_max=n_max,
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
    a = np.eye(4)
    b = np.ones((4, 3))
    with pytest.raises(ValueError, match="non-empty"):
        compute_l_curve_tikhonov((a, b), n_max=1, lambda_values=np.array([]), plot=False)
    with pytest.raises(ValueError, match="finite"):
        compute_l_curve_tikhonov((a, b), n_max=1, lambda_values=np.array([0.0, np.nan]), plot=False)
    with pytest.raises(ValueError, match="non-negative"):
        compute_l_curve_tikhonov((a, b), n_max=1, lambda_values=np.array([-1.0, 0.0]), plot=False)
