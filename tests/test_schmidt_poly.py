"""Test suite for Schmidt quasi-normalized Associated Legendre Polynomials."""

import numpy as np
import pytest
from math import factorial
from scipy.special import lpmv

from shgeomag.core.design_matrix import _compute_schmidt_p_and_dp


def schmidt_norm(n: int, m: int) -> float:
    """Calculate the Schmidt quasi-normalization factor."""
    if m == 0:
        return 1.0
    return np.sqrt(2.0 * factorial(n - m) / factorial(n + m))


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0])
def test_schmidt_polynomials(theta_deg):
    """
    Test that the recursive Schmidt polynomial algorithm matches
    values computed independently using scipy's fully normalized functions.
    """
    n_max = 6
    theta = np.deg2rad(np.array([theta_deg]))
    ct = np.cos(theta[0])

    p_code, dp_code = _compute_schmidt_p_and_dp(theta, n_max)

    for n in range(0, n_max + 1):
        for m in range(0, n + 1):
            # Scipy's lpmv computes un-normalized P_n^m * (-1)^m (Condon-Shortley phase)
            scipy_pnm = float(lpmv(m, n, ct))
            # Remove phase and apply Schmidt normalization
            expected_p = scipy_pnm * ((-1) ** m) * schmidt_norm(n, m)
            actual_p = float(p_code[n, m, 0])
            
            assert np.isclose(actual_p, expected_p, atol=1e-8), \
                f"Mismatch for P[{n},{m}] at {theta_deg} deg: {actual_p} vs {expected_p}"


@pytest.mark.parametrize("theta_deg", [10.0, 45.0, 80.0, 100.0, 135.0, 170.0])
def test_schmidt_derivatives(theta_deg):
    """
    Test derivatives of the Schmidt polynomials via central finite differences.
    """
    n_max = 6
    eps = 1e-6
    theta = np.deg2rad(np.array([theta_deg]))
    
    # Evaluate at theta + eps and theta - eps
    p_plus, _ = _compute_schmidt_p_and_dp(theta + eps, n_max)
    p_minus, _ = _compute_schmidt_p_and_dp(theta - eps, n_max)
    
    _, dp_code = _compute_schmidt_p_and_dp(theta, n_max)

    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            expected_dp = float((p_plus[n, m, 0] - p_minus[n, m, 0]) / (2 * eps))
            actual_dp = float(dp_code[n, m, 0])
            
            assert np.isclose(actual_dp, expected_dp, atol=1e-5), \
                f"Derivative mismatch for dP[{n},{m}] at {theta_deg} deg: {actual_dp} vs {expected_dp}"
