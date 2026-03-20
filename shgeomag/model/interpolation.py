"""Coefficient interpolation and secular-variation extrapolation routines."""

from __future__ import annotations

import warnings

import numpy as np

from .coefficients import GaussCoefficients


def _interp_linear(base: GaussCoefficients, nxt: GaussCoefficients, year: float) -> GaussCoefficients:
    dt = float(nxt.epoch - base.epoch)
    alpha = (year - base.epoch) / dt
    g = base.g + (nxt.g - base.g) * alpha
    h = base.h + (nxt.h - base.h) * alpha
    return GaussCoefficients(
        epoch=year,
        n_max=base.n_max,
        n_array=base.n_array.copy(),
        m_array=base.m_array.copy(),
        g=g,
        h=h,
    )


def _interp_sv(base: GaussCoefficients, year: float) -> GaussCoefficients:
    if base.dg_dt is None or base.dh_dt is None:
        raise ValueError("Secular variation requested but dg_dt/dh_dt are missing")
    delta = year - base.epoch
    return GaussCoefficients(
        epoch=year,
        n_max=base.n_max,
        n_array=base.n_array.copy(),
        m_array=base.m_array.copy(),
        g=base.g + base.dg_dt * delta,
        h=base.h + base.dh_dt * delta,
    )


def interpolate_coefficients(
    year: float,
    coeffs_by_epoch: dict[float, GaussCoefficients],
    valid_from: float,
    valid_to: float,
    sv_coeffs: GaussCoefficients | None = None,
) -> GaussCoefficients:
    """Interpolate or extrapolate Gauss coefficients for a requested year.

    Parameters
    ----------
    year : float
        Decimal year requested.
    coeffs_by_epoch : dict
        Mapping of epoch year to coefficients.
    valid_from, valid_to : float
        Model validity window in decimal years.
    sv_coeffs : GaussCoefficients, optional
        Optional SV row for extrapolation from the last epoch.

    Returns
    -------
    GaussCoefficients
        Interpolated coefficients at ``year``.
    """
    if year < valid_from or year > valid_to:
        warnings.warn(
            f"Requested year {year} outside model validity [{valid_from}, {valid_to}]",
            RuntimeWarning,
            stacklevel=2,
        )

    epochs = sorted(coeffs_by_epoch)
    if len(epochs) == 1:
        base = coeffs_by_epoch[epochs[0]]
        if sv_coeffs is not None:
            return _interp_sv(sv_coeffs, year)
        return GaussCoefficients(
            epoch=year,
            n_max=base.n_max,
            n_array=base.n_array.copy(),
            m_array=base.m_array.copy(),
            g=base.g.copy(),
            h=base.h.copy(),
        )

    if year <= epochs[0]:
        return _interp_linear(coeffs_by_epoch[epochs[0]], coeffs_by_epoch[epochs[1]], year)

    if year >= epochs[-1]:
        if sv_coeffs is not None:
            return _interp_sv(sv_coeffs, year)
        return _interp_linear(coeffs_by_epoch[epochs[-2]], coeffs_by_epoch[epochs[-1]], year)

    idx = int(np.searchsorted(np.asarray(epochs), year, side="right")) - 1
    return _interp_linear(coeffs_by_epoch[epochs[idx]], coeffs_by_epoch[epochs[idx + 1]], year)


__all__ = ["interpolate_coefficients"]
