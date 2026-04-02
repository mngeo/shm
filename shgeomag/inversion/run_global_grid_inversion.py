"""CLI runner for global-grid synthetic inversion experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from shgeomag.inversion.cg import CGInversionResult, invert_gauss_coefficients_cg_tikhonov
from shgeomag.io.reader import load_model
from shgeomag.utils.coord_utils import geodetic_to_geocentric
from shgeomag.utils.grid import global_grid
from shgeomag.utils.time_utils import datetime_to_mjd2000, decimal_year_to_mjd2000, mjd2000_to_decimal_year

RegType = Literal["identity", "Manojs_scheme", "Ohmic_heating"]


@dataclass(slots=True)
class GridInversionRunResult:
    """Container for one global-grid inversion run."""

    output_path: str
    iterations: int
    converged: bool
    final_relative_residual: float
    rms_field_residual: float
    rms_coeff_difference: float
    model_path: str
    year: float
    mjd2000: float


def _default_model_path() -> str:
    for candidate in ("model/igrf14coeffs.txt", "models/igrf14coeffs.txt"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find IGRF file in model/ or models/")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run global-grid synthetic inversion for Gauss coefficients.")
    parser.add_argument("--model-path", type=str, default=None, help="Input model file. Default: auto-detect IGRF.")
    parser.add_argument("--output", type=str, required=True, help="Output coefficient file path.")
    parser.add_argument("--n-max", type=int, default=13, help="Maximum spherical harmonic degree.")
    parser.add_argument("--epoch-year", type=float, default=2020.0, help="Epoch decimal year.")
    parser.add_argument(
        "--date-utc",
        type=str,
        default=None,
        help="Optional ISO UTC datetime (e.g. 2020-01-01T00:00:00+00:00). Overrides epoch-year for field date.",
    )
    parser.add_argument("--noise-percent", type=float, default=0.0, help="Gaussian noise percentage for X/Y/Z.")
    parser.add_argument("--seed", type=int, default=20260330, help="Random seed for noise.")
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="Tikhonov lambda.")
    parser.add_argument(
        "--use-regularization",
        action="store_true",
        help="Enable regularization. If omitted, lambda is forced to 0.0.",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        choices=("identity", "Manojs_scheme", "Ohmic_heating"),
        default="Ohmic_heating",
        help="Regularization type.",
    )
    parser.add_argument("--grid-nx", type=int, default=20, help="Number of longitude points.")
    parser.add_argument("--grid-ny", type=int, default=20, help="Number of latitude points.")
    parser.add_argument("--max-iter", type=int, default=4000, help="Maximum CG iterations.")
    parser.add_argument("--tol", type=float, default=1e-12, help="CG stopping tolerance.")
    return parser


def _resolve_time(epoch_year: float, date_utc: str | None) -> tuple[float, float]:
    if date_utc is not None:
        mjd = float(datetime_to_mjd2000(date_utc).item())
        year = float(mjd2000_to_decimal_year(np.asarray([mjd], dtype=float)).item())
        return year, mjd
    mjd = float(decimal_year_to_mjd2000(np.asarray([epoch_year], dtype=float)).item())
    return float(epoch_year), mjd


def _write_coefficients(path: str, c_est: np.ndarray, n_max: int, header: list[str]) -> None:
    idx = 0
    lines = header.copy()
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            g = float(c_est[idx])
            idx += 1
            if m > 0:
                h = float(c_est[idx])
                idx += 1
            else:
                h = 0.0
            lines.append(f"{n:d} {m:d} {g:.6f} {h:.6f}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_global_grid_inversion(
    *,
    model_path: str | None,
    output_path: str,
    n_max: int,
    epoch_year: float,
    date_utc: str | None,
    noise_percent: float,
    seed: int,
    use_regularization: bool,
    regularization: RegType,
    lambda_reg: float,
    grid_nx: int,
    grid_ny: int,
    max_iter: int,
    tol: float,
) -> GridInversionRunResult:
    """Run synthetic global-grid inversion and write estimated coefficients.

    Parameters
    ----------
    model_path : str or None
        Model file path. If ``None``, auto-detects IGRF in ``model/`` then ``models/``.
    output_path : str
        Output path for retrieved coefficients.
    n_max : int
        Target maximum spherical harmonic degree.
    epoch_year : float
        Decimal year used for interpolation when ``date_utc`` is not provided.
    date_utc : str or None
        Optional ISO datetime string in UTC for exact field date.
    noise_percent : float
        Additive Gaussian noise percentage applied component-wise to ``X, Y, Z``.
    seed : int
        Random seed used to generate noise.
    use_regularization : bool
        Enable Tikhonov regularization. If ``False``, lambda is forced to zero.
    regularization : {"identity", "Manojs_scheme", "Ohmic_heating"}
        Tikhonov regularization matrix type.
    lambda_reg : float
        Regularization weight ``lambda``.
    grid_nx, grid_ny : int
        Longitude/latitude grid size.
    max_iter : int
        Maximum CG iterations.
    tol : float
        Convergence tolerance for CG.

    Returns
    -------
    GridInversionRunResult
        Run diagnostics and metrics.
    """
    if noise_percent < 0.0:
        raise ValueError("noise_percent must be non-negative")
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be non-negative")
    if grid_nx < 2 or grid_ny < 2:
        raise ValueError("grid_nx and grid_ny must be at least 2")
    if n_max <= 0:
        raise ValueError("n_max must be positive")

    resolved_model_path = _default_model_path() if model_path is None else model_path
    year, mjd = _resolve_time(epoch_year=epoch_year, date_utc=date_utc)

    model = load_model(resolved_model_path)
    model.set_truncation(n_max)

    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -90.0, 90.0
    dx = (lon_max - lon_min) / (grid_nx - 1)
    dy = (lat_max - lat_min) / (grid_ny - 1)
    grid = global_grid(
        model=model,
        year=year,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        dx_deg=dx,
        dy_deg=dy,
        h_gps_km=0.0,
        output="xyz",
    )

    lat = grid["lat"].ravel()
    lon = grid["lon"].ravel()
    x = grid["X"].ravel()
    y = grid["Y"].ravel()
    z = grid["Z"].ravel()

    if noise_percent > 0.0:
        frac = noise_percent / 100.0
        rng = np.random.default_rng(seed)
        sig_x = frac * np.maximum(np.abs(x), 1e-12)
        sig_y = frac * np.maximum(np.abs(y), 1e-12)
        sig_z = frac * np.maximum(np.abs(z), 1e-12)
        x = x + rng.normal(0.0, sig_x)
        y = y + rng.normal(0.0, sig_y)
        z = z + rng.normal(0.0, sig_z)

    gc_lat_deg, gc_lon_deg, r_km = geodetic_to_geocentric(lat, lon, 0.0)
    data = np.column_stack(
        [
            np.full_like(gc_lat_deg, mjd, dtype=float),
            gc_lat_deg,
            gc_lon_deg,
            r_km,
            x,
            y,
            z,
        ]
    )

    lambda_eff = float(lambda_reg if use_regularization else 0.0)
    reg_diag = None
    if regularization == "identity":
        # Keep geometry-aware NED inversion path; avoid fallback to generic A@q=B path.
        reg_diag = np.ones(n_max * (n_max + 2), dtype=float)

    result: CGInversionResult = invert_gauss_coefficients_cg_tikhonov(
        data=data,
        n_max=n_max,
        lambda_reg=lambda_eff,
        regularization=regularization,
        reg_diag=reg_diag,
        max_iter=max_iter,
        tol=tol,
        use_cache=True,
        epoch_year=float(epoch_year),
    )

    c_est = result.coefficient_vector
    c_true = model.get_coefficients(float(epoch_year)).coefficient_vector()
    rms_coeff_diff = float(np.sqrt(np.mean((c_est - c_true) ** 2)))
    rms_field_residual = float(np.sqrt(np.mean(result.residual_xyz.reshape(-1) ** 2)))

    date_str = date_utc if date_utc is not None else f"epoch-decimal-year={epoch_year}"
    header = [
        "# estimated coefficients from NED Jacobian-based CG inversion",
        f"# model_path={resolved_model_path}",
        f"# date={date_str}",
        f"# epoch={epoch_year}",
        f"# n_max={n_max}",
        f"# noise_percent={noise_percent}",
        f"# use_regularization={use_regularization}",
        f"# inversion=invert_gauss_coefficients_cg_tikhonov(lambda_reg={lambda_eff},regularization={regularization})",
        "# columns: n m g h",
    ]
    _write_coefficients(output_path, c_est, n_max=n_max, header=header)

    return GridInversionRunResult(
        output_path=output_path,
        iterations=result.iterations,
        converged=result.converged,
        final_relative_residual=result.final_relative_residual,
        rms_field_residual=rms_field_residual,
        rms_coeff_difference=rms_coeff_diff,
        model_path=resolved_model_path,
        year=year,
        mjd2000=mjd,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run global-grid inversion from command line arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_global_grid_inversion(
        model_path=args.model_path,
        output_path=args.output,
        n_max=args.n_max,
        epoch_year=args.epoch_year,
        date_utc=args.date_utc,
        noise_percent=args.noise_percent,
        seed=args.seed,
        use_regularization=args.use_regularization,
        regularization=args.regularization,
        lambda_reg=args.lambda_reg,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print(f"model_path={result.model_path}")
    print(f"decimal_year={result.year:.12f}")
    print(f"mjd2000={result.mjd2000:.12f}")
    print(f"output_file={result.output_path}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"final_relative_residual={result.final_relative_residual:.12e}")
    print(f"rms_error_field_nT={result.rms_field_residual:.12e}")
    print(f"rms_coeff_difference_nT={result.rms_coeff_difference:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
