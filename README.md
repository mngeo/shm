# shgeomag (`shm`)
Vectorized spherical-harmonic geomagnetic modeling for IGRF, WMM, and custom Gauss coefficient files.

## Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
# or use pinned runtime+dev deps:
# pip install -r requirements.txt
```

## Run Full Pipeline
```bash
. .venv/bin/activate
ruff check .
mypy shgeomag/
pytest --cov=shgeomag tests/
cd docs && make doctest html
```

## Usage
```python
import numpy as np
from shgeomag.io.reader import load_model
from shgeomag.core.design_matrix import build_ned_jacobian
from shgeomag.core.field import compute_fdi, compute_sph_cached
from shgeomag.inversion import (
    compute_l_curve_tikhonov,
    invert_gauss_coefficients_cg,
    invert_gauss_coefficients_cg_tikhonov,
)
from shgeomag.utils import global_grid

model = load_model("models/WMM2025.COF")
f, d, i = compute_fdi(
    model,
    gc_lat_deg=np.array([10.0]),
    lon_deg=np.array([20.0]),
    r_km=np.array([6371.2]),
    year=2026.0,
)
print(f, d, i)

grid = global_grid(model, 2026.0, -180, 180, -90, 90, 5.0, 5.0, 0.0)
print(grid["F"].shape)

# Optional: faster spherical components when many rows share
# identical (gc_lat, lon, r) geometry.
br, bt, bp = compute_sph_cached(
    model,
    gc_lat_deg=np.array([10.0, 10.0, -20.0]),
    lon_deg=np.array([30.0, 30.0, 120.0]),
    r_km=np.array([6371.2, 6371.2, 6400.0]),
    year=2026.0,
)
print(br, bt, bp)

# Jacobian of geodetic NED components wrt Gauss coefficients c
# with c ordering [g10, g11, h11, g20, ...].
j_ned = build_ned_jacobian(
    gc_lat_rad=np.deg2rad(np.array([10.0, -20.0])),
    lon_rad=np.deg2rad(np.array([30.0, 120.0])),
    r_km=np.array([6371.2, 6400.0]),
    n_max=model.n_max,
)
print(j_ned.shape)  # (3*N, n_coeff)
```

## Inversion (CG)
```python
import numpy as np
from shgeomag.inversion import invert_gauss_coefficients_cg

# data columns: [MJD2000, gc_lat_deg, lon_deg, r_km, X_nT, Y_nT, Z_nT]
data = np.array([
    [-1000.0, 10.0, 20.0, 6371.2, 30000.0, 500.0, 1000.0],
    [-1000.0, -5.0, 60.0, 6371.2, 28000.0, 300.0, -2000.0],
], dtype=float)

res = invert_gauss_coefficients_cg(
    data,
    n_max=8,
    max_iter=200,
    tol=1e-8,
    damping=0.0,
    use_cache=True,  # cache repeated geometry
)
print(res.converged, res.iterations, res.final_relative_residual)
print(res.coefficient_vector.shape)  # (n_max*(n_max+2),)

# Tikhonov-regularized variant with explicit lambda.
res_reg = invert_gauss_coefficients_cg_tikhonov(
    data,
    n_max=8,
    lambda_reg=1000.0,
    regularization="identity",  # or "Manojs_scheme" / "Ohmic_heating"
    max_iter=200,
    tol=1e-8,
    use_cache=True,
)
print(res_reg.converged, res_reg.iterations, res_reg.final_relative_residual)

# New weighted scheme:
# L = diag((1:n_coeff)^2), R = L.T @ L = diag((1:n_coeff)^4)
res_reg_weighted = invert_gauss_coefficients_cg_tikhonov(
    data,
    n_max=8,
    lambda_reg=1000.0,
    regularization="Manojs_scheme",
)

res_reg_ohmic = invert_gauss_coefficients_cg_tikhonov(
    data,
    n_max=8,
    lambda_reg=1000.0,
    regularization="Ohmic_heating",
)
# Ohmic_heating diagonal used in R:
# diag_i = 4*pi*(Re/Rcmb)^(2*n+3)*(n+1)*(2*n+1)*(2*n+3)/n

# L-curve for user-selected lambdas (plot can be switched off).
solution_norm, residual_norm = compute_l_curve_tikhonov(
    data=data,
    n_max=8,
    lambda_values=np.array([0.0, 1.0, 10.0, 100.0, 1000.0]),
    regularization="identity",
    plot=False,
)
print(solution_norm, residual_norm)
```

## Synthetic Inversion Tests
- Test suite includes epoch `2020.0` synthetic recovery checks for both:
  - `models/igrf14coeffs.txt`
  - `models/WMM2025.COF`
- Metric: RMS difference between recovered and original coefficient vectors.
- Covered cases:
  - clean synthetic data (near-zero RMS expected)
  - 5% Gaussian noise (`sigma = 0.05 * abs(component)`)
  - 10% Gaussian noise (`sigma = 0.10 * abs(component)`)
- Run:
```bash
pytest -q tests/test_inversion.py
```

## Pipeline Reference (2020.0 + L-curve)
- End-to-end observatory preprocessing and inversion recipe:
  - `inversion_pipeline_01.md`
- Includes:
  - data mapping (`obs -> gc_lat/lon/r`)
  - obs-list filter
  - ±1.5 year time window around `2020-01-01`
  - daily mean (`>=3/day`)
  - crustal-bias subtraction
  - degree-12 Tikhonov inversion
  - L-curve (`lambda=[0,10,100,1000,10000]`)

## Performance Notes
- For one-off forward evaluation, prefer `compute_sph(...)`.
- For repeated geometry workloads (many rows sharing the same
  `gc_lat/lon/r`), use `compute_sph_cached(...)` for faster execution.
- `compute_multi_epoch(...)` already uses the cached spherical path
  internally per time slab.
- `build_ned_jacobian(...)` supports the same repeated-geometry optimization
  via `use_cache=True` (default).
- `invert_gauss_coefficients_cg(...)` uses a cached Jacobian operator for
  `J @ v` and `J.T @ v` products when `use_cache=True`.
- `invert_gauss_coefficients_cg_tikhonov(...)` is the explicit
  Tikhonov-regularized API using ``(J^T J + lambda R) c = J^T d`` with
  user-provided `lambda_reg`, where `R` is selected by
  `regularization` (`"identity"`, `"Manojs_scheme"`, or
  `"Ohmic_heating"`), or by
  passing a custom diagonal via `reg_diag`.
  For `"Ohmic_heating"`, the coefficient-space diagonal uses
  ``diag_i = 4*pi*(Re/Rcmb)^(2*n+3)*(n+1)*(2*n+1)*(2*n+3)/n``
  (non-inverted form).
- `compute_l_curve_tikhonov(...)` returns
  ``(solution_norm, residual_norm)`` for a lambda array and can
  optionally plot the L-curve (`plot=True/False`).

## Global-Grid Inversion CLI
Use the synthetic inversion runner to generate/recover Gauss coefficients
on a global grid and write output files:

```bash
. .venv/bin/activate
python -m shgeomag.inversion.run_global_grid_inversion \
  --model-path models/igrf14coeffs.txt \
  --output output/estimated_gauss13_2020_20260330_04.txt \
  --n-max 13 \
  --epoch-year 2020.0 \
  --date-utc 2020-01-01T00:00:00+00:00 \
  --noise-percent 5 \
  --seed 20260330 \
  --use-regularization \
  --regularization Ohmic_heating \
  --lambda-reg 0.0001 \
  --grid-nx 20 \
  --grid-ny 20
```

## Model Files
- WMM example: `models/WMM2025.COF`
- IGRF example: `models/igrf14coeffs.txt`

## Requirements
- Pinned runtime+dev dependencies are in `requirements.txt`.
