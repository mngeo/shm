# Inversion Pipeline 01 (Epoch 2020.0)

This document captures the current observatory inversion pipeline used in this repository for epoch `2020.0`, including Tikhonov inversion and L-curve evaluation.

## Inputs

- `../preprocessing/output/selection.txt`
- `../preprocessing/output/obs_mapping.txt`
- `../preprocessing/input/obs_list.txt` (fallback from `./input/obs_list.txt` if missing)
- `../preprocessing/output/crustal_bias.txt`

## Processing Steps

1. Read `selection.txt` and map each `obs` to `(gc_lat, lon, r_km)` via `obs_mapping.txt`.
2. Keep only observatories present in obs list (3-letter matching: `obs[:3]`).
3. Keep only timestamps within ±1.5 years around `2020-01-01`:
   - Decimal-year window: `[2018.5, 2021.5]`
   - Equivalent MJD2000 window from `decimal_year_to_mjd2000`.
4. Aggregate to daily means per observatory and keep only days with at least 3 samples.
5. Subtract crustal biases (`bias_x`, `bias_y`, `bias_z`) per observatory.
6. Build inversion data matrix:
   - Columns: `[time_since_2000, gc_lat_deg, lon_deg, r_km, X_nT, Y_nT, Z_nT]`
   - `time_since_2000 = decimal_year(day_mjd) - 2000.0`
7. Invert for degree/order `n_max=12` using CG with Tikhonov regularization.

## Inversion Settings

- `n_max = 12`
- `epoch_year = 2020.0`
- `max_iter = 3000`
- `tol = 1e-10`
- `use_cache = True`
- Initial model: random normal vector with seed `20260324`, std `50.0`
- Coefficient count: `n_max * (n_max + 2) = 168`

## Outputs (current run set)

- Coefficients (`lambda=1000`):
  - `output/estimated_gauss12_2020_l1000.txt`
- L-curve norms table (`lambdas = [0, 10, 100, 1000, 10000]`):
  - `output/lcurve_gauss12_2020.txt`
- L-curve plot:
  - `output/lcurve_gauss12_2020.png`

## Python Skeleton

```python
import numpy as np
from shgeomag.inversion import (
    invert_gauss_coefficients_cg_tikhonov,
    compute_l_curve_tikhonov,
)

# inv_data: [time_since_2000, gc_lat_deg, lon_deg, r_km, X, Y, Z]
# built after preprocessing steps above

n_max = 12
x0 = np.random.default_rng(20260324).normal(0.0, 50.0, size=n_max * (n_max + 2))

result = invert_gauss_coefficients_cg_tikhonov(
    data=inv_data,
    n_max=n_max,
    lambda_reg=1000.0,
    max_iter=3000,
    tol=1e-10,
    x0=x0,
    use_cache=True,
    epoch_year=2020.0,
)

lambdas = np.array([0.0, 10.0, 100.0, 1000.0, 10000.0], dtype=float)
solution_norm, residual_norm = compute_l_curve_tikhonov(
    data=inv_data,
    n_max=n_max,
    lambda_values=lambdas,
    max_iter=3000,
    tol=1e-10,
    x0=x0,
    use_cache=True,
    epoch_year=2020.0,
    plot=True,  # use plot=False for headless/CI workflows
)
```
