#AGENTS.md

```markdown
# AGENTS.md — shgeomag Coding Agent Instructions

## Project overview
shgeomag is a Python library for computing geomagnetic field models
(IGRF, WMM, and custom Gauss coefficient files) using vectorised NumPy
operations and a matrix-form design-matrix approach.

## Repository layout (key directories)
- `shgeomag/io/`       — file parsers and writers
- `shgeomag/model/`    — GeomagModel + GaussCoefficients + interpolation
- `shgeomag/core/`     — design matrix, field components, rotations
- `shgeomag/data/`     — InputData container
- `shgeomag/utils/`    — time, coordinate, and grid utilities
- `tests/`             — pytest test suite
- `docs/`              — Sphinx documentation

## Coding conventions
- Python 3.10+. Use type hints on all public functions.
- NumPy-style docstrings on every public function and class.
- No global state. All functions take explicit arguments.
- Physical units: degrees (angles), km (distances), nT (field),
  decimal year (time) — always stated in docstrings.
- Input angles to internal math functions must be in RADIANS.
  Convert at module boundaries; never inside recursion loops.
- Use `np.deg2rad` / `np.rad2deg` at input/output boundaries only.

## Mathematical conventions (DO NOT deviate)
1. Colatitude θ = π/2 − geocentric_latitude_radians (NOT geographic lat).
2. Longitude φ in [0, 2π].
3. Geocentric radius r in km; reference radius a_ref = 6371.2 km.
4. Legendre functions: Schmidt quasi-normalized form (see math guide).
5. Coefficient vector c ordering: for n=1..N_max, m=0..n:
   g_n^m first, then h_n^m if m>0.
   Index formula: i(n,m,type) = n*(n+1) + m - 1 for g; +1 for h (when m>0).
6. B_r positive outward (away from Earth centre).
7. B_theta positive toward south pole (increasing colatitude).
8. B_phi positive eastward.
9. Geodetic NED: X=North, Y=East, Z=Down (positive downward).
10. Inclination I positive downward (I > 0 in Northern Hemisphere).
11. Declination D positive eastward from North.
12. For NED Jacobians, columns represent partial derivatives with respect
    to coefficient vector `c` in order
    `[g10, g11, h11, g20, g21, h21, ...]`.
    `build_ned_jacobian` returns `∂[X,Y,Z]/∂c` (NOT a 3x3 rotation matrix).
13. Regularized inversion uses Tikhonov with identity:
    ``(J^T J + λI)c = J^T d``.
    User-facing regularized API: `invert_gauss_coefficients_cg_tikhonov`
    with explicit `lambda_reg`.

## Performance rules
- NEVER materialise the full G matrix for N_points > 10^5.
  Use the column-loop accumulation strategy (see math guide §2.5).
- For multi-epoch data: sort by decimal year FIRST, then group into
  time-slabs sharing the same interpolated coefficients.
- Legendre recursion outer loop is over degree n (Python loop OK,
  n ≤ 14 for WMM/IGRF). Inner axis is N_points (NumPy vectorised).
- Pre-compute (a/r)^{n+2} using np.cumprod to avoid repeated pow calls.
- Pre-compute cos(mφ), sin(mφ) via iterative complex multiplication,
  not np.cos(m * phi) in a loop.
- For datasets with repeated station geometry, cache by unique
  `(gc_lat, lon, r)` and reuse harmonic basis matrices
  (see `compute_sph_cached` path) instead of recomputing per row.
- Apply the same unique-geometry cache strategy to Jacobian generation
  (`build_ned_jacobian(use_cache=True)`).

## File format rules for io/reader.py
- Format detection must not use file extension — use content inspection.
- IGRF detection: first non-comment line contains 'g/h' as first token.
- WMM detection: first line matches float + 'WMM' string.
- Never hard-code column widths for IGRF — use split().
- Parse SV column in IGRF as the LAST column on each data row.
- For formats (3)-(5): a line beginning with '#' is ALWAYS a comment.
  Lines NOT beginning with '#' could be header or data — decide by
  checking whether the first token is an integer (data) or string (header).
- Emit warnings (not exceptions) for missing dg/dt columns.

## Coordinate utility rules
- geodetic_to_geocentric must use WGS-84 constants from _constants.py.
  Do NOT use spherical earth approximation anywhere in coord conversions.
- sea_level_height_to_radius requires a geoid model. Use pyproj
  with EGM96 as default; document the dependency clearly.
- All conversion functions must be vectorised (accept NumPy arrays).

## Testing rules
- Every new public function MUST have a corresponding test.
- WMM2025 regression test tolerance: 0.01 nT on F; 0.001° on D, I.
- IGRF regression test tolerance: 1 nT on F; 0.01° on D, I.
- Test files for formats (1)-(5) must be placed in tests/data/.
- Use pytest fixtures for model loading (scope="module" for heavy loads).
- Tests must not require network access. Bundle all test data locally.
- Parametrize multi-point and single-point tests separately.
- For new Jacobian code paths, include at least one finite-difference
  derivative test versus direct field evaluation.

## Documentation rules
- Every module has a module-level docstring explaining its purpose.
- Every math-heavy function includes a 'Notes' section with the
  equation it implements (LaTeX in docstrings is rendered by Sphinx
  via `.. math::` in RST; in docstrings use `$...$` for inline).
- Update docs/api/*.rst when adding new public functions.
- Run `make doctest` as part of CI to verify all Examples blocks.

## Forbidden patterns
- No `scipy.special.lpmv` in production code (use internal recursion).
  scipy may be used ONLY in tests for cross-validation.
- No `eval()` or `exec()` in parsers.
- No hardcoded year values — always derive from model or user input.
- Do not use `datetime.now()` inside field computation functions
  (side-effectful); only use it in the public-facing default-argument
  evaluation in io/reader.py.
- No silent swallowing of exceptions in parsers; raise with context.

## Git commit conventions
- Prefix: feat|fix|docs|test|refactor|perf
- Example: `feat(io): add IGRF-14 multi-epoch parser`
- All tests must pass before merging to main.

## CI pipeline (GitHub Actions)
1. Lint: ruff check .
2. Type check: mypy shgeomag/
3. Tests: pytest --cov=shgeomag tests/
4. Docs: cd docs && make html
5. Coverage threshold: 85%
```

---

## 9. Key Implementation Pitfalls to Avoid

1. **Pole singularity in Bφ:** At θ = 0 (north pole) or θ = π (south pole), `sin(θ) = 0`, so the Bφ formula has `1/sin(θ)`. The Schmidt functions `P_n^m(±1) = 0` for `m > 0`, so the product `P_n^m / sin(θ)` remains finite. Use L'Hôpital-style recursion or add a small epsilon guard: `np.where(np.abs(sin_theta) < 1e-10, 0.0, P * m / sin_theta)`.

2. **IGRF secular variation column indexing:** The `SV` column is valid from the last tabulated epoch (e.g., 2025.0) to +5 years (2030.0). It represents `dg/dt` per year, not the total variation. When computing e.g. g(2027) = g(2025) + SV * 2.0.

3. **IGRF epoch column alignment:** There are both `g` and `h` rows interleaved; they share the same set of epoch columns. Make sure both `g` and `h` rows for a given `(n,m)` are associated with the correct column positions.

4. **WMM file terminator:** The `999…9` terminator line must be detected and parsing stopped before it is processed as a data row.

5. **Geocentric vs geodetic latitude confusion:** All field computations use geocentric colatitude θ internally. The rotation to geodetic NED is done only at the output stage. Never mix the two in the same computation.

6. **MJD2000 vs decimal year:** Internally track time as MJD2000 in `InputData`. Convert to decimal year only when interpolating coefficients. Do not use decimal year for time differencing (not uniformly spaced due to leap years).

7. **Coefficient count:** For degree/order `N_max`, the number of `g` coefficients = `N_max*(N_max+2)/2` approximately, but precisely: `Σ_{n=1}^{N_max} (n+1)` = `N_max*(N_max+2)` total coefficients (both g and h). Verify this against the model file line count.

8. **Truncation must reindex:** When `truncate(n_new)` is called, rebuild the coefficient index array; do not just zero out the high-degree terms (zeros in c are not equivalent to a truncated model for the design-matrix column structure).

9. **Regularization semantics:** Use `lambda_reg` only as
   Tikhonov identity weight in coefficient space; do not alter Jacobian
   column ordering or observation units when enabling regularization.
```
