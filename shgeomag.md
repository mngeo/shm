# shgeomag — High-Level Plan, Mathematical Programming Guide & AGENTS.md

---

## 0. Repository Overview

```
shgeomag/
├── shgeomag/
│   ├── __init__.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── reader.py          # model file parser (all 5 format variants)
│   │   └── writer.py          # model text-file writer
│   ├── model/
│   │   ├── __init__.py
│   │   ├── coefficients.py    # GaussCoefficients data class
│   │   ├── model.py           # GeomagModel (the central object)
│   │   └── interpolation.py   # coefficient interpolation helpers
│   ├── core/
│   │   ├── __init__.py
│   │   ├── design_matrix.py   # build G matrix (vectorised)
│   │   ├── field.py           # Br/Btheta/Bphi → BxByBz, F/D/I
│   │   └── rotation.py        # geocentric ↔ geodetic rotations
│   ├── data/
│   │   ├── __init__.py
│   │   └── container.py       # InputData ([MJD2000, gc_lat, lon, R])
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── time_utils.py      # Unix/datetime ↔ MJD2000
│   │   ├── coord_utils.py     # geodetic ↔ geocentric, height → R
│   │   └── grid.py            # global/local grid builder + map output
│   └── _constants.py          # WGS-84, earth radius, etc.
├── tests/
│   ├── test_io.py
│   ├── test_model.py
│   ├── test_design_matrix.py
│   ├── test_field.py
│   ├── test_coords.py
│   ├── test_time.py
│   ├── test_igrf_published.py # regression against IGRF tabulated values
│   └── test_wmm_published.py  # regression against WMM2025 test values
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   └── tutorials/
├── AGENTS.md
├── pyproject.toml
└── README.md
```

---

## 1. Module-by-Module High-Level Plan

### 1.1 `shgeomag/_constants.py`
- WGS-84 semi-major axis `a = 6378.137 km`, flattening `f = 1/298.257223563`
- Reference radius for Gauss coefficients `a_ref = 6371.2 km`
- Derived: semi-minor axis `b = a(1-f)`, first eccentricity squared `e2 = 1-(b/a)²`

---

### 1.2 `shgeomag/io/reader.py` — Model File Parser

**Detection logic (in order):**
1. Read first non-blank, non-`#` line.
2. If it matches the IGRF header pattern (`g/h  n  m  YYYY  YYYY …`), parse as IGRF-14 format.
3. If it matches the WMM header pattern (`YYYY.0  WMM-YYYY  DD/MM/YYYY`), parse as WMM `.COF` format.
4. Count non-`#` columns in the first data row:
   - 4 columns → format (3): `n m g h`
   - 2 columns after a `g/h` marker → format (4): `[g/h] n m value`
   - 6 columns → format (5): `n m g h dg/dt dh/dt`

**IGRF parser specifics:**
- Header row contains the epoch years (1900.0, 1905.0, … 2025.0) plus a final `SV` column.
- Each data line: `g/h  n  m  val_1900  val_1905 … val_2025  SV`
- Store each epoch column as a separate `GaussCoefficients` snapshot; store the `SV` column as a secular variation array valid from the last epoch to +5 years.
- Detect `Nmax` from the maximum `n` appearing in the file.

**WMM `.COF` parser specifics:**
- Header line: `epoch  model_name  date`
- Data lines: `n  m  g  h  dg/dt  dh/dt`  (space separated, 6 values)
- Terminator: a row of `999…9`
- Epoch is the decimal year from the header; valid window is `[epoch, epoch+5]`.

**Format (3) / (4) / (5) parsers:**
- Skip lines beginning with `#`.
- Auto-detect column count.
- If no year is embedded, accept `year` kwarg; default to `datetime.utcnow().year`.
- Valid window: `[year-5, year+5]` unless overridden.

**Return:** a `GeomagModel` object.

---

### 1.3 `shgeomag/model/coefficients.py`

```
GaussCoefficients:
    epoch: float            # decimal year
    n_max: int
    n_array: np.ndarray     # shape (N_coeffs,) int
    m_array: np.ndarray     # shape (N_coeffs,) int
    g: np.ndarray           # shape (N_coeffs,) float64
    h: np.ndarray           # shape (N_coeffs,) float64
    dg_dt: np.ndarray|None
    dh_dt: np.ndarray|None
```

- Indexing convention: coefficients stored in order of increasing `n`, then increasing `|m|`, g before h. This maps directly to the column vector `c` used in `G·c = B`.
- Provide helper `to_gh_dict()` → `{(n,m): (g,h)}` for human-readable access.
- Provide `truncate(n_max_new)` returning a new object with all coefficients where `n > n_max_new` removed.

---

### 1.4 `shgeomag/model/model.py` — `GeomagModel`

**Attributes:**
- `source_file: str`
- `model_name: str`
- `format: str`  ("IGRF", "WMM", "ngmh", "gh", "ngmhdgdh")
- `valid_from: float`, `valid_to: float`  (decimal year)
- `epochs: list[float]`  (one or many)
- `coeffs: dict[float, GaussCoefficients]`  keyed by epoch
- `sv_coeffs: GaussCoefficients|None`  secular variation for extrapolation
- `n_max: int`  (max degree in file)
- `n_truncate: int`  (user-set truncation, defaults to `n_max`)

**Methods:**
- `info()` — print model name, epochs, `n_max`, number of g+h coefficients
- `set_truncation(n)` — set `n_truncate`; raises if `n > n_max`
- `get_coefficients(year: float) -> GaussCoefficients` — calls interpolation module
- `write(path, fmt)` — calls writer

---

### 1.5 `shgeomag/model/interpolation.py`

Two cases:

**Case A — secular variation present:**
```
g'(t) = g(t0) + dg/dt * (t - t0)
h'(t) = h(t0) + dh/dt * (t - t0)
```

**Case B — two epoch snapshots bracketing `t`:**
```
g'(t) = g(t0) + [g(t1) - g(t0)] / (t1 - t0) * (t - t0)
h'(t) = h(t0) + [h(t1) - h(t0)] / (t1 - t0) * (t - t0)
```

Both cases are vectorised over the coefficient arrays using NumPy.

Issue a `warnings.warn(...)` if `t < valid_from` or `t > valid_to`.

---

### 1.6 `shgeomag/core/design_matrix.py` — The G Matrix

This is the heart of the module. See **Section 2** for the full mathematical programming guide.

**Public API:**
```python
def build_design_matrix(
    gc_lat_rad, lon_rad, r_km,
    n_max, n_truncate,
    a_ref=6371.2
) -> np.ndarray:   # shape (3*N_points, N_coeffs)
```

The matrix maps the coefficient vector `c` (g₁₀, g₁₁, h₁₁, g₂₀, …) to the field vector `B` = `[Br₁, Bθ₁, Bφ₁, Br₂, …]`.

---

### 1.7 `shgeomag/core/field.py`

**`compute_sph(model, gc_lat, lon, r, year)`**
- Returns `(Br, Btheta, Bphi)` arrays, shape `(N,)`.
- Internally calls `build_design_matrix` then `G @ c`.

**`compute_geo(model, gc_lat, lon, r, year)`**
- Returns `(Bx, By, Bz)` in geodetic NED frame.
- Calls `compute_sph`, then `rotate_sph_to_geo`.

**`compute_fdi(model, gc_lat, lon, r, year)`**
- Returns `(F, D, I)` — total intensity, declination, inclination.
- Calls `compute_geo` to get geodetic `(Bx, By, Bz)`, then derives F/D/I.

**`compute_multi_epoch(model, data: InputData)`**
- `data` columns: `[MJD2000, gc_lat, lon, R]`
- Strategy: convert all MJD2000 to decimal year, sort by year, find epoch breaks, batch-compute per-epoch slab.

---

### 1.8 `shgeomag/core/rotation.py`

Geocentric spherical → geodetic NED rotation. See **Section 2.6**.

---

### 1.9 `shgeomag/data/container.py` — `InputData`

```python
class InputData:
    data: np.ndarray   # shape (N, 4), columns: MJD2000, gc_lat, lon, R

    @classmethod
    def from_array(cls, arr): ...
    @classmethod
    def from_dataframe(cls, df, col_map): ...
    def validate(self): ...   # check ranges, warn on out-of-range
    def sort_by_time(self): ...
```

---

### 1.10 `shgeomag/utils/time_utils.py`

- `unix_to_mjd2000(ts)` — Unix timestamp (seconds since 1970-01-01) → MJD2000
- `datetime_to_mjd2000(dt)` — Python `datetime` / ISO string → MJD2000
- `mjd2000_to_datetime(mjd)` → `datetime`
- `mjd2000_to_decimal_year(mjd)` → float (accounts for leap years)
- `decimal_year_to_mjd2000(dy)` → float

**MJD2000 definition:** Modified Julian Day 2000 = Julian Day Number − 2451544.5 (i.e., days since 2000-01-01T00:00:00 UTC).

---

### 1.11 `shgeomag/utils/coord_utils.py`

- `geodetic_to_geocentric(lat_gd_deg, lon_deg, h_km)` → `(gc_lat_deg, lon_deg, r_km)`
- `geocentric_to_geodetic(gc_lat_deg, lon_deg, r_km)` → `(lat_gd_deg, lon_deg, h_km)`
- `gps_height_to_radius(lat_gd_deg, h_gps_km)` → `r_km`
  - GPS height = height above WGS-84 ellipsoid = geodetic height
- `sea_level_height_to_radius(lat_gd_deg, h_msl_km)` → `r_km`
  - Requires EGM96/EGM2008 geoid undulation (embed a low-res lookup table or use `pyproj`)
- `radius_to_gps_height(lat_gd_deg, r_km)` → `h_gps_km`
- `radius_to_sea_level_height(lat_gd_deg, r_km)` → `h_msl_km`

---

### 1.12 `shgeomag/utils/grid.py`

```python
def global_grid(model, year, lon_min, lon_max, lat_min, lat_max,
                dx_deg, dy_deg, h_gps_km, output='fdi'):
    ...
    # Returns xarray.Dataset with F, D, I (or BxByBz) on a regular grid

def plot_grid(ds, component, projection='auto', output_path=None):
    ...
    # Uses geopandas + cartopy/matplotlib
    # 'auto' → PlateCarree for global, UTM or LCC for local
```

---

## 2. Mathematical Programming Guide

### 2.1 Coordinate Conventions

The following conventions are used throughout:

| Symbol | Meaning |
|--------|---------|
| `θ` (theta) | Geocentric colatitude = 90° − gc_lat, in radians |
| `φ` (phi)   | Geocentric longitude, 0–360°, in radians |
| `r`         | Geocentric radius in km |
| `a`         | Reference radius = 6371.2 km |
| `n, m`      | Degree and order of Gauss coefficient |

**CRITICAL:** The geocentric colatitude θ = π/2 − gc_lat_rad must be computed consistently. At the geographic north pole, gc_lat = +90°, θ = 0. At the south pole, gc_lat = −90°, θ = π.

---

### 2.2 Associated Legendre Functions — Schmidt Quasi-Normal Form

The key to correct Gauss spherical harmonics is the **Schmidt quasi-normalized** associated Legendre functions `P_n^m(cos θ)`.

**Recursion (preferred over scipy for large N):**

Start with:
```
P_0^0 = 1
P_1^0 = cos(θ)
P_1^1 = sin(θ)
```

For the diagonal (`m = n`):
```
P_n^n = sin(θ) * sqrt((2n-1)/(2n)) * P_{n-1}^{n-1}
```

For the sub-diagonal (`m = n-1`):
```
P_n^{n-1} = cos(θ) * sqrt(2n-1) * P_{n-1}^{n-1}
```

For all other `m < n-1`:
```
P_n^m = cos(θ) * sqrt((4n²-1)/(n²-m²)) * P_{n-1}^m
        - sqrt(((2n+1)(n-1-m)(n-1+m)) / ((2n-3)(n²-m²))) * P_{n-2}^m
```

**Programming this efficiently:**
- Pre-allocate a 2D array `P[n_max+1, n_max+1]`.
- Compute using a double nested loop over `n` (outer) and `m` (inner).
- For vectorised evaluation over N points, use shape `P[n_max+1, n_max+1, N_points]`.
- Alternatively, use a pre-computed recursion coefficient array (constants depending only on n,m) multiplied element-wise by the appropriate sin/cos terms — this makes the inner loop NumPy-efficient.

**The derivative** `dP_n^m / dθ` (needed for Btheta) can be computed from:
```
dP_n^m/dθ = 0.5 * sqrt((n-m)(n+m+1)) * P_n^{m+1}
             - 0.5 * sqrt((n+m)(n-m+1)) * P_n^{m-1}
             [with convention P_n^{m} = 0 for m > n or m < 0]
```

Or by the recurrence:
```
dP_n^m/dθ = -sin(θ) * [recursion_a * P_{n-1}^m - recursion_b * P_{n-2}^m]
             + cos(θ) * recursion_c * P_n^{m}   (varies by form used)
```

The simplest correct form in code: after computing all `P_n^m`, compute `dP/dtheta` as:
```
dPnm = sqrt((n-m)*(n+m+1)) * P_n^{m+1} / 2  -  sqrt((n+m)*(n-m+1)) * P_n^{m-1} / 2
```
with `P_n^{n+1} = 0` and appropriate handling of `m = 0`.

---

### 2.3 Gauss' Spherical Harmonic Potential

The geomagnetic scalar potential is:
```
V(r, θ, φ) = a Σ_{n=1}^{N} (a/r)^{n+1} Σ_{m=0}^{n}
               [g_n^m cos(mφ) + h_n^m sin(mφ)] P_n^m(cos θ)
```

The magnetic field components are `B = −∇V` in spherical coordinates:
```
Br     = +∂V/∂r
         = Σ_{n,m} (n+1) (a/r)^{n+2} [g cos(mφ) + h sin(mφ)] P_n^m / a
           (SIGN: Br = -∂V/∂r is radially outward positive;
            field points inward at north pole so Br < 0 near north pole)

Bθ     = -(1/r) ∂V/∂θ
         = +(1/r) Σ_{n,m} (a/r)^{n+1} [g cos(mφ) + h sin(mφ)] dP_n^m/dθ

Bφ     = -(1/(r sinθ)) ∂V/∂φ
         = +(1/(r sinθ)) Σ_{n,m} m (a/r)^{n+1} [-g sin(mφ) + h cos(mφ)] P_n^m
```

**Sign convention note:**  
The convention used in WMM/IGRF is:
- `Br` is positive radially outward (away from Earth's center)
- `Bθ` is positive southward (increasing θ = towards south pole)
- `Bφ` is positive eastward

---

### 2.4 Building the Design Matrix G

The design matrix `G` converts the coefficient vector `c` to the field vector `B = G · c`, where:

```
c = [g_1^0, g_1^1, h_1^1, g_2^0, g_2^1, h_2^1, g_2^2, h_2^2, …]
```

ordering: for each `n` from 1 to `N_max`, for each `m` from 0 to `n`:
- append `g_n^m` (always), then `h_n^m` (only if `m > 0`)

Total number of coefficients = `N_max * (N_max + 2)`.

**Structure of G for a single point (3 rows per point):**

For each coefficient `(n, m, type)` where `type ∈ {g, h}`, define the partial derivatives:

```
For g_n^m (m=0):
  ∂Br/∂g_n^0    = -(n+1) * (a/r)^{n+2} * P_n^0 * cos(0*φ)
  ∂Bθ/∂g_n^0   = +(a^{n+1}/r^{n+2}) * dP_n^0/dθ * cos(0*φ)     [divide by 1/r outside]
  ∂Bφ/∂g_n^0   = 0   (since m=0, sinφ term = 0)

For g_n^m (m>0):
  ∂Br/∂g_n^m   = -(n+1) * (a/r)^{n+2} * P_n^m * cos(mφ)
  ∂Bθ/∂g_n^m  = +(1/r) * (a/r)^{n+1} * dP_n^m/dθ * cos(mφ)
  ∂Bφ/∂g_n^m  = -(m/(r sinθ)) * (a/r)^{n+1} * P_n^m * (-sin(mφ)) — note: = +m*(a/r)^{n+1}*P_n^m*sin(mφ)/(r*sinθ)

For h_n^m (m>0):
  ∂Br/∂h_n^m   = -(n+1) * (a/r)^{n+2} * P_n^m * sin(mφ)
  ∂Bθ/∂h_n^m  = +(1/r) * (a/r)^{n+1} * dP_n^m/dθ * sin(mφ)
  ∂Bφ/∂h_n^m  = -(m/(r sinθ)) * (a/r)^{n+1} * P_n^m * cos(mφ)
```

**Vectorisation over N points:**

Pre-compute arrays of shape `(N_points,)`:
- `theta` = colatitude in radians
- `sin_theta`, `cos_theta`
- `r_ratio_np2[n]` = `(a/r)^{n+2}` for each n — compute as `(a/r)^3 * (a/r)^{n-1}` using `cumprod`
- `cos_mphi[m]`, `sin_mphi[m]` for m = 0…N_max — use `np.outer` or loop + cumulative angle

Then the design matrix is assembled as `G[3i, j]`, `G[3i+1, j]`, `G[3i+2, j]` where `i` indexes the point and `j` indexes the coefficient.

For large N_points, build G in blocks or use sparse operations. For very large inputs (>10⁶ points), consider `numba` JIT or chunk-wise matrix multiplication to avoid materialising the full matrix.

---

### 2.5 Efficient Computation for Large Inputs

**Strategy:**
1. **Batch by epoch:** For multi-year data, sort `InputData` by decimal year. For IGRF-style models with 5-year epochs, coefficients change only at epoch boundaries. Group rows with the same interpolated-coefficient epoch, compute `G · c` once per group.
2. **Avoid materialising G:** For a pure field evaluation (not inversion), evaluate `G · c` column-by-column without storing G:
   ```
   Br_total = 0
   for each (n, m, g, h):
       Br_total += g * partial_Br_g + h * partial_Br_h
   ```
   This reduces memory from `O(N * N_coeffs)` to `O(N)` intermediates.
3. **Vectorise Legendre recursion:** Loop over degree `n` (≤ 13 for WMM, ≤ 14 for IGRF) in Python; inner operations are NumPy over `N_points`.
4. **Pre-compute (a/r)^n:** `np.cumprod(a/r)` repeated to get the correct powers.
5. **Pre-compute trigonometric terms:** `np.cos(m*phi)` and `np.sin(m*phi)` for each `m` using complex exponential trick: `e_mphi = exp(1j * phi)`, then `cos_mphi[m] = real(e_mphi^m)`, `sin_mphi[m] = imag(e_mphi^m)`, computed via iterative complex multiplication.

---

### 2.6 Geocentric → Geodetic Rotation

The three geocentric spherical components `(Br, Bθ, Bφ)` need to be rotated to geodetic north-east-down `(Bx, By, Bz)` = `(X, Y, Z)` = `(North, East, Down)`.

**Step 1: Convert geocentric spherical → geocentric Cartesian (ECEF)**

Let `θ_c` = geocentric colatitude, `φ` = longitude:
```
B_ECEF_x = Br * sin(θ_c)*cos(φ) + Bθ * cos(θ_c)*cos(φ) - Bφ * sin(φ)
B_ECEF_y = Br * sin(θ_c)*sin(φ) + Bθ * cos(θ_c)*sin(φ) + Bφ * cos(φ)
B_ECEF_z = Br * cos(θ_c)        - Bθ * sin(θ_c)
```

**Step 2: Rotate ECEF → geodetic NED at geodetic latitude `φ_d`**

Given geodetic latitude `φ_d` (in radians), the local North-East-Down unit vectors in ECEF are:
```
North = [-sin(φ_d)*cos(λ), -sin(φ_d)*sin(λ), cos(φ_d)]
East  = [-sin(λ),           cos(λ),            0       ]
Down  = [-cos(φ_d)*cos(λ), -cos(φ_d)*sin(λ), -sin(φ_d)]
```
where `λ` = longitude.

Therefore:
```
Bx (North) = B_ECEF · North
By (East)  = B_ECEF · East
Bz (Down)  = B_ECEF · Down
```

**Alternative direct formula** (combining both steps, commonly used in WMM code):

Given the angle `ψ` = geodetic latitude − geocentric latitude = `φ_d − (π/2 − θ_c)`:
```
Bx = -Bθ * cos(ψ) - Br * sin(ψ)    [North, positive northward]
By =  Bφ                              [East,  positive eastward ]
Bz = +Bθ * sin(ψ) - Br * cos(ψ)    [Down,  positive downward  = -Up]
```

The angle `ψ` is the angular offset between geocentric and geodetic "vertical" directions.

**Computing `ψ`:**
Given geodetic latitude `φ_d` and geocentric colatitude `θ_c`:
```
ψ = φ_d - (π/2 - θ_c) = φ_d + θ_c - π/2
```
or equivalently `ψ = φ_d − φ_c` where `φ_c` = geocentric latitude = `π/2 − θ_c`.

---

### 2.7 Geodetic ↔ Geocentric Conversion

**Geodetic → Geocentric:**

Given geodetic latitude `φ_d`, longitude `λ`, and height above ellipsoid `h` (km):

WGS-84 constants:
```
a_wgs = 6378.137 km
b_wgs = a_wgs * (1 - f)   # f = 1/298.257223563
e2 = 1 - (b_wgs/a_wgs)²   # first eccentricity squared ≈ 0.00669437999014
```

Normal radius of curvature:
```
N(φ_d) = a_wgs / sqrt(1 - e2 * sin²(φ_d))
```

ECEF coordinates:
```
X = (N + h) * cos(φ_d) * cos(λ)
Y = (N + h) * cos(φ_d) * sin(λ)
Z = (N * (1 - e2) + h) * sin(φ_d)
```

Geocentric radius and latitude:
```
r = sqrt(X² + Y² + Z²)
φ_c = arcsin(Z / r)   (geocentric latitude)
λ stays the same
```

**Geocentric → Geodetic:**

Use Bowring's iterative method or the direct Zhu (1994) / Fukushima method for numerical stability near poles.

Bowring's method (iterate 2–3 times):
```
β = arctan(Z / ((1-f) * sqrt(X²+Y²)))   # initial guess (reduced latitude)
φ_d = arctan((Z + e2 * b_wgs * sin³(β)) / (sqrt(X²+Y²) - e2 * a_wgs * cos³(β)))
β = arctan((1-f) * tan(φ_d))
(repeat 2 more times)
h = sqrt(X²+Y²)*cos(φ_d) + Z*sin(φ_d) - a_wgs * sqrt(1 - e2*sin²(φ_d))
```

---

### 2.8 F, D, I from Geodetic Components

Given geodetic NED components `(X, Y, Z)` = `(North, East, Down)`:
```
H = sqrt(X² + Y²)          # horizontal intensity
F = sqrt(X² + Y² + Z²)     # total intensity
D = arctan2(Y, X)           # declination (positive east of north)
I = arctan2(Z, H)           # inclination (positive downward = dipping angle)
```

All in degrees for output:
```
D_deg = degrees(D)
I_deg = degrees(I)
```

---

### 2.9 Time Conversions

**MJD2000** = Julian Day Number − 2451544.5 = days since 2000-01-01 12:00:00 UTC.

```
JD(2000-01-01 12:00 UTC) = 2451545.0
MJD = JD - 2400000.5
MJD2000 = MJD - 51544.0 = JD - 2451544.5
```

**Unix timestamp → MJD2000:**
```
JD_unix_epoch = 2440587.5   # JD of 1970-01-01 00:00 UTC
MJD2000 = (unix_ts / 86400.0) + (2440587.5 - 2451544.5)
         = unix_ts / 86400.0 - 10957.0
```

**Decimal year from MJD2000:**
```
dt = mjd2000_to_datetime(mjd)
year_start = datetime(dt.year, 1, 1)
year_end   = datetime(dt.year+1, 1, 1)
frac = (dt - year_start).total_seconds() / (year_end - year_start).total_seconds()
decimal_year = dt.year + frac
```

Use `calendar.isleap` to correctly handle 365/366 day years.

---

## 3. I/O — Writer (`shgeomag/io/writer.py`)

Support output formats:
- **plain text** with columns `n m g h` or `n m g h dg/dt dh/dt`
- **WMM `.COF`** format (header + 6-column data + 999 terminator)
- **comment header** with model name, epoch, date written

---

## 4. Grid and Mapping (`shgeomag/utils/grid.py`)

**Grid generation algorithm:**
1. Accept bounding box `(lon_min, lon_max, lat_min, lat_max)`, step sizes `(dx, dy)` in degrees, GPS height `h_gps`.
2. Use `np.meshgrid` to generate `(lat, lon)` arrays.
3. Convert each point: `geodetic_to_geocentric(lat, lon, h_gps)` → `(gc_lat, lon, r)`.
4. Flatten to 1D arrays; build `InputData`; call `compute_fdi` or `compute_geo`.
5. Reshape output back to 2D grid.
6. Package into `xarray.Dataset` with proper coordinate metadata.

**Projection logic:**
- If `|lon_max - lon_min| > 180` or `|lat_max - lat_min| > 90` → global, use `PlateCarree` (or `Mollweide` for full globe).
- Otherwise local; if mid-latitude > 60° use `LambertConformalConic`; else `Mercator`.

**Mapping:**
- Use `geopandas` + `matplotlib` with `cartopy` for projection handling.
- Overlay coastlines and country borders from Natural Earth via `geopandas.datasets`.
- Color scales: use `cmocean.cm.balance` or `matplotlib.cm.RdBu` for declination; `viridis` for intensity.
- Save to PNG/PDF via `fig.savefig()`.

---

## 5. Sphinx Documentation Plan

### 5.1 Structure
```
docs/
├── conf.py
├── index.rst
├── installation.rst
├── quickstart.rst
├── api/
│   ├── io.rst
│   ├── model.rst
│   ├── core.rst
│   ├── data.rst
│   └── utils.rst
├── tutorials/
│   ├── load_wmm.rst
│   ├── load_igrf.rst
│   ├── compute_global_grid.rst
│   └── time_series.rst
└── references.rst
```

### 5.2 `conf.py` settings
```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google/NumPy style docstrings
    "sphinx.ext.mathjax",       # LaTeX math in docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",                  # Jupyter notebooks as tutorials
]
html_theme = "pydata_sphinx_theme"
```

### 5.3 Docstring conventions
- Use **NumPy style** throughout.
- Every public function: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`.
- Math sections use `.. math::` directives with LaTeX.
- Example blocks use `doctest` format so they are runnable.

### 5.4 LaTeX equations in docs
For example, the potential expansion:

```rst
.. math::

   V(r, \theta, \phi) = a \sum_{n=1}^{N} \left(\frac{a}{r}\right)^{n+1}
   \sum_{m=0}^{n} \left[g_n^m \cos(m\phi) + h_n^m \sin(m\phi)\right]
   P_n^m(\cos\theta)
```

### 5.5 Build instructions
```bash
cd docs
sphinx-apidoc -o api/ ../shgeomag --separate --module-first
make html
```

Include CI step in `pyproject.toml` to build docs on every commit.

---

## 6. Test Plan

### 6.1 `tests/test_io.py`
- Load each of the 5 format variants from small synthetic files.
- Check `model.n_max`, `model.epochs`, `model.valid_from/to`.
- Check that comment lines and non-`#` headers are skipped correctly.
- Check that the WMM 999-terminator row is not parsed as a data row.
- Test `write()` round-trip: read → write → read → compare coefficients.

### 6.2 `tests/test_model.py`
- Test `truncate(n)`: verify coefficients with n > truncation value are absent.
- Test `info()` output.
- Test `get_coefficients(year)` on single-epoch and multi-epoch models.
- Test warning emission when `year` is outside `valid_from/to`.

### 6.3 `tests/test_design_matrix.py`
- For n=1, m=0, verify G analytically: g₁₀ multiplied by specific G entries should reproduce the zonal dipole field.
- Check that G shape = `(3*N_points, N_coeffs)` for N_coeffs = n_max*(n_max+2).
- Verify reciprocity: field at a specific point computed via G·c matches direct sum.

### 6.4 `tests/test_field.py`
- Verify `compute_sph` at geographic north pole: Bφ = 0, Bθ = 0 for axisymmetric (m=0 only) model.
- Verify `compute_geo` rotates correctly: at equator, Bz(Down) should equal radial component with appropriate sign.
- Verify `compute_fdi`: F = sqrt(X²+Y²+Z²), check at multiple known points.

### 6.5 `tests/test_coords.py`
- Round-trip: geodetic → geocentric → geodetic; check |error| < 1e-9 degrees.
- Verify `gps_height_to_radius` at equator matches analytic formula `r = a_wgs + h_gps` (approximately).
- Test at poles and equator as edge cases.

### 6.6 `tests/test_time.py`
- `unix_to_mjd2000(0)` should equal `(2440587.5 - 2451544.5)` = −10957.0 days.
- `datetime_to_mjd2000(datetime(2000,1,1,12))` = 0.5.
- `mjd2000_to_decimal_year` at Jan 1 of any year = exact integer.
- Round-trip: MJD2000 → datetime → MJD2000 within 1e-9.

### 6.7 `tests/test_wmm_published.py` — WMM2025 regression tests

Use `WMM2025_TestValues.txt` (bundled in `tests/data/`).

For each row in the file:
- Parse: `decimal_year, h_above_ellipsoid_km, lat_gd_deg, lon_deg, D_deg, I_deg, H_nT, X_nT, Y_nT, Z_nT, F_nT, ...`
- Convert: `geodetic_to_geocentric(lat_gd_deg, lon_deg, h_above_ellipsoid_km)` → gc_lat, lon, r
- Call `compute_fdi(model, gc_lat, lon, r, decimal_year)` → computed F, D, I
- Also compute X, Y, Z via `compute_geo`
- Tolerance: `|F_computed - F_published| < 0.01 nT`, `|D_computed - D_published| < 0.001°`

### 6.8 `tests/test_igrf_published.py` — IGRF-14 regression tests

Use published IGRF test values from IAGA website or embedded reference table. Test at:
- Geographic north pole, year 2020.0, r = 6371.2 km
- Equatorial points at several longitudes
- Mid-latitude points in each quadrant
- Tolerance: `< 1 nT` on F, `< 0.01°` on D, consistent with published IGRF validator output

---

## 7. Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",          # for special functions (validation only)
    "pandas>=2.0",
    "xarray",
    "geopandas",
    "matplotlib",
    "cartopy",
    "pyproj",               # geodetic transformations
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "sphinx", "pydata-sphinx-theme",
       "sphinx-autodoc-typehints", "nbsphinx", "numba"]
```

