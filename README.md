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
from shgeomag.core.field import compute_fdi, compute_sph_cached
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
```

## Performance Notes
- For one-off forward evaluation, prefer `compute_sph(...)`.
- For repeated geometry workloads (many rows sharing the same
  `gc_lat/lon/r`), use `compute_sph_cached(...)` for faster execution.
- `compute_multi_epoch(...)` already uses the cached spherical path
  internally per time slab.

## Model Files
- WMM example: `models/WMM2025.COF`
- IGRF example: `models/igrf14coeffs.txt`

## Requirements
- Pinned runtime+dev dependencies are in `requirements.txt`.
