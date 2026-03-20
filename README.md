# shgeomag (`shm`)
Vectorized spherical-harmonic geomagnetic modeling for IGRF, WMM, and custom Gauss coefficient files.

## Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Run Full Pipeline
```bash
. .venv/bin/activate
ruff check .
mypy shgeomag/
pytest --cov=shgeomag tests/
cd docs && make html
```

## Usage
```python
import numpy as np
from shgeomag.io.reader import load_model
from shgeomag.core.field import compute_fdi
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
```

## Model Files
- WMM example: `models/WMM2025.COF`
- IGRF example: `models/igrf14coeffs.txt`

## Requirements
- Pinned runtime+dev dependencies are in `requirements.txt`.
