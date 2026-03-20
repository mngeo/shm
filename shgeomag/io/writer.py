"""Model writer utilities for custom plain text and WMM-style outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from shgeomag.model.model import GeomagModel


def _iter_rows(model: GeomagModel, year: float):
    coeffs = model.get_coefficients(year)
    gh = coeffs.to_gh_dict()
    sv_lookup: dict[tuple[int, int], tuple[float, float]] = {}
    if model.sv_coeffs is not None and model.sv_coeffs.dg_dt is not None and model.sv_coeffs.dh_dt is not None:
        sv_lookup = {
            (int(nn), int(mm)): (float(dgn), float(dhn))
            for nn, mm, dgn, dhn in zip(
                model.sv_coeffs.n_array,
                model.sv_coeffs.m_array,
                model.sv_coeffs.dg_dt,
                model.sv_coeffs.dh_dt,
                strict=True,
            )
        }
    for n in range(1, coeffs.n_max + 1):
        for m in range(0, n + 1):
            g, h = gh[(n, m)]
            dg = 0.0
            dh = 0.0
            if sv_lookup:
                dg, dh = sv_lookup[(n, m)]
            yield n, m, g, h, dg, dh


def write_model(model: GeomagModel, path: str, fmt: str = "ngmhdgdh", year: float | None = None) -> None:
    """Write a model to a text file.

    Parameters
    ----------
    model : GeomagModel
        Model to export.
    path : str
        Output path.
    fmt : str, default="ngmhdgdh"
        Output format: ``ngmh``, ``ngmhdgdh``, or ``wmm``.
    year : float, optional
        Coefficient epoch to export; defaults to model's first epoch.
    """
    y = float(model.epochs[0] if year is None else year)
    out = []
    out.append(f"# model={model.model_name} epoch={y} written={datetime.now(timezone.utc).isoformat()}")

    if fmt.lower() == "wmm":
        out.append(f"{y:10.1f} {model.model_name:>16s} {datetime.now(timezone.utc).strftime('%m/%d/%Y')}")
        for n, m, g, h, dg, dh in _iter_rows(model, y):
            out.append(f"{n:3d} {m:2d} {g:11.1f} {h:11.1f} {dg:10.1f} {dh:10.1f}")
        out.append("999999999999999999999999999999999999999999999999")
    elif fmt.lower() == "ngmh":
        for n, m, g, h, _, _ in _iter_rows(model, y):
            out.append(f"{n:d} {m:d} {g:.6f} {h:.6f}")
    else:
        for n, m, g, h, dg, dh in _iter_rows(model, y):
            out.append(f"{n:d} {m:d} {g:.6f} {h:.6f} {dg:.6f} {dh:.6f}")

    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


__all__ = ["write_model"]
