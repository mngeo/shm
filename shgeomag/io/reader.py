"""Content-driven model file parser for IGRF, WMM, and simple custom formats."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np

from shgeomag.model.coefficients import GaussCoefficients
from shgeomag.model.model import GeomagModel


def _is_int_token(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _build_coeff(epoch: float, n_max: int, g_mat: np.ndarray, h_mat: np.ndarray, dg: np.ndarray | None = None, dh: np.ndarray | None = None) -> GaussCoefficients:
    n_vals: list[int] = []
    m_vals: list[int] = []
    g_vals: list[float] = []
    h_vals: list[float] = []
    dg_vals: list[float] = []
    dh_vals: list[float] = []

    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            n_vals.append(n)
            m_vals.append(m)
            g_vals.append(g_mat[n, m])
            h_vals.append(h_mat[n, m])
            if dg is not None and dh is not None:
                dg_vals.append(dg[n, m])
                dh_vals.append(dh[n, m])

    return GaussCoefficients(
        epoch=epoch,
        n_max=n_max,
        n_array=np.asarray(n_vals, dtype=int),
        m_array=np.asarray(m_vals, dtype=int),
        g=np.asarray(g_vals, dtype=float),
        h=np.asarray(h_vals, dtype=float),
        dg_dt=np.asarray(dg_vals, dtype=float) if dg_vals else None,
        dh_dt=np.asarray(dh_vals, dtype=float) if dh_vals else None,
    )


def _parse_igrf(path: Path, lines: list[str]) -> GeomagModel:
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("g/h"))
    header_tokens = lines[header_idx].split()
    epoch_tokens = header_tokens[3:-1]
    epochs = [float(tok) for tok in epoch_tokens]

    rows = []
    for line in lines[header_idx + 1 :]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if toks[0] not in {"g", "h"}:
            continue
        rows.append(toks)

    n_max = max(int(r[1]) for r in rows)
    coeffs_by_epoch: dict[float, GaussCoefficients] = {}

    g_epochs = {ep: np.zeros((n_max + 1, n_max + 1), dtype=float) for ep in epochs}
    h_epochs = {ep: np.zeros((n_max + 1, n_max + 1), dtype=float) for ep in epochs}
    dg = np.zeros((n_max + 1, n_max + 1), dtype=float)
    dh = np.zeros((n_max + 1, n_max + 1), dtype=float)

    for toks in rows:
        kind = toks[0]
        n = int(toks[1])
        m = int(toks[2])
        vals = [float(v) for v in toks[3:]]
        if len(vals) != len(epochs) + 1:
            raise ValueError(f"Malformed IGRF row for n={n}, m={m}: expected {len(epochs)+1} values")
        epoch_vals = vals[:-1]
        sv = vals[-1]

        for ep, v in zip(epochs, epoch_vals, strict=True):
            if kind == "g":
                g_epochs[ep][n, m] = v
            else:
                h_epochs[ep][n, m] = v

        if kind == "g":
            dg[n, m] = sv
        else:
            dh[n, m] = sv

    for ep in epochs:
        coeffs_by_epoch[ep] = _build_coeff(ep, n_max, g_epochs[ep], h_epochs[ep])

    sv_coeff = _build_coeff(epochs[-1], n_max, g_epochs[epochs[-1]], h_epochs[epochs[-1]], dg=dg, dh=dh)
    return GeomagModel(
        source_file=str(path),
        model_name="IGRF",
        format="IGRF",
        valid_from=epochs[0],
        valid_to=epochs[-1] + 5.0,
        epochs=epochs,
        coeffs=coeffs_by_epoch,
        sv_coeffs=sv_coeff,
        n_max=n_max,
    )


def _parse_wmm(path: Path, lines: list[str]) -> GeomagModel:
    header = lines[0].split()
    epoch = float(header[0])
    model_name = header[1]

    rows = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("999"):
            break
        toks = s.split()
        if len(toks) < 6:
            continue
        rows.append(toks[:6])

    n_max = max(int(r[0]) for r in rows)
    g_mat = np.zeros((n_max + 1, n_max + 1), dtype=float)
    h_mat = np.zeros((n_max + 1, n_max + 1), dtype=float)
    dg = np.zeros((n_max + 1, n_max + 1), dtype=float)
    dh = np.zeros((n_max + 1, n_max + 1), dtype=float)

    for toks in rows:
        n, m = int(toks[0]), int(toks[1])
        g_mat[n, m] = float(toks[2])
        h_mat[n, m] = float(toks[3])
        dg[n, m] = float(toks[4])
        dh[n, m] = float(toks[5])

    base = _build_coeff(epoch, n_max, g_mat, h_mat)
    sv = _build_coeff(epoch, n_max, g_mat, h_mat, dg=dg, dh=dh)

    return GeomagModel(
        source_file=str(path),
        model_name=model_name,
        format="WMM",
        valid_from=epoch,
        valid_to=epoch + 5.0,
        epochs=[epoch],
        coeffs={epoch: base},
        sv_coeffs=sv,
        n_max=n_max,
    )


def _parse_custom(path: Path, lines: list[str], year: float | None) -> GeomagModel:
    data_rows: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if _is_int_token(toks[0]) or toks[0] in {"g", "h"}:
            data_rows.append(toks)

    if not data_rows:
        raise ValueError("No data rows found in custom model file")

    first = data_rows[0]
    fmt = ""
    if first[0] in {"g", "h"} and len(first) >= 4:
        fmt = "gh"
    elif len(first) == 4:
        fmt = "ngmh"
    elif len(first) >= 6:
        fmt = "ngmhdgdh"
    else:
        raise ValueError("Unable to detect custom format")

    if year is None:
        year = float(datetime.now(timezone.utc).year)

    parsed: list[tuple[int, int, float, float, float, float]] = []
    if fmt == "gh":
        tmp: dict[tuple[int, int], dict[str, float]] = {}
        for toks in data_rows:
            kind = toks[0]
            n, m = int(toks[1]), int(toks[2])
            val = float(toks[3])
            key = (n, m)
            tmp.setdefault(key, {})[kind] = val
        for (n, m), vals in tmp.items():
            parsed.append((n, m, vals.get("g", 0.0), vals.get("h", 0.0), 0.0, 0.0))
    elif fmt == "ngmh":
        for toks in data_rows:
            n, m = int(toks[0]), int(toks[1])
            parsed.append((n, m, float(toks[2]), float(toks[3]), 0.0, 0.0))
    else:
        for toks in data_rows:
            n, m = int(toks[0]), int(toks[1])
            parsed.append((n, m, float(toks[2]), float(toks[3]), float(toks[4]), float(toks[5])))

    n_max = max(n for n, _, _, _, _, _ in parsed)
    g_mat = np.zeros((n_max + 1, n_max + 1), dtype=float)
    h_mat = np.zeros((n_max + 1, n_max + 1), dtype=float)
    dg = np.zeros((n_max + 1, n_max + 1), dtype=float)
    dh = np.zeros((n_max + 1, n_max + 1), dtype=float)

    for n, m, g, h, dgi, dhi in parsed:
        g_mat[n, m] = g
        h_mat[n, m] = h
        dg[n, m] = dgi
        dh[n, m] = dhi

    if fmt != "ngmhdgdh":
        warnings.warn("Custom format missing dg/dt columns; SV arrays set to zero", RuntimeWarning, stacklevel=2)

    base = _build_coeff(year, n_max, g_mat, h_mat)
    sv = _build_coeff(year, n_max, g_mat, h_mat, dg=dg, dh=dh)
    return GeomagModel(
        source_file=str(path),
        model_name=path.stem,
        format=fmt,
        valid_from=year - 5.0,
        valid_to=year + 5.0,
        epochs=[year],
        coeffs={year: base},
        sv_coeffs=sv,
        n_max=n_max,
    )


def load_model(path: str, year: float | None = None) -> GeomagModel:
    """Load a geomagnetic model from file content.

    Parameters
    ----------
    path : str
        Input file path.
    year : float, optional
        Fallback year for custom formats that have no epoch metadata.

    Returns
    -------
    GeomagModel
        Parsed geomagnetic model object.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()

    non_comment = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not non_comment:
        raise ValueError("Model file has no parseable content")

    first_tokens = non_comment[0].split()
    if any(line.split()[0] == "g/h" for line in non_comment if line.split()):
        return _parse_igrf(p, lines)

    if len(first_tokens) >= 2:
        try:
            float(first_tokens[0])
            if "WMM" in first_tokens[1]:
                return _parse_wmm(p, non_comment)
        except ValueError:
            pass

    return _parse_custom(p, lines, year)


__all__ = ["load_model"]
