import numpy as np

from shgeomag.core.design_matrix import build_design_matrix, build_ned_jacobian
from shgeomag.core.field import compute_geo, compute_sph
from shgeomag.io.reader import load_model
from shgeomag.model.coefficients import GaussCoefficients
from shgeomag.model.model import GeomagModel


def test_design_matrix_shape():
    lat = np.deg2rad(np.array([10.0, 20.0]))
    lon = np.deg2rad(np.array([30.0, 60.0]))
    r = np.array([6371.2, 6371.2])
    g = build_design_matrix(lat, lon, r, n_max=3)
    assert g.shape == (6, 15)


def test_design_matrix_matches_compute_sph_regression():
    rng = np.random.default_rng(42)

    cases = [
        ("models/WMM2025.COF", 2026.0),
        ("models/igrf14coeffs.txt", 2020.0),
    ]
    for model_path, year in cases:
        model = load_model(model_path)
        gc_lat_deg = rng.uniform(-89.0, 89.0, 128)
        lon_deg = rng.uniform(0.0, 360.0, 128)
        r_km = rng.uniform(6200.0, 7000.0, 128)

        br_ref, bt_ref, bp_ref = compute_sph(model, gc_lat_deg, lon_deg, r_km, year)

        coeffs = model.get_coefficients(year)
        g_mat = build_design_matrix(
            np.deg2rad(gc_lat_deg),
            np.deg2rad(lon_deg),
            r_km,
            n_max=coeffs.n_max,
            n_truncate=coeffs.n_max,
        )
        b_vec = g_mat @ coeffs.coefficient_vector()
        br_dm = b_vec[0::3]
        bt_dm = b_vec[1::3]
        bp_dm = b_vec[2::3]

        assert np.allclose(br_ref, br_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(bt_ref, bt_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(bp_ref, bp_dm, rtol=1e-11, atol=1e-8)


def test_ned_jacobian_shape():
    lat = np.deg2rad(np.array([10.0, 20.0]))
    lon = np.deg2rad(np.array([30.0, 60.0]))
    r = np.array([6371.2, 6371.2])
    j = build_ned_jacobian(lat, lon, r, n_max=3)
    assert j.shape == (6, 15)


def test_ned_jacobian_matches_compute_geo_regression():
    rng = np.random.default_rng(123)

    cases = [
        ("models/WMM2025.COF", 2026.0),
        ("models/igrf14coeffs.txt", 2020.0),
    ]
    for model_path, year in cases:
        model = load_model(model_path)
        gc_lat_deg = rng.uniform(-89.0, 89.0, 128)
        lon_deg = rng.uniform(0.0, 360.0, 128)
        r_km = rng.uniform(6200.0, 7000.0, 128)

        x_ref, y_ref, z_ref = compute_geo(model, gc_lat_deg, lon_deg, r_km, year)
        coeffs = model.get_coefficients(year)

        j = build_ned_jacobian(
            np.deg2rad(gc_lat_deg),
            np.deg2rad(lon_deg),
            r_km,
            n_max=coeffs.n_max,
            n_truncate=coeffs.n_max,
        )
        b_vec = j @ coeffs.coefficient_vector()
        x_dm = b_vec[0::3]
        y_dm = b_vec[1::3]
        z_dm = b_vec[2::3]

        assert np.allclose(x_ref, x_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(y_ref, y_dm, rtol=1e-11, atol=1e-8)
        assert np.allclose(z_ref, z_dm, rtol=1e-11, atol=1e-8)


def test_ned_jacobian_cache_matches_no_cache():
    rng = np.random.default_rng(7)
    gc_lat_deg = rng.uniform(-70.0, 70.0, 32)
    lon_deg = rng.uniform(0.0, 360.0, 32)
    r_km = np.full(32, 6371.2)
    # Duplicate geometry to exercise unique-geometry expansion.
    gc_lat_deg = np.repeat(gc_lat_deg[:8], 4)
    lon_deg = np.repeat(lon_deg[:8], 4)
    r_km = np.repeat(r_km[:8], 4)

    lat = np.deg2rad(gc_lat_deg)
    lon = np.deg2rad(lon_deg)
    j_cache = build_ned_jacobian(lat, lon, r_km, n_max=8, use_cache=True)
    j_no_cache = build_ned_jacobian(lat, lon, r_km, n_max=8, use_cache=False)
    assert np.allclose(j_cache, j_no_cache, rtol=0.0, atol=0.0)


def _column_to_nm_kind(n_max: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            out.append((n, m, "g"))
            if m > 0:
                out.append((n, m, "h"))
    return out


def test_ned_jacobian_matches_finite_difference_wrt_gh_coeffs():
    model = load_model("models/WMM2025.COF")
    year = 2026.0
    coeff = model.get_coefficients(year)
    mapping = _column_to_nm_kind(coeff.n_max)

    gc_lat_deg = np.array([12.0, -35.0, 57.5])
    lon_deg = np.array([25.0, 190.0, 301.0])
    r_km = np.array([6371.2, 6400.0, 6700.0])

    j = build_ned_jacobian(np.deg2rad(gc_lat_deg), np.deg2rad(lon_deg), r_km, n_max=coeff.n_max)
    eps = 1e-4
    cols_to_check = [0, 1, 2, 10, 25, 40]

    def _with_perturb(col: int, sign: float) -> GeomagModel:
        n, m, kind = mapping[col]
        idx = np.where((coeff.n_array == n) & (coeff.m_array == m))[0]
        assert idx.size == 1
        k = int(idx[0])
        g = coeff.g.copy()
        h = coeff.h.copy()
        if kind == "g":
            g[k] += sign * eps
        else:
            h[k] += sign * eps

        c = GaussCoefficients(
            epoch=year,
            n_max=coeff.n_max,
            n_array=coeff.n_array.copy(),
            m_array=coeff.m_array.copy(),
            g=g,
            h=h,
        )
        return GeomagModel(
            source_file=model.source_file,
            model_name=model.model_name,
            format=model.format,
            valid_from=year - 1.0,
            valid_to=year + 1.0,
            epochs=[year],
            coeffs={year: c},
            sv_coeffs=None,
            n_max=coeff.n_max,
            n_truncate=coeff.n_max,
        )

    for col in cols_to_check:
        model_p = _with_perturb(col, +1.0)
        model_m = _with_perturb(col, -1.0)
        xp, yp, zp = compute_geo(model_p, gc_lat_deg, lon_deg, r_km, year)
        xm, ym, zm = compute_geo(model_m, gc_lat_deg, lon_deg, r_km, year)
        d_num = np.empty(3 * gc_lat_deg.size, dtype=float)
        d_num[0::3] = (xp - xm) / (2.0 * eps)
        d_num[1::3] = (yp - ym) / (2.0 * eps)
        d_num[2::3] = (zp - zm) / (2.0 * eps)
        assert np.allclose(j[:, col], d_num, rtol=2e-5, atol=1e-4)
