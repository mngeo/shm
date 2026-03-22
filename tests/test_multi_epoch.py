import numpy as np

from shgeomag.core.field import compute_fdi, compute_geo, compute_multi_epoch
from shgeomag.data.container import InputData


def test_compute_multi_epoch(wmm_model):
    arr = np.array(
        [
            [9500.0, 10.0, 20.0, 6371.2],
            [9600.0, 11.0, 21.0, 6371.3],
        ]
    )
    data = InputData.from_array(arr)
    out = compute_multi_epoch(wmm_model, data)
    assert set(out) == {"X", "Y", "Z", "F", "D", "I"}
    assert out["F"].shape == (2,)


def test_compute_multi_epoch_matches_direct_calls(wmm_model):
    # Multiple rows share identical geometries to hit the caching path.
    arr = np.array(
        [
            [9500.0, 10.0, 20.0, 6371.2],
            [9500.0, 10.0, 20.0, 6371.2],
            [9600.0, -15.0, 150.0, 6400.0],
            [9600.0, -15.0, 150.0, 6400.0],
        ]
    )
    data = InputData.from_array(arr)
    out = compute_multi_epoch(wmm_model, data)

    # Recreate per-row reference results using existing public functions.
    from shgeomag.utils.time_utils import mjd2000_to_decimal_year

    years = mjd2000_to_decimal_year(arr[:, 0])
    x_ref = np.empty(arr.shape[0], dtype=float)
    y_ref = np.empty(arr.shape[0], dtype=float)
    z_ref = np.empty(arr.shape[0], dtype=float)
    f_ref = np.empty(arr.shape[0], dtype=float)
    d_ref = np.empty(arr.shape[0], dtype=float)
    i_ref = np.empty(arr.shape[0], dtype=float)

    for i, yr in enumerate(years):
        x, y, z = compute_geo(wmm_model, arr[i : i + 1, 1], arr[i : i + 1, 2], arr[i : i + 1, 3], float(yr))
        f, d, inc = compute_fdi(wmm_model, arr[i : i + 1, 1], arr[i : i + 1, 2], arr[i : i + 1, 3], float(yr))
        x_ref[i], y_ref[i], z_ref[i] = x[0], y[0], z[0]
        f_ref[i], d_ref[i], i_ref[i] = f[0], d[0], inc[0]

    assert np.allclose(out["X"], x_ref, atol=1e-10, rtol=1e-12)
    assert np.allclose(out["Y"], y_ref, atol=1e-10, rtol=1e-12)
    assert np.allclose(out["Z"], z_ref, atol=1e-10, rtol=1e-12)
    assert np.allclose(out["F"], f_ref, atol=1e-10, rtol=1e-12)
    assert np.allclose(out["D"], d_ref, atol=1e-10, rtol=1e-12)
    assert np.allclose(out["I"], i_ref, atol=1e-10, rtol=1e-12)
