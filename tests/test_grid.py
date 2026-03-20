from shgeomag.utils.grid import global_grid


def test_global_grid_fdi(wmm_model):
    out = global_grid(wmm_model, 2026.0, -10, 10, -10, 10, 10, 10, 0.0, output="fdi")
    assert "F" in out and "D" in out and "I" in out
    assert out["F"].shape == out["lat"].shape
