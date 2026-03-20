import numpy as np


def test_model_info(wmm_model):
    txt = wmm_model.info()
    assert "WMM" in txt
    assert "n_max=12" in txt


def test_truncation(wmm_model):
    wmm_model.set_truncation(8)
    c = wmm_model.get_coefficients(2026.0)
    assert c.n_max == 8
    assert np.max(c.n_array) == 8
    wmm_model.set_truncation(12)
