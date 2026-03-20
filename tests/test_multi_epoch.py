import numpy as np

from shgeomag.core.field import compute_multi_epoch
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
