import numpy as np

from shgeomag.data.container import InputData


def test_input_data_validate_and_sort():
    arr = np.array(
        [
            [10.0, 0.0, 0.0, 6371.2],
            [5.0, 1.0, 10.0, 6372.0],
        ]
    )
    data = InputData.from_array(arr)
    out = data.sort_by_time()
    assert out.data[0, 0] == 5.0
