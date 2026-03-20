from datetime import datetime, timezone

import numpy as np

from shgeomag.utils.time_utils import (
    datetime_to_mjd2000,
    decimal_year_to_mjd2000,
    mjd2000_to_datetime,
    mjd2000_to_decimal_year,
    unix_to_mjd2000,
)


def test_unix_to_mjd2000_zero():
    assert np.isclose(unix_to_mjd2000(0.0), -10957.0)


def test_datetime_roundtrip():
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    mjd = datetime_to_mjd2000(dt)
    back = mjd2000_to_datetime(mjd)[0]
    assert back == dt


def test_decimal_year_roundtrip():
    dy = np.array([2024.0, 2024.5])
    mjd = decimal_year_to_mjd2000(dy)
    dy2 = mjd2000_to_decimal_year(mjd)
    assert np.allclose(dy2, dy, atol=1e-7)
