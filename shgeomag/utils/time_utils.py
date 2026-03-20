"""Time conversion utilities for MJD2000 and decimal year values."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np

_EPOCH_2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)


DatetimeLike = Union[datetime, str]


def unix_to_mjd2000(unix_seconds: float | np.ndarray) -> np.ndarray:
    """Convert Unix timestamp seconds to MJD2000 days.

    Parameters
    ----------
    unix_seconds : float or ndarray
        Unix timestamp in seconds since 1970-01-01T00:00:00 UTC.

    Returns
    -------
    ndarray
        Time in MJD2000 days.
    """
    ts = np.asarray(unix_seconds, dtype=float)
    return ts / 86400.0 - 10957.0


def datetime_to_mjd2000(value: DatetimeLike | np.ndarray) -> np.ndarray:
    """Convert datetime or ISO strings to MJD2000.

    Parameters
    ----------
    value : datetime, str, or ndarray
        UTC-aware or naive datetimes (naive assumed UTC), or ISO strings.

    Returns
    -------
    ndarray
        MJD2000 time in days.
    """
    if isinstance(value, (datetime, str)):
        arr = np.asarray([value], dtype=object)
    else:
        arr = np.asarray(value, dtype=object)
    out = np.empty(arr.shape, dtype=float)
    for idx, item in np.ndenumerate(arr):
        dt = datetime.fromisoformat(item) if isinstance(item, str) else item
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        out[idx] = (dt - _EPOCH_2000).total_seconds() / 86400.0
    return out


def mjd2000_to_datetime(mjd2000: float | np.ndarray) -> np.ndarray:
    """Convert MJD2000 days to UTC datetimes.

    Parameters
    ----------
    mjd2000 : float or ndarray
        MJD2000 values in days.

    Returns
    -------
    ndarray
        UTC datetime objects.
    """
    arr = np.asarray(mjd2000, dtype=float)
    out = np.empty(arr.shape, dtype=object)
    for idx, item in np.ndenumerate(arr):
        out[idx] = _EPOCH_2000 + timedelta(days=float(item))
    return out


def mjd2000_to_decimal_year(mjd2000: float | np.ndarray) -> np.ndarray:
    """Convert MJD2000 days to decimal year.

    Parameters
    ----------
    mjd2000 : float or ndarray
        MJD2000 values in days.

    Returns
    -------
    ndarray
        Decimal year values.
    """
    dts = mjd2000_to_datetime(mjd2000)
    out = np.empty(dts.shape, dtype=float)
    for idx, dt in np.ndenumerate(dts):
        start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
        frac = (dt - start).total_seconds() / (end - start).total_seconds()
        out[idx] = dt.year + frac
    return out


def decimal_year_to_mjd2000(decimal_year: float | np.ndarray) -> np.ndarray:
    """Convert decimal year to MJD2000 days.

    Parameters
    ----------
    decimal_year : float or ndarray
        Decimal year values.

    Returns
    -------
    ndarray
        MJD2000 days.
    """
    arr = np.asarray(decimal_year, dtype=float)
    out = np.empty(arr.shape, dtype=float)
    for idx, dy in np.ndenumerate(arr):
        year = int(np.floor(dy))
        frac = float(dy - year)
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        seconds = frac * (end - start).total_seconds()
        dt = start + timedelta(seconds=seconds)
        out[idx] = (dt - _EPOCH_2000).total_seconds() / 86400.0
    return out


__all__ = [
    "unix_to_mjd2000",
    "datetime_to_mjd2000",
    "mjd2000_to_datetime",
    "mjd2000_to_decimal_year",
    "decimal_year_to_mjd2000",
]
