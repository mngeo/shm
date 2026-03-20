"""Container for vectorized field input data in MJD2000/geocentric format."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class InputData:
    """Container for field computation input rows.

    Parameters
    ----------
    data : ndarray
        Array with shape ``(N, 4)`` and columns
        ``[mjd2000, geocentric_lat_deg, lon_deg, r_km]``.
    """

    data: np.ndarray

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "InputData":
        """Construct InputData from a raw array."""
        obj = cls(np.asarray(arr, dtype=float))
        obj.validate()
        return obj

    @classmethod
    def from_dataframe(cls, df, col_map: dict[str, str]) -> "InputData":
        """Construct InputData from a dataframe-like object.

        Parameters
        ----------
        df : DataFrame
            Dataframe with source columns.
        col_map : dict
            Mapping with keys ``mjd2000``, ``gc_lat``, ``lon``, ``r``.
        """
        arr = np.column_stack(
            [
                np.asarray(df[col_map["mjd2000"]], dtype=float),
                np.asarray(df[col_map["gc_lat"]], dtype=float),
                np.asarray(df[col_map["lon"]], dtype=float),
                np.asarray(df[col_map["r"]], dtype=float),
            ]
        )
        return cls.from_array(arr)

    def validate(self) -> None:
        """Validate shape and basic range assumptions for data values."""
        if self.data.ndim != 2 or self.data.shape[1] != 4:
            raise ValueError("InputData.data must have shape (N, 4)")

        if np.any(self.data[:, 1] < -90.0) or np.any(self.data[:, 1] > 90.0):
            raise ValueError("Geocentric latitude must be in [-90, 90] degrees")

        if np.any(self.data[:, 3] <= 0.0):
            raise ValueError("Geocentric radius must be positive")

    def sort_by_time(self) -> "InputData":
        """Return a new InputData sorted by MJD2000 time."""
        idx = np.argsort(self.data[:, 0])
        return InputData(self.data[idx].copy())


__all__ = ["InputData"]
