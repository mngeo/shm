"""Physical and geodetic constants used by shgeomag."""

from __future__ import annotations

WGS84_A_KM: float = 6378.137
WGS84_F: float = 1.0 / 298.257223563
WGS84_B_KM: float = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2: float = 1.0 - (WGS84_B_KM / WGS84_A_KM) ** 2
A_REF_KM: float = 6371.2

__all__ = [
    "WGS84_A_KM",
    "WGS84_F",
    "WGS84_B_KM",
    "WGS84_E2",
    "A_REF_KM",
]
