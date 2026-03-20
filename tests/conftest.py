from pathlib import Path

import pytest

from shgeomag.io.reader import load_model


@pytest.fixture(scope="module")
def wmm_model():
    return load_model(str(Path(__file__).parent / "data" / "WMM2025.COF"))


@pytest.fixture(scope="module")
def igrf_model():
    return load_model(str(Path(__file__).parent / "data" / "igrf14coeffs.txt"))
