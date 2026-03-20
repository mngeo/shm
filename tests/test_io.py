from pathlib import Path

from shgeomag.io.reader import load_model


def test_load_wmm():
    model = load_model(str(Path(__file__).parent / "data" / "WMM2025.COF"))
    assert model.format == "WMM"
    assert model.n_max == 12
    assert model.valid_from == 2025.0
    assert model.valid_to == 2030.0


def test_load_igrf():
    model = load_model(str(Path(__file__).parent / "data" / "igrf14coeffs.txt"))
    assert model.format == "IGRF"
    assert model.n_max == 13
    assert min(model.epochs) == 1900.0
    assert max(model.epochs) == 2025.0


def test_detect_custom_format(tmp_path):
    p = tmp_path / "custom.txt"
    p.write_text("# comment\n1 0 -30000 0\n1 1 -1500 4500\n", encoding="utf-8")
    model = load_model(str(p), year=2024.0)
    assert model.format == "ngmh"
    assert model.epochs == [2024.0]
