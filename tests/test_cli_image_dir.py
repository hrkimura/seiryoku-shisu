from pathlib import Path

from seiryoku.cli import monthly_image_dir


def test_monthly_image_dir_uses_latest_data_month(tmp_path):
    history = [
        {"year": 2026, "month": 9},
        {"year": 2025, "month": 12},
        {"year": 2026, "month": 1},
    ]
    assert monthly_image_dir(history, base=tmp_path) == tmp_path / "images" / "2026-09"


def test_monthly_image_dir_is_outside_notes():
    d = monthly_image_dir([{"year": 2026, "month": 9}])
    assert "notes" not in Path(d).parts
