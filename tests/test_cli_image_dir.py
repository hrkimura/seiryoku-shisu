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


def test_done_marker_is_per_calendar_month(tmp_path):
    from datetime import date

    from seiryoku.cli import done_marker

    assert done_marker(date(2026, 11, 7), tmp_path) == done_marker(date(2026, 11, 30), tmp_path)
    assert done_marker(date(2026, 11, 7), tmp_path) != done_marker(date(2026, 12, 7), tmp_path)


def test_ready_to_run_waits_for_this_months_crawl():
    from datetime import date

    from seiryoku.cli import ready_to_run

    assert not ready_to_run(date(2026, 11, 2), date(2026, 10, 7))
    assert not ready_to_run(date(2026, 11, 3), None)
    assert ready_to_run(date(2026, 11, 4), date(2026, 11, 4))
    assert not ready_to_run(date(2026, 11, 20), date(2026, 10, 7))


class _Ran(Exception):
    pass


def _patch_main(tmp_path, monkeypatch, refresh_done_at):
    from seiryoku import cli, refresh_cli

    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cli, "done_marker", lambda today, base=tmp_path: base / "done")
    monkeypatch.setattr(refresh_cli, "done_at", lambda: refresh_done_at)
    monkeypatch.setattr(cli, "run", lambda: (_ for _ in ()).throw(_Ran()))
    return cli


def test_main_skips_when_done_or_waiting_and_force_overrides(tmp_path, monkeypatch):
    import pytest

    cli = _patch_main(tmp_path, monkeypatch, None)
    cli.main([])  # クロール待ち
    (tmp_path / "done").write_text("x")
    cli.main([])  # 済んでいる
    with pytest.raises(_Ran):
        cli.main(["--force"])
