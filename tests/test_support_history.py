from seiryoku import support_history


def test_record_snapshot_appends_once_per_update_date(tmp_path, monkeypatch):
    monkeypatch.setattr(support_history, "HISTORY_PATH", tmp_path / "s.json")
    snap = {"updated": "2026-10-06", "S_p": {"自由民主党": 33.2}, "source": "u"}
    assert support_history.record_snapshot(snap, fetched="2026-10-08") is True
    assert support_history.record_snapshot(snap, fetched="2026-10-09") is False
    assert support_history.load_history() == [
        {"updated": "2026-10-06", "fetched": "2026-10-08", "S_p": {"自由民主党": 33.2}, "source": "u"},
    ]


def test_monthly_series_takes_last_snapshot_in_each_month():
    history = [
        {"updated": "2026-10-20", "S_p": {"a": 2.0}},
        {"updated": "2026-10-06", "S_p": {"a": 1.0}},
        {"updated": "2026-12-01", "S_p": {"a": 3.0}},
    ]
    assert support_history.monthly_series(history) == {
        (2026, 10): {"a": 2.0}, (2026, 12): {"a": 3.0},
    }
