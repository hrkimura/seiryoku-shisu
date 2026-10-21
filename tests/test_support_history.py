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


def test_cli_support_facts_uses_latest_snapshot(tmp_path, monkeypatch):
    from seiryoku import cli

    monkeypatch.setattr(support_history, "HISTORY_PATH", tmp_path / "s.json")
    support_history.record_snapshot({"updated": "2026-10-06", "S_p": {"自由民主党": 33.2}, "source": "u"}, "2026-10-08")
    support_history.record_snapshot({"updated": "2026-11-03", "S_p": {"自由民主党": 32.0}, "source": "u"}, "2026-11-05")
    facts = cli.support_facts({"n_support_months": 2, "fetch_error": None, "by_party": {}})
    assert facts["latest"]["updated"] == "2026-11-03"
    assert facts["n_months"] == 2


def test_value_at_interpolates_between_updates():
    from datetime import date

    history = [
        {"updated": "2026-10-19", "S_p": {"a": 2.0, "b": 1.0}},
        {"updated": "2026-11-08", "S_p": {"a": 4.0}},
    ]
    v = support_history.value_at(history, date(2026, 10, 31))
    assert v["method"] == "interpolated"
    assert v["S_p"] == {"a": 2.0 + 2.0 * 12 / 20}
    assert v["dates"] == ["2026-10-19", "2026-11-08"]
    assert v["offset_days"] == 8


def test_value_at_exact_update_on_month_end():
    from datetime import date

    v = support_history.value_at([{"updated": "2026-10-31", "S_p": {"a": 1.0}}], date(2026, 10, 31))
    assert v["method"] == "exact" and v["offset_days"] == 0


def test_value_at_uses_nearest_only_for_its_own_month_end():
    from datetime import date

    history = [{"updated": "2026-10-06", "S_p": {"a": 1.0}}, {"updated": "2026-10-19", "S_p": {"a": 2.0}}]
    v = support_history.value_at(history, date(2026, 9, 30))
    assert v["method"] == "nearest" and v["dates"] == ["2026-10-06"] and v["offset_days"] == 6
    assert support_history.value_at(history, date(2026, 8, 31)) is None
    # 10/19に最も近い月末は10/31なので、10月末には使える
    v = support_history.value_at(history, date(2026, 10, 31))
    assert v["method"] == "nearest" and v["dates"] == ["2026-10-19"]
    assert support_history.value_at(history, date(2026, 11, 30)) is None
    assert support_history.value_at([], date(2026, 10, 31)) is None


def test_support_fetch_records_and_reports_failure(tmp_path, monkeypatch, capsys):
    from seiryoku import support_fetch
    from seiryoku.fetch import miharu

    monkeypatch.setattr(support_history, "HISTORY_PATH", tmp_path / "s.json")
    snap = {"updated": "2026-10-19", "S_p": {"自由民主党": 32.1}, "source": "u", "unknown_labels": ["新党"]}
    monkeypatch.setattr(miharu, "fetch_support", lambda: snap)
    assert support_fetch.main() == 0
    assert support_fetch.main() == 0
    assert len(support_history.load_history()) == 1
    out = capsys.readouterr().out
    assert "追記しました" in out and "記録済み" in out and "新党" in out

    def fail():
        raise ConnectionError("offline")

    monkeypatch.setattr(miharu, "fetch_support", fail)
    assert support_fetch.main() == 1
