import json

from seiryoku import refresh_cli, refresh_watchdog


def test_main_does_nothing_when_already_done(monkeypatch):
    monkeypatch.setattr(refresh_cli, "is_done", lambda: True)
    monkeypatch.setattr(refresh_cli, "is_running", lambda: (_ for _ in ()).throw(AssertionError("呼ばれるべきでない")))
    monkeypatch.setattr(refresh_cli, "main", lambda: (_ for _ in ()).throw(AssertionError("呼ばれるべきでない")))

    refresh_watchdog.main()  # 例外が飛ばなければOK


def test_main_does_nothing_when_already_running(monkeypatch):
    monkeypatch.setattr(refresh_cli, "is_done", lambda: False)
    monkeypatch.setattr(refresh_cli, "is_running", lambda: True)
    monkeypatch.setattr(refresh_cli, "main", lambda: (_ for _ in ()).throw(AssertionError("呼ばれるべきでない")))

    refresh_watchdog.main()


def test_main_resumes_when_not_done_and_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_watchdog, "WATCHDOG_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(refresh_cli, "is_done", lambda: False)
    monkeypatch.setattr(refresh_cli, "is_running", lambda: False)

    calls = []
    monkeypatch.setattr(refresh_cli, "main", lambda: calls.append(1))

    refresh_watchdog.main()

    assert calls == [1]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["attempts"] == 1


def test_main_stops_after_max_attempts_and_notifies(tmp_path, monkeypatch):
    from datetime import date

    from seiryoku import monthly_article

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"cycle": "2026-11", "attempts": refresh_watchdog.MAX_ATTEMPTS}), encoding="utf-8")
    monkeypatch.setattr(refresh_watchdog, "WATCHDOG_STATE_PATH", state_path)
    monkeypatch.setattr(refresh_cli, "is_done", lambda: False)
    monkeypatch.setattr(refresh_cli, "is_running", lambda: False)
    monkeypatch.setattr(refresh_cli, "main", lambda: (_ for _ in ()).throw(AssertionError("呼ばれるべきでない")))
    notes = []
    monkeypatch.setattr(monthly_article, "notify", lambda *a: notes.append(a))

    refresh_watchdog.main(date(2026, 11, 20))

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["attempts"] == refresh_watchdog.MAX_ATTEMPTS + 1
    assert len(notes) == 1


def test_main_counts_attempts_per_month(tmp_path, monkeypatch):
    from datetime import date

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"cycle": "2026-10", "attempts": refresh_watchdog.MAX_ATTEMPTS + 1}), encoding="utf-8")
    monkeypatch.setattr(refresh_watchdog, "WATCHDOG_STATE_PATH", state_path)
    monkeypatch.setattr(refresh_cli, "is_done", lambda: False)
    monkeypatch.setattr(refresh_cli, "is_running", lambda: False)
    calls = []
    monkeypatch.setattr(refresh_cli, "main", lambda: calls.append(1))

    refresh_watchdog.main(date(2026, 11, 3))

    assert calls == [1]
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"cycle": "2026-11", "attempts": 1}
