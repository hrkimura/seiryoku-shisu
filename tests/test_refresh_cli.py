import json
import os

import pytest

from seiryoku import municipal_registry, refresh_cli, turnover


def test_is_running_false_when_no_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_cli, "LOCK_PATH", tmp_path / "refresh_cli.lock")
    assert refresh_cli.is_running() is False


def test_is_running_true_when_lock_pid_is_alive(tmp_path, monkeypatch):
    lock_path = tmp_path / "refresh_cli.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(refresh_cli, "LOCK_PATH", lock_path)
    assert refresh_cli.is_running() is True


def test_is_running_false_when_lock_pid_is_dead(tmp_path, monkeypatch):
    lock_path = tmp_path / "refresh_cli.lock"
    lock_path.write_text("999999999", encoding="utf-8")  # 存在しないはずのPID
    monkeypatch.setattr(refresh_cli, "LOCK_PATH", lock_path)
    assert refresh_cli.is_running() is False


def test_is_done_reflects_marker_file_existence(tmp_path, monkeypatch):
    marker_path = tmp_path / "refresh_cli_done.json"
    monkeypatch.setattr(refresh_cli, "DONE_MARKER_PATH", marker_path)
    assert refresh_cli.is_done() is False
    marker_path.write_text("{}", encoding="utf-8")
    assert refresh_cli.is_done() is True


def _patch_marker_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh_cli, "LOCK_PATH", tmp_path / "refresh_cli.lock")
    monkeypatch.setattr(refresh_cli, "DONE_MARKER_PATH", tmp_path / "refresh_cli_done.json")
    monkeypatch.setattr(refresh_cli, "HEAD_DONE_MARKER_PATH", tmp_path / "refresh_cli_head_done.json")
    monkeypatch.setattr(refresh_cli, "GIKAI_DONE_MARKER_PATH", tmp_path / "refresh_cli_gikai_done.json")


def test_main_acquires_and_releases_lock_and_writes_done_marker(tmp_path, monkeypatch):
    _patch_marker_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(turnover, "refresh_and_rebuild_term_chains", lambda: {"checked": 3, "stale": [1]})
    monkeypatch.setattr(municipal_registry, "refresh_and_rebuild_gikai_term_chains", lambda: {"checked": 5, "stale": []})

    refresh_cli.main()

    assert not refresh_cli.LOCK_PATH.exists()
    assert refresh_cli.is_done() is True
    # 完走後は次回サイクルに備えてサイクル内スキップ用マーカーが片付いている
    assert not refresh_cli.HEAD_DONE_MARKER_PATH.exists()
    assert not refresh_cli.GIKAI_DONE_MARKER_PATH.exists()


def test_main_releases_lock_even_if_an_error_occurs(tmp_path, monkeypatch):
    _patch_marker_paths(monkeypatch, tmp_path)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(turnover, "refresh_and_rebuild_term_chains", boom)

    with pytest.raises(RuntimeError):
        refresh_cli.main()

    assert not refresh_cli.LOCK_PATH.exists()
    assert refresh_cli.is_done() is False


def test_main_skips_head_phase_when_already_done_this_cycle(tmp_path, monkeypatch):
    """首長側が完走済み(サイクル内マーカーあり)で議会側の途中にプロセスが落ちた
    ケースを模す。再開時に首長側を再スキャンしてはいけない(2026-09に発覚した
    不具合: 首長側の完了は恒久データ側から消えるため、これが無いと毎回
    フルスキャンし直してしまう)。"""
    _patch_marker_paths(monkeypatch, tmp_path)
    refresh_cli.HEAD_DONE_MARKER_PATH.write_text(
        json.dumps({"checked": 1785, "stale": [3962]}), encoding="utf-8"
    )

    def boom():
        raise AssertionError("首長側を再スキャンしてしまった")

    monkeypatch.setattr(turnover, "refresh_and_rebuild_term_chains", boom)
    monkeypatch.setattr(municipal_registry, "refresh_and_rebuild_gikai_term_chains", lambda: {"checked": 5, "stale": []})

    refresh_cli.main()

    assert refresh_cli.is_done() is True


def test_main_skips_gikai_phase_when_already_done_this_cycle(tmp_path, monkeypatch):
    _patch_marker_paths(monkeypatch, tmp_path)
    refresh_cli.GIKAI_DONE_MARKER_PATH.write_text(json.dumps({"checked": 1788, "stale": []}), encoding="utf-8")

    def boom():
        raise AssertionError("議会側を再スキャンしてしまった")

    monkeypatch.setattr(turnover, "refresh_and_rebuild_term_chains", lambda: {"checked": 3, "stale": []})
    monkeypatch.setattr(municipal_registry, "refresh_and_rebuild_gikai_term_chains", boom)

    refresh_cli.main()

    assert refresh_cli.is_done() is True


def test_main_leaves_head_done_marker_intact_when_gikai_phase_fails(tmp_path, monkeypatch):
    """議会側で失敗しても、既に完走した首長側のサイクル内マーカーは消さない
    (次回の再開時にスキップできるようにするため)。"""
    _patch_marker_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(turnover, "refresh_and_rebuild_term_chains", lambda: {"checked": 3, "stale": []})

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(municipal_registry, "refresh_and_rebuild_gikai_term_chains", boom)

    with pytest.raises(RuntimeError):
        refresh_cli.main()

    assert refresh_cli.HEAD_DONE_MARKER_PATH.exists()
    assert not refresh_cli.GIKAI_DONE_MARKER_PATH.exists()
    assert refresh_cli.is_done() is False


def test_done_at_reads_marker_date(tmp_path, monkeypatch):
    from datetime import date

    from seiryoku import refresh_cli

    monkeypatch.setattr(refresh_cli, "DONE_MARKER_PATH", tmp_path / "done.json")
    assert refresh_cli.done_at() is None
    (tmp_path / "done.json").write_text("{}")
    assert refresh_cli.done_at() == date.today()
