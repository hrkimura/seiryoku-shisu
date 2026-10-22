import pytest
import requests

from seiryoku.fetch import util


class _FakeClock:
    """time.time()/time.sleep()を差し替え、実時間を待たずに経過時間だけ進める。"""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _FakeResponse:
    def __init__(self, content=b"ok", headers=None, status=200):
        self.content = content
        self.headers = headers or {}
        self.status_code = status

    def raise_for_status(self):
        pass


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(util, "CACHE_DIR", tmp_path)
    clock = _FakeClock()
    monkeypatch.setattr(util.time, "time", clock.time)
    monkeypatch.setattr(util.time, "sleep", clock.sleep)
    util._blocked_until.clear()
    util._last_request_time.clear()
    return clock


def test_fetch_success_no_retry_needed(monkeypatch, tmp_path):
    clock = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResponse(b"hello"))

    result = util.fetch("https://example.com/a")

    assert result == b"hello"
    assert clock.slept == [] or all(s < 1 for s in clock.slept)  # throttleのみ


def test_fetch_caches_second_call_without_network(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: calls.append(1) or _FakeResponse(b"x"))

    util.fetch("https://example.com/b")
    util.fetch("https://example.com/b")

    assert len(calls) == 1


def test_fetch_retries_after_waf_challenge_then_succeeds(monkeypatch, tmp_path):
    clock = _setup(monkeypatch, tmp_path)
    responses = [
        _FakeResponse(b"", headers={"x-amzn-waf-action": "challenge"}, status=202),
        _FakeResponse(b"real content"),
    ]
    monkeypatch.setattr(requests, "get", lambda *a, **kw: responses.pop(0))

    result = util.fetch("https://go2senkyo.com/c", retries=1)

    assert result == b"real content"
    assert util._WAF_COOLDOWN_SECONDS in clock.slept


def test_fetch_retries_after_connection_error_then_succeeds(monkeypatch, tmp_path):
    clock = _setup(monkeypatch, tmp_path)
    calls = {"n": 0}

    def fake_get(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("boom")
        return _FakeResponse(b"recovered")

    monkeypatch.setattr(requests, "get", fake_get)

    result = util.fetch("https://go2senkyo.com/d", retries=1)

    assert result == b"recovered"
    assert calls["n"] == 2


def test_fetch_raises_after_exhausting_retries(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: _FakeResponse(b"", headers={"x-amzn-waf-action": "challenge"}, status=202)
    )

    try:
        util.fetch("https://go2senkyo.com/e", retries=1)
        assert False, "WafChallengeErrorが送出されるべき"
    except util.WafChallengeError:
        pass


def test_shared_cooldown_avoids_double_wait_for_second_call(monkeypatch, tmp_path):
    """同じホストへの2つ目のfetch呼び出しは、1つ目が設定した共有クールダウンの
    残り時間だけ待てばよく、フルの11分を待ち直さない。"""
    clock = _setup(monkeypatch, tmp_path)

    # 1つ目: 即座にWAFチャレンジで恒久エラーになる(retries=0)
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: _FakeResponse(b"", headers={"x-amzn-waf-action": "challenge"}, status=202)
    )
    try:
        util.fetch("https://go2senkyo.com/f", retries=0)
    except util.WafChallengeError:
        pass

    # ここで少しだけ時間を進める(クールダウンの一部だけ経過した状態を模す)
    clock.now += 60

    # 2つ目: 今度は成功する。共有クールダウンの「残り」だけ待つはず。
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResponse(b"ok"))
    slept_before = list(clock.slept)
    util.fetch("https://go2senkyo.com/g", retries=0)
    new_sleeps = clock.slept[len(slept_before):]

    # 残りのクールダウン(660 - 60 = 600秒前後)だけ待ち、フルの660秒を待ち直していない
    assert any(500 < s <= 600 for s in new_sleeps)
    assert util._WAF_COOLDOWN_SECONDS not in new_sleeps


def test_fetch_without_wait_raises_instead_of_waiting_out_a_block(monkeypatch, tmp_path):
    clock = _setup(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: calls.append(1) or _FakeResponse(b"x"))
    util._blocked_until["example.com"] = clock.now + 660

    with pytest.raises(util.WafChallengeError):
        util.fetch("https://example.com/c", wait=False)

    assert calls == []
    assert clock.slept == []
