import json
from datetime import date, timedelta

import numpy as np
import pytest

from seiryoku import by_elections as be
from seiryoku import state_space_glm as glm
from seiryoku.fetch.go2senkyo import Candidate, ElectionHistoryRow
from seiryoku.trend import Event

START = date(2018, 1, 7)


def _row(name, vote_date="2022/04/24", turnout="30.1%", url="https://example.com/x"):
    return ElectionHistoryRow(vote_date, "", name, turnout, 2, url)


def test_completed_by_elections_keeps_only_contested_by_elections_since_min_year():
    rows = [
        _row("A市議会議員補欠選挙"),
        _row("A市議会議員選挙"),
        _row("A市議会議員補欠選挙", turnout="-%"),
        _row("A市議会議員補欠選挙", vote_date="未定"),
        _row("A市議会議員補欠選挙", vote_date="2017/04/23"),
        _row("A市議会議員補欠選挙", url=None),
    ]
    assert be._completed_by_elections(rows, 2018) == rows[:1]


def _params(**kw):
    return glm.GlmParams(q_x=1e-4, phi={}, ell=0.05, **kw)


def _obs(level, k):
    events = [Event(START, "衆院選", 1.2e8, 465.0, {"A": 0.3, "B": 0.7}, ""),
              Event(START + timedelta(days=30), level, 1e5, 1.0, {"A": k, "B": 1 - k}, "X")]
    return glm.glm_observations(events, "A", START, filter_levels=False)


def test_national_by_election_moves_the_national_state_and_council_by_election_barely_does():
    """国政の補選は国政選挙と同じ負荷1で党勢zにかかり、議会の補選は地方の負荷gamma(ここでは0.05)で
    かかる。同じ1議席の勝ちでも、国政の補選のほうがzを大きく動かす。"""
    moves = {}
    for level in (be.SHUGIIN_BY, be.MUNI_COUNCIL_BY):
        trace = []
        glm.run_filter(_obs(level, 1.0), _params(), trace=trace)
        # traceの最後は補選の時刻の[t, 予測平均, 予測分散, 更新後平均, 更新後分散]
        moves[level] = trace[-1][3][0] - trace[-1][1][0]
    assert moves[be.SHUGIIN_BY] > 0
    assert moves[be.SHUGIIN_BY] > 5 * abs(moves[be.MUNI_COUNCIL_BY])


def test_party_loading_ignores_by_elections():
    from seiryoku import state_space as ss

    base = [Event(START, "衆院選", 1.2e8, 465.0, {"A": 0.3, "B": 0.7}, "")]
    councils = [Event(START + timedelta(days=d), "市区町村議会", 1e5, 20.0, {"A": 0.25, "B": 0.75}, "")
                for d in range(10, 300, 10)]
    by = [Event(START + timedelta(days=d), be.MUNI_COUNCIL_BY, 1e5, 2.0, {"A": 1.0}, "")
          for d in range(15, 300, 10)]
    pts = [ss.SeatVotePoint(date(2017, 10, 22), p, x, v) for p, x, v in
           (("A", 0.3, 0.35), ("B", 0.6, 0.45), ("C", 0.1, 0.2))]
    pts += [ss.SeatVotePoint(date(2014, 12, 14), p, x, v) for p, x, v in
            (("A", 0.32, 0.36), ("B", 0.55, 0.42), ("C", 0.13, 0.22))]
    without = glm.party_loading(glm.glm_observations(base + councils, "A", START, False), pts, [], "A", date.max)
    with_by = glm.party_loading(glm.glm_observations(base + councils + by, "A", START, False), pts, [], "A", date.max)
    assert without is not None and with_by == pytest.approx(without)


def test_paired_difference_clusters_by_election_date():
    a = {"A": [{"vote_date": d, "pred": 0.3, "y": 0.3} for d in ("d1", "d2", "d3")],
         "B": [{"vote_date": d, "pred": 0.2, "y": 0.2} for d in ("d1", "d2", "d3")]}
    b = {"A": [{"vote_date": "d1", "pred": 0.4, "y": 0.3}, {"vote_date": "d2", "pred": 0.3, "y": 0.3},
               {"vote_date": "d3", "pred": 0.3, "y": 0.3}],
         "B": [{"vote_date": "d1", "pred": 0.3, "y": 0.2}, {"vote_date": "d2", "pred": 0.2, "y": 0.2},
               {"vote_date": "d3", "pred": 0.2, "y": 0.2}]}
    d = be.paired_difference(a, b)
    assert d["n"] == 6 and d["elections"] == 3
    assert d["mean"] == pytest.approx(0.02 / 6)
    assert d["by_election"]["d1"] == pytest.approx(0.02)
    # 選挙ごとの和は(0.02, 0, 0)、件数は各2。残差は0.02-2m, -2m, -2m(m=0.02/6)
    m = 0.02 / 6
    resid = np.array([0.02 - 2 * m, -2 * m, -2 * m])
    assert d["se"] == pytest.approx(np.sqrt(3 / 2 * np.sum(resid ** 2)) / 6)


def test_build_gikai_cache_resumes_and_retries_failures(tmp_path, monkeypatch):
    from seiryoku import municipal_registry
    from seiryoku.fetch import go2senkyo

    monkeypatch.setattr(be, "GIKAI_PATH", tmp_path / "gikai_by.json")
    calls = []
    fail = {2}

    def history(jid, kind):
        calls.append(jid)
        if jid in fail:
            raise RuntimeError("waf")
        return [_row(f"{jid}市議会議員補欠選挙", url=f"https://example.com/{jid}"), _row(f"{jid}市議会議員選挙")]

    monkeypatch.setattr(go2senkyo, "jurisdiction_history", history)
    monkeypatch.setattr(go2senkyo, "parse_candidates",
                        lambda url: [Candidate("a", "自由民主党", True), Candidate("b", "日本共産党", False)])
    state = be.build_gikai_cache([1, 2, 3])
    assert state["elections"]["1"] == [{"vote_date": "2022-04-24", "name": "1市議会議員補欠選挙",
                                        "seats": {"自由民主党": 1}}]
    assert sorted(state["done_ids"]) == [1, 3] and "2" in state["errors"]
    fail.clear()
    calls.clear()
    state = be.build_gikai_cache([1, 2, 3])
    assert calls == [2]
    assert sorted(state["done_ids"]) == [1, 2, 3]
    assert json.loads((tmp_path / "gikai_by.json").read_text())["elections"].keys() == {"1", "2", "3"}
    assert "2" not in state["errors"]


def test_build_gikai_cache_refresh_adds_new_by_elections_and_keeps_old_ones_on_failure(tmp_path, monkeypatch):
    from seiryoku.fetch import go2senkyo

    monkeypatch.setattr(be, "GIKAI_PATH", tmp_path / "gikai_by.json")
    rows = {1: [_row("1市議会議員補欠選挙", url="https://example.com/1a")],
            2: [_row("2市議会議員補欠選挙", url="https://example.com/2a")]}
    fail = set()

    def history(jid, kind):
        if jid in fail:
            raise RuntimeError("503")
        return rows[jid]

    monkeypatch.setattr(go2senkyo, "jurisdiction_history", history)
    monkeypatch.setattr(go2senkyo, "parse_candidates", lambda url: [Candidate("a", "自由民主党", True)])
    be.build_gikai_cache([1, 2])
    rows[1] = [_row("1市議会議員補欠選挙", vote_date="2026/10/04", url="https://example.com/1b")] + rows[1]
    fail.add(2)
    state = be.build_gikai_cache([1, 2])
    assert [r["vote_date"] for r in state["elections"]["1"]] == ["2022-04-24"]
    state = be.build_gikai_cache([1, 2], refresh=True)
    assert [r["vote_date"] for r in state["elections"]["1"]] == ["2026-10-04", "2022-04-24"]
    assert len(state["elections"]["2"]) == 1 and "2" in state["errors"]
    assert sorted(state["done_ids"]) == [1, 2]


def test_by_election_levels_are_folded_into_regular_levels_for_charts():
    assert set(be.PARENT_LEVEL) == set(be.BY_LEVELS) - {be.NATIONAL_VOTE_BY}


def test_national_by_election_data_file():
    data = json.loads(be.NATIONAL_PATH.read_text(encoding="utf-8"))["elections"]
    assert len(data) == 21
    assert all(r["house"] in ("衆議院", "参議院") and r["party"] and r["source"].startswith("https://") for r in data)
    events = be.national_events({"自由民主党", "立憲民主党", "日本維新の会"})
    assert {e.level for e in events} == set(be.NATIONAL_BY_LEVELS)
    assert all(e.n == 1.0 and sum(e.shares.values()) == pytest.approx(1.0) for e in events)


def test_backtest_variants_drop_the_right_levels():
    used = {name: set(be.BY_LEVELS) - drop for name, drop in be.VARIANTS.items()}
    assert used == {"なし": set(), "国政補選": set(be.NATIONAL_BY_LEVELS), "国政補選得票": {be.NATIONAL_VOTE_BY},
                    "議会補選": set(be.COUNCIL_BY_LEVELS), "両方": set(be.NATIONAL_BY_LEVELS + be.COUNCIL_BY_LEVELS)}


def test_vote_observations_only_for_parties_that_fielded_a_candidate():
    events = [Event(START, "衆院選", 1.2e8, 465.0, {"A": 0.3, "B": 0.7}, ""),
              Event(START + timedelta(days=30), be.NATIONAL_VOTE_BY, 0.0, 1.0, {"A": 0.45}, "X区")]
    a = glm.glm_observations(events, "A", START, filter_levels=False)
    b = glm.glm_observations(events, "B", START, filter_levels=False)
    assert a.level == ["衆院選", be.NATIONAL_VOTE_BY] and a.y[1] == pytest.approx(0.45)
    assert b.level == ["衆院選"]


def test_vote_observation_is_a_kalman_update_through_the_seat_vote_slope():
    """logit v = z/k + alpha + e の線形正規の観測なので、zの更新はカルマンフィルタの式に一致する。"""
    events = [Event(START, "衆院選", 1.2e8, 465.0, {"A": 0.3, "B": 0.7}, ""),
              Event(START + timedelta(days=30), be.NATIONAL_VOTE_BY, 0.0, 1.0, {"A": 0.45}, "X区")]
    obs = glm.glm_observations(events, "A", START, filter_levels=False)
    p = glm.GlmParams(q_x=1e-4, phi={}, ell=0.5, vote_load=0.5, vote_var=0.04)
    trace = []
    glm.run_filter(obs, p, trace=trace)
    m, C = trace[-1][1], trace[-1][2]
    h = np.zeros(len(m))
    h[0] = 0.5
    h[2 + sorted(set(obs.level) - {"衆院選"}).index(be.NATIONAL_VOTE_BY)] = 1.0
    S = h @ C @ h + 0.04
    expected = m + C @ h / S * (np.log(0.45 / 0.55) - h @ m)
    assert trace[-1][3] == pytest.approx(expected)


def test_vote_variance_is_recovered_from_simulated_by_elections():
    rng = np.random.default_rng(3)
    events = []
    for y in range(6):
        day = START + timedelta(days=365 * y)
        events.append(Event(day, "衆院選", 1.2e8, 465.0, {"A": 140 / 465, "B": 325 / 465}, ""))
        z = np.log(140 / 325)
        for j in range(25):
            v = 1 / (1 + np.exp(-(0.5 * z + 0.3 + rng.normal(0, 0.4))))
            events.append(Event(day + timedelta(days=60 + j), be.NATIONAL_VOTE_BY, 0.0, 1.0, {"A": v}, f"{y}-{j}"))
    obs = glm.glm_observations(events, "A", START, filter_levels=False)
    params, _ = glm.fit_params(obs, ell=0.5, san={"vote_load": 0.5}, maxiter=60, starts=1)
    assert np.sqrt(params.vote_var) == pytest.approx(0.4, rel=0.25)


def test_national_vote_events_attribute_votes_to_the_filing_party():
    events = be.national_vote_events({"自由民主党", "立憲民主党", "日本維新の会", "日本共産党"})
    by = {(e.vote_date.isoformat(), e.name): e for e in events}
    tokyo = by[("2024-04-28", "東京都第15区")]
    assert tokyo.shares["立憲民主党"] == pytest.approx(49476 / 170745, rel=1e-3)
    assert "自由民主党" not in tokyo.shares
    assert all(0 < v < 1 for e in events for v in e.shares.values())
