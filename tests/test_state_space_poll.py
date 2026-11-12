import math
from dataclasses import replace
from datetime import date

import numpy as np

from seiryoku import state_space as ss
from seiryoku import state_space_glm as glm
from seiryoku import state_space_poll as sp
from seiryoku.fetch.wikipedia_polls import Poll
from seiryoku.trend import Event


def _poll(end, firm="NHK", n=1000, **shares):
    return Poll(start=end, end=end, firm=firm, n=n, n_approx=False, shares=shares)


def test_decay_state_prediction_and_transition():
    p = ss.Params(q_x=0.0, q_b=0.0, sigma2={}, a=0.0, slope="none",
                  decay_idx=(-1,), decay_rate=0.5, decay_var=0.2)
    m = np.array([1.0, 0.0, 2.0])
    C = np.diag([0.1, 0.0, 0.0])
    m2, C2 = ss._predict(m, C, 2.0, p)
    f = math.exp(-1.0)
    assert m2[0] == 1.0 and math.isclose(m2[2], 2.0 * f)
    assert math.isclose(C2[2, 2], 0.2 * (1 - f ** 2))
    assert math.isclose(ss._transition(2.0, p, 3)[2, 2], f)
    # 定常分散から始めれば分散は変わらない
    _, C3 = ss._predict(m, np.diag([0.1, 0.0, 0.2]), 7.0, p)
    assert math.isclose(C3[2, 2], 0.2)


def test_party_share_branch_and_alliance():
    p = _poll("2026-10-01", **{"立憲民主党": 3.0, "民主改革の会": 1.0, "自由民主党": 36.0})
    assert math.isclose(sp.party_share(p, "立憲民主党"), 4.0 / 40.0)
    assert sp.party_share(p, "参政党") is None
    # 中道改革連合の名義があった期間は、出身政党の観測に使わない
    during = _poll("2026-05-01", **{"立憲民主党": 1.0, "公明党": 1.0, "中道改革連合": 8.0, "自由民主党": 40.0})
    assert sp.party_share(during, "立憲民主党") is None and sp.party_share(during, "公明党") is None
    assert sp.party_share(during, "自由民主党") == 40.0 / 50.0
    after = _poll("2026-10-01", **{"公明党": 2.0, "中道改革連合": 0.3, "自由民主党": 37.7})
    assert sp.party_share(after, "公明党") == 2.0 / 40.0
    combined = Poll("2026-02-01", "2026-02-02", "Kyodo News", 1000, False, {"自由民主党": 40.0, "立憲民主党": 1.0},
                    combined=[{"parties": ["立憲民主党", "公明党"], "value": 10.0}])
    assert sp.party_share(combined, "立憲民主党") is None


def test_poll_events_fill_missing_sample_size():
    polls = [_poll("2020-01-10", "ANN", None, **{"自由民主党": 30.0, "日本共産党": 3.0}),
             _poll("2023-01-10", "ANN", 1200, **{"自由民主党": 30.0, "日本共産党": 3.0}),
             _poll("2023-01-12", "NHK", 800, **{"自由民主党": 30.0, "日本共産党": 3.0}),
             _poll("2017-01-12", "NHK", 800, **{"自由民主党": 30.0})]
    ev = sp.poll_events(polls, "日本共産党", start=date(2018, 1, 1))
    assert [e.name for e in ev] == ["ANN", "ANN", "NHK"]
    assert math.isclose(ev[0].n, 1200 * 0.33)
    assert math.isclose(ev[0].shares["日本共産党"], 3.0 / 33.0)


def _synthetic(with_polls: bool):
    """党勢は一定のまま、選挙は最初の2回だけ。支持率は途中から一段上がる。"""
    events = []
    obs = glm.GlmObservations(t=np.array([0.0, 12.0]), level=["衆院選", "衆院選"], k=np.array([20.0, 20.0]),
                              n=np.array([400.0, 400.0]), y=np.array([0.05, 0.05]),
                              events=[Event(date(2018, 1, 1), "衆院選", 0, 400, {}, ""),
                                      Event(date(2019, 1, 1), "衆院選", 0, 400, {}, "")])
    rng = np.random.default_rng(0)
    for month in range(0, 48):
        d = date(2018 + month // 12, month % 12 + 1, 15)
        share = 0.05 if month < 30 else 0.08
        for firm in ("A", "B"):
            n = 600.0
            events.append(Event(d, sp.POLL, 0.0, n, {"x": float(rng.binomial(n, share) / n)}, firm))
    return sp.with_polls(obs, events) if with_polls else obs


def test_polls_move_the_nowcast():
    base = glm.GlmParams(q_x=0.01, phi={}, ell=1.0)
    params = replace(base, poll_firms=["A", "B"], poll_load=1.0, poll_decay=0.01, poll_boom_var=1e-4,
                     poll_house_var=1e-4, poll_extra_var=1e-4)
    t_end = 47.5
    without = glm.run_filter(_synthetic(False), base, until=t_end)
    with_p = glm.run_filter(_synthetic(True), params, until=t_end)
    g0, g1 = base.national_vector(len(without.mean)), params.national_vector(len(with_p.mean))
    assert len(with_p.mean) == len(without.mean) + 1 + 3  # 世論調査の水準差・2社の癖・ブーム
    assert abs(glm._expit(g0 @ without.mean) - 0.05) < 0.01
    assert glm._expit(g1 @ with_p.mean) > 0.065
    assert g1 @ with_p.cov @ g1 < g0 @ without.cov @ g0


def test_fit_poll_params_runs():
    base = glm.GlmParams(q_x=0.01, phi={}, ell=1.0)
    fitted, ll = sp.fit_poll_params(_synthetic(True), base, maxiter=20)
    assert fitted.poll_firms == ["A", "B"] and math.isfinite(ll)
    assert fitted.q_x == base.q_x


def test_months_before():
    assert sp._months_before(date(2026, 3, 31), 3) == date(2025, 12, 28)
    assert sp._months_before(date(2026, 3, 10), 0) == date(2026, 3, 10)


def test_lagged_online_check_hides_recent_local_elections():
    d = [date(2018, 1, 10), date(2019, 1, 10), date(2020, 1, 10)]
    local = [date(2019, 11, 1), date(2018, 6, 1)]
    evs = ([Event(x, "衆院選", 0, 400, {}, "") for x in d]
           + [Event(x, "市区町村議会", 0, 20, {}, f"m{j}") for j, x in enumerate(local)])
    evs.sort(key=lambda e: e.vote_date)
    obs = glm.GlmObservations(t=np.array([ss._t(e.vote_date) for e in evs]), level=[e.level for e in evs],
                              k=np.array([20.0 if e.n == 400 else 1.0 for e in evs]),
                              n=np.array([float(e.n) for e in evs]), y=np.array([0.05] * len(evs)), events=evs)
    polls = [_poll(f"{2018 + m // 12}-{m % 12 + 1:02d}-15", "A", 1000, x=5.0, y=95.0) for m in range(24)]
    base = glm.GlmParams(q_x=0.01, phi={}, ell=1.0)
    pp = replace(base, poll_firms=["A"], poll_boom_var=1e-3, poll_house_var=1e-3, poll_extra_var=1e-3)
    rows = sp.lagged_online_check({"x": obs}, {"x": base}, {"x": pp}, polls, {"x": d[0]}, lags=(0, 3), workers=1)
    assert len(rows) == 2 * 2 * 2  # 予測する選挙2回 x 遅れ2通り x 2モデル
    assert {r["model"] for r in rows} == {"v3", "v4"}
    s = sp.summarize_check(rows)
    assert [(r["model"], r["lag"]) for r in s] == [("v3", 0), ("v4", 0), ("v3", 3), ("v4", 3)]
    assert all(r["n"] == 2 for r in s)


def test_fit_all_checkpoint_skips_done_parties(tmp_path, monkeypatch):
    ck = tmp_path / "ck.json"
    import json
    ck.write_text(json.dumps({"x": {"poll_load": 1.0, "loglik": 0.0, "firms": []}}))
    obs = _synthetic(False)
    out = sp.fit_all({"x": obs}, {"x": glm.GlmParams(q_x=0.01, phi={}, ell=1.0)}, [], {"x": date(2018, 1, 1)},
                     workers=1, checkpoint=ck)
    assert out == {"x": {"poll_load": 1.0, "loglik": 0.0, "firms": []}}


def test_fit_out_of_sample_fits_each_election_before_its_date(tmp_path):
    d = [date(2018, 1, 10), date(2019, 1, 10), date(2020, 1, 10)]
    evs = [Event(x, "衆院選", 0, 400, {}, "") for x in d]
    obs = glm.GlmObservations(t=np.array([ss._t(x) for x in d]), level=["衆院選"] * 3, k=np.array([20.0] * 3),
                              n=np.array([400.0] * 3), y=np.array([0.05] * 3), events=evs)
    # 会社Bは2019年の選挙の後にだけ出るので、2019年の選挙の前のパラメータには入らない
    polls = ([_poll(f"{2018 + m // 12}-{m % 12 + 1:02d}-15", "A", 1000, x=5.0, y=95.0) for m in range(24)]
             + [_poll("2019-06-15", "B", 1000, x=5.0, y=95.0)])
    base = glm.GlmParams(q_x=0.01, phi={}, ell=1.0)
    ck = tmp_path / "ck.json"
    out = sp.fit_out_of_sample({"x": obs}, {"x": base}, polls, {"x": d[0]}, ck, workers=1)
    assert set(out) == {"2019-01-10|x", "2020-01-10|x"}
    assert out["2019-01-10|x"]["firms"] == ["A"]
    assert out["2020-01-10|x"]["firms"] == ["A", "B"]
    # 書き出し済みの鍵は推定し直さない
    before = ck.stat().st_mtime_ns
    assert sp.fit_out_of_sample({"x": obs}, {"x": base}, polls, {"x": d[0]}, ck, workers=1) == out
    assert ck.stat().st_mtime_ns == before
    params = {k: sp._params_from_row(base, v) for k, v in out.items()}
    rows = sp.lagged_online_check({"x": obs}, {"x": base}, params, polls, {"x": d[0]}, lags=(0,), workers=1,
                                  out_of_sample=True)
    assert len(rows) == 2 * 2


def test_attach_polls_refits_only_when_v3_params_change(tmp_path, monkeypatch):
    import json
    calls = []

    def fake_fit_all(obs_by_party, base, polls, first, until=None, workers=None, checkpoint=None):
        calls.append(len(polls))
        return {p: {"poll_load": 1.0, "poll_decay": 0.1, "poll_boom_var": 0.01, "poll_house_var": 0.01,
                    "poll_extra_var": 0.01, "loglik": 0.0, "firms": ["A"]} for p in obs_by_party}

    monkeypatch.setattr(sp, "fit_all", fake_fit_all)
    obs = _synthetic(False)
    hist = tmp_path / "polls.json"
    from seiryoku.fetch.wikipedia_polls import to_json
    hist.write_text(json.dumps(to_json([_poll("2018-03-15", "A", 1000, x=5.0, y=95.0),
                                        _poll("2030-01-15", "A", 1000, x=5.0, y=95.0)])))
    path = tmp_path / "pp.json"
    base = {"x": glm.GlmParams(q_x=0.01, phi={}, ell=1.0)}
    args = ({"x": obs}, base, {"estimated_at": "2026-10-31"}, {"x": date(2018, 1, 1)}, date(2026, 10, 31))
    params, merged, info = sp.attach_polls(*args, path=path, polls_path=hist)
    assert calls == [1]  # tauより後の調査は使わない
    assert info["refit_reason"] == "保存済みの世論調査のパラメータがない"
    assert params["x"].poll_firms == ["A"] and len(merged["x"].t) == len(obs.t) + 1
    sp.attach_polls(*args, path=path, polls_path=hist)
    assert calls == [1]
    sp.attach_polls(args[0], base, {"estimated_at": "2027-02-28"}, *args[3:], path=path, polls_path=hist)
    assert calls == [1, 1]
