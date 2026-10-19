import math
from datetime import date, timedelta

import numpy as np
import pytest

from seiryoku import state_space as ss
from seiryoku import state_space_glm as glm
from seiryoku.trend import UNCLASSIFIED_MAYOR, Event

START = date(2018, 1, 7)


@pytest.mark.parametrize("k,n", [(0.0, 12.0), (3.0, 12.0), (12.0, 12.0), (1.0, 1.0), (0.5, 1.0)])
def test_local_loglik_derivatives_match_finite_differences(k, n):
    eta, h = -0.7, 1e-5
    ll, g1, g2 = glm._local_loglik(eta, k, n, 15.0)
    lp, g1p, _ = glm._local_loglik(eta + h, k, n, 15.0)
    lm, g1m, _ = glm._local_loglik(eta - h, k, n, 15.0)
    assert g1 == pytest.approx((lp - lm) / (2 * h), rel=1e-5, abs=1e-8)
    assert g2 == pytest.approx((g1p - g1m) / (2 * h), rel=1e-4, abs=1e-7)


def test_beta_binomial_with_one_seat_is_bernoulli():
    mu = 0.3
    eta = math.log(mu / (1 - mu))
    for k in (0.0, 1.0):
        bb = glm._local_loglik(eta, k, 1.0 + 1e-6, 7.0)[0]
        assert bb == pytest.approx(math.log(mu if k else 1 - mu), abs=1e-4)


def test_laplace_pseudo_observation_reproduces_the_posterior_mode():
    m, s2, k, n, phi = 0.2, 0.5, 2.0, 10.0, 30.0
    y_p, R, _ = glm._laplace(m, s2, k, n, phi)
    eta_hat = (m / s2 + y_p / R) / (1 / s2 + 1 / R)
    _, g1, _ = glm._local_loglik(eta_hat, k, n, phi)
    assert g1 - (eta_hat - m) / s2 == pytest.approx(0.0, abs=1e-6)


def _simulate(jump=None, n_days=6 * 365, seed=0, k_sv=2.0, alpha=-0.5, phi=30.0):
    """真の党勢x(t)の世界で、議会は議席数をベータ二項分布、首長はベルヌーイ分布、
    国政選挙はxをそのまま観測する。"""
    rng = np.random.default_rng(seed)
    events = []
    for d in range(0, n_days, 7):
        day = START + timedelta(days=d)
        x = 0.3 + (jump[1] if jump and day >= jump[0] else 0.0)
        mu = 1 / (1 + math.exp(-(math.log(x / (1 - x)) / k_sv + alpha)))
        # 実データの月あたり件数(議会・首長とも30〜40件)に近づける
        for _ in range(8):
            n = int(rng.integers(5, 30))
            k = rng.binomial(n, rng.beta(mu * phi, (1 - mu) * phi))
            events.append(Event(day, "市区町村議会", 1e5, float(n), {"A": k / n, "B": 1 - k / n}, ""))
            won = rng.random() < mu
            events.append(Event(day, "市区町村長", 1e5, 1.0, {"A": 1.0} if won else {"B": 1.0}, ""))
        if d // 365 != (d - 7) // 365:
            seats = round(x * 465)
            events.append(Event(day, "衆院選", 1.2e8, 465.0, {"A": seats / 465, "B": 1 - seats / 465}, ""))
    return events


def test_recovers_level_and_tracks_a_jump_between_national_elections():
    jump_day = START + timedelta(days=4 * 365 + 60)
    events = _simulate(jump=(jump_day, 0.15), n_days=5 * 365)
    obs = glm.glm_observations(events, "A", START)
    params, _ = glm.fit_params(obs, ell=0.5, jumps=True, maxiter=60, starts=1, phi_mode="level")
    tau = START + timedelta(days=5 * 365 - 7)
    nc = glm.nowcast(obs, params, tau)
    assert nc.x == pytest.approx(0.45, abs=0.04)
    assert nc.x_lo < 0.45 < nc.x_hi
    assert params.phi["市区町村議会"] == pytest.approx(30.0, rel=0.5)
    assert "市区町村長" not in params.phi


def test_phi_modes_set_the_number_of_dispersion_parameters():
    events = _simulate(n_days=2 * 365, seed=1)
    events += [Event(e.vote_date, "都道府県議会", e.P, 40.0, e.shares, "")
               for e in events if e.level == "市区町村議会"][::4]
    obs = glm.glm_observations(events, "A", START)
    level, ll_level = glm.fit_params(obs, ell=0.5, maxiter=40, starts=1, phi_mode="level")
    shared, ll_shared = glm.fit_params(obs, ell=0.5, maxiter=40, starts=1, phi_mode="shared")
    none, ll_none = glm.fit_params(obs, ell=0.5, maxiter=40, starts=1, phi_mode="none")
    assert set(level.phi) == {"市区町村議会", "都道府県議会"}
    assert len(set(shared.phi.values())) == 1 and set(shared.phi) == set(level.phi)
    assert none.phi == {}
    assert ll_level >= ll_shared - 1e-3 and ll_shared >= ll_none - 1e-3


def test_nowcast_stays_inside_the_unit_interval_for_a_tiny_party():
    events = _simulate(n_days=3 * 365, seed=2)
    events = [Event(e.vote_date, e.level, e.P, e.n,
                    {"A": e.shares["A"] * 0.02, "B": 1 - e.shares["A"] * 0.02} if e.level == "衆院選" else e.shares,
                    e.name) for e in events]
    obs = glm.glm_observations(events, "A", START)
    params, _ = glm.fit_params(obs, ell=0.5, maxiter=40, starts=1)
    nc = glm.nowcast(obs, params, events[-1].vote_date)
    assert 0 < nc.x_lo <= nc.x <= nc.x_hi < 1


def test_municipality_effects_are_learned_from_repeated_elections():
    """自治体ごとに固有の水準があり、同じ自治体が2年おきに選挙をする世界で、
    自治体の効果を入れると尤度が上がり、その分散を復元する。"""
    rng = np.random.default_rng(7)
    u = rng.normal(0, 1.0, 120)
    events = []
    for d in range(0, 6 * 365, 7):
        day = START + timedelta(days=d)
        week = d // 7
        for m in range(120):
            if (week + m) % 104 == 0:
                mu = 1 / (1 + math.exp(-(math.log(0.3 / 0.7) / 2.0 - 0.5 + u[m])))
                n = 20
                k = rng.binomial(n, rng.beta(mu * 50, (1 - mu) * 50))
                events.append(Event(day, "市区町村議会", 1e5, float(n), {"A": k / n, "B": 1 - k / n}, f"市{m}"))
        if d // 365 != (d - 7) // 365:
            events.append(Event(day, "衆院選", 1.2e8, 465.0, {"A": 140 / 465, "B": 325 / 465}, "衆院選"))
    obs = glm.glm_observations(events, "A", START)
    _, ll0 = glm.fit_params(obs, ell=0.5, maxiter=40, starts=1)
    p1, ll1 = glm.fit_params(obs, ell=0.5, muni=True, maxiter=40, starts=1)
    assert ll1 > ll0 + 20
    assert math.sqrt(p1.tau2["市区町村議会"]) == pytest.approx(1.0, rel=0.35)


def _pts(k, c, dates, v_a=None):
    out = []
    for j, d in enumerate(dates):
        for party, v in (("A", v_a or 0.4), ("中", 0.2 + 0.05 * j), ("小", 0.05 + 0.02 * j)):
            x = 1 / (1 + math.exp(-(k * ss._logit(v) + c)))
            out.append(ss.SeatVotePoint(d, party, x, v))
    return out


def test_sangiin_link_maps_the_two_seat_vote_curves_through_the_vote_share():
    shu = _pts(2.0, 1.0, (date(2017, 10, 22), date(2021, 10, 31)))
    san = _pts(1.5, 0.6, (date(2019, 7, 21), date(2022, 7, 10)))
    link = glm.sangiin_link(shu, san)
    v = 0.3
    z = 2.0 * ss._logit(v) + 1.0
    assert link["san_load"] * z + link["san_offset"] == pytest.approx(1.5 * ss._logit(v) + 0.6, abs=1e-3)
    assert link["san_prior_var"] == pytest.approx(0.0, abs=1e-6)


def test_beta_binomial_switches_continuously_to_binomial_for_large_precision():
    below = glm._local_loglik(-0.4, 3.0, 12.0, 0.99e7)
    above = glm._local_loglik(-0.4, 3.0, 12.0, 1.01e7)
    assert np.allclose(below, above, rtol=1e-4)


def test_pooled_loading_interpolates_the_profile_maximum(monkeypatch):
    def fake_fit(obs, ell, **kw):
        return None, -(math.log(ell) - math.log(0.4)) ** 2
    monkeypatch.setattr(glm, "fit_params", fake_fit)
    best, _ = glm.pooled_loading({"A": None, "B": None}, grid=(0.3, 0.45, 0.6))
    assert best == pytest.approx(0.4, rel=1e-6)


class _SerialExecutor:
    map = staticmethod(map)


def test_pooled_fit_interpolates_the_summed_profile_and_widens_the_grid(monkeypatch):
    def fake_fit(obs, ell, q_x=None, **kw):
        return glm.GlmParams(q_x=q_x, phi={}, ell=ell), -(math.log(q_x) - math.log(obs)) ** 2
    monkeypatch.setattr(glm, "fit_params", fake_fit)
    # 2党の最大点が0.01と0.0225なので、合計の最大点はその幾何平均0.015
    q, params = glm.pooled_fit(_SerialExecutor(), {"A": 0.01, "B": 0.0225}, {"A": 0.5, "B": 0.5},
                               {"A": {}, "B": {}})
    assert q == pytest.approx(0.015, rel=1e-6) and params["B"].q_x == q
    q, _ = glm.pooled_fit(_SerialExecutor(), {"A": 1e-3}, {"A": 0.5}, {"A": {}})
    assert q == pytest.approx(1e-3, rel=1e-6)


def test_municipality_effects_are_placed_only_on_councils_and_q_can_be_fixed():
    events = _simulate(n_days=2 * 365, seed=3)
    obs = glm.glm_observations(events, "A", START)
    assert "市区町村長" in obs.level
    p, _ = glm.fit_params(obs, ell=0.5, muni=True, maxiter=30, q_x=0.03)
    assert set(p.tau2) == {"市区町村議会"} and p.q_x == pytest.approx(0.03)


def test_interpolation_check_holds_out_each_national_election_after_the_first():
    events = _simulate(n_days=4 * 365, seed=5)
    obs = {"A": glm.glm_observations(events, "A", START)}
    nat = sorted({e.vote_date for e in events if e.level == "衆院選"})
    shu = [p for d in nat for p in _pts(2.0, 1.0, (d,))]
    san = _pts(1.5, 0.6, (date(2019, 7, 21), date(2022, 7, 10)))
    res = glm.national_interpolation_check(obs, shu, san, workers=1)
    assert [r["vote_date"] for r in res["A"]] == [d.isoformat() for d in nat[1:]]
    assert all(r["lo"] <= r["pred"] <= r["hi"] and r["q"] > 0 for r in res["A"])
    s = glm.interpolation_summary(res)
    assert s["n"] == len(nat) - 1 and 0 <= s["coverage"] <= 1


def test_level_movement_sums_the_absolute_updates_by_month_and_level():
    events = _simulate(n_days=365, seed=6)
    obs = glm.glm_observations(events, "A", START)
    params = glm.GlmParams(q_x=0.02, phi={}, ell=0.5, tau2={"市区町村議会": 0.3})
    ends = [d for d in (START + timedelta(days=30 * k) for k in range(1, 14))]
    mv = glm.level_movement(obs, params, ends)
    c: list = []
    glm.run_filter(obs, params, contrib=c)
    total = sum(abs(dz) for i, _, dz in c if obs.events[i].vote_date <= ends[-1])
    assert sum(sum(m.values()) for m in mv) == pytest.approx(total)
    assert set().union(*mv) <= set(obs.level)


def test_filtered_series_matches_the_nowcast_at_each_month_end():
    events = _simulate(n_days=2 * 365, seed=8)
    obs = glm.glm_observations(events, "A", START)
    params = glm.GlmParams(q_x=0.05, phi={}, ell=0.5, tau2={"市区町村議会": 0.3})
    ends = [START + timedelta(days=91 * k) for k in range(1, 8)]
    for end, (z, s) in zip(ends, glm.filtered_series(obs, params, ends)):
        nc = glm.nowcast(obs, params, end)
        assert z == pytest.approx(nc.z, abs=1e-9) and s == pytest.approx(nc.z_se, rel=1e-6)


def test_loading_is_recovered_when_estimated():
    # 負荷は地方と国政が同時に動いたときにしか識別できないので、跳びの後にも国政選挙を置く
    jump_day = START + timedelta(days=3 * 365 + 60)
    events = _simulate(jump=(jump_day, 0.15), n_days=6 * 365, k_sv=2.0)
    obs = glm.glm_observations(events, "A", START)
    params, _ = glm.fit_params(obs, maxiter=80, starts=1)
    assert params.ell == pytest.approx(0.5, rel=0.3)


def test_party_loading_reduces_to_inverse_elasticity_when_local_share_equals_vote_share():
    shu = _pts(2.0, 1.0, (date(2017, 10, 22), date(2018, 10, 22)), v_a=0.25)
    san = _pts(1.5, 0.6, (date(2019, 7, 21), date(2019, 8, 21)), v_a=0.25)
    events = [Event(date(2019, 1, 1) + timedelta(days=i), "市区町村議会", 1e5, 20.0,
                    {"A": 0.25, "B": 0.75}, f"市{i}") for i in range(10)]
    obs = glm.glm_observations(events, "A", date(2019, 1, 1), filter_levels=False)
    assert glm.party_loading(obs, shu, san, "A", date(2020, 1, 1)) == pytest.approx(0.5, rel=1e-4)
    # 地方の占有率が得票率より低い政党は、地方の変化を国政に小さく写す
    low = [Event(e.vote_date, e.level, e.P, e.n, {"A": 0.05, "B": 0.95}, e.name) for e in events]
    obs = glm.glm_observations(low, "A", date(2019, 1, 1), filter_levels=False)
    assert glm.party_loading(obs, shu, san, "A", date(2020, 1, 1)) > 1.0


def test_smoother_bridges_two_national_elections_like_a_brownian_bridge():
    d0 = date(2020, 1, 1)
    events = [Event(d0, "衆院選", 1.2e8, 465.0, {"A": 100 / 465, "B": 365 / 465}, ""),
              Event(d0 + timedelta(days=730), "衆院選", 1.2e8, 465.0, {"A": 200 / 465, "B": 265 / 465}, "")]
    obs = glm.glm_observations(events, "A", d0, filter_levels=False)
    params = glm.GlmParams(q_x=0.2, phi={}, ell=0.5, sigma_national=1e-6)
    t_mid = (obs.t[0] + obs.t[1]) / 2
    (m, C), (m_end, _) = glm.smooth(obs, params, [t_mid, obs.t[1]])
    z0, _ = glm.national_logit(obs.k[0], obs.n[0])
    z1, _ = glm.national_logit(obs.k[1], obs.n[1])
    assert m[0] == pytest.approx((z0 + z1) / 2, abs=1e-6)
    assert C[0, 0] == pytest.approx(0.2 * (obs.t[1] - obs.t[0]) / 4, rel=1e-4)
    assert m_end[0] == pytest.approx(z1, abs=1e-6)


def test_monthly_contributions_add_up_to_the_change_of_the_estimate():
    events = _simulate(n_days=2 * 365, seed=3)
    obs = glm.glm_observations(events, "A", START)
    params = glm.GlmParams(q_x=0.05, phi={"市区町村議会": 30.0, "市区町村長": 1e8}, ell=0.5,
                           tau2={"市区町村議会": 0.3})
    a, b = date(2019, 3, 31), date(2019, 4, 30)
    parts = glm.monthly_contributions(obs, params, a, b)
    assert parts and all(a < date.fromisoformat(p["vote_date"]) <= b for p in parts)
    dz = glm.nowcast(obs, params, b).z - glm.nowcast(obs, params, a).z
    assert sum(p["dz"] for p in parts) == pytest.approx(dz, abs=1e-9)


def test_national_only_component_absorbs_moves_invisible_in_local_elections():
    """地方選挙は動かず国政選挙だけが動く世界では、国政だけに現れる部分を入れると尤度が上がり、
    国政の党勢の推定は国政選挙の結果に合う。"""
    events = _simulate(n_days=6 * 365, seed=5)
    rng = np.random.default_rng(1)
    out = []
    for e in events:
        if e.level == "衆院選":
            x = 0.3 + 0.12 * rng.standard_normal()
            e = Event(e.vote_date, e.level, e.P, e.n, {"A": round(x * 465) / 465, "B": 1 - round(x * 465) / 465}, "")
        out.append(e)
    obs = glm.glm_observations(out, "A", START)
    p0, ll0 = glm.fit_params(obs, ell=0.5, maxiter=60, starts=1)
    p1, ll1 = glm.fit_params(obs, ell=0.5, national_dev=True, maxiter=60, starts=1)
    assert ll1 > ll0
    assert p1.q_nat > 0.01
    last = [i for i, lv in enumerate(obs.level) if lv == "衆院選"][-1]
    assert glm.nowcast(obs, p1, obs.events[last].vote_date).x == pytest.approx(obs.y[last], abs=0.01)


def test_filter_with_municipality_effects_matches_the_exact_filter():
    """地盤を状態ベクトルに入れた厳密なフィルタと、状態の外に置く近似の結果がほぼ一致する。"""
    rng = np.random.default_rng(11)
    M = 30
    u = rng.normal(0, 1.0, M)
    events = []
    z = math.log(0.3 / 0.7)
    for d in range(0, 6 * 365, 7):
        day = START + timedelta(days=d)
        z += rng.normal(0, math.sqrt(0.05 * 7 / 365.25))
        for m in range(M):
            if (d // 7 + 3 * m) % 104 == 0:
                n = int(rng.integers(5, 30))
                k = rng.binomial(n, 1 / (1 + math.exp(-(0.5 * z - 0.5 + u[m]))))
                events.append(Event(day, "市区町村議会", 1e5, float(n), {"A": k / n, "B": 1 - k / n}, f"市{m}"))
        if d % 364 == 0:
            s = round(465 / (1 + math.exp(-z)))
            events.append(Event(day, "衆院選", 1.2e8, 465.0, {"A": s / 465, "B": 1 - s / 465}, ""))
    obs = glm.glm_observations(events, "A", START, filter_levels=False)
    params = glm.GlmParams(q_x=0.05, phi={"市区町村議会": 1e8}, ell=0.5, tau2={"市区町村議会": 1.0})
    names = sorted({e.name for e in obs.events if e.level != "衆院選"})
    col = {nm: 3 + i for i, nm in enumerate(names)}
    K = 3 + M
    m_ = np.zeros(K)
    C = np.eye(K) * params.prior_var
    C[1, 1] = 0.0
    for j in range(3, K):
        C[j, j] = 1.0
    t_prev = obs.t[0]
    for i in range(len(obs.t)):
        C[0, 0] += params.q_x * (obs.t[i] - t_prev)
        t_prev = obs.t[i]
        h = np.zeros(K)
        if obs.level[i] == "衆院選":
            h[0] = 1.0
            y, sd = glm.national_logit(obs.k[i], obs.n[i], params.sigma_national)
            R = sd * sd
        else:
            h[0], h[2], h[col[obs.events[i].name]] = 0.5, 1.0, 1.0
            y, R, _ = glm._laplace(float(h @ m_), float(h @ C @ h), obs.k[i], obs.n[i], 1e8)
        Ch = C @ h
        S = h @ Ch + R
        m_ = m_ + Ch * (y - h @ m_) / S
        C = C - np.outer(Ch, Ch) / S
    r = glm.run_filter(obs, params, state_levels=["市区町村議会"])
    assert r.mean[0] == pytest.approx(m_[0], abs=0.1 * math.sqrt(C[0, 0]))
    assert r.cov[0, 0] == pytest.approx(C[0, 0], rel=0.1)


def test_coalition_mayors_are_reallocated_in_proportion_to_the_win_probability():
    """強い党と弱い党の相乗りの首長選では、等分の取り分が当選確率に比例して強い党に寄り、
    組の合計は変わらない。単独推薦の首長選と議会選挙はそのまま。"""
    d0 = date(2019, 1, 1)
    events = []
    for j in range(40):
        d = d0 + timedelta(days=20 * j)
        events.append(Event(d, "市区町村議会", 1.0, 20.0, {"A": 0.6, "B": 0.1}, f"m{j}"))
        events.append(Event(d, "市区町村長", 1.0, 1.0, {"A": 1.0} if j % 2 else {"B": 1.0} if j % 5 == 0 else {}, f"m{j}"))
    coal = Event(d0 + timedelta(days=900), "市区町村長", 1.0, 1.0, {"A": 0.5, "B": 0.5}, "x")
    events.append(coal)
    obs = {p: glm.glm_observations(events, p, d0, filter_levels=False) for p in "AB"}
    params = {p: glm.GlmParams(q_x=0.01, phi={"市区町村議会": 30.0, "市区町村長": 1e8}, ell=1.0) for p in "AB"}
    out = glm.coalition_weighted(obs, params)
    ia, ib = len(obs["A"].t) - 1, len(obs["B"].t) - 1
    assert out["A"].y[ia] + out["B"].y[ib] == pytest.approx(1.0)
    assert out["A"].y[ia] > 0.7
    np.testing.assert_allclose(out["A"].y[:ia], obs["A"].y[:ia])
    np.testing.assert_allclose(out["A"].k, out["A"].y * out["A"].n)


def test_mayors_before_the_endorsement_list_are_not_observed():
    d = date(2026, 1, 10)
    events = [Event(d, "市区町村議会", 1.0, 10.0, {"A": 0.3}, "m"),
              Event(d + timedelta(days=1), UNCLASSIFIED_MAYOR, 1.0, 1.0, {"A": 1.0}, "m")]
    obs = glm.glm_observations(events, "A", d, filter_levels=False)
    assert obs.level == ["市区町村議会"]


def test_saved_parameters_round_trip(tmp_path):
    p = glm.GlmParams(q_x=0.02, phi={"市区町村議会": 1e6}, ell=0.7, tau2={"市区町村議会": 0.4},
                      san_load=0.84, san_offset=-0.1)
    dev = glm.GlmParams(q_x=0.02, phi={}, ell=0.7, q_nat=0.05)
    path = tmp_path / "params.json"
    glm.save_params(path, {"A": p}, {"estimated_at": "2026-09-30"}, {"A": dev})
    params, meta, nd = glm.load_params(path)
    assert params["A"] == p and nd["A"] == dev and meta["estimated_at"] == "2026-09-30"


def test_estimate_all_fits_every_party_in_two_stages():
    events = _simulate(n_days=3 * 365, seed=4)
    obs = {"A": glm.glm_observations(events, "A", START)}
    params, dev = glm.estimate_all(obs, {"A": 0.5}, {"A": {"maxiter": 20, "starts": 1}}, workers=1)
    assert set(params) == {"A"} and params["A"].ell == 0.5 and dev == {}


def test_refit_is_triggered_by_a_national_election_after_the_saved_parameters(tmp_path):
    path = tmp_path / "params.json"
    d = date(2025, 7, 20)
    events = [Event(d, "参院選", 1e8, 125.0, {"A": 0.3}, ""),
              Event(d + timedelta(days=30), "市区町村議会", 1.0, 10.0, {"A": 0.3}, "m")]
    assert glm.refit_reason(events, path) is not None
    p = glm.GlmParams(q_x=0.02, phi={}, ell=0.7)
    glm.save_params(path, {"A": p}, {"estimated_at": "2025-08-31", "last_national": glm.latest_national(events)})
    assert glm.refit_reason(events, path) is None
    events.append(Event(date(2026, 2, 8), "衆院選", 1e8, 465.0, {"A": 0.4}, ""))
    assert "2026-02-08" in glm.refit_reason(events, path)


def test_binomial_national_error_is_wider_for_a_party_with_few_seats():
    _, sd_small = glm.national_logit(1.0, 465.0, None)
    y, sd_large = glm.national_logit(300.0, 465.0, None)
    assert sd_small > 5 * sd_large
    x = 300.5 / 466
    assert sd_large * x * (1 - x) == pytest.approx(math.sqrt(x * (1 - x) / 465), rel=0.01)
