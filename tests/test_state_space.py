import math
from datetime import date, timedelta

import numpy as np
import pytest

from seiryoku import state_space as ss
from seiryoku.trend import Event

START = date(2018, 1, 7)


def _simulate(slope, alphas, sigma_local, national_every_days=365, n_days=6 * 365, seed=0, jump=None, lam=1.0):
    """真の党勢 x(t)=0.3+slope*t(月)の世界で、国政選挙はxを、地方選挙はx+alphaを観測する。"""
    rng = np.random.default_rng(seed)
    events = []
    for d in range(0, n_days, 7):
        day = START + timedelta(days=d)
        months = ss._t(day) - ss._t(START)
        x = 0.3 + slope * months
        if jump and day >= jump[0]:
            x += jump[1]
        for level, alpha in alphas.items():
            for _ in range(3):
                y = lam * x + alpha + rng.normal(0, sigma_local)
                events.append(Event(day, level, 1e5, 20.0, {"A": y, "B": 1 - y}, ""))
        # 投票日は7日刻みなので、national_every_daysの区切りをまたいだ最初の週に国政選挙を置く
        if d // national_every_days != (d - 7) // national_every_days:
            events.append(Event(day, "衆院選", 1.2e8, 465.0, {"A": x, "B": 1 - x}, "衆院選"))
    return events


def _fit(events, tau):
    obs = ss.party_observations(events, "A", START)
    params, _ = ss.fit_params(obs, maxiter=80)
    return obs, params, ss.nowcast(obs, params, "A", tau)


def test_recovers_slope_and_level_offsets():
    events = _simulate(0.002, {"市区町村議会": -0.1, "市区町村長": 0.15}, 0.05)
    tau = events[-1].vote_date
    _, _, nc = _fit(events, tau)
    assert nc.D == pytest.approx(0.002, abs=0.0005)
    assert nc.alpha["市区町村議会"] == pytest.approx(-0.1, abs=0.01)
    assert nc.alpha["市区町村長"] == pytest.approx(0.15, abs=0.01)
    true_x = 0.3 + 0.002 * (ss._t(tau) - ss._t(START))
    assert nc.x == pytest.approx(true_x, abs=0.01)


def test_national_election_pins_the_level():
    """国政選挙の直後、xの推定は国政選挙の値にほぼ一致し、標準誤差はσ_国程度になる。"""
    events = _simulate(0.0, {"市区町村議会": -0.1}, 0.1)
    obs = ss.party_observations(events, "A", START)
    params = ss.Params(q_x=1e-5, q_b=1e-7, sigma2={"市区町村議会": 0.01}, a=0.0)
    national = [e for e in events if e.level == "衆院選"][-1]
    r = ss.run_filter(obs, params, until=ss._t(national.vote_date))
    assert r.mean[0] == pytest.approx(national.shares["A"], abs=3 * ss.SIGMA_NATIONAL)
    assert math.sqrt(r.cov[0, 0]) < 2 * ss.SIGMA_NATIONAL


def test_local_elections_track_a_jump_between_national_elections():
    """国政選挙の間に党勢が跳んでも、地方選挙だけで次の国政選挙の前に追従する。"""
    jump_day = START + timedelta(days=4 * 365 + 60)
    events = _simulate(0.0, {"市区町村議会": -0.1}, 0.05, national_every_days=2 * 365,
                       n_days=5 * 365, jump=(jump_day, 0.1))
    tau = START + timedelta(days=5 * 365 - 7)
    _, _, nc = _fit(events, tau)
    assert nc.x == pytest.approx(0.4, abs=0.03)


def test_party_observations_skip_all_independent_elections_and_pre_founding():
    day = date(2020, 1, 1)
    # 階層の使用条件(MIN_LEVEL_OBS)に掛からないよう、常に使われる国政の階層で確かめる
    events = [
        Event(day, "衆院選", 1e5, 1, {"無所属": 1.0}, ""),
        Event(day, "衆院選", 1e5, 1, {"無所属": 0.5, "A": 0.5}, ""),
        Event(day - timedelta(days=30), "衆院選", 1e5, 1, {"A": 1.0}, ""),
    ]
    obs = ss.party_observations(sorted(events, key=lambda e: e.vote_date), "A", day)
    assert list(obs.y) == [1.0]


def test_predict_matches_dense_matrix_form():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    C = A @ A.T
    m = rng.normal(size=4)
    p = ss.Params(q_x=0.01, q_b=0.002, sigma2={}, a=0.0)
    dt = 2.5
    m2, C2 = ss._predict(m, C, dt, p)
    F = np.eye(4)
    F[0, 1] = dt
    Q = np.zeros((4, 4))
    Q[:2, :2] = p.q_x * np.array([[dt, 0], [0, 0]]) + p.q_b * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    assert np.allclose(m2, F @ m)
    assert np.allclose(C2, F @ C @ F.T + Q)


def test_levels_where_the_party_almost_never_wins_are_not_used():
    day = date(2020, 1, 1)
    events = [Event(day + timedelta(days=i), "市区町村長", 1e5, 1, {"B": 1.0}, "") for i in range(50)]
    events += [Event(day + timedelta(days=i), "市区町村議会", 1e5, 20, {"A": 0.2, "B": 0.8}, "") for i in range(50)]
    events.append(Event(day, "衆院選", 1.2e8, 465, {"A": 0.0, "B": 1.0}, ""))
    obs = ss.party_observations(sorted(events, key=lambda e: e.vote_date), "A", day)
    assert set(obs.level) == {"市区町村議会", "衆院選"}


def test_loadings_recover_damped_local_response():
    """地方の占有率が国政の変化の3割しか動かない世界で、負荷を推定すると0.3に近く、
    次の国政選挙の予測は負荷1のモデルより当たる。"""
    jump_day = START + timedelta(days=4 * 365 + 60)
    # 負荷は国政と地方の共変動からしか学べないので、跳びの前にも趨勢を持たせる
    events = _simulate(0.003, {"市区町村議会": 0.1}, 0.03, national_every_days=365,
                       n_days=6 * 365 + 30, jump=(jump_day, 0.15), lam=0.3)
    obs = ss.party_observations(events, "A", START)
    params, _ = ss.fit_params(obs, loadings=True, drift=True)
    assert params.lam["市区町村議会"] == pytest.approx(0.3, abs=0.08)
    checks = {flag: ss.national_forecast_check(obs, loadings=flag, drift=flag, since=START + timedelta(days=5 * 365))
              for flag in (False, True)}
    err = {flag: abs(c[-1]["y"] - c[-1]["pred"]) for flag, c in checks.items()}
    assert err[True] < err[False]


def test_alpha_drift_is_added_only_to_local_levels():
    p = ss.Params(q_x=0.0, q_b=0.0, sigma2={}, a=0.0, q_alpha=0.01)
    m, C = ss._predict(np.zeros(4), np.zeros((4, 4)), 2.0, p, local_idx=[3])
    assert C[3, 3] == pytest.approx(0.02)
    assert C[2, 2] == 0.0


def _curve_points(k=2.0, c=1.0, dates=(date(2017, 10, 22), date(2021, 10, 31))):
    vs = {"大": (0.4, 0.35), "中": (0.2, 0.25), "小": (0.05, 0.07)}
    pts = []
    for j, d in enumerate(dates):
        for party in vs:
            v = vs[party][j]
            x = 1 / (1 + math.exp(-(k * ss._logit(v) + c)))
            pts.append(ss.SeatVotePoint(d, party, x, v))
    return pts


def test_seat_vote_curve_recovers_elasticity_and_intercept():
    curve = ss.seat_vote_curve(_curve_points())
    assert curve.k == pytest.approx(2.0, rel=1e-4)
    assert curve.c == pytest.approx(1.0, abs=1e-3)
    assert curve.resid_var == pytest.approx(0.0, abs=1e-6)


def test_seat_vote_curve_weights_points_by_the_number_of_seats():
    """議席の少ない党の議席が偶然1つ多くても、議席の多い党の点が弾力性を決める。"""
    pts = _curve_points()
    pts = [ss.SeatVotePoint(p.vote_date, p.party, p.x, p.v, 465) for p in pts]
    noisy = [ss.SeatVotePoint(p.vote_date, p.party, p.x + (2 / 465 if p.party == "小" and p.vote_date.year == 2021 else 0),
                              p.v, 465) for p in pts]
    assert ss.seat_vote_curve(noisy).k == pytest.approx(2.0, rel=0.1)


def test_seat_vote_loading_is_below_one_for_large_and_above_one_for_small_parties():
    pts = _curve_points()
    after = date(2022, 1, 1)
    assert ss.seat_vote_loading(pts, "大", after) < 1 < ss.seat_vote_loading(pts, "小", after)


def test_seat_vote_loading_uses_only_earlier_elections_and_falls_back_to_the_curve():
    pts = _curve_points()
    assert ss.seat_vote_loading(pts, "大", date(2017, 10, 22)) is None
    known = ss.seat_vote_loading(pts, "中", date(2022, 1, 1))
    pts = [ss.SeatVotePoint(p.vote_date, p.party, p.x, None if p.party == "中" and p.vote_date.year == 2021 else p.v)
           for p in pts]
    assert ss.seat_vote_loading(pts, "中", date(2022, 1, 1)) == pytest.approx(known, rel=1e-3)


@pytest.mark.parametrize("slope,beta", [("rw", 0.0), ("damped", 0.3), ("damped", 1e-5)])
def test_two_short_steps_equal_one_long_step(slope, beta):
    p = ss.Params(q_x=0.01, q_b=0.002, sigma2={}, a=0.0, slope=slope, beta=beta)
    m0, C0 = np.array([0.3, 0.01, 0.0]), np.diag([0.02, 0.001, 0.05])
    m1, C1 = ss._predict(*ss._predict(m0, C0, 0.7, p), 2.1, p)
    m2, C2 = ss._predict(m0, C0, 2.8, p)
    assert np.allclose(m1, m2)
    assert np.allclose(C1, C2, rtol=1e-6, atol=1e-12)


def test_damped_slope_approaches_random_walk_slope_as_beta_vanishes():
    rw = ss._slope_transition(3.0, ss.Params(0.0, 0.002, {}, 0.0))
    damped = ss._slope_transition(3.0, ss.Params(0.0, 0.002, {}, 0.0, slope="damped", beta=1e-6))
    assert np.allclose(rw, damped, rtol=1e-4)


def test_slope_none_keeps_the_slope_at_zero():
    events = _simulate(0.003, {"市区町村議会": 0.0}, 0.03)
    obs = ss.party_observations(events, "A", START)
    params, _ = ss.fit_params(obs, maxiter=50, slope="none")
    nc = ss.nowcast(obs, params, "A", events[-1].vote_date)
    assert nc.D == 0.0 and nc.D_se == 0.0


def test_collapse_matches_mixture_moments():
    comps = [(0.25, np.array([0.0, 1.0]), np.eye(2)), (0.75, np.array([2.0, 1.0]), 2 * np.eye(2))]
    m, C = ss._collapse(comps)
    assert np.allclose(m, [1.5, 1.0])
    assert C[0, 0] == pytest.approx(0.25 * 1 + 0.75 * 2 + 0.25 * 1.5 ** 2 + 0.75 * 0.5 ** 2)
    assert C[1, 1] == pytest.approx(1.75)


def test_jump_model_fits_a_flat_world_with_one_jump_better_than_a_random_walk():
    jump_day = START + timedelta(days=3 * 365 + 100)
    events = _simulate(0.0, {"市区町村議会": -0.1}, 0.05, national_every_days=365,
                       n_days=6 * 365, jump=(jump_day, 0.12), seed=3)
    obs = ss.party_observations(events, "A", START)
    _, ll_rw = ss.fit_params(obs, maxiter=100, slope="none")
    p_jump, ll_jump = ss.fit_params(obs, maxiter=100, slope="none", jumps=True)
    assert ll_jump > ll_rw
    assert p_jump.jump_var > 10 * p_jump.q_x


def test_outlier_component_recovers_contamination_and_protects_the_level():
    rng = np.random.default_rng(5)
    events = _simulate(0.0, {"市区町村議会": -0.1}, 0.03, seed=4)
    contaminated = []
    for e in events:
        if e.level == "市区町村議会" and rng.random() < 0.1:
            y = 1.0
            e = Event(e.vote_date, e.level, e.P, e.n, {"A": y, "B": 0.0}, "")
        contaminated.append(e)
    obs = ss.party_observations(contaminated, "A", START)
    p_plain, ll_plain = ss.fit_params(obs, maxiter=100, slope="none")
    p_out, ll_out = ss.fit_params(obs, maxiter=100, slope="none", outliers=True)
    assert ll_out > ll_plain
    assert p_out.outlier_prob == pytest.approx(0.1, abs=0.04)
    assert p_out.sigma2["市区町村議会"] < p_plain.sigma2["市区町村議会"]


def test_binomial_variance_discounts_shares_computed_from_few_party_seats():
    day = date(2020, 1, 1)
    base = [Event(day + timedelta(days=i), "市区町村議会", 1e5, 20, {"A": 0.2, "B": 0.8}, "") for i in range(30)]
    tiny = Event(day + timedelta(days=40), "市区町村議会", 1e5, 12, {"無所属": 11 / 12, "A": 1 / 12}, "")
    obs = ss.party_observations(base + [tiny], "A", day)
    assert obs.n_eff[-1] == pytest.approx(1.0)
    p = ss.Params(q_x=1e-6, q_b=0.0, sigma2={"市区町村議会": 1e-4}, a=0.0, slope="none")
    shift = {}
    for flag in (False, True):
        p.binomial = flag
        before = ss.run_filter(obs, p, until=ss._t(tiny.vote_date) - 0.01).mean
        after = ss.run_filter(obs, p).mean
        shift[flag] = after[0] + after[2] - (before[0] + before[2])
    assert 0 < shift[True] < 0.2 * shift[False]


def test_rts_smoother_bridges_two_national_elections():
    from datetime import date, timedelta
    from seiryoku.trend import Event
    d0 = date(2020, 1, 1)
    events = [Event(d0, "衆院選", 1.2e8, 465.0, {"A": 0.2, "B": 0.8}, ""),
              Event(d0 + timedelta(days=730), "衆院選", 1.2e8, 465.0, {"A": 0.4, "B": 0.6}, "")]
    obs = ss.party_observations(events, "A", d0)
    params = ss.Params(q_x=1e-3, q_b=0.0, sigma2={}, a=0.0, slope="none")
    trace = []
    ss.run_filter(obs, params, trace=trace)
    t_mid = (obs.t[0] + obs.t[1]) / 2
    (m, C), = ss.rts_smooth(trace, params, [], [t_mid])
    r = ss.SIGMA_NATIONAL ** 2
    assert m[0] == pytest.approx(0.3, abs=1e-3)
    assert C[0, 0] == pytest.approx(1e-3 * (obs.t[1] - obs.t[0]) / 4 + r / 2, rel=0.05)
