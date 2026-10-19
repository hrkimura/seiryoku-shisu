import math
from datetime import date, timedelta

import numpy as np
import pytest

from seiryoku import trend
from seiryoku.monthly_index import MonthEntries, _level_aggregate
from seiryoku.trend import Event, fit_trends, first_appearance, normalized_shares

TAU = date(2026, 9, 30)


def _ev(days_before, level, share_a, P=1e5, n=1.0, name="x"):
    shares = {"A": share_a, "B": 1.0 - share_a}
    return Event(TAU - timedelta(days=days_before), level, P, n, shares, name)


def _linear_events(slope_per_month, levels_offsets, n_per_level=60, rng=None):
    """各階層に水準の違う直線の趨勢を持たせたイベント列。投票日は階層ごとにずらす。"""
    events = []
    for li, (level, offset) in enumerate(levels_offsets.items()):
        for i in range(n_per_level):
            days = 7 * i + li
            months_ago = days / trend.DAYS_PER_MONTH
            a = 0.4 + offset - slope_per_month * months_ago
            if rng is not None:
                a += rng.normal(0, 0.02)
            events.append(_ev(days, level, a, P=1e5 * (1 + (i % 7))))
    events.sort(key=lambda e: e.vote_date)
    return events


def test_recovers_known_slope_despite_level_differences():
    events = _linear_events(0.003, {"市区町村長": 0.2, "市区町村議会": -0.1})
    fits = fit_trends(events, TAU, 12, ["A", "B"], first_dates={"A": date(2000, 1, 1), "B": date(2000, 1, 1)})
    assert fits["A"].D == pytest.approx(0.003, abs=1e-9)
    assert fits["B"].D == pytest.approx(-0.003, abs=1e-9)


def test_level_composition_shift_does_not_create_a_slope():
    """前半は水準の高い階層、後半は低い階層だけ、という偏りがあっても傾きは0。"""
    events = []
    for i in range(40):
        events.append(_ev(300 + 5 * i, "市区町村長", 0.5))
        events.append(_ev(5 * i, "市区町村議会", 0.3))
        events.append(_ev(150 + 5 * i, "市区町村長", 0.5))
        events.append(_ev(150 + 5 * i + 2, "市区町村議会", 0.3))
    events.sort(key=lambda e: e.vote_date)
    fits = fit_trends(events, TAU, 12, ["A"], first_dates={"A": date(2000, 1, 1)})
    assert fits["A"].D == pytest.approx(0.0, abs=1e-12)


def test_single_event_level_does_not_affect_slope():
    rng = np.random.default_rng(1)
    base = _linear_events(0.002, {"市区町村長": 0.0, "市区町村議会": 0.1}, rng=rng)
    first = {"A": date(2000, 1, 1)}
    without = fit_trends(base, TAU, 12, ["A"], first_dates=first)["A"]
    shugiin = Event(TAU - timedelta(days=200), "衆院選", 1.2e8, 465, {"A": 0.95, "B": 0.05}, "衆院選")
    with_ = fit_trends(sorted(base + [shugiin], key=lambda e: e.vote_date), TAU, 12, ["A"], first_dates=first)["A"]
    assert with_.D == pytest.approx(without.D, abs=1e-12)


def test_slopes_sum_to_zero_across_parties():
    rng = np.random.default_rng(2)
    events = _linear_events(0.001, {"市区町村長": 0.0, "市区町村議会": 0.1}, rng=rng)
    fits = fit_trends(events, TAU, 9, ["A", "B"], first_dates={"A": date(2000, 1, 1), "B": date(2000, 1, 1)})
    assert fits["A"].D + fits["B"].D == pytest.approx(0.0, abs=1e-12)


def test_contributions_sum_to_D():
    rng = np.random.default_rng(3)
    events = _linear_events(0.001, {"市区町村長": 0.0, "市区町村議会": 0.1}, rng=rng)
    fit = fit_trends(events, TAU, 12, ["A"], first_dates={"A": date(2000, 1, 1)}, keep_details=True)["A"]
    assert sum(v for v, _ in fit.contributions) == pytest.approx(fit.D, abs=1e-12)


def test_level_only_fit_with_window_kernel_reproduces_current_index():
    """Kを12か月の窓に、次数を0にすると、階層の切片をW_ellで合成した値が現行のC_pになる。"""
    rng = np.random.default_rng(4)
    events = []
    for level, n in [("市区町村長", 1.0), ("市区町村議会", 20.0), ("都道府県知事", 1.0)]:
        for i in range(15):
            a = float(rng.uniform(0, 1))
            events.append(Event(TAU - timedelta(days=int(rng.integers(0, 360))), level,
                                float(rng.uniform(1e4, 1e6)), n, {"A": a, "B": 1 - a}, ""))
    events.sort(key=lambda e: e.vote_date)
    # イベントは全て窓内に置き、hを非常に大きくしてK(s)≈1にする
    fits = fit_trends(events, TAU, 1e9, ["A"], degree=0, first_dates={"A": date(2000, 1, 1)})
    C_p, share = _level_aggregate([(e.level, e.P, e.n, e.shares) for e in events])
    alpha = fits["A"].alpha
    combined = sum(share[lv] * alpha[lv] for lv in alpha)
    assert combined == pytest.approx(C_p["A"], abs=1e-9)


def test_normalized_shares_excludes_independents():
    assert normalized_shares({"無所属": 0.5, "A": 0.25, "B": 0.25}) == {"A": 0.5, "B": 0.5}
    assert normalized_shares({"無所属": 1.0}) is None


def test_party_is_excluded_before_its_first_appearance():
    events = [_ev(100, "市区町村長", 0.0), _ev(50, "市区町村長", 0.0)]
    events.append(Event(TAU - timedelta(days=10), "市区町村長", 1e5, 1, {"C": 1.0}, ""))
    first = first_appearance(sorted(events, key=lambda e: e.vote_date))
    assert first["C"] == TAU - timedelta(days=10)
    assert "C" not in fit_trends(sorted(events, key=lambda e: e.vote_date), TAU - timedelta(days=20), 12, ["C"], first_dates=first)


def test_unclassified_mayor_elections_get_their_own_level():
    me = MonthEntries(
        entries={(2026, 1): [("市区町村長", 1e5, 1.0, {"A": 1.0}), ("市区町村長", 1e5, 1.0, {"B": 1.0})]},
        labels={(2026, 1): [{"name": "x", "vote_date": "2025-12-20"}, {"name": "y", "vote_date": "2026-01-18"}]},
        months=[(2026, 1)],
    )
    events = trend.events_from_month_entries(me, classified_until=date(2025, 12, 22))
    assert [e.level for e in events] == ["市区町村長", trend.UNCLASSIFIED_MAYOR]


def test_wild_bootstrap_p_value_is_small_for_a_strong_trend_and_large_for_none():
    rng = np.random.default_rng(5)
    first = {"A": date(2000, 1, 1)}
    strong = _linear_events(0.01, {"市区町村長": 0.0, "市区町村議会": 0.1}, rng=rng)
    flat = _linear_events(0.0, {"市区町村長": 0.0, "市区町村議会": 0.1}, rng=rng)
    p_strong = fit_trends(strong, TAU, 12, ["A"], first_dates=first, bootstrap=499)["A"].p_wild
    p_flat = fit_trends(flat, TAU, 12, ["A"], first_dates=first, bootstrap=499)["A"].p_wild
    assert p_strong < 0.01
    assert p_flat > 0.01


def test_backtest_prefers_trend_model_when_data_has_a_trend():
    rng = np.random.default_rng(6)
    events = _linear_events(0.01, {"市区町村長": 0.0, "市区町村議会": 0.1}, n_per_level=100, rng=rng)
    taus = [TAU - timedelta(days=d) for d in (120, 150, 180)]
    bt = trend.backtest(events, ["A"], taus, [12], [0, 1])
    assert bt["mean_loss"]["h=12,deg=1"] < bt["mean_loss"]["h=12,deg=0"]


@pytest.mark.parametrize("mode", trend.WEIGHT_MODES)
def test_every_weight_mode_recovers_slope_and_sums_to_zero(mode):
    events = _linear_events(0.002, {"市区町村長": 0.2, "市区町村議会": -0.1})
    first = {"A": date(2000, 1, 1), "B": date(2000, 1, 1)}
    fits = fit_trends(events, TAU, 12, ["A", "B"], first_dates=first, weights=mode)
    assert fits["A"].D == pytest.approx(0.002, abs=1e-9)
    assert fits["A"].D + fits["B"].D == pytest.approx(0.0, abs=1e-12)


def test_fgls_recovers_population_exponent_of_the_noise():
    """誤差の分散がP^{-1/2}に比例するデータでは、推定される指数aが0.5に近い。"""
    rng = np.random.default_rng(7)
    events = []
    for i in range(3000):
        P = float(np.exp(rng.uniform(np.log(1e3), np.log(1e6))))
        sd = 0.3 * (P / 1e3) ** -0.25
        a = 0.5 + rng.normal(0, sd)
        events.append(Event(TAU - timedelta(days=int(rng.integers(0, 700))), "市区町村議会", P, 20.0,
                            {"A": a, "B": 1 - a}, ""))
    events.sort(key=lambda e: e.vote_date)
    fit = fit_trends(events, TAU, 12, ["A"], first_dates={"A": date(2000, 1, 1)}, weights="fgls")["A"]
    assert fit.a_hat == pytest.approx(0.5, abs=0.1)


def test_separate_national_levels_gives_each_national_election_its_own_level():
    events = [_ev(10, "衆院選", 0.5), _ev(400, "衆院選", 0.3), _ev(20, "市区町村長", 0.4)]
    levels = {e.level for e in trend.separate_national_levels(events)}
    assert len([lv for lv in levels if lv.startswith("衆院選@")]) == 2
    assert "市区町村長" in levels


def test_alliance_seats_are_returned_to_the_original_parties():
    d = date(2026, 2, 8)
    shu = trend.Event(d, "衆院選", 1e8, 465.0, {"自由民主党": 0.6, "中道改革連合": 0.098}, "")
    gov = trend.Event(d, "都道府県知事", 1e6, 1.0, {"中道改革連合": 1.0}, "x県")
    a, b = trend.split_alliances([shu, gov])
    assert a.shares["立憲民主党"] == pytest.approx(0.098 * 21 / 49)
    assert a.shares["公明党"] == pytest.approx(0.098 * 28 / 49)
    assert "中道改革連合" not in a.shares and a.shares["自由民主党"] == 0.6
    assert b.shares == {"立憲民主党": 0.5, "公明党": 0.5}
