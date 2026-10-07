import numpy as np

from seiryoku import support_analysis as sa


def _months(n, start=(2020, 1)):
    y, m = start
    out = []
    for _ in range(n):
        out.append((y, m))
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


def _inputs(c, s, party="A"):
    months = _months(len(c))
    cp_history = [{"year": y, "month": m, "C_p": {party: v / 100}} for (y, m), v in zip(months, c)]
    support = {ym: {party: v} for ym, v in zip(months, s)}
    return cp_history, support


def test_contiguous_pairs_uses_latest_unbroken_run_and_percent_scale():
    cp_history, support = _inputs([10, 20, 30, 40], [1, 2, 3, 4])
    del support[(2020, 2)]
    pairs = sa.contiguous_pairs(cp_history, support, "A")
    assert [p[0] for p in pairs] == [(2020, 3), (2020, 4)]
    assert pairs[0][1] == 30.0


def test_short_series_is_reported_as_insufficient():
    cp_history, support = _inputs([10, 11], [1, 2])
    r = sa.analyze_all(cp_history, support)["A"]
    assert r["status"] == "insufficient"
    assert r["primary"] is None
    assert r["level_corr"] is None


def test_stationary_pair_is_analyzed_in_levels():
    rng = np.random.default_rng(0)
    common = rng.normal(size=120)
    c = 10 + common + 0.5 * rng.normal(size=120)
    s = 5 + common + 0.5 * rng.normal(size=120)
    r = sa.analyze_party(sa.contiguous_pairs(*_inputs(c, s), "A"))
    assert r["status"] == "I(0)-I(0)"
    assert r["primary"] == "level"
    assert r["level_corr"]["r"] > 0.5


def test_cointegrated_random_walks_are_detected_with_beta():
    rng = np.random.default_rng(1)
    s = 20 + np.cumsum(rng.normal(size=150))
    c = 3 + 2.0 * s + rng.normal(size=150)
    r = sa.analyze_party(sa.contiguous_pairs(*_inputs(c, s), "A"))
    assert r["status"] == "I(1)-I(1) cointegrated"
    assert r["primary"] == "level"
    assert abs(r["cointegration"]["beta"] - 2.0) < 0.1


def test_independent_random_walks_fall_back_to_differences():
    rng = np.random.default_rng(5)
    s = 50 + np.cumsum(rng.normal(size=150))
    c = 50 + np.cumsum(rng.normal(size=150))
    r = sa.analyze_party(sa.contiguous_pairs(*_inputs(c, s), "A"))
    assert r["status"] == "I(1)-I(1) not cointegrated"
    assert r["primary"] == "difference"
    assert abs(r["diff_corr"]["r"]) < 0.3


def test_lag_peak_finds_support_leading_by_two_months():
    rng = np.random.default_rng(3)
    ds = rng.normal(size=150)
    dc = np.roll(ds, 2) + 0.3 * rng.normal(size=150)
    s, c = 50 + np.cumsum(ds), 50 + np.cumsum(dc)
    r = sa.analyze_party(sa.contiguous_pairs(*_inputs(c, s), "A"))
    assert r["lag_peak"] == 2
    assert r["lag_corr"]["2"]["r"] > 0.8


def test_analyze_all_skips_independents_and_parties_missing_on_one_side():
    months = _months(3)
    cp_history = [{"year": y, "month": m, "C_p": {"A": 0.1, "無所属": 0.9}} for y, m in months]
    support = {ym: {"A": 1.0, "無党派": 40.0, "B": 2.0} for ym in months}
    assert list(sa.analyze_all(cp_history, support)) == ["A"]


def test_provisional_months_are_counted():
    cp_history, support = _inputs([1, 2, 3], [1, 2, 3])
    r = sa.analyze_party(sa.contiguous_pairs(cp_history, support, "A"), provisional_start=(2020, 2))
    assert r["n_provisional_months"] == 2
