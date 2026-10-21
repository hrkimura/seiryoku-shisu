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


def _v2_result():
    def series(xs):
        return [{"tau": t, "x": x, "x_f": x} for t, x in zip(["2026-08-31", "2026-09-30", "2026-10-31"], xs)]
    return {"parties": {
        "自由民主党": {"series": series([0.50, 0.52, 0.55])},
        "参政党": {"series": series([0.05, 0.04, 0.03])},
        "立憲民主党": {"series": series([0.10, 0.11, 0.10])},
        "日本共産党": {"series": series([0.03, 0.03, 0.04])},
        "公明党": {"series": series([0.04, 0.04, 0.04])},
    }}


_SUPPORT = [
    {"updated": "2026-10-06", "S_p": {"無党派": 40.0, "自由民主党": 30.0, "参政党": 5.0, "立憲民主党": 3.0,
                                      "日本共産党": 2.0, "チームみらい": 2.0}},
    {"updated": "2026-11-05", "S_p": {"無党派": 40.0, "自由民主党": 32.0, "参政党": 4.0, "立憲民主党": 3.0,
                                      "日本共産党": 1.0, "チームみらい": 2.0}},
]


def test_trend_support_diffs_pairs_month_ends():
    d = sa.trend_support_diffs(_v2_result(), _SUPPORT)
    # 8月末にそろえられる支持率はなく、公明党とチームみらいは片方にしかないので、9月末→10月末の4党だけ
    assert d["periods"] == [("2026-09-30", "2026-10-31")]
    assert sorted(p["party"] for p in d["points"]) == ["参政党", "日本共産党", "立憲民主党", "自由民主党"]
    ldp = next(p for p in d["points"] if p["party"] == "自由民主党")
    assert abs(ldp["dx"] - 3.0) < 1e-9
    # 無党派を除いた合計は9月末が42、10月末が(10/06と11/05の補間)、自民の割合の差
    w = 25 / 30
    oct_ldp = (1 - w) * 30 + w * 32
    oct_total = 42.0 + w * (42.0 - 42.0)
    assert abs(ldp["ds"] - (oct_ldp / oct_total * 100 - 30 / 42 * 100)) < 1e-9
    assert d["alignment"]["2026-09-30"]["method"] == "nearest"
    assert d["alignment"]["2026-10-31"]["method"] == "interpolated"
    assert d["n"] == 4 and d["pearson"]["ci95"] is not None and d["spearman"] is not None


def test_trend_support_diffs_without_two_month_ends():
    d = sa.trend_support_diffs(_v2_result(), _SUPPORT[:1])
    assert d["points"] == [] and d["pearson"] is None and d["spearman"] is None


def test_trend_support_diffs_longer_horizon_uses_non_overlapping_periods():
    taus = ["2026-04-30", "2026-05-31", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30", "2026-10-31"]
    result = {"parties": {p: {"series": [{"tau": t, "x": 0.1 + k * i / 100, "x_f": 0.1 + k * i / 100}
                                         for i, t in enumerate(taus)]}
                          for k, p in enumerate(["自由民主党", "参政党"], start=1)}}
    history = [{"updated": t, "S_p": {"無党派": 40.0, "自由民主党": 30.0 + i, "参政党": 5.0}}
               for i, t in enumerate(taus)]
    d = sa.trend_support_diffs(result, history, horizon=3)
    # 最新の10月末から3か月ずつ区切る。4月末の端は残っても使わない区切り方にならないこと
    assert d["periods"] == [("2026-04-30", "2026-07-31"), ("2026-07-31", "2026-10-31")]
    assert d["horizon"] == 3 and d["n"] == 4
    ldp = [p for p in d["points"] if p["party"] == "自由民主党"]
    assert all(abs(p["dx"] - 3.0) < 1e-9 for p in ldp)
    assert sa.trend_support_diffs(result, history, horizon=6)["periods"] == [("2026-04-30", "2026-10-31")]


def test_trend_support_diffs_keeps_only_the_largest_parties(monkeypatch):
    monkeypatch.setattr(sa, "MAX_DIFF_PARTIES", 3)
    d = sa.trend_support_diffs(_v2_result(), _SUPPORT)
    # 10月末のv2の値は自民0.55、立憲0.10、共産0.04、公明0.04、参政0.03で、同じ値は元の並び順を保つ
    assert sorted(p["party"] for p in d["points"]) == ["日本共産党", "立憲民主党", "自由民主党"]
