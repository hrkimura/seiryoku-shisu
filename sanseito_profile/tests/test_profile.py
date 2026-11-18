import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import party_distance as pdist  # noqa: E402
import profile_utas as pu  # noqa: E402


def test_wilson_known_values():
    lo, hi = pu.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = pu.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-4)


def test_wilson_empty_is_nan():
    assert all(math.isnan(v) for v in pu.wilson(0, 0))


def test_newcombe_method10_matches_paper_example():
    # Newcombe (1998) Stat Med 17:873 の例 56/70 - 48/80、方法10は 0.0524〜0.3339
    d, lo, hi = pu.newcombe_diff(56, 70, 48, 80)
    assert d == pytest.approx(0.2, abs=1e-12)
    assert lo == pytest.approx(0.0524, abs=1e-4)
    assert hi == pytest.approx(0.3339, abs=1e-4)


def test_newcombe_is_antisymmetric():
    d1, lo1, hi1 = pu.newcombe_diff(9, 30, 20, 40)
    d2, lo2, hi2 = pu.newcombe_diff(20, 40, 9, 30)
    assert d1 == pytest.approx(-d2)
    assert lo1 == pytest.approx(-hi2)
    assert hi1 == pytest.approx(-lo2)


def test_holm_hand_computed():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert pu.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_nan_and_caps_at_one():
    out = pu.holm(np.array([0.5, np.nan, 0.4]))
    assert math.isnan(out[1])
    assert out[0] == pytest.approx(0.8)
    assert out[2] == pytest.approx(0.8)
    assert pu.holm(np.array([0.6, 0.7])) == pytest.approx([1.0, 1.0])


def test_indicator_default_excludes_nonresponse_only():
    it = pu.Item("g", "x", "c", frozenset({1, 2}))
    out = pu.indicator(pd.Series([1, 2, 3, 99, np.nan]), it)
    assert out.iloc[:3].tolist() == [1.0, 1.0, 0.0]
    assert out.iloc[3:].isna().all()


def test_indicator_explicit_valid_set():
    it = pu.Item("g", "x", "c", frozenset({3}), frozenset({1, 2, 3, 4}))
    out = pu.indicator(pd.Series([3, 4, 66, 99]), it)
    assert out.iloc[:2].tolist() == [1.0, 0.0]
    assert out.iloc[2:].isna().all()


def test_age_group_boundaries():
    g = pu.age_group(pd.Series([1, 3, 4, 5, 6, 9, 10, 11, 12, 15, 99]))
    assert g.tolist()[:10] == ["18-29", "18-29", "30s", "30s", "40s", "50s", "60s", "60s", "70+", "70+"]
    assert pd.isna(g.iloc[10])


def _synthetic(n_s=60, n_o=240, p_s=0.7, p_o=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_s + n_o
    return pd.DataFrame({
        "sanseito": [1] * n_s + [0] * n_o,
        "Q4": [6] * n_s + list(rng.choice([1, 2, 3], n_o)),
        "age_g": rng.choice(["18-29", "40s", "70+"], n),
        "male": rng.integers(0, 2, n),
        "X": np.r_[rng.random(n_s) < p_s, rng.random(n_o) < p_o].astype(int) + 1,
    })


def test_profile_table_counts_and_direction():
    df = _synthetic()
    df.loc[0, "X"] = 99
    item = pu.Item("g", "賛成", "X", frozenset({2}))
    t = pu.profile_table(df, [item], adjust=False)
    r = t.iloc[0]
    assert r.n_s == 59 and r.n_o == 240
    assert r.p_s == pytest.approx((df.loc[1:59, "X"] == 2).mean())
    assert r["diff"] > 0.2 and r["diff_lo"] > 0
    assert r.p_holm == pytest.approx(r.p)


def test_adjusted_or_detects_effect_and_refuses_sparse():
    df = _synthetic(n_s=150, n_o=450, p_s=0.8, p_o=0.2)
    y = (df["X"] == 2).astype(float)
    or_, lo, hi, p = pu.adjusted_or(y, df)
    assert lo > 1 and or_ > 5 and p < 1e-6
    y2 = y.copy()
    y2[df.sanseito == 1] = 0
    y2.iloc[0] = 1
    assert all(math.isnan(v) for v in pu.adjusted_or(y2, df))


def test_rms_distance_in_points():
    assert pdist.rms_distance(np.array([0.2, 0.5]), np.array([0.6, 0.5])) == pytest.approx(100 * math.sqrt(0.08))


def test_unbiased_distance_subtracts_sampling_noise():
    x = pd.DataFrame({"a": [1, 0, 1, 0, 1, 1, 0, 0]})
    party = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
    # p1=p2=0.5 なので補正後の二乗差は -(0.25/3)*2、符号つきの平方根を返す
    assert pdist.unbiased_distance(x, party, 1, 2) == pytest.approx(-100 * math.sqrt(0.5 / 3))


def test_unbiased_distance_close_to_truth_for_large_groups():
    rng = np.random.default_rng(3)
    n = 20000
    x = pd.DataFrame({"a": np.r_[rng.random(n) < 0.2, rng.random(n) < 0.6].astype(float),
                      "b": np.r_[rng.random(n) < 0.5, rng.random(n) < 0.5].astype(float)})
    party = pd.Series([1] * n + [2] * n)
    assert pdist.unbiased_distance(x, party, 1, 2) == pytest.approx(100 * math.sqrt(0.16 / 2), abs=1.0)


def test_pair_distances_all_pairs():
    sh = pd.DataFrame({"a": [0.1, 0.5, 0.9]}, index=[1, 2, 3])
    d = pdist.pair_distances(sh)
    assert set(d) == {(1, 2), (1, 3), (2, 3)}
    assert d[(1, 3)] == pytest.approx(80)


def test_item_labels_used_by_distance_exist():
    labels = {it.label for it in pu.ITEMS}
    assert set(pdist.DEMO) <= labels and set(pdist.ATTITUDE) <= labels
    assert len(pdist.ATTITUDE) == 20


def test_range_set_is_inclusive():
    assert pu._r(1, 3) == frozenset({1, 2, 3})
    assert pu.AGREE == frozenset({1, 2})


def test_age_group_every_code():
    codes = list(range(1, 16))
    expect = ["18-29"] * 3 + ["30s"] * 2 + ["40s"] * 2 + ["50s"] * 2 + ["60s"] * 2 + ["70+"] * 4
    assert pu.age_group(pd.Series(codes)).tolist() == expect
    assert pd.isna(pu.age_group(pd.Series([16])).iloc[0])


def test_holm_accepts_integer_list():
    assert pu.holm([0, 1]).tolist() == [0.0, 1.0]


def test_adjusted_or_interval_matches_statsmodels():
    import statsmodels.formula.api as smf
    df = _synthetic(n_s=150, n_o=450, p_s=0.6, p_o=0.4, seed=5)
    y = (df["X"] == 2).astype(float)
    or_, lo, hi, p = pu.adjusted_or(y, df)
    d = df.assign(y=y)
    fit = smf.logit("y ~ sanseito + C(age_g) + male", data=d).fit(disp=0)
    b, se = fit.params["sanseito"], fit.bse["sanseito"]
    assert or_ == pytest.approx(math.exp(b))
    assert lo == pytest.approx(math.exp(b - 1.96 * se))
    assert hi == pytest.approx(math.exp(b + 1.96 * se))
    assert p == pytest.approx(fit.pvalues["sanseito"])


def test_adjusted_or_sparse_threshold_both_sides():
    df = _synthetic(n_s=150, n_o=450, p_s=0.5, p_o=0.5, seed=6)
    s = df.sanseito == 1
    for n_rare in (2, 3):
        for rare_value in (1.0, 0.0):
            y = (df["X"] == 2).astype(float)
            y[s] = 1.0 - rare_value
            y[s & (np.cumsum(s) <= n_rare)] = rare_value
            out = pu.adjusted_or(y, df)
            assert len(out) == 4
            assert all(math.isnan(v) for v in out) == (n_rare < 3)


def test_profile_table_fisher_p_and_default_adjust():
    from scipy import stats
    df = _synthetic(seed=7)
    item = pu.Item("g", "賛成", "X", frozenset({2}))
    t = pu.profile_table(df, [item])
    r = t.iloc[0]
    k1, n1 = int((df.loc[df.sanseito == 1, "X"] == 2).sum()), int((df.sanseito == 1).sum())
    k2, n2 = int((df.loc[df.sanseito == 0, "X"] == 2).sum()), int((df.sanseito == 0).sum())
    assert r.p == pytest.approx(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    assert "or_adj" in t.columns and "p_adj_holm" in t.columns


def test_profile_table_every_column():
    df = _synthetic(n_s=150, n_o=450, p_s=0.6, p_o=0.4, seed=8)
    items = [pu.Item("属性", "賛成", "X", frozenset({2})), pu.Item("属性", "反対", "X", frozenset({1}))]
    t = pu.profile_table(df, items)
    assert t["group"].tolist() == ["属性", "属性"] and t["label"].tolist() == ["賛成", "反対"]
    r = t.iloc[0]
    s = df.sanseito == 1
    k1, n1 = int((df.loc[s, "X"] == 2).sum()), int(s.sum())
    k2, n2 = int((df.loc[~s, "X"] == 2).sum()), int((~s).sum())
    d, lo, hi = pu.newcombe_diff(k1, n1, k2, n2)
    assert (r.n_s, r.n_o) == (n1, n2)
    assert (r.p_s, r.p_o) == (pytest.approx(k1 / n1), pytest.approx(k2 / n2))
    assert (r["diff"], r.diff_lo, r.diff_hi) == (pytest.approx(d), pytest.approx(lo), pytest.approx(hi))
    expect = pu.adjusted_or((df["X"] == 2).astype(float), df)
    assert (r.or_adj, r.or_lo, r.or_hi, r.p_adj) == tuple(pytest.approx(v) for v in expect)
    assert t["p_adj_holm"].tolist() == pytest.approx(pu.holm(t["p_adj"].to_numpy()).tolist())


def _three_parties(monkeypatch, n=300, seed=9):
    """党1・6・8。属性は6と8が同じで1だけ違い、態度は6と8が正反対。"""
    rng = np.random.default_rng(seed)
    male = {1: 0.9, 6: 0.5, 8: 0.5}
    defense = {1: 0.5, 6: 0.9, 8: 0.1}
    q4 = np.repeat([1, 6, 8], n)
    df = pd.DataFrame({"Q4": q4,
                       "Q39": [1 if rng.random() < male[p] else 2 for p in q4],
                       "Q35_1": [1 if rng.random() < defense[p] else 5 for p in q4]})
    monkeypatch.setattr(pdist, "PARTIES", [1, 6, 8])
    monkeypatch.setattr(pdist, "DEMO", ["男性"])
    monkeypatch.setattr(pdist, "ATTITUDE", ["防衛力を強化すべき"])
    return df


def test_rank_probability_recovers_twin_pair(monkeypatch):
    df = _three_parties(monkeypatch)
    r = pdist.rank_probability(df, (6, 8), n_boot=50)
    assert r["demo_rank_median"] == 1 and r["att_rank_median"] == 1
    assert r["att_top3"] == 1.0 and r["att_top5"] == 1.0
    assert r["demo_rank1"] > 0.6 and r["demo_top3"] == 1.0
    r2 = pdist.rank_probability(df, (1, 6), n_boot=50)
    assert r2["demo_rank1"] < 0.4 and r2["att_rank_median"] > 1


def test_bootstrap_point_and_interval(monkeypatch):
    df = _three_parties(monkeypatch)
    t = pdist.bootstrap(df, n_boot=50).set_index("pair")
    assert t.loc["自民-参政", "demo"] == pytest.approx(
        pdist.rms_distance(np.array([(df[df.Q4 == 1].Q39 == 1).mean()]), np.array([(df[df.Q4 == 6].Q39 == 1).mean()])))
    assert t.loc["参政-れいわ", "att"] > 70
    assert (t["demo_lo"] <= t["demo_hi"]).all() and (t["att_lo"] <= t["att"] + 1e-9).all()
    assert (t["att"] <= t["att_hi"] + 1e-9).all()


def test_resample_within_keeps_party_sizes_and_draws_with_replacement():
    party = np.array([1] * 200 + [6] * 50 + [8] * 10)
    idx = pdist.resample_within(party, [1, 6, 8], np.random.default_rng(0))
    assert len(idx) == len(party)
    for p, n in ((1, 200), (6, 50), (8, 10)):
        assert (party[idx] == p).sum() == n
    assert len(np.unique(idx[:200])) < 200


def test_interval95_percentiles():
    assert pdist.interval95(np.arange(1001)) == pytest.approx((25.0, 975.0))


def test_summarize_ranks_thresholds():
    r = pdist.summarize_ranks(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6]))
    assert r["demo_rank1"] == 0.25 and r["demo_top3"] == 0.75 and r["demo_rank_median"] == 2.5
    assert r["att_top3"] == 0.25 and r["att_top5"] == 0.75 and r["att_rank_median"] == 4.5


def test_clip_at_zero_only_distance_columns():
    import figures
    t = pd.DataFrame({"a": [6, 1], "b": [8, 2], "demo": [-3.7, 5.0], "demo_lo": [-5.4, 2.0], "demo_hi": [13.4, 8.0],
                      "att": [21.9, -1.0], "att_lo": [17.4, -2.0], "att_hi": [29.0, 3.0]})
    c = figures.clip_at_zero(t)
    assert c["demo"].tolist() == [0.0, 5.0] and c["demo_lo"].tolist() == [0.0, 2.0]
    assert c["att"].tolist() == [21.9, 0.0] and c["att_hi"].tolist() == [29.0, 3.0]
    assert c["a"].tolist() == [6, 1] and t["demo"].iloc[0] == -3.7
