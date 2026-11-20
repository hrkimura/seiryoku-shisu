import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import figures  # noqa: E402
import lca  # noqa: E402
import subgroups as sg  # noqa: E402


def test_encode_maps_answers_and_missing():
    df = pd.DataFrame({c: [1, 5, 99, np.nan] for c in lca.ITEMS})
    x = lca.encode(df)
    assert x.dtype.kind == "i"
    assert x[:, 0].tolist() == [0, 4, -1, -1]


def test_onehot_rows_and_missing():
    x = np.array([[0, -1], [4, 2]])
    oh = lca._onehot(x)
    assert oh.shape == (2, 2, 5)
    assert oh[0, 0].tolist() == [1, 0, 0, 0, 0]
    assert oh[0, 1].sum() == 0
    assert oh[1, 0, 4] == 1 and oh[1, 1, 2] == 1 and oh.sum() == 3


def _two_class_data(n=600, j=8, frac=0.3, missing=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < frac).astype(int)
    probs = {0: [0.7, 0.2, 0.06, 0.02, 0.02], 1: [0.02, 0.02, 0.06, 0.2, 0.7]}
    x = np.array([[rng.choice(5, p=probs[zi]) for _ in range(j)] for zi in z])
    x[rng.random(x.shape) < missing] = -1
    return x, z


@pytest.mark.parametrize("missing", [0.0, 0.2])
def test_em_recovers_two_classes(missing):
    x, z = _two_class_data(missing=missing)
    ll, pi, theta, resp = lca.fit(x, 2, n_starts=3)
    labels = resp.argmax(axis=1)
    assert adjusted_rand_score(z, labels) > 0.95
    assert sorted(pi) == pytest.approx([0.3, 0.7], abs=0.05)
    assert np.allclose(theta.sum(axis=2), 1, atol=1e-6)
    assert np.allclose(resp.sum(axis=1), 1)
    hi = int(np.argmax(pi))
    assert theta[hi, :, 0].mean() == pytest.approx(0.7, abs=0.06)


def test_em_loglik_matches_direct_computation():
    x, _ = _two_class_data(n=200, j=4, seed=1)
    ll, pi, theta, _ = lca.em(x, 2, seed=3)
    oh = lca._onehot(x)
    logp = np.log(pi)[None, :] + np.einsum("njc,kjc->nk", oh, np.log(theta + lca.EPS))
    direct = float(np.log(np.exp(logp).sum(axis=1)).sum())
    # em は最後の更新の前の対数尤度を返すので、更新後の値以下で、差は収束判定の幅程度
    assert ll <= direct + 1e-6
    assert direct - ll < 1e-3


def test_fit_keeps_best_start():
    x, _ = _two_class_data(n=200, j=4, seed=2)
    best = lca.fit(x, 3, n_starts=4, seed=0)
    singles = [lca.em(x, 3, s)[0] for s in range(4)]
    assert best[0] == pytest.approx(max(singles))


def test_n_params_and_bic():
    assert lca.n_params(2) == 1 + 2 * len(lca.ITEMS) * 4
    assert lca.bic(-100.0, 2, 50) == pytest.approx(200 + lca.n_params(2) * math.log(50))


def test_sanseito_by_cluster_counts_voters_only():
    df = pd.DataFrame({"Q4": [6, 6, 1, 13, 6, 2, 2, 99]})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    t = lca.sanseito_by_cluster(df, labels).set_index("cluster")
    assert t.loc[0, "n"] == 4 and t.loc[0, "n_voted"] == 3 and t.loc[0, "n_sanseito"] == 2
    assert t.loc[1, "n_voted"] == 3 and t.loc[1, "n_sanseito"] == 1
    assert t.loc[0, "share"] == pytest.approx(2 / 3) and t.loc[0, "share_rest"] == pytest.approx(1 / 3)
    assert t.loc[0, "share_of_all_sanseito"] == pytest.approx(2 / 3)
    assert t.loc[0, "p"] == pytest.approx(stats.fisher_exact([[2, 1], [1, 2]])[1])
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())


def test_share_valid_sets():
    s = pd.Series([1, 2, 3, 99, np.nan])
    assert sg.share(s, {1}, None) == (1, 3)
    assert sg.share(s, {1, 2}, {1, 2}) == (2, 2)


def test_compare_values_and_fisher():
    a = pd.DataFrame({"Q2": [7, 7, 1, 99]})
    b = pd.DataFrame({"Q2": [1, 1, 7, 1]})
    t = sg.compare(a, b, {"参政党を支持": ("Q2", {7}, None)}).iloc[0]
    assert (t.n_a, t.p_a, t.n_b, t.p_b) == (3, pytest.approx(2 / 3), 4, pytest.approx(0.25))
    assert t.p == pytest.approx(stats.fisher_exact([[2, 1], [1, 3]])[1])


def test_compare_thermo_excludes_nonresponse():
    a = pd.DataFrame({c: [10, 8, 99] for c in sg.THERMO.values()})
    b = pd.DataFrame({c: [2, 4, 6] for c in sg.THERMO.values()})
    t = sg.compare_thermo(a, b)
    assert t["n_a"].tolist() == [2, 2, 2] and t["mean_a"].tolist() == [9.0, 9.0, 9.0]
    assert t["mean_b"].tolist() == [4.0, 4.0, 4.0]
    assert t["p"].iloc[0] == pytest.approx(stats.mannwhitneyu([10, 8], [2, 4, 6]).pvalue)


def test_mean_ci_t_interval():
    m, lo, hi = figures.mean_ci(pd.Series([1.0, 2.0, 3.0]))
    h = stats.t.ppf(0.975, 2) / math.sqrt(3)
    assert (m, lo, hi) == (2.0, pytest.approx(2 - h), pytest.approx(2 + h))


def test_encode_every_answer_code():
    df = pd.DataFrame({c: [1, 2, 3, 4, 5, 0, 6] for c in lca.ITEMS})
    assert lca.encode(df)[:, 3].tolist() == [0, 1, 2, 3, 4, -1, -1]


def test_sanseito_by_cluster_asymmetric_and_empty():
    q4 = [6, 6, 6, 1] + [6, 2, 2, 2, 2, 2, 2] + [13, 99]
    labels = np.array([0] * 4 + [1] * 7 + [2] * 2)
    t = lca.sanseito_by_cluster(pd.DataFrame({"Q4": q4}), labels).set_index("cluster")
    assert t.loc[0, "share_rest"] == pytest.approx(1 / 7)
    assert t.loc[0, "p"] == pytest.approx(stats.fisher_exact([[3, 1], [1, 6]])[1])
    assert t.loc[1, "p"] == pytest.approx(stats.fisher_exact([[1, 6], [3, 1]])[1])
    assert t.loc[2, "n_voted"] == 0 and math.isnan(t.loc[2, "share"])


def test_compare_uses_valid_set_and_columns():
    a = pd.DataFrame({"Q40": [6, 7, 10, 99, 16]})
    b = pd.DataFrame({"Q40": [10, 11, 6, 99, 20]})
    items = {"40〜59歳": ("Q40", set(range(6, 10)), set(range(1, 16)))}
    t = sg.compare(a, b, items)
    r = t.iloc[0]
    assert r["label"] == "40〜59歳" and (r.n_a, r.n_b) == (3, 3)
    assert r.p_a == pytest.approx(2 / 3) and r.p_b == pytest.approx(1 / 3)
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())


def test_compare_thermo_bounds_and_columns():
    a = pd.DataFrame({c: [0, 10, 11, -1] for c in sg.THERMO.values()})
    b = pd.DataFrame({c: [0, 0, 10, 11] for c in sg.THERMO.values()})
    t = sg.compare_thermo(a, b)
    assert t["label"].tolist() == list(sg.THERMO)
    assert t["n_a"].tolist() == [2] * 3 and t["mean_a"].tolist() == [5.0] * 3
    assert t["n_b"].tolist() == [3] * 3 and t["mean_b"].tolist() == pytest.approx([10 / 3] * 3)
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())


def test_composition_groups_and_other():
    s = pd.Series([7, 7, 1, 3, 14, 99, np.nan, 7])
    c = figures.composition(s, {"参政党": {7}, "自民党": {1}}, "その他")
    assert c.index.tolist() == ["参政党", "自民党", "その他"]
    assert c.tolist() == pytest.approx([37.5, 12.5, 50.0])
    empty = figures.composition(pd.Series([1, 1]), {"参政党": {7}, "自民党": {1}}, "その他")
    assert empty.tolist() == pytest.approx([0.0, 100.0, 0.0])


def test_survey_vs_actual_shares_among_voters():
    df = pd.DataFrame({"Q4": [1, 1, 6, 2, 13, 99]})
    t = figures.survey_vs_actual(df).set_index("party")
    assert t.loc["自民", "survey"] == pytest.approx(50.0)
    assert t.loc["参政", "survey"] == pytest.approx(25.0)
    assert t.loc["共産", "survey"] == 0.0
    assert t.loc["参政", "actual"] == 12.55 and len(t) == len(figures.ACTUAL_2025)
    lo, hi = figures.wilson(1, 4)
    assert t.loc["参政", "lo"] == pytest.approx(lo * 100) and t.loc["参政", "hi"] == pytest.approx(hi * 100)
    assert t.loc["共産", "lo"] == pytest.approx(0.0) and t.loc["共産", "hi"] > 0


def test_significant_uses_unadjusted_for_age():
    t = pd.DataFrame({"p_holm": [0.01, 0.01, 0.2], "p_adj_holm": [0.2, 0.2, 0.01]},
                     index=["40〜59歳", "男性", "新聞"])
    assert figures.significant(t.loc["40〜59歳"]) is True
    assert figures.significant(t.loc["男性"]) is False
    assert figures.significant(t.loc["新聞"]) is True
    t2 = pd.DataFrame({"p_holm": [0.05], "p_adj_holm": [0.05]}, index=["男性"])
    assert figures.significant(t2.loc["男性"]) is False


def test_dislike_vs_cabinet_groups_and_correlation():
    d = pd.DataFrame({
        "class": [0] * 6 + [1] * 3,
        "Q34_5": [0, 1, 3, 5, 10, 99, 0, 4, 10],
        "Q34_7": [2, 3, 5, 6, 9, 5, 1, 1, 1],
        "Q1": [2, 1, 2, 1, 1, 1, 3, 2, 1],
    })
    t, cor = sg.dislike_vs_cabinet(d)
    g0 = t[t["cls"] == 0].set_index("group")
    names = [b[0] for b in sg.DISLIKE_BINS]
    assert g0.loc[names[0], "n"] == 2 and g0.loc[names[0], "k_support"] == 1
    assert g0.loc[names[0], "support"] == pytest.approx(0.5) and g0.loc[names[0], "takaichi"] == pytest.approx(2.5)
    assert g0.loc[names[1], "n"] == 1 and g0.loc[names[2], "n"] == 2 and g0.loc[names[2], "support"] == 1.0
    c = cor.set_index("cls")
    assert c.loc[0, "n"] == 5
    assert c.loc[0, "rho"] == pytest.approx(stats.spearmanr([0, 1, 3, 5, 10], [2, 3, 5, 6, 9]).statistic)
    assert c.loc[0, "p"] == pytest.approx(stats.spearmanr([0, 1, 3, 5, 10], [2, 3, 5, 6, 9]).pvalue)
    assert cor["p_holm"].tolist() == pytest.approx(lca.holm(cor["p"].to_numpy()).tolist(), nan_ok=True)


def test_dislike_bins_cover_scale_without_overlap():
    covered = [v for _, lo, hi in sg.DISLIKE_BINS for v in range(lo, hi + 1)]
    assert sorted(covered) == list(range(11))


def test_dislike_vs_cabinet_bounds_and_counts():
    d = pd.DataFrame({
        "class": [5] * 8,
        "Q34_5": [0, 0, 0, 10, 11, 1, 1, 7],
        "Q34_7": [0, 10, 4, 10, 5, 11, 3, 8],
        "Q1": [1, 1, 3, 2, 1, 1, 99, 3],
    })
    t, cor = sg.dislike_vs_cabinet(d)
    g = t.set_index("group")
    names = [b[0] for b in sg.DISLIKE_BINS]
    # 0点・10点は含み、11点と Q1=99 は除き、Q1=3(その他・答えない)は分母に入る
    assert g.loc[names[0], "n"] == 3 and g.loc[names[0], "k_support"] == 2
    assert g.loc[names[0], "support"] == pytest.approx(2 / 3)
    assert g.loc[names[0], "takaichi"] == pytest.approx(14 / 3)
    assert g.loc[names[1], "n"] == 0 and math.isnan(g.loc[names[1], "support"])
    assert g.loc[names[2], "n"] == 2 and g.loc[names[2], "k_support"] == 0
    assert cor.iloc[0]["n"] == 5


def test_collapse3_maps_and_keeps_missing():
    x = np.array([[0, 1, 2, 3, 4, -1]])
    assert lca.collapse3(x).tolist() == [[0, 0, 1, 2, 2, -1]]


def test_em_with_three_categories():
    x, z = _two_class_data(n=400, j=6, seed=4)
    x3 = lca.collapse3(x)
    ll, pi, theta, resp = lca.fit(x3, 2, n_starts=3, n_cat=3)
    assert theta.shape == (2, 6, 3)
    assert adjusted_rand_score(z, resp.argmax(axis=1)) > 0.9
    assert lca.n_params(2, 3) == 1 + 2 * len(lca.ITEMS) * 2
    assert lca.bic(-10.0, 2, 20, 3) == pytest.approx(20 + lca.n_params(2, 3) * math.log(20))


def test_restrictive_class_share_hard_and_soft():
    d = pd.DataFrame({"Q4": [6, 6, 6, 1, 1, 2]})
    resp = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    r = lca.restrictive_class_share(d, resp)
    # クラス0: 投票者3人のうち参政2人、クラス1: 3人のうち参政1人 → 中核はクラス0
    assert r["core"] == 0 and r["n_core"] == 3 and r["share_in_core"] == pytest.approx(2 / 3)
    assert r["hard"] == pytest.approx(2 / 3)
    assert r["soft"] == pytest.approx((0.9 + 0.6 + 0.2) / 3)


def test_adjust_rating_level_removes_pure_response_style():
    rng = np.random.default_rng(0)
    n = 600
    style = rng.normal(5, 2, n)
    d = pd.DataFrame({"class": np.repeat([0, 1], n // 2)})
    # 好き嫌いはすべて「点の付け方」だけで決まり、参政党と高市氏の間に固有の関係はない
    for c in ["Q34_5", "Q34_7"] + sg.OTHER_THERMO:
        d[c] = np.clip(np.round(style + rng.normal(0, 1, n)), 0, 10)
    d["Q1"] = np.where(rng.random(n) < 0.5, 1, 2)
    raw = stats.spearmanr(d["Q34_5"], d["Q34_7"]).statistic
    out = sg.adjust_rating_level(d)
    assert raw > 0.5
    assert out["partial_rho"].abs().max() < 0.2
    assert out["n"].tolist() == [300, 300]
    assert out["partial_p_holm"].tolist() == pytest.approx(lca.holm(out["partial_p"].to_numpy()).tolist())
    assert out["coef_p_holm"].tolist() == pytest.approx(lca.holm(out["coef_p"].to_numpy()).tolist())


def test_restrictive_class_share_three_classes():
    d = pd.DataFrame({"Q4": [6, 1, 1, 1, 6, 6, 1, 6, 1, 2, 2]})
    resp = np.zeros((11, 3))
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    resp[np.arange(11), labels] = 1.0
    r = lca.restrictive_class_share(d, resp)
    # 参政の割合: クラス0=1/4、クラス1=2/3、クラス2=1/4 → 中核はクラス1(人数3)
    assert r["core"] == 1 and r["n_core"] == 3
    assert r["hard"] == pytest.approx(2 / 4) and r["soft"] == pytest.approx(2 / 4)


def test_adjust_rating_level_matches_direct_computation():
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(1)
    n = 400
    style = rng.normal(5, 2, n)
    shared = rng.normal(0, 1.5, n)
    d = pd.DataFrame({"class": np.repeat([3, 7], n // 2)})
    for c in sg.OTHER_THERMO:
        d[c] = np.clip(np.round(style + rng.normal(0, 1, n)), 0, 10)
    d["Q34_5"] = np.clip(np.round(style + shared + rng.normal(0, 1, n)), 0, 10)
    d["Q34_7"] = np.clip(np.round(style + shared + rng.normal(0, 1, n)), 0, 10)
    d["Q1"] = np.where(rng.random(n) < 1 / (1 + np.exp(-(d["Q34_5"] - 5))), 1, np.where(rng.random(n) < 0.8, 2, 3))
    d.loc[0, "Q34_5"] = 11
    d.loc[1, "Q34_7"] = 11
    d.loc[2, "Q1"] = 99
    out = sg.adjust_rating_level(d).set_index("cls")
    t = d.iloc[3:].copy()
    t["G"] = t[sg.OTHER_THERMO].mean(axis=1)
    t["sup"] = (t["Q1"] == 1).astype(int)
    for c, g in t.groupby("class"):
        r5 = smf.ols("Q34_5 ~ G", data=g).fit().resid
        r7 = smf.ols("Q34_7 ~ G", data=g).fit().resid
        expect = stats.spearmanr(r5, r7)
        m = smf.logit("sup ~ Q34_5 + G", data=g).fit(disp=0)
        assert out.loc[c, "n"] == len(g)
        assert out.loc[c, "partial_rho"] == pytest.approx(expect.statistic, abs=1e-9)
        assert out.loc[c, "partial_p"] == pytest.approx(expect.pvalue, rel=1e-6)
        assert out.loc[c, "coef"] == pytest.approx(m.params["Q34_5"], rel=1e-5)
        assert out.loc[c, "coef_p"] == pytest.approx(m.pvalues["Q34_5"], rel=1e-4)
        assert out.loc[c, "coef"] > 0 and out.loc[c, "partial_rho"] > 0.3


def test_adjust_rating_level_keeps_partial_other_ratings():
    rng = np.random.default_rng(2)
    n = 60
    d = pd.DataFrame({"class": [0] * n, "Q1": rng.choice([1, 2], n),
                      "Q34_5": rng.integers(0, 11, n), "Q34_7": rng.integers(0, 11, n)})
    for c in sg.OTHER_THERMO:
        d[c] = rng.integers(0, 11, n)
    d.loc[0, sg.OTHER_THERMO[0]] = 99
    assert sg.adjust_rating_level(d).iloc[0]["n"] == n


def test_loyalty_by_party_counts_same_party_and_ldp():
    raw = pd.DataFrame({"Q4": [6, 6, 6, 6, 2, 2, 1],
                        "Q2": [7, 1, 14, 7, 2, 1, 1],
                        "Q3": [5, 1, 12, 12, 2, 3, 1]})
    t = figures.loyalty_by_party(raw).set_index(["party", "measure"])
    assert t.loc[("参政", "support"), "pct"] == pytest.approx(50.0)
    assert t.loc[("参政", "ldp"), "pct"] == pytest.approx(25.0)
    assert t.loc[("参政", "vote26"), "pct"] == pytest.approx(25.0)
    assert t.loc[("参政", "ldp26"), "pct"] == pytest.approx(25.0)
    assert t.loc[("参政", "support"), "n"] == 4
    lo, hi = figures.wilson(2, 4)
    assert t.loc[("参政", "support"), "lo"] == pytest.approx(lo * 100)
    assert t.loc[("参政", "support"), "hi"] == pytest.approx(hi * 100)
    assert t.loc[("立憲", "support"), "pct"] == pytest.approx(50.0)
    assert t.loc[("立憲", "vote26"), "pct"] == pytest.approx(50.0)
    assert ("自民", "ldp") not in t.index and ("自民", "ldp26") not in t.index
    assert t.loc[("自民", "support"), "pct"] == pytest.approx(100.0)
    assert t.loc[("自民", "vote26"), "pct"] == pytest.approx(100.0)
    assert "共産" not in t.index.get_level_values("party")
