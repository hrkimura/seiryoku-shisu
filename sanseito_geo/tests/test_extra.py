import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
import composition as co  # noqa: E402
import extra  # noqa: E402
import tourism as tr  # noqa: E402


def test_labels_and_groups():
    assert co.label("01_023：中国") == "中国"
    assert co.label("16：技術・人文知識・国際業務") == "技術・人文知識・国際業務"
    assert co.label("ベトナム") == "ベトナム"
    assert co.nation_group("01_037：ベトナム") == "ベトナム"
    assert co.nation_group("01_039：ミャンマー") == co.OTHER
    assert co.status_group("26：技能実習２号ロ") == "技能実習"
    assert co.status_group("技能") == "専門・技術"
    assert co.status_group("37：永住者の配偶者等") == "永住・定住・配偶者"
    assert co.status_group("39：特別永住者") == "特別永住者"
    assert co.status_group("35：永住者") == "永住・定住・配偶者"
    assert co.status_group("研修") == co.OTHER


def test_per_unit_sums_codes_handles_hamamatsu_and_missing():
    table = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, 100.0], "b": [0.0, 1.0, 0.0, 1.0, 50.0]},
                         index=pd.Index(["01101", "01102", "22131", "22138", "22100"], name="code"))
    out = co.per_unit(table, ["01101+01102", "22130", "99999", "01101"])
    assert out.loc[0, "a"] == 3 and out.loc[0, "b"] == 1
    assert out.loc[1, "a"] == 12 and out.loc[1, "b"] == 1
    assert out.loc[2].isna().all() and list(out.columns) == ["a", "b"] and out["a"].dtype == float
    assert out.loc[3, "a"] == 1
    assert list(out.index) == [0, 1, 2, 3]


def test_wide_2022_and_long_2025(tmp_path):
    d = pd.DataFrame([["表題", None, None, None, None, None],
                      ["市区町村コード", "都道府県市区町村", "総数", "中国", "ミャンマー", "タイ", "スリランカ"],
                      [None, "総数", 9, 9, 9, 9, 9],
                      ["01000", "北海道", 9, 9, 9, 9, 9],
                      ["01101", "札幌市中央区", 6, 3, "2", "1", 4],
                      ["01102", "札幌市北区", 6, 3, "2", None, "-"],
                      ["注", None, None, None, None, None, None]])
    f = tmp_path / "w.xlsx"
    d.to_excel(f, header=False, index=False)
    w = co.wide_2022(f, co.nation_group)
    assert w.loc["01101", "中国"] == 3 and w.loc["01101", co.OTHER] == 6
    assert w.loc["01102", "タイ"] == 0 and "01000" in w.index and w.index.name == "code"
    assert w.loc["01101", "タイ"] == 1 and "総数" not in w.columns and "注" not in w.index
    c = tmp_path / "l.csv"
    pd.DataFrame({"市区町村コード": ["01101", "01101", "01102"], "国籍・地域": ["01_023：中国", "01_039：ミャンマー", "01_023：中国"],
                  "在留資格": ["31：留学", "31：留学", "26：技能実習２号ロ"], "在留外国人数": [5, 2, 7]}).to_csv(c, index=False)
    ln = co.long_2025(c, "国籍・地域", co.nation_group)
    assert ln.loc["01101", "中国"] == 5 and ln.loc["01101", co.OTHER] == 2 and ln.loc["01102", co.OTHER] == 0
    assert ln.index.name == "code" and (ln.dtypes == float).all()
    ls = co.long_2025(c, "在留資格", co.status_group)
    assert ls.loc["01101", "留学"] == 7 and ls.loc["01102", "技能実習"] == 7


def test_tourism_parse_table_reads_prefecture_rows():
    d = pd.DataFrame([[None] * 17 for _ in range(6)])
    d.iat[4, 0], d.iat[4, 1], d.iat[4, 16] = "　48全国", 7, 7
    d.iat[5, 0], d.iat[5, 1], d.iat[5, 16] = "01北海道(再掲)", 9, 9
    d.iat[0, 0] = "令和6年 1～12月  計"
    d.iat[1, 0], d.iat[1, 1], d.iat[1, 16] = "　01北海道", 100, 10
    d.iat[2, 0], d.iat[2, 1], d.iat[2, 16] = "　47沖縄県", 50, 5
    d.iat[3, 0], d.iat[3, 1], d.iat[3, 16] = "　99その他", 1, 1
    p = tr.parse_table(d)
    assert p["pref"].tolist() == [1, 47]
    assert p["nights"].tolist() == [100, 50] and p["foreign_nights"].tolist() == [10, 5]


def _units(n=200, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"pref": np.repeat(np.arange(10), n // 10), "valid25": rng.uniform(1, 2, n),
                      "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    t["d_share"] = 1.5 * t["a"] - 0.5 * t["b"] + 0.3 * t["c"] + rng.normal(scale=0.1, size=n)
    return t


def test_joint_effects_recovers_coefficients():
    t = _units()
    t["unused"] = np.nan
    one = extra.joint_effects(t, ["a"], ["b", "c"]).iloc[0]
    fit = analysis.ols(t.assign(y=t["d_share"]), "d_share", "a", ["b", "c"])
    assert one["coef"] == pytest.approx(fit.coef) and one["lo"] == pytest.approx(fit.ci[0])
    assert one["hi"] == pytest.approx(fit.ci[1])
    r = extra.joint_effects(t, ["a", "b"], ["c"]).set_index("group")
    assert r.loc["a", "coef"] == pytest.approx(1.5, abs=0.05)
    assert r.loc["b", "coef"] == pytest.approx(-0.5, abs=0.05)
    assert r.loc["a", "lo"] < 1.5 < r.loc["a", "hi"] and r.loc["a", "hi"] - r.loc["a", "lo"] < 0.2
    assert r.loc["a", "mean_increase"] == pytest.approx(np.average(t["a"], weights=t["valid25"]))
    assert (r["n"] == 200).all()


def test_prefecture_table_and_effect():
    t = pd.DataFrame({"pref": [1, 1, 2, 3], "votes22": [1, 1, 2, 1], "valid22": [10, 10, 20, 10],
                      "votes25": [3, 1, 6, 2], "valid25": [10, 10, 20, 10], "for22": [1, 1, 2, 1],
                      "pop22": [100, 100, 100, 100], "for25": [2, 2, 2, 1], "pop25": [100, 100, 100, 100]})
    base = pd.DataFrame({"pref": [1, 2, 3], "foreign_nights": [100.0, 50.0, 10.0]})
    now = pd.DataFrame({"pref": [1, 2, 3], "foreign_nights": [400.0, 100.0, 20.0]})
    p = extra.prefecture_table(t, base, now).set_index("pref")
    assert p.loc[1, "d_share"] == pytest.approx(10.0) and p.loc[1, "share22"] == pytest.approx(10.0)
    assert p.loc[1, "d_fshare"] == pytest.approx(1.0)
    assert p.loc[1, "fn_now"] == pytest.approx(2.0) and p.loc[1, "d_fn"] == pytest.approx(1.5)
    assert p.loc[2, "log_fn_now"] == pytest.approx(0.0)
    p = p.reset_index()
    p["x"] = [0.0, 1.0, 2.0]
    p["d_share"] = [3.0, 5.5, 7.0]
    p["valid25"] = [1.0, 10.0, 3.0]
    import statsmodels.api as sm
    ref = sm.WLS(p["d_share"], sm.add_constant(p[["x"]]), weights=p["valid25"]).fit()
    c, lo, hi = extra.prefecture_effect(p, "x", [])
    assert c == pytest.approx(float(np.asarray(ref.params)[1])) and lo < c < hi


def test_density_split_has_three_groups_two_methods():
    rng = np.random.default_rng(0)
    n = 300
    t = pd.DataFrame({"pref": np.repeat(np.arange(10), 30), "valid25": 1.0, "log_density": rng.normal(size=n),
                      "d_fshare": rng.normal(size=n), "pred_fshare": rng.normal(size=n)})
    for c in analysis.CONTROLS + analysis.REGION:
        t[c] = rng.normal(size=n)
    t["d_share"] = t["d_fshare"] + rng.normal(size=n)
    r = extra.density_split(t)
    assert len(r) == 6 and set(r["method"]) == {"最小二乗", "操作変数"}
    assert list(r.columns) == ["group", "method", "coef", "lo", "hi", "n", "first_stage_F"]
    assert list(dict.fromkeys(r["group"])) == ["農村(密度の下位3分の1)", "中間", "都市(密度の上位3分の1)"]
    assert (r["n"] == 100).all() and (r["lo"] < r["hi"]).all()
    assert r["n"].sum() == 2 * n
    assert r[r.method == "最小二乗"]["coef"].between(0.7, 1.3).all()


def test_reiwa_overlap_reports_both_specs():
    rng = np.random.default_rng(2)
    n = 300
    t = pd.DataFrame({"pref": np.repeat(np.arange(10), 30), "valid25": 1.0, "reiwa22": rng.normal(size=n)})
    for c in ["share22", "log_pop22"] + analysis.REGION:
        t[c] = rng.normal(size=n)
    t["d_share"] = 0.5 * t["reiwa22"] + rng.normal(scale=0.1, size=n)
    r = extra.reiwa_overlap(t)
    assert r["spec"].tolist() == ["統制なし", "地域の特徴をそろえる"]
    assert r["coef"].between(0.45, 0.55).all() and (r["lo"] < r["hi"]).all()
    assert r["n"].tolist() == [n, n]


def test_individual_outcomes_and_exposure():
    import individual as ind

    r = pd.DataFrame({"Q35_8": [1, 4, 5, 99], "Q36_6": [5, 2, 3, 4], "Q4": [6, 1, 13, 99]})
    o = ind.outcomes(r)
    assert o["受け入れ推進に反対"].tolist()[:3] == [0.0, 1.0, 1.0] and np.isnan(o["受け入れ推進に反対"].iloc[3])
    assert o["文化が脅かされる"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert o["参政党に投票"].tolist()[:2] == [1.0, 0.0] and o["参政党に投票"].iloc[2:].isna().all()
    u = pd.DataFrame({"pref": [1, 1, 2], "for22": [1.0, 1.0, 2.0], "pop22": [100.0, 100.0, 100.0],
                      "for25": [3.0, 1.0, 2.0], "pop25": [100.0, 100.0, 100.0]})
    e = ind.prefecture_exposure(u)
    assert e.loc[1, "外国人の割合(2025年)"] == pytest.approx(2.0)
    assert e.loc[1, "外国人の割合の増加"] == pytest.approx(1.0) and e.loc[2, "外国人の割合の増加"] == 0


def test_holm_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = extra.holm(p)
    assert adj.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])
    r = pd.DataFrame({"coef": [1.96, 0.0], "lo": [0.0, -1.96], "hi": [3.92, 1.96]})
    h = extra.with_holm(r)
    assert h["p"].tolist() == pytest.approx([0.05, 1.0], abs=1e-3)
    assert h["p_holm"].tolist() == pytest.approx([0.1, 1.0], abs=1e-3)


def test_level_effect_and_vote_distribution():
    rng = np.random.default_rng(5)
    n = 300
    t = pd.DataFrame({"pref": np.repeat(np.arange(10), 30), "valid25": 1.0, "fshare25": rng.uniform(0, 8, n),
                      "log_pop22": rng.normal(size=n)})
    for c in analysis.REGION:
        t[c] = rng.normal(size=n)
    t["share25"] = 10 + 0.2 * t["fshare25"] + rng.normal(scale=0.1, size=n)
    r = extra.level_effect(t)
    assert r["spec"].tolist() == ["都道府県の違いをそろえる", "地域の特徴もそろえる"]
    assert r["coef"].between(0.18, 0.22).all() and (r["lo"] < r["hi"]).all()
    d = pd.DataFrame({"fshare25": [0.5, 2.0, 4.0, 6.0], "votes25": [10.0, 20.0, 30.0, 40.0],
                      "pop25": [1.0, 1.0, 1.0, 1.0]})
    v = extra.vote_distribution(d, (1, 3, 5)).set_index("below")
    assert v.loc[1, "votes_pct"] == pytest.approx(10.0) and v.loc[3, "votes_pct"] == pytest.approx(30.0)
    assert v.loc[5, "votes_pct"] == pytest.approx(60.0) and v.loc[5, "pop_pct"] == pytest.approx(75.0)
    d.loc[0, "fshare25"] = 1.0
    assert extra.vote_distribution(d, (1,)).loc[0, "votes_pct"] == 0
