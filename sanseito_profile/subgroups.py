"""参政党の2025年の投票者を二通りに分けて比べる(探索的)。

- 潜在クラス分析の中核(外国人の受け入れに強く反対する右派のクラス)にいる人と、穏健保守のクラスにいる人
- 2026年衆院選で参政党に残った人と、自民党へ移った人
比べる項目は結果を見る前に固定し、それぞれの比較の中で Holm 法で補正する。
"""
from __future__ import annotations

import sys

import pandas as pd
from scipy import stats

from profile_utas import DATA, OUT, holm

# 26年衆院選(Q3)の符号
Q3_SANSEITO, Q3_LDP = 5, 1

BINARY = {
    "参政党を支持": ("Q2", {7}, None),
    "長期的に参政党寄り": ("Q33", {7}, None),
    "高市内閣を支持": ("Q1", {1}, None),
    "40〜59歳": ("Q40", set(range(6, 10)), set(range(1, 16))),
    "60歳以上": ("Q40", set(range(10, 16)), set(range(1, 16))),
    "男性": ("Q39", {1}, {1, 2}),
    "大学・大学院卒": ("Q43", {5, 6}, set(range(1, 7))),
    "情報源:動画サイト": ("Q49_6", {1}, None),
    "情報源:新聞": ("Q49_1", {1}, None),
    "政治は腐敗、強い指導者が必要": ("Q35_13", {1, 2}, set(range(1, 6))),
    "在留外国人で社会が豊かになる": ("Q36_6", {1, 2}, set(range(1, 6))),
    "首相は靖国に参拝してほしい": ("Q35_2", {1, 2}, set(range(1, 6))),
    "消費税率を引き下げるべき": ("Q35_12", {1, 2}, set(range(1, 6))),
}


def share(s: pd.Series, yes: set, valid: set | None) -> tuple[int, int]:
    ok = s.isin(valid) if valid is not None else s.notna() & (s != 99)
    return int(s[ok].isin(yes).sum()), int(ok.sum())


def compare(a: pd.DataFrame, b: pd.DataFrame, items: dict = BINARY) -> pd.DataFrame:
    rows = []
    for label, (col, yes, valid) in items.items():
        k1, n1 = share(a[col], yes, valid)
        k2, n2 = share(b[col], yes, valid)
        p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
        rows.append(dict(label=label, n_a=n1, p_a=k1 / n1, n_b=n2, p_b=k2 / n2, p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    return t


THERMO = {"自民党": "Q34_1", "参政党": "Q34_5", "高市早苗": "Q34_7"}


def compare_thermo(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """好き嫌い(0〜10点)の平均を Mann-Whitney 検定で比べ、Holm 法で補正する。"""
    rows = []
    for label, col in THERMO.items():
        x, y = a[col][a[col].between(0, 10)], b[col][b[col].between(0, 10)]
        rows.append(dict(label=label, n_a=len(x), mean_a=x.mean(), n_b=len(y), mean_b=y.mean(),
                         p=stats.mannwhitneyu(x, y).pvalue))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    return t


# 参政党の好き嫌い(0〜10点)の段階。0〜1は「とても嫌い」に当たる強い拒否、5以上は中間から好意的
DISLIKE_BINS = [("強く嫌う(0〜1点)", 0, 1), ("嫌う(2〜4点)", 2, 4), ("中間〜好き(5〜10点)", 5, 10)]


def dislike_vs_cabinet(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """クラスごとに、参政党の好感度の段階別の高市内閣の支持率と高市氏の好感度、
    およびクラス内での参政党と高市氏の好感度の順位相関を返す。d は class 列を持つ。"""
    ok = d["Q34_5"].between(0, 10) & d["Q34_7"].between(0, 10) & d["Q1"].isin([1, 2, 3])
    d = d[ok]
    rows, cors = [], []
    for c, g in d.groupby("class"):
        r = stats.spearmanr(g["Q34_5"], g["Q34_7"])
        cors.append(dict(cls=int(c), n=len(g), rho=float(r.statistic), p=float(r.pvalue)))
        for name, lo, hi in DISLIKE_BINS:
            x = g[g["Q34_5"].between(lo, hi)]
            # 人数0の段階では平均が自然に NaN になる
            rows.append(dict(cls=int(c), group=name, n=len(x), k_support=int((x["Q1"] == 1).sum()),
                             support=(x["Q1"] == 1).mean(), takaichi=x["Q34_7"].mean()))
    cor = pd.DataFrame(cors)
    cor["p_holm"] = holm(cor["p"].to_numpy())
    return pd.DataFrame(rows), cor


# 参政党・高市氏・自民党(高市氏と強く結びつく)を除いた、好き嫌いを聞いた残りの対象
OTHER_THERMO = ["Q34_2", "Q34_3", "Q34_4", "Q34_6", "Q34_8"]


def adjust_rating_level(d: pd.DataFrame) -> pd.DataFrame:
    """どの対象にも低い(高い)点を付ける回答の癖を、他の対象への好き嫌いの平均 G で調整する。
    クラスごとに、G を除いた参政党と高市氏の好感度の偏相関(順位)と、
    内閣支持を参政党の好感度と G で説明するロジットの参政党の係数を返す。"""
    import numpy as np
    import statsmodels.formula.api as smf
    t = d[["class", "Q1", "Q34_5", "Q34_7"] + OTHER_THERMO].copy()
    for c in ["Q34_5", "Q34_7"] + OTHER_THERMO:
        t[c] = t[c].where(t[c].between(0, 10))
    t["G"] = t[OTHER_THERMO].mean(axis=1)
    t = t.dropna(subset=["Q34_5", "Q34_7", "G"])
    t = t[t["Q1"].isin([1, 2, 3])].assign(sup=lambda x: (x["Q1"] == 1).astype(int))
    rows = []
    for c, g in t.groupby("class"):
        r5 = g["Q34_5"] - np.polyval(np.polyfit(g["G"], g["Q34_5"], 1), g["G"])
        r7 = g["Q34_7"] - np.polyval(np.polyfit(g["G"], g["Q34_7"], 1), g["G"])
        pc = stats.spearmanr(r5, r7)
        m = smf.logit("sup ~ Q34_5 + G", data=g).fit(disp=0)
        rows.append(dict(cls=int(c), n=len(g), partial_rho=float(pc.statistic), partial_p=float(pc.pvalue),
                         coef=float(m.params["Q34_5"]), coef_p=float(m.pvalues["Q34_5"])))
    out = pd.DataFrame(rows)
    out["partial_p_holm"] = holm(out["partial_p"].to_numpy())
    out["coef_p_holm"] = holm(out["coef_p"].to_numpy())
    return out


def main() -> None:
    df = pd.read_csv(DATA)
    cls = pd.read_csv(OUT / "utas2026_lca.csv")
    s = df[df["Q4"] == 6].merge(cls, on="ID", how="left")
    core_class = int(pd.read_csv(OUT / "utas2026_lca_sanseito.csv").sort_values("share").iloc[-1]["cluster"])
    pd.set_option("display.width", 200)
    print(f"中核クラス={core_class}")
    t1 = compare(s[s["class"] == core_class], s[s["class"].isin(_moderate_classes(core_class))])
    print("中核(a) 対 穏健保守(b)"); print(t1.round(3).to_string(index=False))
    t2 = compare(s[s["Q3"] == Q3_SANSEITO], s[s["Q3"] == Q3_LDP])
    print("26年も参政(a) 対 26年は自民(b)"); print(t2.round(3).to_string(index=False))
    t3 = compare_thermo(s[s["Q3"] == Q3_SANSEITO], s[s["Q3"] == Q3_LDP])
    print("好感度: 26年も参政(a) 対 26年は自民(b)"); print(t3.round(3).to_string(index=False))
    t3.to_csv(OUT / "utas2026_stay_vs_ldp_thermo.csv", index=False)
    t1.to_csv(OUT / "utas2026_core_vs_moderate.csv", index=False)
    t2.to_csv(OUT / "utas2026_stay_vs_ldp.csv", index=False)
    t4, cor = dislike_vs_cabinet(df.merge(pd.read_csv(OUT / "utas2026_lca.csv"), on="ID"))
    print("参政党の好き嫌い別の内閣支持"); print(t4.round(2).to_string(index=False)); print(cor.round(3).to_string(index=False))
    t4.to_csv(OUT / "utas2026_dislike_vs_cabinet.csv", index=False)
    cor.to_csv(OUT / "utas2026_dislike_vs_takaichi_cor.csv", index=False)
    adj = adjust_rating_level(df.merge(pd.read_csv(OUT / "utas2026_lca.csv"), on="ID"))
    print("回答の癖で調整"); print(adj.round(3).to_string(index=False))
    adj.to_csv(OUT / "utas2026_dislike_adjusted.csv", index=False)


def _moderate_classes(core: int) -> list[int]:
    """参政党の割合が全体並みのクラス(Fisher で全体と差がない)を穏健保守とする。"""
    t = pd.read_csv(OUT / "utas2026_lca_sanseito.csv")
    return [int(c) for c in t.loc[(t["p_holm"] >= 0.05) & (t["cluster"] != core), "cluster"]]


if __name__ == "__main__":
    sys.exit(main())
