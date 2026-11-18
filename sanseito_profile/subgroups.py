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


def _moderate_classes(core: int) -> list[int]:
    """参政党の割合が全体並みのクラス(Fisher で全体と差がない)を穏健保守とする。"""
    t = pd.read_csv(OUT / "utas2026_lca_sanseito.csv")
    return [int(c) for c in t.loc[(t["p_holm"] >= 0.05) & (t["cluster"] != core), "cluster"]]


if __name__ == "__main__":
    sys.exit(main())
