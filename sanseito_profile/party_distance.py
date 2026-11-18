"""政党の投票者どうしの「属性の距離」と「政策態度の距離」を全ペアについて測る。

距離は、項目ごとの該当割合の差の二乗平均平方根(百分率ポイント)。項目の組は結果を見る前に固定した。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from profile_utas import ITEMS, OUT, PARTY_Q4, indicator, load

PARTIES = [1, 2, 3, 4, 5, 6, 7, 8]
DEMO = ["男性", "18〜39歳", "40〜59歳", "大学・大学院卒", "中学・高校卒", "既婚(配偶者あり)",
        "生活水準が中の下以下", "無職", "主婦・主夫", "事務・技術職", "製造・サービス職"]
ATTITUDE = [it.label for it in ITEMS if it.group == "政策"] + ["憲法を変える必要がある"]


def indicator_matrix(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    by_label = {it.label: it for it in ITEMS}
    return pd.DataFrame({lab: indicator(df[by_label[lab].col], by_label[lab]) for lab in labels}, index=df.index)


def shares(x: pd.DataFrame, party: pd.Series) -> pd.DataFrame:
    return x.groupby(party).mean()


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) * 100


def unbiased_distance(x: pd.DataFrame, party: pd.Series, p: int, q: int) -> float:
    """標本誤差による上方の偏りを除いた距離。

    (p1-p2)^2 の期待値は真の差の二乗に p1(1-p1)/n1 + p2(1-p2)/n2 を足したものなので、その分を引く。
    小さい党どうしでは素の距離が誤差だけで数ポイントになるため、こちらで比べる。
    """
    mask = np.asarray(party)
    a, b = x[mask == p], x[mask == q]
    pa, pb = a.mean(), b.mean()
    na, nb = a.notna().sum(), b.notna().sum()
    sq = (pa - pb) ** 2 - pa * (1 - pa) / (na - 1) - pb * (1 - pb) / (nb - 1)
    m = float(sq.mean())
    return float(np.sign(m) * np.sqrt(abs(m))) * 100


def resample_within(party: np.ndarray, parties: list[int], rng: np.random.Generator) -> np.ndarray:
    """党ごとに、その党の回答者の中から同じ人数を復元抽出した行番号。"""
    return np.concatenate([rng.choice(np.where(party == p)[0], size=int((party == p).sum()), replace=True)
                           for p in parties])


def interval95(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_ranks(demo_rank: np.ndarray, att_rank: np.ndarray) -> dict[str, float]:
    demo_rank, att_rank = np.asarray(demo_rank), np.asarray(att_rank)
    return {"demo_rank1": float((demo_rank == 1).mean()), "demo_top3": float((demo_rank <= 3).mean()),
            "demo_rank_median": float(np.median(demo_rank)),
            "att_top3": float((att_rank <= 3).mean()), "att_top5": float((att_rank <= 5).mean()),
            "att_rank_median": float(np.median(att_rank))}


def pair_distances(sh: pd.DataFrame) -> dict[tuple[int, int], float]:
    return {(p, q): rms_distance(sh.loc[p].to_numpy(), sh.loc[q].to_numpy())
            for p, q in itertools.combinations(sh.index, 2)}


def bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    demo, att = indicator_matrix(df, DEMO), indicator_matrix(df, ATTITUDE)
    party = df["Q4"]
    point = {k: pair_distances(shares(m, party)) for k, m in (("demo", demo), ("att", att))}
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        idx = resample_within(party.to_numpy(), PARTIES, rng)
        pb = party.iloc[idx].to_numpy()
        for k, m in (("demo", demo), ("att", att)):
            draws[k].append(pair_distances(m.iloc[idx].groupby(pb).mean()))
    rows = []
    for pair in point["demo"]:
        row = {"pair": f"{PARTY_Q4[pair[0]]}-{PARTY_Q4[pair[1]]}"}
        for k in ("demo", "att"):
            v = np.array([d[pair] for d in draws[k]])
            row[k] = point[k][pair]
            row[f"{k}_lo"], row[f"{k}_hi"] = interval95(v)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_probability(df: pd.DataFrame, focus: tuple[int, int], n_boot: int = 1000, seed: int = 1) -> dict[str, float]:
    """偏りを除いた距離で、focus のペアが属性で最も近い・態度で上位3位以内に遠い確率(党内で復元抽出)。"""
    rng = np.random.default_rng(seed)
    demo, att = indicator_matrix(df, DEMO), indicator_matrix(df, ATTITUDE)
    party = df["Q4"].to_numpy()
    pairs = list(itertools.combinations(PARTIES, 2))
    k = pairs.index(focus)
    demo_rank, att_rank = [], []
    for _ in range(n_boot):
        idx = resample_within(party, PARTIES, rng)
        pb = party[idx]
        d = np.array([unbiased_distance(demo.iloc[idx], pb, p, q) for p, q in pairs])
        a = np.array([unbiased_distance(att.iloc[idx], pb, p, q) for p, q in pairs])
        demo_rank.append(int((d < d[k]).sum()) + 1)
        att_rank.append(int((a > a[k]).sum()) + 1)
    return summarize_ranks(np.array(demo_rank), np.array(att_rank))


def main() -> None:
    df = load()
    df = df[df["Q4"].isin(PARTIES)]
    t = bootstrap(df)
    demo, att = indicator_matrix(df, DEMO), indicator_matrix(df, ATTITUDE)
    pairs = list(itertools.combinations(PARTIES, 2))
    t["demo_ub"] = [unbiased_distance(demo, df["Q4"], p, q) for p, q in pairs]
    t["att_ub"] = [unbiased_distance(att, df["Q4"], p, q) for p, q in pairs]
    t.to_csv(OUT / "utas2026_party_distance.csv", index=False)
    t["rank_demo_close"] = t["demo_ub"].rank()
    t["rank_att_far"] = t["att_ub"].rank(ascending=False)
    print(t.sort_values("att_ub", ascending=False)[["pair", "demo", "demo_ub", "rank_demo_close", "att", "att_ub",
                                                    "rank_att_far"]].round(1).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
