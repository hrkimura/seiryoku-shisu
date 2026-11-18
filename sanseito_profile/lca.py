"""政策態度19項目(各5段階)の潜在クラス分析。投票先は使わず、後からクラスごとの参政党の割合を調べる。

5段階の回答は離散値なので、混合ガウスでは「ある項目で全員が同じ選択肢」という退化したクラスタができる。
各クラスの中で項目ごとの選択肢の確率(カテゴリ分布)を推定する潜在クラス分析を使う。
無回答は尤度から外す(欠測をそのまま扱える)。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from profile_utas import DATA, OUT, PARTY_Q4, SANSEITO, VOTED, holm

ITEMS = [f"Q35_{i}" for i in range(1, 14)] + [f"Q36_{i}" for i in range(1, 7)]
MAX_MISSING = 3
# クラスの番号は fit の乱数の種(0)で決まる。種や項目を変えたら中身を見て付け直す
CLASS_NAMES = {0: "穏健な保守", 1: "はっきり答える保守", 2: "中間の回答が多い層", 3: "リベラル",
               4: "外国人の受け入れに強く反対する右派"}

N_CAT = 5
K_RANGE = range(2, 11)
# 確率0のセルで対数尤度が発散しないための小さな下駄。結果の解釈に効かない大きさ
EPS = 1e-6


def encode(df: pd.DataFrame) -> np.ndarray:
    """回答を 0〜4 に、無回答・範囲外を -1 にした整数行列。"""
    x = df[ITEMS].to_numpy()
    out = np.where(np.isin(x, [1, 2, 3, 4, 5]), x - 1, -1)
    return out.astype(int)


def _onehot(x: np.ndarray) -> np.ndarray:
    """(n, J, C) の0/1配列。欠測の行はすべて0。"""
    n, j = x.shape
    oh = np.zeros((n, j, N_CAT))
    rows, cols = np.nonzero(x >= 0)
    oh[rows, cols, x[rows, cols]] = 1.0
    return oh


def em(x: np.ndarray, k: int, seed: int, n_iter: int = 500, tol: float = 1e-6):
    rng = np.random.default_rng(seed)
    oh = _onehot(x)
    n, j, _ = oh.shape
    pi = np.full(k, 1 / k)
    theta = rng.dirichlet(np.ones(N_CAT), size=(k, j))
    prev = -np.inf
    for _ in range(n_iter):
        logp = np.log(pi)[None, :] + np.einsum("njc,kjc->nk", oh, np.log(theta + EPS))
        m = logp.max(axis=1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))).sum())
        resp = np.exp(logp - m)
        resp /= resp.sum(axis=1, keepdims=True)
        pi = resp.mean(axis=0)
        num = np.einsum("nk,njc->kjc", resp, oh)
        theta = num / np.maximum(num.sum(axis=2, keepdims=True), EPS)
        if ll - prev < tol:
            break
        prev = ll
    return ll, pi, theta, resp


def fit(x: np.ndarray, k: int, n_starts: int = 10, seed: int = 0):
    """乱数の初期値を n_starts 通り試し、対数尤度が最大のものを返す。"""
    best = None
    for s in range(n_starts):
        r = em(x, k, seed * 1000 + s)
        if best is None or r[0] > best[0]:
            best = r
    return best


def sanseito_by_cluster(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """比例で投票した人に占める参政党の割合を、クラスタとそれ以外で比べる(Fisher、Holm補正)。"""
    voted = df["Q4"].isin(VOTED).to_numpy()
    s = (df["Q4"] == SANSEITO).to_numpy()
    rows = []
    for c in np.unique(labels):
        inc = labels == c
        a, n_in = int((s & inc & voted).sum()), int((inc & voted).sum())
        b, n_out = int((s & ~inc & voted).sum()), int((~inc & voted).sum())
        p = stats.fisher_exact([[a, n_in - a], [b, n_out - b]])[1]
        rows.append(dict(cluster=int(c), n=int(inc.sum()), n_voted=n_in, n_sanseito=a,
                         share=a / n_in if n_in else np.nan, share_rest=b / n_out, p=p,
                         share_of_all_sanseito=a / int((s & voted).sum())))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    return t


def n_params(k: int) -> int:
    return (k - 1) + k * len(ITEMS) * (N_CAT - 1)


def bic(ll: float, k: int, n: int) -> float:
    return -2 * ll + n_params(k) * np.log(n)


def main() -> None:
    df = pd.read_csv(DATA)
    x = encode(df)
    keep = (x < 0).sum(axis=1) <= MAX_MISSING
    d, x = df[keep].reset_index(drop=True), x[keep]
    n = len(x)
    fits = {k: fit(x, k) for k in K_RANGE}
    bics = {k: bic(fits[k][0], k, n) for k in K_RANGE}
    k = min(bics, key=bics.get)
    print(f"対象 {n} 人、BIC最小の k={k}", {kk: round(v) for kk, v in bics.items()})
    labels = fits[k][3].argmax(axis=1)
    aris = [adjusted_rand_score(labels, fit(x, k, seed=s)[3].argmax(axis=1)) for s in range(1, 6)]
    print("初期値の組を変えたときの一致度(ARI)", np.round(aris, 2))
    t = sanseito_by_cluster(d, labels)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    theta = fits[k][2]
    agree = pd.DataFrame(theta[:, :, :2].sum(axis=2).T, index=ITEMS).round(2)
    print("賛成・Aに近い側(1・2)の確率")
    print(agree.to_string())
    votes = pd.crosstab(labels, d["Q4"].map(PARTY_Q4).fillna("棄権・不明"), normalize="index").round(2)
    print(votes.to_string())
    pd.DataFrame({"ID": d["ID"], "class": labels}).to_csv(OUT / "utas2026_lca.csv", index=False)
    t.to_csv(OUT / "utas2026_lca_sanseito.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
