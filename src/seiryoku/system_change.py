"""国政選挙の制度の変化(定数の増減、小選挙区と比例の比率の変更など)を、新しい国政選挙の結果から検知する。

党勢の状態空間モデルは、議席と得票の曲線logit x=k logit v+cを全期間で共通とし、国政選挙の議席を
党勢の観測にする(state_space.seat_vote_curve)。比例だけを削るような改正で増幅率kが変わると、
改正後の最初の選挙で、制度の効果を党勢の変化と読んでしまう。総定数を見ても比率の変更は分からない
ので、曲線そのものが変わったかを検定する。

検定は、最新の選挙だけに別のkとcを許した曲線と、全選挙で共通の曲線の尤度比による。議席数を
二項分布とみなすと過分散で毎回棄却されるので、最新の選挙より前の選挙だけの当てはまりから
過分散の大きさ(ピアソンのカイ二乗/自由度)を推定し、尤度比をそれで割ってカイ二乗分布(自由度2)と比べる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .state_space import SeatVotePoint, _logit

# 参院の2025年の選挙は制度の改正がないのにp=0.037で検知される(多党化で増幅率が上がった)。
# 曲線が共通という仮定が崩れたこと自体はモデルに効くので、制度の改正に限らず知らせる
ALPHA = 0.05
HOUSE_NAMES = {"shugiin": "衆院", "sangiin": "参院"}


@dataclass
class CurveChange:
    house: str
    vote_date: date
    k_before: float
    k_latest: float
    dispersion: float
    statistic: float
    p_value: float


def _arrays(points: list[SeatVotePoint]):
    pts = [p for p in points if p.v is not None and 0 < p.v < 1 and 0 <= p.x <= 1]
    lv = np.array([_logit(p.v) for p in pts])
    n = np.array([float(p.total or 1) for p in pts])
    k = np.array([p.x for p in pts]) * n
    return pts, lv, n, k


def _fit(lv, n, k, X):
    """eta=X b の二項分布の最尤推定。戻り値は(係数, 対数尤度(定数項を除く))。"""
    def nll(b):
        eta = X @ b
        r = k - n / (1 + np.exp(-eta))
        return -float(np.sum(k * eta - n * np.logaddexp(0, eta))), -(X.T @ r)

    b0 = np.zeros(X.shape[1])
    b0[0] = 1.0
    res = minimize(nll, b0, jac=True, method="L-BFGS-B")
    return res.x, -res.fun


def curve_change(points: list[SeatVotePoint], house: str) -> CurveChange | None:
    """最新の選挙で議席と得票の曲線が変わったかの検定。前の選挙が2回未満、または最新の選挙で
    得票率の分かる党が3つ未満なら検定できないのでNone。"""
    dates = sorted({p.vote_date for p in points})
    if len(dates) < 3:
        return None
    latest = dates[-1]
    pts, lv, n, k = _arrays(points)
    is_new = np.array([p.vote_date == latest for p in pts], float)
    if is_new.sum() < 3 or (1 - is_new).sum() < 4:
        return None
    ones = np.ones(len(pts))
    old = is_new == 0
    b_old, _ = _fit(lv[old], n[old], k[old], np.column_stack([lv[old], ones[old]]))
    mu = n[old] / (1 + np.exp(-(np.column_stack([lv[old], ones[old]]) @ b_old)))
    pearson = float(np.sum((k[old] - mu) ** 2 / np.maximum(mu * (1 - mu / n[old]), 1e-9)))
    dispersion = max(pearson / max(int(old.sum()) - 2, 1), 1.0)
    _, ll_common = _fit(lv, n, k, np.column_stack([lv, ones]))
    b_sep, ll_sep = _fit(lv, n, k, np.column_stack([lv, ones, lv * is_new, is_new]))
    stat = max(2 * (ll_sep - ll_common), 0.0) / dispersion
    return CurveChange(house, latest, float(b_old[0]), float(b_sep[0] + b_sep[2]), dispersion, stat,
                       float(chi2.sf(stat, 2)))


def seat_totals(house: str) -> dict[date, int]:
    """国政選挙ごとの議席の総数(無所属を含む。参院はその選挙の改選数)。"""
    from .fetch.diet_history import fetch_sangiin_election_history, fetch_shugiin_history
    from .trend import parse_date

    snaps = fetch_shugiin_history(48) if house == "shugiin" else fetch_sangiin_election_history(24)
    return {parse_date(s.date): int(sum(s.seats.values())) for s in snaps}


def total_change(totals: dict[date, int]) -> tuple[date, int, int] | None:
    """最新の選挙の議席の総数が前回と違えば(投票日, 前回, 今回)。"""
    ds = sorted(totals)
    if len(ds) < 2 or totals[ds[-1]] == totals[ds[-2]]:
        return None
    return ds[-1], totals[ds[-2]], totals[ds[-1]]


def warnings_for(points_by_house: dict[str, list[SeatVotePoint]],
                 totals_by_house: dict[str, dict[date, int]]) -> list[str]:
    out = []
    for house, totals in totals_by_house.items():
        tc = total_change(totals)
        if tc:
            out.append(f"[要確認] 選挙制度の変化: {HOUSE_NAMES[house]}の議席の総数が変わった({tc[0].isoformat()}の選挙で"
                       f"{tc[1]}→{tc[2]})。定数・区割り・小選挙区と比例の比率の改正がなかったか調べる。")
    for house, points in points_by_house.items():
        cc = curve_change(points, house)
        if cc and cc.p_value < ALPHA:
            out.append(f"[要確認] 選挙制度の変化: {HOUSE_NAMES[house]}の{cc.vote_date.isoformat()}の選挙で、議席と得票の曲線が"
                       f"それまでの選挙と違う(増幅率k {cc.k_before:.2f}→{cc.k_latest:.2f}、"
                       f"過分散で割った尤度比 {cc.statistic:.1f}、p={cc.p_value:.3f})。"
                       "制度の改正(小選挙区と比例の比率、区割りなど)か、政党の構図の変化(多党化など)で"
                       "増幅の仕方が変わった可能性がある。")
    return out


def check() -> list[str]:
    """月次実行から呼ぶ。保存済みの国政選挙の結果で検知する。"""
    from .state_space import load_seat_vote_points

    points = {h: load_seat_vote_points(house=h) for h in HOUSE_NAMES}
    totals = {h: seat_totals(h) for h in HOUSE_NAMES}
    return warnings_for(points, totals)
