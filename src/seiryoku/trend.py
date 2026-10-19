"""党勢の趨勢$D_p(\\tau)$を、指数重み付きの局所2次回帰で推定する
(notes/local_regression_proposal.tex)。

選挙イベント1件を回帰の1点とし、無所属を除いて正規化した占有率
$\\tilde\\phi_p(e)$を、階層ごとの切片と全階層共通の2次式で当てはめる。
$$\\tilde\\phi_p(e)=\\alpha_{p,\\ell(e)}+\\beta_p s_e+\\gamma_p s_e^2+\\varepsilon_e,
  \\qquad s_e=(t_e-\\tau)/h$$
重みは$K(s_e)=e^{s_e}$と、現行の$C_p(T)$と同じ階層内・階層間の重みの積で、
$N_\\ell$・$\\Pi_\\ell$はいずれも$K$で重み付けした全期間の合計にする。
$D_p(\\tau)=\\hat\\beta_p/h$(月あたりの占有率の変化)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

INDEPENDENT = "無所属"
UNCLASSIFIED_MAYOR = "市区町村長(未仕分け)"
DAYS_PER_MONTH = 365.25 / 12
# e^{-12}未満の重みは結果に影響しないので、行列を小さくするために落とす
MIN_LOG_KERNEL = -12.0
UNIFIED_LOCAL_ELECTION_MONTHS = {(2019, 4), (2023, 4)}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "trend"


@dataclass(frozen=True)
class Event:
    vote_date: date
    level: str
    P: float
    n: float
    shares: dict[str, float]
    name: str


def parse_date(s: str) -> date:
    y, m, d = s.replace("/", "-")[:10].split("-")
    return date(int(y), int(m), int(d))


def events_from_month_entries(month_entries, classified_until: date | None) -> list[Event]:
    """build_month_entriesの結果を、投票日つきのイベント列にする。

    classified_untilより後の市区町村長選挙は、推薦・支持の名簿がまだ公開されて
    おらず無所属のまま残るので、別の階層として専用の切片を与える(案の\\S7)。
    """
    events = []
    for ym in month_entries.months:
        for (level, P, n, shares), label in zip(month_entries.entries[ym], month_entries.labels[ym]):
            if not label.get("vote_date"):
                continue
            d = parse_date(label["vote_date"])
            if level == "市区町村長" and classified_until is not None and d > classified_until:
                level = UNCLASSIFIED_MAYOR
            events.append(Event(d, level, float(P), float(n), dict(shares), label.get("name") or ""))
    events.sort(key=lambda e: e.vote_date)
    return events


def endorsement_cutoff(by_year) -> date | None:
    """名簿の全年版に収録された市区町村長選挙のうち、最も新しい投票日。
    年版の収録期間は年によって暦年からずれる(2021年版は2022年3月まで)ので、
    年版のラベルではなく実際の投票日から決める。"""
    latest = None
    for entries in by_year.values():
        for e in entries.values():
            if not e.vote_date or len(e.vote_date) < 10:
                continue
            try:
                d = parse_date(e.vote_date)
            except ValueError:
                continue
            latest = d if latest is None or d > latest else latest
    return latest


def normalized_shares(shares: dict[str, float]) -> dict[str, float] | None:
    """無所属を除いて正規化する。当選者が全員無所属なら定義できないのでNone。"""
    rest = 1.0 - shares.get(INDEPENDENT, 0.0)
    if rest <= 1e-12:
        return None
    return {p: v / rest for p, v in shares.items() if p != INDEPENDENT}


def first_appearance(events: list[Event]) -> dict[str, date]:
    """各政党が初めて占有率を得た投票日。結党日の代わりに使う。"""
    first: dict[str, date] = {}
    for e in events:
        for p, v in e.shares.items():
            if p != INDEPENDENT and v > 0 and p not in first:
                first[p] = e.vote_date
    return first


def separate_national_levels(events: list[Event]) -> list[Event]:
    """衆院選・参院選に1回ごとの専用の切片を与え、傾きに寄与しないようにする。"""
    from dataclasses import replace

    return [replace(e, level=f"{e.level}@{e.vote_date.isoformat()}") if e.level in ("衆院選", "参院選") else e
            for e in events]


def months_between(a: date, b: date) -> float:
    return (b - a).days / DAYS_PER_MONTH


@dataclass
class TrendFit:
    party: str
    tau: date
    h: float
    degree: int
    D: float
    se: float
    beta: float
    gamma: float | None
    alpha: dict[str, float]
    n_events: int
    n_eff: float
    max_weight_share: float
    n_clusters: int
    eff_clusters: float
    p_wild: float | None = None
    a_hat: float | None = None
    contributions: list[tuple[float, Event]] = field(default_factory=list, repr=False)
    residuals: np.ndarray | None = field(default=None, repr=False)
    used_events: list[Event] = field(default_factory=list, repr=False)
    weights: np.ndarray | None = field(default=None, repr=False)

    def predict(self, event: Event) -> float | None:
        if event.level not in self.alpha:
            return None
        s = months_between(self.tau, event.vote_date) / self.h
        y = self.alpha[event.level] + self.beta * s
        if self.gamma is not None:
            y += self.gamma * s * s
        return y


def _level_weights(events: list[Event], kernel: np.ndarray) -> np.ndarray:
    """$\\omega_e$の、時間方向の重みを除いた部分 $W_\\ell/\\sum W \\cdot P_e/\\Pi_\\ell$。"""
    N: dict[str, float] = {}
    Pi: dict[str, float] = {}
    for e, k in zip(events, kernel):
        N[e.level] = N.get(e.level, 0.0) + k * e.n * e.P
        Pi[e.level] = Pi.get(e.level, 0.0) + k * e.P
    W = {lv: math.sqrt(v) for lv, v in N.items() if v > 0}
    W_total = sum(W.values())
    return np.array([W[e.level] / W_total * e.P / Pi[e.level] for e in events])


WEIGHT_MODES = ("design", "sqrt_pop", "equal", "fgls")


def _weights(events: list[Event], kernel: np.ndarray, mode: str) -> np.ndarray:
    """時間方向の重みを含めた$\omega_e$。designは$C_p(T)$と同じ階層内・階層間の重み。
    sqrt_pop($\sqrt{P_e}$のみ)とequal(選挙1件ずつ同じ重み)は、designでは少数の
    大きな選挙が重みを占めて実効標本数が小さくなりすぎるため、比較用に用意した。"""
    if mode == "design":
        return kernel * _level_weights(events, kernel)
    if mode == "sqrt_pop":
        return kernel * np.sqrt([e.P for e in events])
    if mode in ("equal", "fgls"):
        # fglsは政党ごとに分散を推定してから_fgls_weightsで掛け直す
        return kernel.copy()
    raise ValueError(mode)


def _fgls_weights(X_levels: np.ndarray, P: np.ndarray, y: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, float]:
    """誤差の分散を$\sigma_\ell^2P_e^{-a}$と置き、傾きを入れない水準だけの当てはめの
    残差から$\sigma_\ell^2$と$a$を推定して、その逆数を重みにする。分散の推定に
    傾きを使わないのは、重みの選択が$D_p$の値に引きずられないようにするため。
    戻り値は(重み, a)。"""
    theta, *_ = np.linalg.lstsq(X_levels * np.sqrt(kernel)[:, None], y * np.sqrt(kernel), rcond=None)
    r2 = (y - X_levels @ theta) ** 2
    if not np.average(r2, weights=kernel) > 0:
        return kernel.copy(), float("nan")
    Z = np.column_stack([X_levels, np.log(P)])
    # log(r^2)の回帰。r=0の点で発散しないよう、全体の平均二乗残差の1%を下駄にする
    target = np.log(r2 + 0.01 * np.average(r2, weights=kernel))
    coef, *_ = np.linalg.lstsq(Z * np.sqrt(kernel)[:, None], target * np.sqrt(kernel), rcond=None)
    log_var = Z @ coef
    return kernel * np.exp(-(log_var - log_var.min())), float(-coef[-1])


_WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def fit_trends(
    events: list[Event],
    tau: date,
    h: float,
    parties: list[str],
    degree: int = 2,
    first_dates: dict[str, date] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
    keep_details: bool = False,
    weights: str = "design",
) -> dict[str, TrendFit]:
    """評価時点tauでの各政党の傾きを推定する。推定できない政党は結果に含めない。

    標準誤差は投票日をクラスタとする頑健な標準誤差(CR1)。bootstrap>0なら、
    帰無仮説$\\beta_p=0$のもとでのワイルド・クラスタ・ブートストラップ(Webb重み)の
    p値も求める。重みが人口のべき分布に引きずられて正規近似が怪しいため。
    """
    pool = []
    for e in events:
        if e.vote_date > tau:
            break
        s = months_between(tau, e.vote_date) / h
        if s < MIN_LOG_KERNEL:
            continue
        y = normalized_shares(e.shares)
        if y is None:
            continue
        pool.append((e, s, y))
    if not pool:
        return {}

    s_all = np.array([s for _, s, _ in pool])
    kernel = np.exp(s_all)
    omega_all = _weights([e for e, _, _ in pool], kernel, weights)

    results: dict[str, TrendFit] = {}
    for party in parties:
        start = (first_dates or {}).get(party)
        if start is None or start > tau:
            continue
        idx = [i for i, (e, _, _) in enumerate(pool) if e.vote_date >= start]
        if not idx:
            continue
        fit = _fit_one(
            party, tau, h, degree, [pool[i] for i in idx], s_all[idx], omega_all[idx],
            bootstrap, seed, keep_details, weights == "fgls",
        )
        if fit is not None:
            results[party] = fit
    return results


def _fit_one(party, tau, h, degree, rows, s, omega, bootstrap, seed, keep_details, fgls=False):
    events = [e for e, _, _ in rows]
    y = np.array([yy.get(party, 0.0) for _, _, yy in rows])
    levels = sorted({e.level for e in events})
    level_idx = {lv: i for i, lv in enumerate(levels)}
    L = len(levels)
    X = np.zeros((len(events), L + degree))
    for i, e in enumerate(events):
        X[i, level_idx[e.level]] = 1.0
    for d in range(1, degree + 1):
        X[:, L + d - 1] = s ** d
    a_hat = None
    if fgls:
        omega, a_hat = _fgls_weights(X[:, :L], np.array([e.P for e in events]), y, omega)

    sw = np.sqrt(omega)
    Xw = X * sw[:, None]
    XtWX = Xw.T @ Xw
    if np.linalg.matrix_rank(XtWX) < X.shape[1]:
        return None
    bread = np.linalg.inv(XtWX)
    P = bread @ (X * omega[:, None]).T  # theta = P @ y
    theta = P @ y
    resid = y - X @ theta
    # 次数0(水準のみ)は予測誤差の比較の基準として使うだけで、傾きは0とする
    c = P[L] if degree >= 1 else np.zeros(len(y))

    clusters = {}
    for i, e in enumerate(events):
        clusters.setdefault(e.vote_date, []).append(i)
    groups = list(clusters.values())
    G = len(groups)
    n, k = X.shape
    if G < 2 or n <= k:
        return None
    adj = G / (G - 1) * (n - 1) / (n - k)
    se = math.sqrt(adj * sum(float(c[g] @ resid[g]) ** 2 for g in groups))

    g_weight = np.array([omega[g].sum() for g in groups])
    eff_clusters = g_weight.sum() ** 2 / (g_weight ** 2).sum()
    n_eff = omega.sum() ** 2 / (omega ** 2).sum()

    beta = float(theta[L]) if degree >= 1 else 0.0
    p_wild = None
    if bootstrap > 0 and se > 0 and degree >= 1:
        p_wild = _wild_cluster_p(X, omega, y, L, groups, beta / se, adj, bootstrap, seed, P, c)

    contributions = []
    if keep_details:
        contrib = c * y / h
        contributions = sorted(zip(contrib.tolist(), events), key=lambda t: -abs(t[0]))
    return TrendFit(
        party=party, tau=tau, h=h, degree=degree,
        D=beta / h, se=se / h, beta=beta,
        gamma=float(theta[L + 1]) if degree >= 2 else None,
        alpha={lv: float(theta[i]) for lv, i in level_idx.items()},
        n_events=n, n_eff=float(n_eff), max_weight_share=float(omega.max() / omega.sum()),
        n_clusters=G, eff_clusters=float(eff_clusters), p_wild=p_wild, a_hat=a_hat,
        contributions=contributions,
        residuals=resid if keep_details else None,
        used_events=events if keep_details else [],
        weights=omega if keep_details else None,
    )


def _wild_cluster_p(X, omega, y, L, groups, t_obs, adj, B, seed, P, c):
    """帰無仮説(sの1次の係数が0)を課した制約付き残差でのワイルド・クラスタ・
    ブートストラップ(WCR)。$\\hat\\beta^*=c\\cdot y^*$と線形なので、反復ごとの
    再推定は行列積だけで済む。"""
    Xr = np.delete(X, L, axis=1)
    sw = np.sqrt(omega)
    theta_r, *_ = np.linalg.lstsq(Xr * sw[:, None], y * sw, rcond=None)
    fitted_r = Xr @ theta_r
    resid_r = y - fitted_r

    rng = np.random.default_rng(seed)
    v = rng.choice(_WEBB, size=(len(groups), B))
    V = np.empty((len(y), B))
    for gi, g in enumerate(groups):
        V[g] = v[gi]
    Y = fitted_r[:, None] + resid_r[:, None] * V
    theta_b = P @ Y
    R = Y - X @ theta_b
    beta_b = theta_b[L]
    cR = c[:, None] * R
    var_b = np.zeros(B)
    for g in groups:
        var_b += cR[g].sum(axis=0) ** 2
    se_b = np.sqrt(adj * var_b)
    t_b = beta_b / np.where(se_b > 0, se_b, np.nan)
    return float(np.nanmean(np.abs(t_b) >= abs(t_obs)))


def forecast_loss(
    events: list[Event], fits: dict[str, TrendFit], tau: date, horizon_months: float,
    first_dates: dict[str, date], weights: str = "design",
) -> tuple[float, int] | None:
    """tauの後horizon_months以内の選挙について、当てはめ曲線を延ばした予測の
    重み付き二乗誤差(全政党の合計)と、評価に使ったイベント数を返す。
    重みは予測対象の選挙だけで計算し、合計1に正規化する(時間方向の重みなし)。"""
    future = []
    for e in events:
        if e.vote_date <= tau:
            continue
        if months_between(tau, e.vote_date) > horizon_months:
            break
        y = normalized_shares(e.shares)
        if y is not None:
            future.append((e, y))
    if not future:
        return None
    w = _weights([e for e, _ in future], np.ones(len(future)), weights)
    w = w / w.sum()
    loss = 0.0
    used = set()
    for party, fit in fits.items():
        for i, (e, y) in enumerate(future):
            if first_dates.get(party, tau) > e.vote_date:
                continue
            pred = fit.predict(e)
            if pred is None:
                continue
            loss += w[i] * (y.get(party, 0.0) - pred) ** 2
            used.add(i)
    return loss, len(used)


def month_ends(start: tuple[int, int], end: tuple[int, int]) -> list[date]:
    result = []
    y, m = start
    while (y, m) <= end:
        ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
        result.append(date.fromordinal(date(ny, nm, 1).toordinal() - 1))
        y, m = ny, nm
    return result


def backtest(
    events: list[Event], parties: list[str], taus: list[date], hs: list[float],
    degrees: list[int], horizon_months: float = 3.0,
    fit_weights: str = "design", loss_weights: str = "design",
) -> dict:
    """(h, 次数)の組ごとに、各評価時点での予測誤差を集計する。同じ評価時点では
    全ての組が同じ予測対象を使うよう、どれかの組で予測できなかった時点は除く。"""
    first = first_appearance(events)
    per_tau: dict[str, list[float]] = {}
    valid = []
    for tau in taus:
        row = {}
        for h in hs:
            for deg in degrees:
                fits = fit_trends(events, tau, h, parties, degree=deg, first_dates=first, weights=fit_weights)
                r = forecast_loss(events, fits, tau, horizon_months, first, weights=loss_weights)
                if r is None or r[1] == 0:
                    row = None
                    break
                row[f"h={h:g},deg={deg}"] = r[0]
            if row is None:
                break
        if row is None:
            continue
        valid.append(tau.isoformat())
        for key, v in row.items():
            per_tau.setdefault(key, []).append(v)
    summary = {key: float(np.mean(v)) for key, v in per_tau.items()}
    return {"taus": valid, "per_tau": per_tau, "mean_loss": summary}


def diagnostics(fit: TrendFit) -> dict:
    """前提の確認: 統一地方選挙とそれ以外の残差の差、隣り合う投票日の残差の相関。"""
    events, r, w = fit.used_events, fit.residuals, fit.weights
    unified = np.array([(e.vote_date.year, e.vote_date.month) in UNIFIED_LOCAL_ELECTION_MONTHS
                        and e.level in ("市区町村長", "市区町村議会", "都道府県知事", "都道府県議会")
                        for e in events])
    local = np.array([e.level in ("市区町村長", "市区町村議会") for e in events])
    out = {}
    if unified.any() and (local & ~unified).any():
        out["resid_unified_minus_other"] = float(
            np.average(r[unified], weights=w[unified]) - np.average(r[local & ~unified], weights=w[local & ~unified])
        )
    by_date: dict[date, list[int]] = {}
    for i, e in enumerate(events):
        by_date.setdefault(e.vote_date, []).append(i)
    dates = sorted(by_date)
    means = np.array([np.average(r[by_date[d]], weights=w[by_date[d]]) for d in dates])
    gw = np.array([w[by_date[d]].sum() for d in dates])
    if len(means) > 3:
        a, b = means[:-1], means[1:]
        ww = np.sqrt(gw[:-1] * gw[1:])
        ma, mb = np.average(a, weights=ww), np.average(b, weights=ww)
        cov = np.average((a - ma) * (b - mb), weights=ww)
        out["lag1_date_resid_corr"] = float(
            cov / math.sqrt(np.average((a - ma) ** 2, weights=ww) * np.average((b - mb) ** 2, weights=ww))
        )
    return out


def split_alliances(events: list[Event]) -> list[Event]:
    """合流した名義の議席を元の党に戻す。国政選挙は当選者の出身で分け、それ以外の選挙の
    推薦は元の党で等分する(相乗りの首長選と同じ扱いになる)。"""
    from .lineage import ALLIANCE_MEMBERS, ALLIANCE_ORIGIN_SEATS

    out = []
    for e in events:
        if not any(a in e.shares for a in ALLIANCE_MEMBERS):
            out.append(e)
            continue
        shares = {p: v for p, v in e.shares.items() if p not in ALLIANCE_MEMBERS}
        for a, members in ALLIANCE_MEMBERS.items():
            v = e.shares.get(a, 0.0)
            if not v:
                continue
            seats = ALLIANCE_ORIGIN_SEATS.get((e.vote_date.isoformat(), a)) if e.level in ("衆院選", "参院選") else None
            weights = {p: s / sum(seats.values()) for p, s in seats.items()} if seats else \
                {p: 1.0 / len(members) for p in members}
            for p, w in weights.items():
                shares[p] = shares.get(p, 0.0) + v * w
        out.append(Event(e.vote_date, e.level, e.P, e.n, shares, e.name))
    return out


def load_events() -> tuple[list[Event], date | None]:
    from .fetch import jichisoken
    from .monthly_index import build_month_entries

    cutoff = endorsement_cutoff(jichisoken.municipal_head_endorsements_by_year())
    return split_alliances(events_from_month_entries(build_month_entries(), cutoff)), cutoff


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="党勢の趨勢D_pの後ろ向き検証と最新値")
    ap.add_argument("--hs", default="6,9,12,18")
    ap.add_argument("--bootstrap", type=int, default=1999)
    args = ap.parse_args(argv)
    hs = [float(x) for x in args.hs.split(",")]

    events, cutoff = load_events()
    first = first_appearance(events)
    parties = sorted(p for p in first if p != INDEPENDENT)
    last = events[-1].vote_date
    latest_tau = month_ends((last.year, last.month), (last.year, last.month))[-1]

    start_year = 2018 + math.ceil(2 * max(hs) / 12)
    taus = [t for t in month_ends((start_year, 1), (last.year, last.month))
            if months_between(t, last) >= 3]
    bt = backtest(events, parties, taus, hs, [0, 1, 2])
    best = min(((k, v) for k, v in bt["mean_loss"].items() if k.endswith("deg=2")), key=lambda kv: kv[1])
    h_best = float(best[0].split(",")[0].split("=")[1])

    fits = fit_trends(events, latest_tau, h_best, parties, first_dates=first,
                      bootstrap=args.bootstrap, keep_details=True)
    series = {}
    for tau in month_ends((2018 + math.ceil(2 * h_best / 12), 1), (latest_tau.year, latest_tau.month)):
        for p, f in fit_trends(events, tau, h_best, parties, first_dates=first).items():
            series.setdefault(p, []).append({"tau": tau.isoformat(), "D": f.D, "se": f.se})

    latest = {}
    for p, f in fits.items():
        latest[p] = {
            "D": f.D, "se": f.se, "p_wild": f.p_wild, "n_events": f.n_events, "n_eff": f.n_eff,
            "max_weight_share": f.max_weight_share, "n_clusters": f.n_clusters,
            "eff_clusters": f.eff_clusters, "diagnostics": diagnostics(f),
            "top_contributions": [
                {"D_contrib": v, "vote_date": e.vote_date.isoformat(), "level": e.level, "name": e.name}
                for v, e in f.contributions[:8]
            ],
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "classified_until": cutoff.isoformat() if cutoff else None,
        "latest_tau": latest_tau.isoformat(), "h_selected": h_best,
        "backtest_mean_loss": bt["mean_loss"], "backtest_n_taus": len(bt["taus"]),
        "backtest_taus": [bt["taus"][0], bt["taus"][-1]] if bt["taus"] else [],
        "latest": latest, "series": series,
    }
    path = OUTPUT_DIR / f"trend_{latest_tau.isoformat()}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    main()
