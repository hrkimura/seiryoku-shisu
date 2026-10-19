"""全政党をまとめた党勢の状態空間モデル。

政党ごとのモデル(state_space_glm)と同じく、政党pの党勢をz_p=logit x_p、階層の水準差を
alpha_pl、自治体の地盤をu_plmとする。違いは地方選挙の観測で、追跡する政党とその他の議席数の
組を、1つの多項分布Mult(n*, softmax(eta, 0))で観測する。eta_p=gamma_p z_p+alpha_pl+u_plm。
ある党が議席を得れば他の党の議席が減るという関係が、観測の中に入る。
政党ごとのモデルで議会のベータ二項分布の精度phiが二項分布と区別できないほど大きく推定されたので、
その極限の多項分布を使う。国政選挙は政党ごとに独立にz_pを観測する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.special import gammaln

from . import state_space as ss
from . import state_space_glm as glm
from .trend import INDEPENDENT, UNCLASSIFIED_MAYOR, Event, normalized_shares

MAYOR_LEVELS = glm.MAYOR_LEVELS


@dataclass
class JointData:
    parties: list[str]
    first: dict[str, date]
    t: np.ndarray
    level: list[str]
    name: list[str]
    n: np.ndarray
    k: np.ndarray
    events: list[Event]
    # 追跡しない政党の議席数(n*から追跡する政党の分を引いたもの)
    others: np.ndarray = field(default_factory=lambda: np.zeros(0))


def joint_data(events: list[Event], parties: list[str], first: dict[str, date]) -> JointData:
    """選挙ごとに、無所属を除いた議席数n*と、追跡する政党の議席数の組を作る。
    追跡する政党がまだ現れていない時期の選挙も含める(その党の議席は0)。"""
    start = min(first[p] for p in parties)
    rows = []
    for e in events:
        if e.level == UNCLASSIFIED_MAYOR:
            continue
        if e.vote_date < start:
            continue
        sh = normalized_shares(e.shares)
        if sh is None:
            continue
        n = e.n * (1.0 - e.shares.get(INDEPENDENT, 0.0))
        rows.append((e, n, [sh.get(p, 0.0) * n for p in parties]))
    return JointData(parties=list(parties), first=dict(first),
                     t=np.array([ss._t(e.vote_date) for e, _, _ in rows]),
                     level=[e.level for e, _, _ in rows], name=[e.name for e, _, _ in rows],
                     n=np.array([n for _, n, _ in rows]), k=np.array([k for _, _, k in rows]),
                     events=[e for e, _, _ in rows],
                     others=np.array([max(n - sum(k), 0.0) for _, n, k in rows]))


@dataclass
class JointParams:
    q: dict[str, float]
    ell: dict[str, float]
    # 地盤の分散。キーは(政党, 階層)。
    tau2: dict[tuple[str, str], float] = field(default_factory=dict)
    san_load: float = 1.0
    san_offset: float = 0.0
    san_prior_var: float = glm.PRIOR_VAR_ALPHA
    sigma_national: float | None = None
    prior_var: float = glm.PRIOR_VAR_Z
    # 政党どうしのz_pの増分の相関行列(partiesの順)。Noneなら独立。
    corr: np.ndarray | None = None

    def noise(self, parties: list[str]) -> np.ndarray:
        """単位時間あたりのz_pの増分の共分散行列。"""
        sq = np.sqrt([self.q[p] for p in parties])
        R = np.eye(len(parties)) if self.corr is None else self.corr
        return sq[:, None] * R * sq[None, :]


@dataclass
class JointFilterResult:
    loglik: float
    mean: np.ndarray
    cov: np.ndarray
    levels: list[str]


class _Layout:
    """状態ベクトルの並び。政党ごとに[z_p, alpha_p(衆院選以外の階層)]のブロックを並べる。"""

    def __init__(self, parties: list[str], levels: list[str]):
        self.parties = parties
        self.levels = levels
        self.block = 1 + len(levels)
        self.dim = len(parties) * self.block

    def z(self, a: int) -> int:
        return a * self.block

    def alpha(self, a: int, level: str) -> int | None:
        if level == ss.REFERENCE_LEVEL:
            return None
        return a * self.block + 1 + self.levels.index(level)


def _softmax(eta: np.ndarray) -> tuple[np.ndarray, float]:
    top = max(eta.max(), 0.0)
    e = np.exp(eta - top)
    z0 = math.exp(-top)
    return e / (e.sum() + z0), z0 / (e.sum() + z0)


def _psd(W: np.ndarray) -> np.ndarray:
    lam, vec = np.linalg.eigh(0.5 * (W + W.T))
    return (vec * np.maximum(lam, 0.0)) @ vec.T


def _counts_loglik(k: np.ndarray, k0: float, n: float):
    """議席数の組の多項分布。戻り値の関数はetaから(対数尤度, 勾配, 負のヘッセ行列)を返す。"""
    const = gammaln(n + 1) - gammaln(k + 1).sum() - gammaln(k0 + 1)

    def f(eta):
        mu, mu0 = _softmax(eta)
        ll = const + float(k @ np.log(mu)) + (k0 * math.log(mu0) if k0 > 0 else 0.0)
        return ll, k - n * mu, n * (np.diag(mu) - np.outer(mu, mu))
    return f


def _set_loglik(in_set: np.ndarray, others_in: bool):
    """当選者が政党の組Sのどれかの陣営である、という首長選の観測(相乗り)。P=sum_{c in S} mu_c。
    党pへの配分r_p=mu_p/Pは、直前の党勢から見込まれる強さに比例する。"""
    def f(eta):
        mu, mu0 = _softmax(eta)
        P = float(mu[in_set].sum()) + (mu0 if others_in else 0.0)
        r = np.where(in_set, mu, 0.0) / P
        W = (np.diag(mu) - np.outer(mu, mu)) - (np.diag(r) - np.outer(r, r))
        return math.log(P), r - mu, W
    return f


def _laplace(m: np.ndarray, S: np.ndarray, f):
    """事前分布N(m, S)と対数尤度fの積を最大化し、事後分布N(eta, V)と対数周辺尤度を返す。
    負のヘッセ行列が半正定値でないところは0で抑える(その方向には情報がないとみなす)。"""
    Sinv = np.linalg.inv(S)
    eta = m.copy()
    for _ in range(60):
        _, g, W = f(eta)
        W = _psd(W)
        step = np.linalg.solve(W + Sinv, g - Sinv @ (eta - m))
        big = np.abs(step).max()
        if big > 2.0:
            step *= 2.0 / big
        eta = eta + step
        if big < 1e-9:
            break
    ll, _, W = f(eta)
    W = _psd(W)
    V = np.linalg.inv(Sinv + W)
    dev = eta - m
    _, logdet = np.linalg.slogdet(np.eye(len(m)) + S @ W)
    return eta, V, Sinv, ll - 0.5 * float(dev @ Sinv @ dev) - 0.5 * logdet


def run_filter(data: JointData, params: JointParams, until: float | None = None, trace: list | None = None,
               contrib: list | None = None, levels: list[str] | None = None,
               mayor_sets: bool = False) -> JointFilterResult:
    """traceとcontribの意味はglm.run_filterと同じ。contribの要素は(添字, 政党ごとのzの変化)。
    mayor_setsなら、首長選の相乗りを等分せず、当選者が推薦・支持した党のどれかの陣営である
    という観測として扱う(_set_loglik)。
    地盤u_plmは状態の外に置き、平均・分散と状態との共分散を持つ(自治体どうし、政党どうしの
    地盤の共分散は捨てる)。"""
    parties = data.parties
    if levels is None:
        levels = sorted({lv for lv in data.level if lv != ss.REFERENCE_LEVEL})
    lay = _Layout(parties, levels)
    D = lay.dim
    m = np.zeros(D)
    C = np.eye(D) * params.prior_var
    if glm.SANGIIN in levels:
        for a in range(len(parties)):
            j = lay.alpha(a, glm.SANGIIN)
            m[j] = params.san_offset
            C[j, j] = params.san_prior_var
    zi = np.array([lay.z(a) for a in range(len(parties))])
    Q = params.noise(parties)
    zz = np.ix_(zi, zi)
    rows: dict[tuple[str, str, str], int] = {}
    for i, lv in enumerate(data.level):
        if lv in ss.NATIONAL_LEVELS:
            continue
        for p in parties:
            if (p, lv) in params.tau2:
                rows.setdefault((p, lv, data.name[i]), len(rows))
    um = np.zeros(len(rows))
    uv = np.array([params.tau2[(p, g)] for p, g, _ in rows])
    cc = np.zeros((len(rows), D))
    touched = np.zeros(len(rows), bool)
    t_prev = data.t[0] if len(data.t) else 0.0
    loglik = 0.0
    seen: set[str] = set()
    n_seen = 0
    if trace is not None and len(data.t):
        trace.append([t_prev, m.copy(), C.copy(), m.copy(), C.copy()])
    for i in range(len(data.t)):
        t = data.t[i]
        if until is not None and t > until:
            break
        dt = t - t_prev
        if dt > 0:
            C[zz] += Q * dt
            t_prev = t
            if trace is not None:
                trace.append([t, m.copy(), C.copy(), m.copy(), C.copy()])
        lv = data.level[i]
        d = data.events[i].vote_date
        active = [a for a, p in enumerate(parties) if data.first[p] <= d]
        m_before = m.copy()
        if lv in ss.NATIONAL_LEVELS:
            load = params.san_load if lv == glm.SANGIIN else 1.0
            H = np.zeros((len(active), D))
            y = np.zeros(len(active))
            R = np.zeros(len(active))
            for r_, a in enumerate(active):
                H[r_, lay.z(a)] = load
                j = lay.alpha(a, lv)
                if j is not None:
                    H[r_, j] = 1.0
                y[r_], sd = glm.national_logit(data.k[i, a], data.n[i], params.sigma_national)
                R[r_] = sd * sd
            cov_core = C @ H.T
            S = H @ cov_core + np.diag(R)
            v = y - H @ m
            Sinv = np.linalg.inv(S)
            sign, logdet = np.linalg.slogdet(S)
            ll_i = -0.5 * (len(y) * math.log(2 * math.pi) + logdet + v @ Sinv @ v)
            r_idx = np.zeros(0, int)
            A_vec, B = Sinv @ v, Sinv
        else:
            n = data.n[i]
            if n <= 1e-9 or not active:
                continue
            H = np.zeros((len(active), D))
            for r_, a in enumerate(active):
                H[r_, lay.z(a)] = params.ell[parties[a]]
                H[r_, lay.alpha(a, lv)] = 1.0
            r_idx = np.array([rows.get((parties[a], lv, data.name[i]), -1) for a in active])
            has_u = r_idx >= 0
            if has_u.any():
                # glm.run_filterと同じく、状態との共分散を地盤の分散と矛盾しない大きさに縮める
                Cp = np.linalg.pinv(C)
                for r in r_idx[has_u]:
                    qf = float(cc[r] @ Cp @ cc[r])
                    if uv[r] <= 0:
                        cc[r] = 0.0
                    elif qf > 0.99 * uv[r]:
                        cc[r] *= math.sqrt(0.99 * uv[r] / qf)
            cov_core = C @ H.T
            m_eta = H @ m
            if has_u.any():
                ru = r_idx[has_u]
                cov_core[:, has_u] += cc[ru].T
                m_eta[has_u] += um[ru]
            S_eta = H @ C @ H.T
            if has_u.any():
                HcT = H @ cc[ru].T
                S_eta[:, has_u] += HcT
                S_eta[has_u, :] += HcT.T
                S_eta[np.ix_(has_u, has_u)] += np.diag(uv[ru])
            # 地盤どうしの共分散を捨てているので、S_etaが正定値から外れうる。固有値を下から抑える
            lam_, vec_ = np.linalg.eigh(0.5 * (S_eta + S_eta.T))
            lam_ = np.maximum(lam_, 1e-9 * max(lam_.max(), 1e-12))
            S_eta = (vec_ * lam_) @ vec_.T
            k = np.array([data.k[i, a] for a in active])
            if mayor_sets and lv in MAYOR_LEVELS:
                in_set = k > 1e-12
                f = _set_loglik(in_set, data.others[i] > 1e-12)
            else:
                f = _counts_loglik(k, max(n - k.sum(), 0.0), n)
            eta, V, S_eta_inv, ll_i = _laplace(m_eta, S_eta, f)
            # 擬似観測を経由せず、etaの事後分布から状態を直接条件づける
            A_vec = S_eta_inv @ (eta - m_eta)
            B = S_eta_inv @ (S_eta - V) @ S_eta_inv
        m = m + cov_core @ A_vec
        C = glm._psd_project(C - cov_core @ B @ cov_core.T)
        if len(rows):
            # 今回観測した自治体の地盤は、自分の分散の分も観測と相関する
            obs_rows = [(c_, r) for c_, r in enumerate(r_idx) if r >= 0]
            act = np.flatnonzero(touched)
            extra = [r for _, r in obs_rows if not touched[r]]
            if extra:
                act = np.concatenate([act, np.array(extra, int)])
            wv_full = cc[act] @ H.T
            pos = {r: j for j, r in enumerate(act)}
            for c_, r in obs_rows:
                wv_full[pos[r], c_] += uv[r]
            WB = wv_full @ B
            um[act] += wv_full @ A_vec
            uv[act] = np.maximum(uv[act] - np.einsum("ij,ij->i", WB, wv_full), 0.0)
            cc[act] -= WB @ cov_core.T
            touched[act] = True
        if trace is not None:
            trace[-1][3] = m.copy()
            trace[-1][4] = C.copy()
        if contrib is not None:
            contrib.append((i, {parties[a]: float(m[lay.z(a)] - m_before[lay.z(a)]) for a in range(len(parties))}))
        if n_seen >= 2 and lv in seen:
            loglik += ll_i
        seen.add(lv)
        n_seen += 1
    if until is not None and until > t_prev:
        C = C.copy()
        C[zz] += Q * (until - t_prev)
    return JointFilterResult(loglik, m, C, levels)


def smooth(data: JointData, params: JointParams, times: list[float], levels: list[str] | None = None,
           mayor_sets: bool = False) -> tuple[list[tuple[np.ndarray, np.ndarray]], _Layout]:
    """固定区間平滑化。遷移は単位行列で、雑音はz_pにだけ入る。"""
    trace: list = []
    r = run_filter(data, params, trace=trace, levels=levels, mayor_sets=mayor_sets)
    lay = _Layout(data.parties, r.levels)
    zi = np.array([lay.z(a) for a in range(len(data.parties))])
    Q = params.noise(data.parties)
    n = len(trace)
    ms = [None] * n
    Cs = [None] * n
    ms[-1], Cs[-1] = trace[-1][3], trace[-1][4]
    for k in range(n - 2, -1, -1):
        J = trace[k][4] @ np.linalg.pinv(trace[k + 1][2])
        ms[k] = trace[k][3] + J @ (ms[k + 1] - trace[k + 1][1])
        Cs[k] = trace[k][4] + J @ (Cs[k + 1] - trace[k + 1][2]) @ J.T
    ts = np.array([tr[0] for tr in trace])
    out = []
    for tau in times:
        k = int(np.searchsorted(ts, tau, side="right")) - 1
        if k < 0:
            out.append((trace[0][1], trace[0][2]))
            continue
        if ts[k] == tau:
            out.append((ms[k], Cs[k]))
            continue
        mp = trace[k][3]
        Cp = trace[k][4].copy()
        Cp[np.ix_(zi, zi)] += Q * (tau - ts[k])
        if k == n - 1:
            out.append((mp, Cp))
            continue
        J = Cp @ np.linalg.pinv(trace[k + 1][2])
        out.append((mp + J @ (ms[k + 1] - trace[k + 1][1]), Cp + J @ (Cs[k + 1] - trace[k + 1][2]) @ J.T))
    return out, lay


def params_from_party_fits(fits: dict[str, glm.GlmParams], san: dict) -> JointParams:
    """政党ごとのモデルの推定値から、まとめたモデルのパラメータを作る。"""
    return JointParams(q={p: f.q_x for p, f in fits.items()}, ell={p: f.ell for p, f in fits.items()},
                       tau2={(p, g): v for p, f in fits.items() for g, v in f.tau2.items()}, **san)


def national_interpolation_check(data: JointData, params_for: callable, seat_vote: list, seat_vote_san: list,
                                 mayor_sets: bool = False) -> list[dict]:
    """国政選挙を1回ずつ除き、残りの全期間のデータで平滑化して、その選挙の各党の結果を再現できるか。
    params_for(除いた選挙の日付, 除いた後のデータ, 衆院の点, 参院の点)がパラメータを返す。"""
    national = [i for i, lv in enumerate(data.level) if lv in ss.NATIONAL_LEVELS]
    levels_all = sorted({lv for lv in data.level if lv != ss.REFERENCE_LEVEL})
    results = []
    for i in national[1:]:
        e = data.events[i]
        shu = [p for p in seat_vote if p.vote_date != e.vote_date]
        san = [p for p in seat_vote_san if p.vote_date != e.vote_date]
        keep = np.ones(len(data.t), bool)
        keep[i] = False
        idx = np.flatnonzero(keep)
        sub = JointData(parties=data.parties, first=data.first, t=data.t[idx],
                        level=[data.level[j] for j in idx], name=[data.name[j] for j in idx],
                        n=data.n[idx], k=data.k[idx], events=[data.events[j] for j in idx],
                        others=data.others[idx])
        params = params_for(e.vote_date, sub, shu, san)
        (m, C), = smooth(sub, params, [data.t[i]], levels=levels_all, mayor_sets=mayor_sets)[0]
        lay = _Layout(data.parties, levels_all)
        load = params.san_load if e.level == glm.SANGIIN else 1.0
        for a, p in enumerate(data.parties):
            if data.first[p] > e.vote_date or data.first[p] == e.vote_date:
                continue
            h = np.zeros(lay.dim)
            h[lay.z(a)] = load
            j = lay.alpha(a, e.level)
            if j is not None:
                h[j] = 1.0
            yl, sd_nat = glm.national_logit(data.k[i, a], data.n[i], params.sigma_national)
            mu = float(h @ m)
            s_tot = math.hypot(math.sqrt(max(float(h @ C @ h), 0.0)), sd_nat)
            results.append({"party": p, "vote_date": e.vote_date.isoformat(), "level": e.level,
                            "y": float(data.k[i, a] / data.n[i]), "pred": glm._expit(mu),
                            "lo": glm._expit(mu - 2 * s_tot), "hi": glm._expit(mu + 2 * s_tot),
                            "z": (yl - mu) / s_tot})
    return results
