"""国政の党勢を地方選挙で補間する状態空間モデル(notes/state_space_proposal.tex)。

状態は国政での党勢の水準$x$、傾き$b$、階層ごとの水準差$\\alpha_\\ell$
(衆院選を基準に0とする)。国政選挙は$x$を誤差$\\sigma_{国}$で、地方選挙は
$\\lambda_px+\\alpha_\\ell$を誤差$\\sigma_\\ell^2P_e^{-a}$で観測する。負荷$\\lambda_p$は
衆院選の議席と比例代表の得票の関係から推定の外で決める(seat_vote_loading)。$\\sigma_{国}$は
「国政選挙の結果を党勢の定義とみなす」というスタンスとして固定し、残りは
全期間のデータから最尤法で推定する。
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize

from .trend import INDEPENDENT, Event, first_appearance, normalized_shares

REFERENCE_LEVEL = "衆院選"
NATIONAL_LEVELS = ("衆院選", "参院選")
SIGMA_NATIONAL = 0.005
PRIOR_VAR = 1.0
EPOCH = date(2018, 1, 1)
# 人口を1e5で割ってから冪を取り、sigma_ellを人口10万人の自治体での誤差の大きさとして読めるようにする
POP_SCALE = 1e5
# 地方の階層は、政党がその階層の選挙の一定割合以上で占有率を得ている場合だけ観測に使う。
# ほとんど常に0の階層を正規分布で扱うと、分散が0に推定されてx+alphaを0に固定する
# 「正確な観測」になり、党勢の変化を打ち消してしまうため。
MIN_NONZERO_SHARE = 0.1
MIN_LEVEL_OBS = 10


def _t(d: date) -> float:
    return (d - EPOCH).days / (365.25 / 12)


@dataclass
class Observations:
    t: np.ndarray
    level: list[str]
    P: np.ndarray
    y: np.ndarray
    events: list[Event]
    # 無所属を除いた議席数。正規化した占有率は、この数の議席から計算した比率である
    n_eff: np.ndarray = field(default_factory=lambda: np.zeros(0))


def party_observations(events: list[Event], party: str, start: date, filter_levels: bool = True) -> Observations:
    rows = []
    for e in events:
        if e.vote_date < start:
            continue
        shares = normalized_shares(e.shares)
        if shares is None:
            continue
        rows.append((e, shares.get(party, 0.0)))
    counts: dict[str, list[int]] = {}
    for e, y in rows:
        c = counts.setdefault(e.level, [0, 0])
        c[0] += 1
        c[1] += y > 0
    usable = {lv for lv, (n, nz) in counts.items()
              if lv in NATIONAL_LEVELS or not filter_levels
              or (n >= MIN_LEVEL_OBS and nz / n >= MIN_NONZERO_SHARE)}
    rows = [(e, y) for e, y in rows if e.level in usable]
    return Observations(
        t=np.array([_t(e.vote_date) for e, _ in rows]),
        level=[e.level for e, _ in rows],
        P=np.array([e.P for e, _ in rows]),
        y=np.array([y for _, y in rows]),
        events=[e for e, _ in rows],
        n_eff=np.array([e.n * (1.0 - e.shares.get(INDEPENDENT, 0.0)) for e, _ in rows]),
    )


@dataclass
class Params:
    q_x: float
    q_b: float
    sigma2: dict[str, float]
    a: float
    # 地方の階層の因子負荷。観測はlam*x+alpha。衆院選は小選挙区制で議席が得票の差を
    # 増幅するので、地方の占有率は国政の議席占有率ほどは動かないと考えられる。
    # 空なら全階層1(負荷を推定しないモデル)。
    lam: dict[str, float] = field(default_factory=dict)
    # 地方の階層の水準差alpha_ellのランダムウォークの分散(月あたり)。0なら定数。
    # 負荷を推定するときに0のままだと、lam->0・q_x->無限大という退化した解で
    # 地方独自のゆっくりした変動をxに負わせてしまうので、地方独自の変動の受け皿にする。
    q_alpha: float = 0.0
    # 傾きbの動き方。"rw": ランダムウォーク、"const": 一定(全期間で1つのドリフト)、
    # "none": 傾きなし(b=0)、"damped": 減衰率betaで0に戻る(db=-beta*b*dt+dW_b)。
    slope: str = "rw"
    beta: float = 0.0
    # 水準の跳び。月あたりの発生率jump_rateで、分散jump_varの跳びがxに加わる。
    # 0なら跳びなし(正規分布のランダムウォークだけ)。
    jump_rate: float = 0.0
    jump_var: float = 0.0
    # 地方選挙の観測の外れ値。確率outlier_probで、誤差の分散がoutlier_scale倍になる。
    # 無所属が大半の町村議会では無所属を除く正規化で占有率が1近くに膨らみ、
    # 首長選は0か1しかとらないので、正規分布1つでは裾が足りない。
    outlier_prob: float = 0.0
    outlier_scale: float = 1.0
    # 地方選挙の誤差に、無所属を除いた議席数n_effでの二項分布の分散p(1-p)/n_effを加える。
    # pは予測した占有率。無所属が大半の議会で正規化した占有率が1近くに膨らんでも、
    # n_effが小さいので情報の少ない観測として扱われる。
    binomial: bool = False

    def obs_var(self, level: str, P: float) -> float:
        if level in NATIONAL_LEVELS:
            return SIGMA_NATIONAL ** 2
        return self.sigma2[level] * (P / POP_SCALE) ** (-self.a)


@dataclass
class FilterResult:
    loglik: float
    mean: np.ndarray
    cov: np.ndarray
    state_levels: list[str]
    innovations: list[tuple[Event, float, float]] = field(default_factory=list, repr=False)


def _local_levels(obs: Observations) -> list[str]:
    return sorted({lv for lv in obs.level if lv not in NATIONAL_LEVELS})


def _state_levels(obs: Observations) -> list[str]:
    """状態に持つ水準差の階層。衆院選は基準なので含めない。"""
    return sorted({lv for lv in obs.level if lv != REFERENCE_LEVEL})


def run_filter(
    obs: Observations, params: Params, until: float | None = None, keep_innovations: bool = False,
    state_levels: list[str] | None = None, trace: list | None = None,
) -> FilterResult:
    """カルマンフィルタ。untilを与えると、それ以前の観測だけを使い、時刻untilまで
    予測した状態を返す。各階層の最初の観測と全体の最初の2観測は、事前分布に
    支配されるので対数尤度に含めない(散漫初期化の近似)。traceにリストを渡すと、観測のある
    時刻ごとに[t, 予測平均, 予測分散, 更新後平均, 更新後分散]を積む(rts_smooth用)。"""
    levels = state_levels if state_levels is not None else _state_levels(obs)
    idx = {lv: 2 + i for i, lv in enumerate(levels)}
    local_idx = [2 + i for i, lv in enumerate(levels) if lv not in NATIONAL_LEVELS] if params.q_alpha > 0 else []
    k = 2 + len(levels)
    m = np.zeros(k)
    C = np.eye(k) * PRIOR_VAR
    if params.slope == "none":
        C[1, 1] = 0.0
    elif params.slope == "damped":
        # 減衰する傾きは平均0の定常過程なので、定常分布を事前分布にする
        C[1, 1] = params.q_b / (2 * params.beta)
    else:
        C[1, 1] = 0.01  # 傾きの事前分散: 月あたり10ポイントの標準偏差で十分に散漫
    t_prev = obs.t[0] if len(obs.t) else 0.0
    loglik = 0.0
    seen: set[str] = set()
    n_seen = 0
    innov = []
    # 跳びを扱うときは「この間隔で跳ばなかった/跳んだ」の2成分の正規混合で持ち、
    # 次の時刻に進む前に1つの正規分布にまとめ直す(GPB1)。同じ投票日の観測は
    # 同じ成分のまま続けて更新する。
    comps = [(1.0, m, C)]
    if trace is not None and len(obs.t):
        trace.append([t_prev, m, C, m, C])
    for i in range(len(obs.t)):
        t = obs.t[i]
        if until is not None and t > until:
            break
        dt = t - t_prev
        if dt > 0:
            comps = _split(_collapse(comps), dt, params, local_idx)
            t_prev = t
            if trace is not None:
                mp, Cp = _collapse(comps)
                trace.append([t, mp, Cp, mp, Cp])
        lv = obs.level[i]
        j = idx.get(lv)
        lam = params.lam.get(lv, 1.0)
        r = params.obs_var(lv, obs.P[i])
        hyps = [(1.0, r)]
        if params.outlier_prob > 0 and lv not in NATIONAL_LEVELS:
            hyps = [(1.0 - params.outlier_prob, r), (params.outlier_prob, r * params.outlier_scale)]
        new = []
        v_mix = S_mix = 0.0
        for w, m, C in comps:
            if w <= 0:
                continue
            # h = lam*e_0 (+ e_j)
            Ch = lam * C[:, 0]
            pred = lam * m[0]
            if j is not None:
                Ch = Ch + C[:, j]
                pred += m[j]
            S0 = lam * Ch[0] + (Ch[j] if j is not None else 0.0)
            if params.binomial and lv not in NATIONAL_LEVELS:
                p_hat = min(max(pred, 0.01), 0.99)
                S0 += p_hat * (1 - p_hat) / max(obs.n_eff[i], 1.0)
            v = obs.y[i] - pred
            sub = []
            for wh, rh in hyps:
                S = S0 + rh
                v_mix += w * wh * v
                S_mix += w * wh * (S + v * v)
                K = Ch / S
                logp = math.log(w) + math.log(wh) - 0.5 * (math.log(2 * math.pi * S) + v * v / S)
                sub.append((logp, m + K * v, C - np.outer(K, Ch)))
            if len(sub) == 1:
                new.append(sub[0])
                continue
            # 外れ値かどうかの2仮説は、状態の成分ごとにその場でまとめる
            top = max(lp for lp, _, _ in sub)
            lp_sum = top + math.log(sum(math.exp(lp - top) for lp, _, _ in sub))
            mi, Ci = _collapse([(math.exp(lp - lp_sum), m2, C2) for lp, m2, C2 in sub])
            new.append((lp_sum, mi, Ci))
        top = max(lp for lp, _, _ in new)
        ll_i = top + math.log(sum(math.exp(lp - top) for lp, _, _ in new))
        comps = [(math.exp(lp - ll_i), m2, C2) for lp, m2, C2 in new]
        if trace is not None:
            trace[-1][3:5] = _collapse(comps)
        if n_seen >= 2 and lv in seen:
            loglik += ll_i
        if keep_innovations:
            innov.append((obs.events[i], v_mix, S_mix - v_mix * v_mix))
        seen.add(lv)
        n_seen += 1
    m, C = _collapse(comps)
    if until is not None and until > t_prev:
        m, C = _collapse(_split((m, C), until - t_prev, params, local_idx))
    return FilterResult(loglik, m, C, levels, innov)


def _collapse(comps) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(comps, tuple):
        return comps
    if len(comps) == 1:
        return comps[0][1], comps[0][2]
    m = sum(w * mi for w, mi, _ in comps)
    C = sum(w * (Ci + np.outer(mi - m, mi - m)) for w, mi, Ci in comps)
    return m, C


def _split(mc, dt: float, params: Params, local_idx) -> list:
    m, C = _predict(*mc, dt, params, local_idx)
    if params.jump_rate <= 0:
        return [(1.0, m, C)]
    # 間隔dtに1回以上跳ぶ確率。2回以上跳ぶ場合も分散jump_varの1回の跳びで近似する
    p_jump = -math.expm1(-params.jump_rate * dt)
    Cj = C.copy()
    Cj[0, 0] += params.jump_var
    return [(1.0 - p_jump, m, C), (p_jump, m, Cj)]


def _slope_transition(dt: float, params: Params) -> tuple[float, float, float, float, float]:
    """傾きの推移の係数(g, phi)と、傾きの雑音による共分散(Vxx, Vxb, Vbb)。
    推移はx += g*b, b *= phi。"""
    q = params.q_b if params.slope in ("rw", "damped") else 0.0
    if params.slope != "damped":
        return dt, 1.0, q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt
    bt = params.beta
    one_m_phi = -math.expm1(-bt * dt)
    phi = 1.0 - one_m_phi
    one_m_phi2 = -math.expm1(-2 * bt * dt)
    g = one_m_phi / bt
    if bt * dt < 1e-3:
        # 下の閉じた形は桁落ちするので、betaについての展開を使う
        vxx = q * (dt ** 3 / 3 - bt * dt ** 4 / 4)
    else:
        vxx = q / bt ** 2 * (dt - 2 * one_m_phi / bt + one_m_phi2 / (2 * bt))
    return g, phi, vxx, q * one_m_phi ** 2 / (2 * bt ** 2), q * one_m_phi2 / (2 * bt)


def _predict(m, C, dt, params: Params, local_idx: list[int] = ()):
    g, phi, vxx, vxb, vbb = _slope_transition(dt, params)
    m = m.copy()
    m[0] += g * m[1]
    m[1] *= phi
    # F C F^T を、F が左上2x2だけ単位行列でないことを使って更新する
    C = C.copy()
    C[0, :] += g * C[1, :]
    C[1, :] *= phi
    C[:, 0] += g * C[:, 1]
    C[:, 1] *= phi
    C[0, 0] += params.q_x * dt + vxx
    C[0, 1] += vxb
    C[1, 0] += vxb
    C[1, 1] += vbb
    for i in local_idx:
        C[i, i] += params.q_alpha * dt
    return m, C


def _transition(dt: float, params: Params, k_state: int) -> np.ndarray:
    g, phi, *_ = _slope_transition(dt, params)
    F = np.eye(k_state)
    F[0, 1], F[1, 1] = g, phi
    return F


def rts_smooth(trace: list, params: Params, local_idx: list[int], times: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """run_filterのtraceから、各時刻の状態の平均と分散を固定区間平滑化(RTS)で求める。
    跳びの混合は畳んだ1成分で近似する。local_idxは時間とともに分散q_alphaで動く状態の添字。"""
    k_state = len(trace[0][1])
    n = len(trace)
    ms = [None] * n
    Cs = [None] * n
    ms[-1], Cs[-1] = trace[-1][3], trace[-1][4]
    for k in range(n - 2, -1, -1):
        _, _, _, mf, Cf = trace[k]
        t1, mp1, Cp1 = trace[k + 1][:3]
        F = _transition(t1 - trace[k][0], params, k_state)
        J = Cf @ F.T @ np.linalg.pinv(Cp1)
        ms[k] = mf + J @ (ms[k + 1] - mp1)
        Cs[k] = Cf + J @ (Cs[k + 1] - Cp1) @ J.T
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
        mp, Cp = _collapse(_split((trace[k][3], trace[k][4]), tau - ts[k], params, local_idx))
        if k == n - 1:
            out.append((mp, Cp))
            continue
        F = _transition(ts[k + 1] - tau, params, k_state)
        J = Cp @ F.T @ np.linalg.pinv(trace[k + 1][2])
        out.append((mp + J @ (ms[k + 1] - trace[k + 1][1]), Cp + J @ (Cs[k + 1] - trace[k + 1][2]) @ J.T))
    return out


def _unpack(
    z: np.ndarray, local_levels: list[str], loadings: bool, drift: bool, slope: str = "rw", jumps: bool = False,
    outliers: bool = False, binomial: bool = False,
) -> Params:
    L = len(local_levels)
    k = 3 + L
    lam = {}
    if loadings:
        lam = {lv: float(z[k + i]) for i, lv in enumerate(local_levels)}
        k += L
    q_alpha = 0.0
    if drift:
        q_alpha = math.exp(z[k])
        k += 1
    beta = 0.0
    if slope == "damped":
        beta = math.exp(z[k])
        k += 1
    jump_rate = jump_var = 0.0
    if jumps:
        jump_rate, jump_var = math.exp(z[k]), math.exp(z[k + 1])
        k += 2
    outlier_prob, outlier_scale = 0.0, 1.0
    if outliers:
        outlier_prob, outlier_scale = 1 / (1 + math.exp(-z[k])), math.exp(z[k + 1])
    return Params(
        q_x=math.exp(z[0]), q_b=math.exp(z[1]),
        sigma2={lv: math.exp(z[2 + i]) for i, lv in enumerate(local_levels)},
        a=float(z[2 + L]), lam=lam, q_alpha=q_alpha, slope=slope, beta=beta,
        jump_rate=jump_rate, jump_var=jump_var,
        outlier_prob=outlier_prob, outlier_scale=outlier_scale, binomial=binomial,
    )


def fit_params(
    obs: Observations, until: float | None = None, maxiter: int = 500, loadings: bool = False,
    drift: bool = False, fixed_lam: float | None = None, slope: str = "rw", jumps: bool = False,
    outliers: bool = False, binomial: bool = False,
) -> tuple[Params, float]:
    """q_x, q_b, sigma_ell^2, a(loadingsならlam_ell、driftならq_alphaも)を最尤法で推定する。"""
    local = _local_levels(obs)
    state_levels = _state_levels(obs)
    L = len(local)
    z0 = [math.log(1e-5), math.log(1e-6)] + [math.log(0.05)] * L + [0.0]
    bounds = [(-30, 2), (-30, 2)] + [(-12, 2)] * L + [(-2, 2)]
    if loadings:
        z0 += [1.0] * L
        bounds += [(-1.0, 3.0)] * L
    if drift:
        z0 += [math.log(1e-6)]
        bounds += [(-30, 2)]
    if slope in ("const", "none"):
        # q_bを使わないので、最適化の変数から外れるよう値を固定する
        z0[1] = -30.0
        bounds[1] = (-30.0, -30.0)
    if slope == "damped":
        z0 += [math.log(0.1)]
        bounds += [(math.log(1e-3), math.log(5.0))]
    if jumps:
        # 跳びは年0.1回から月1回、大きさは標準偏差1ポイントから30ポイントの範囲
        z0 += [math.log(0.05), math.log(0.01)]
        bounds += [(math.log(0.1 / 12), 0.0), (math.log(1e-4), math.log(0.09))]
    if outliers:
        # 外れ値の確率は0.1%〜50%(logit)、分散の倍率は2倍〜1万倍
        z0 += [math.log(0.05 / 0.95), math.log(20.0)]
        bounds += [(math.log(0.001 / 0.999), 0.0), (math.log(2.0), math.log(1e4))]

    def unpack(z):
        p = _unpack(z, local, loadings, drift, slope, jumps, outliers, binomial)
        if fixed_lam is not None:
            p.lam = {lv: fixed_lam for lv in local}
        return p

    def nll(z):
        return -run_filter(obs, unpack(z), until=until, state_levels=state_levels).loglik

    res = minimize(nll, np.array(z0), method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    return unpack(res.x), -res.fun


@dataclass
class Nowcast:
    party: str
    tau: date
    x: float
    x_se: float
    D: float
    D_se: float
    alpha: dict[str, float]
    params: Params


def nowcast(obs: Observations, params: Params, party: str, tau: date) -> Nowcast:
    r = run_filter(obs, params, until=_t(tau), state_levels=_state_levels(obs))
    return Nowcast(
        party=party, tau=tau,
        x=float(r.mean[0]), x_se=math.sqrt(max(r.cov[0, 0], 0.0)),
        D=float(r.mean[1]), D_se=math.sqrt(max(r.cov[1, 1], 0.0)),
        alpha={lv: float(r.mean[2 + i]) for i, lv in enumerate(r.state_levels)},
        params=params,
    )


def predict_national(obs: Observations, params: Params, until: float, level: str) -> tuple[float, float]:
    """時刻untilまでの観測から、その時点の国政選挙(level)の占有率を予測する。
    戻り値は(予測値, 予測の標準偏差)。"""
    r = run_filter(obs, params, until=until, state_levels=_state_levels(obs))
    h = np.zeros(len(r.mean))
    h[0] = 1.0
    if level != REFERENCE_LEVEL and level in r.state_levels:
        h[2 + r.state_levels.index(level)] = 1.0
    return float(h @ r.mean), math.sqrt(float(h @ r.cov @ h) + SIGMA_NATIONAL ** 2)


def national_forecast_check(
    obs: Observations, loadings: bool = False, since: date = date(2022, 1, 1), drift: bool = False,
    fixed_lam: float | None = None, lam_before: Callable[[date], float | None] | None = None,
    slope: str = "rw", jumps: bool = False, outliers: bool = False, binomial: bool = False,
) -> list[dict]:
    """各国政選挙の前日までのデータだけでパラメータを推定し直し、その選挙の結果を
    予測する。地方選挙が国政の変化を捉えているかの、事後の当てはめに頼らない検証。
    比較の基準として、同じ院の前回の結果をそのまま予測値とした場合も返す。
    lam_beforeを与えると、その選挙の前日までに分かっている情報で負荷を決め直す。"""
    results = []
    last_by_level: dict[str, float] = {}
    for i, e in enumerate(obs.events):
        if e.level not in NATIONAL_LEVELS:
            continue
        if e.vote_date >= since and e.level in last_by_level:
            until = obs.t[i] - 1 / 30
            lam = lam_before(e.vote_date) if lam_before is not None else fixed_lam
            params, _ = fit_params(obs, until=until, loadings=loadings, drift=drift, fixed_lam=lam,
                                   slope=slope, jumps=jumps, outliers=outliers,
                                   binomial=binomial)
            pred, sd = predict_national(obs, params, until, e.level)
            results.append({
                "vote_date": e.vote_date.isoformat(), "level": e.level, "y": float(obs.y[i]),
                "pred": pred, "sd": sd, "naive": last_by_level[e.level],
                "lam": dict(params.lam),
            })
        last_by_level[e.level] = float(obs.y[i])
    return results


def national_interpolation_check(obs: Observations, party: str, seat_vote: list, slope: str = "none",
                                 **fit_kw) -> list[dict]:
    """国政選挙を1回ずつ除き、残りの全期間のデータで平滑化してその結果を再現できるかを見る
    (state_space_glm.national_interpolation_checkと同じ手順)。負荷lam_pは除いた選挙の点を
    使わずに決める。"""
    national = np.array([lv in NATIONAL_LEVELS for lv in obs.level])
    levels_all = _state_levels(obs)
    results = []
    for i in np.flatnonzero(national)[1:]:
        e = obs.events[i]
        pts = [p for p in seat_vote if p.vote_date != e.vote_date]
        lam = seat_vote_loading(pts, party, date.max)
        if lam is None:
            continue
        keep = np.ones(len(obs.t), bool)
        keep[i] = False
        idx = np.flatnonzero(keep)
        sub = Observations(t=obs.t[idx], level=[obs.level[j] for j in idx], P=obs.P[idx], y=obs.y[idx],
                           events=[obs.events[j] for j in idx], n_eff=obs.n_eff[idx])
        params, _ = fit_params(sub, fixed_lam=lam, slope=slope, **fit_kw)
        trace: list = []
        run_filter(sub, params, state_levels=levels_all, trace=trace)
        local_idx = [2 + j for j, lv in enumerate(levels_all) if lv not in NATIONAL_LEVELS] if params.q_alpha > 0 else []
        (m, C), = rts_smooth(trace, params, local_idx, [obs.t[i]])
        h = np.zeros(len(m))
        h[0] = 1.0
        if e.level != REFERENCE_LEVEL:
            h[2 + levels_all.index(e.level)] = 1.0
        sd = math.sqrt(max(float(h @ C @ h), 0.0) + SIGMA_NATIONAL ** 2)
        pred = float(h @ m)
        results.append({"vote_date": e.vote_date.isoformat(), "level": e.level, "y": float(obs.y[i]),
                        "pred": pred, "lo": pred - 2 * sd, "hi": pred + 2 * sd, "z": (obs.y[i] - pred) / sd})
    return results


@dataclass
class SeatVotePoint:
    vote_date: date
    party: str
    x: float
    v: float | None
    # 無所属を除いた議席の総数。0なら議席数が分からず、点ごとの重みを等しくする
    total: int = 0


@dataclass(frozen=True)
class SeatVoteCurve:
    """logit(議席の確率)=k*logit(v)+cの当てはめ結果。"""
    k: float
    c: float
    resid_var: float


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def seat_vote_curve(points: list[SeatVotePoint]) -> SeatVoteCurve:
    """議席数をBin(総数, logit^{-1}(k*logit(v)+c))として、弾力性kと切片cを全政党に共通で
    最尤推定する(3乗比の法則の一般形)。議席数で重みづけるので、議席の少ない党のlogitの
    大きなぶれに引きずられず、0議席も扱える。切片を政党ごとにすると(固定効果でもランダム
    効果でも)標本内の当てはまりは良くなるが、選挙を1回ずつ外して当てる誤差は悪くなった。
    1党あたり2〜4点しかなく、切片が手元の点に合わせすぎるため。"""
    from scipy.optimize import minimize

    pts = [p for p in points if p.v is not None and 0 < p.v < 1 and 0 <= p.x <= 1]
    if len({p.v for p in pts}) < 2:
        raise ValueError("得票の異なる点が足りず、弾力性を決められない")
    lv = np.array([_logit(p.v) for p in pts])
    n = np.array([float(p.total or 1) for p in pts])
    k_obs = np.array([p.x for p in pts]) * n

    def nll(b):
        eta = b[0] * lv + b[1]
        r = k_obs - n / (1 + np.exp(-eta))
        return -float(np.sum(k_obs * eta - n * np.logaddexp(0, eta))), -np.array([r @ lv, r.sum()])

    res = minimize(nll, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[(0.05, 10.0), (-30.0, 30.0)])
    k, c = float(res.x[0]), float(res.x[1])
    inner = [p for p in pts if 0 < p.x < 1]
    r = [_logit(p.x) - k * _logit(p.v) - c for p in inner]
    df = len(inner) - 2
    resid_var = float(np.sum(np.square(r)) / df) if df > 0 else 0.0
    return SeatVoteCurve(k, c, resid_var)


def seat_vote_loading(points: list[SeatVotePoint], party: str, before: date) -> float | None:
    """衆院選の議席xと比例代表の得票率vの関係から、地方の負荷lam_p=dv/dxを決める。

    地方議会の占有率は得票にほぼ比例して動き、衆院の議席は得票の変化をdx/dv倍に
    増幅・圧縮して動く、という見立てによる。dx/dv=k*x(1-x)/(v(1-v))は、その政党の
    直近の(x, v)での曲線の傾き。beforeより前の衆院選だけを使う。"""
    past = [p for p in points if p.vote_date < before]
    try:
        curve = seat_vote_curve(past)
    except ValueError:
        return None
    own = [p for p in past if p.party == party and 0 < p.x < 1]
    if not own:
        return None
    last = max(own, key=lambda p: p.vote_date)
    v = last.v
    if v is None or not 0 < v < 1:
        # 総務省の比例代表の集計表から得票率を読み取れなかった政党は、曲線上の点とみなす
        v = 1 / (1 + math.exp(-(_logit(last.x) - curve.c) / curve.k))
    return v * (1 - v) / (curve.k * last.x * (1 - last.x))


def load_seat_vote_points(min_election: int | None = None, house: str = "shugiin",
                          endorsed: bool = False) -> list[SeatVotePoint]:
    """総務省の公表値から、国政選挙ごとの政党別の議席占有率(無所属を除いて正規化)と
    比例代表の得票率を集める。参院選(house="sangiin")の議席はその選挙での改選分。
    endorsedなら、無所属の当選者を支援した党に等分してから占有率を求める(national_endorsements)。"""
    from .fetch.diet_history import fetch_sangiin_election_history, fetch_shugiin_history
    from .fetch.vote_share import _district_vote_pct, sangiin_index_url, shugiin_index_url
    from .lineage import canonicalize
    from .trend import parse_date

    if house == "shugiin":
        snaps = fetch_shugiin_history(min_election or 48)
        index_url, exclude = shugiin_index_url, ("選挙区別", "都道府県別")
    else:
        snaps = fetch_sangiin_election_history(min_election or 24)
        index_url, exclude = sangiin_index_url, ("選挙区別", "都道府県別", "得票順")
    points = []
    for snap in snaps:
        ordinal = int(snap.session.strip("第回"))
        seats: dict[str, int] = {}
        for p, n in snap.seats.items():
            seats[canonicalize(p)] = seats.get(canonicalize(p), 0) + n
        if endorsed:
            from .national_endorsements import apply, general_seat_deltas

            seats = apply(seats, {canonicalize(p): v for p, v in general_seat_deltas(
                parse_date(snap.date).isoformat()).items()})
        seats.pop(INDEPENDENT, None)
        total = sum(seats.values())
        pct = {canonicalize(p.strip()): v / 100 for p, v in _district_vote_pct(
            index_url(ordinal), "党派別得票数（比例代表）", exclude).items()}
        d = parse_date(snap.date)
        for p, n in seats.items():
            points.append(SeatVotePoint(d, p, n / total, pct.get(p), total))
    return points


def tracked_parties(events: list[Event], min_events: int = 30) -> list[str]:
    counts: dict[str, int] = {}
    for e in events:
        for p, v in e.shares.items():
            if p != INDEPENDENT and v > 0:
                counts[p] = counts.get(p, 0) + 1
    return sorted(p for p, n in counts.items() if n >= min_events)


def innovation_diagnostics(obs: Observations, params: Params) -> dict:
    """標準化した予測誤差の、階層ごとの平均と分散。モデルが正しければ平均0・分散1。"""
    r = run_filter(obs, params, keep_innovations=True, state_levels=_state_levels(obs))
    by_level: dict[str, list[float]] = {}
    national = []
    for e, v, S in r.innovations[2:]:
        z = v / math.sqrt(S)
        by_level.setdefault(e.level, []).append(z)
        if e.level in NATIONAL_LEVELS:
            national.append({"vote_date": e.vote_date.isoformat(), "level": e.level,
                             "innovation": v, "pred_sd": math.sqrt(S)})
    return {
        "by_level": {lv: {"n": len(z), "mean": float(np.mean(z)), "var": float(np.var(z))}
                     for lv, z in by_level.items()},
        "national": national,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from .trend import OUTPUT_DIR, load_events, month_ends

    ap = argparse.ArgumentParser(description="状態空間モデルによる党勢の現在推定と変化率")
    ap.parse_args(argv)
    events, cutoff = load_events()
    first = first_appearance(events)
    last = events[-1].vote_date
    tau = month_ends((last.year, last.month), (last.year, last.month))[-1]
    out = {"tau": tau.isoformat(), "classified_until": cutoff.isoformat() if cutoff else None,
           "sigma_national": SIGMA_NATIONAL, "parties": {}}
    seat_vote = load_seat_vote_points()
    curve = seat_vote_curve(seat_vote)
    out["seat_vote_curve"] = {"k": curve.k, "c": curve.c}
    for party in tracked_parties(events):
        obs = party_observations(events, party, first[party])
        lam = seat_vote_loading(seat_vote, party, tau)
        params, ll = fit_params(obs, fixed_lam=lam)
        series = []
        for m_end in month_ends((2019, 1), (tau.year, tau.month)):
            if m_end < first[party]:
                continue
            nc = nowcast(obs, params, party, m_end)
            series.append({"tau": m_end.isoformat(), "x": nc.x, "x_se": nc.x_se, "D": nc.D, "D_se": nc.D_se})
        nc = nowcast(obs, params, party, tau)
        out["parties"][party] = {
            "x": nc.x, "x_se": nc.x_se, "D": nc.D, "D_se": nc.D_se, "alpha": nc.alpha,
            "params": {"q_x": params.q_x, "q_b": params.q_b, "sigma2": params.sigma2, "a": params.a,
                       "lam": lam},
            "loglik": ll, "diagnostics": innovation_diagnostics(obs, params), "series": series,
        }
        print(party, f"lam={lam} x={nc.x:.3f}±{nc.x_se:.3f} D={nc.D * 100:+.3f}±{nc.D_se * 100:.3f}%/月", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"state_space_{tau.isoformat()}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    main()
