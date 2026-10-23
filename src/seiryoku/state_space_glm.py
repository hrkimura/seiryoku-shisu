"""党勢をlogitの空間に置き、地方選挙を議席数の分布で観測する状態空間モデル。

状態は$z=\\operatorname{logit}x$と階層ごとの水準差$\\alpha_\\ell$。
国政選挙は$z$をほぼ正確に観測し、参院選は衆参の議席と得票の曲線を得票率でつないだ
負荷と水準差の事前分布で観測する。地方選挙は議会選挙だけを使い、無所属を除いた議席$n^*$の
うち政党の議席$k$を、平均$\\mu=\\operatorname{logit}^{-1}(\\gamma z+\\alpha_\\ell+u_m)$、
精度$\\phi_\\ell$のベータ二項分布で観測する。
$u_m$は自治体ごとの固有の水準、$\\gamma$(コードではell)は政党ごとの負荷で、議席と得票の
曲線と地方の占有率から決める(party_loading)。
非正規の観測は、観測ごとに1次元のラプラス近似で正規分布に直して更新する。
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, psi, zeta

from . import state_space as ss
from .by_elections import BY_LEVELS, NATIONAL_BY_LEVELS, NATIONAL_VOTE_BY, SANGIIN_BY
from .trend import Event, first_appearance

PRIOR_VAR_Z = 10.0
PRIOR_VAR_ALPHA = 10.0
PRIOR_VAR_SLOPE = 0.01
SANGIIN = "参院選"
MAYOR_LEVELS = ("市区町村長", "都道府県知事")
COUNCIL_LEVELS = ("市区町村議会", "都道府県議会")



def sangiin_link(shu: list, san: list, before: date | None = None) -> dict:
    """衆参の議席と得票の曲線から、参院選の観測の負荷と水準差の事前分布を決める。
    logit x_参=(k_参/k_衆)(z-c_衆)+c_参。vote_loadは得票率の観測の負荷1/k_衆
    (logit v=(z-c_衆)/k_衆、切片は階層の水準差が受け持つ)。"""
    if before is not None:
        shu = [p for p in shu if p.vote_date < before]
        san = [p for p in san if p.vote_date < before]
    a, b = ss.seat_vote_curve(shu), ss.seat_vote_curve(san)
    load = b.k / a.k
    return {"san_load": load, "san_offset": b.c - load * a.c,
            "san_prior_var": b.resid_var + load ** 2 * a.resid_var, "vote_load": 1 / a.k}


@dataclass
class GlmObservations:
    t: np.ndarray
    level: list[str]
    k: np.ndarray
    n: np.ndarray
    y: np.ndarray
    events: list[Event]


def glm_observations(events: list[Event], party: str, start: date, filter_levels: bool = True) -> GlmObservations:
    """国政選挙と議会選挙を議席数(k, n*)に直す。filter_levels=Falseなら、
    政党がほとんど議席を得ない階層も含める。
    首長選は使わない。党派は推薦・支持の名簿から取るしかなく、名簿の公開は1年以上遅れ、
    その時期も自治体の系統ごとにばらばらだからである。国政選挙の補間の精度は、首長選を
    除いても落ちない。補欠選挙(by_elections)は、イベントに含まれていれば通常選挙と別の階層として使う。
    国政の補選の得票率は、その党が候補を立てた選挙だけを観測にし、yに得票率をそのまま入れる。"""
    votes = [e for e in events if e.level == NATIONAL_VOTE_BY and e.vote_date >= start and 0 < e.shares.get(party, 0.0) < 1]
    events = [e for e in events if e.level in ss.NATIONAL_LEVELS or e.level in COUNCIL_LEVELS
              or (e.level in BY_LEVELS and e.level != NATIONAL_VOTE_BY)]
    obs = ss.party_observations(events, party, start, filter_levels=filter_levels)
    t = np.concatenate([obs.t, [ss._t(e.vote_date) for e in votes]])
    y = np.concatenate([obs.y, [e.shares[party] for e in votes]])
    n = np.concatenate([obs.n_eff, np.ones(len(votes))])
    order = np.argsort(t, kind="stable")
    level, evs = obs.level + [e.level for e in votes], obs.events + votes
    return GlmObservations(t=t[order], level=[level[i] for i in order], k=(y * n)[order], n=n[order],
                           y=y[order], events=[evs[i] for i in order])


def party_loading(obs: GlmObservations, seat_vote: list, seat_vote_san: list, party: str,
                  before: date) -> float | None:
    """政党ごとの地方の負荷gamma_p。正規分布のモデルのlam_p(地方の占有率muの変化は得票率vの
    変化に等しい)をlogitの空間に直したもので、d logit mu/dz=v(1-v)/(k_衆 mu(1-mu))。
    vは衆参いずれかの直近の国政選挙の比例代表の得票率、muはbeforeより前の地方議会の
    議席の合計による占有率。"""
    k = None
    latest = None
    for house in (seat_vote, seat_vote_san):
        past = [p for p in house if p.vote_date < before]
        try:
            curve = ss.seat_vote_curve(past)
        except ValueError:
            continue
        if house is seat_vote:
            k = curve.k
        for p in past:
            if p.party != party:
                continue
            if p.v is not None and 0 < p.v < 1:
                v = p.v
            elif 0 < p.x < 1:
                # 比例代表の得票率を読み取れなかった選挙は、その院の曲線上の点とみなす
                v = _expit((_logit(p.x) - curve.c) / curve.k)
            else:
                continue
            if latest is None or p.vote_date > latest[0]:
                latest = (p.vote_date, v)
    if k is None or latest is None:
        return None
    v = latest[1]
    local = [i for i, e in enumerate(obs.events)
             if e.level in COUNCIL_LEVELS and obs.n[i] > 1 and e.vote_date < before]
    n = float(obs.n[local].sum()) if local else 0.0
    if n == 0:
        return None
    mu = float(obs.k[local].sum()) / n
    if not 0 < mu < 1:
        return None
    return v * (1 - v) / (k * mu * (1 - mu))


@dataclass
class GlmParams:
    q_x: float
    phi: dict[str, float]
    # 地方の負荷。logit mu=ell*z+alpha+u。尤度ではほとんど決まらないので、通常はparty_loadingで
    # 推定の外から与える。
    ell: float
    slope: str = "none"
    jump_rate: float = 0.0
    jump_var: float = 0.0
    # 自治体ごとの固有の水準u_mの分散(階層ごと)。空なら自治体の効果を入れない。
    # u_mは状態に入れず、自治体ごとに平均・分散と状態との共分散を覚えておき、次にその自治体の
    # 選挙があったときの事前分布にする。自治体どうしの共分散だけを捨てる近似である。
    tau2: dict[str, float] = field(default_factory=dict)
    # 参院選の観測はload_san*z+alpha_参。衆参それぞれの議席と得票の曲線を得票率でつなぐと
    # logit x_参=(k_参/k_衆)(z-c_衆)+c_参となるので、負荷をk_参/k_衆とし、alpha_参の
    # 事前分布を曲線から決まる値の周りに、2つの曲線の残差の大きさで置く。
    san_load: float = 1.0
    san_offset: float = 0.0
    san_prior_var: float = PRIOR_VAR_ALPHA
    sigma_national: float | None = None
    # 国政の補選の得票率の観測logit v=vote_load*z+alpha+e、eの分散がvote_var(sangiin_link参照)
    vote_load: float = 1.0
    vote_var: float = 0.0
    prior_var: float = PRIOR_VAR_Z
    # 国政選挙にだけ現れる党勢の部分z_Nのランダムウォークの分散。正なら状態の末尾にz_Nを加え、
    # 国政の党勢をz+z_N、地方選挙の負荷はzだけにかける。z_Nは始点で0とする(水準はzが持つ)。
    q_nat: float = 0.0

    def dynamics(self) -> ss.Params:
        return ss.Params(q_x=self.q_x, q_b=0.0, sigma2={}, a=0.0, slope=self.slope,
                         jump_rate=self.jump_rate, jump_var=self.jump_var, q_alpha=self.q_nat)

    def national_vector(self, k_state: int) -> np.ndarray:
        """状態から国政の党勢(logit)を取り出すベクトル。"""
        g = np.zeros(k_state)
        g[0] = 1.0
        if self.q_nat > 0:
            g[-1] = 1.0
        return g


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _psd_project(C: np.ndarray) -> np.ndarray:
    """対称化し、数値誤差や近似で負になった固有値を0に戻す。"""
    C = 0.5 * (C + C.T)
    lam, vec = np.linalg.eigh(C)
    if lam.min() >= 0:
        return C
    return (vec * np.maximum(lam, 0.0)) @ vec.T


def national_logit(k: float, n: float, sigma: float | None = None) -> tuple[float, float]:
    """国政選挙の観測値をlogitに直し、その誤差の標準偏差を返す。sigmaがNoneなら、議席数を
    二項分布の観測とみなした経験logitの分散を使う。占有率の尺度で全党一律の誤差sigmaを
    与えると、議席の少ない党では国政選挙がほぼ無情報になり、補間が地方選挙に引っ張られる。
    議席0の政党でもlogitが有限になるよう、0.5議席を足す。"""
    p = (k + 0.5) / (n + 1.0)
    if sigma is None:
        return _logit(p), math.sqrt(1 / (k + 0.5) + 1 / (n - k + 0.5))
    return _logit(p), sigma / (p * (1 - p))


def _local_loglik(eta: float, k: float, n: float, phi: float) -> tuple[float, float, float]:
    """地方選挙1件の対数尤度と、線形予測子etaについての1階・2階微分。"""
    mu = min(max(_expit(eta), 1e-12), 1 - 1e-12)
    dmu = mu * (1 - mu)
    if n <= 1.0 + 1e-9:
        # 首長は勝ち負けの1回の試行。推薦が複数政党に割れた選挙はkが0.5などの割合になるので、
        # ベルヌーイ分布の尤度を割合のまま使う
        y = k / max(n, 1e-12)
        return y * math.log(mu) + (1 - y) * math.log(1 - mu), y - mu, -dmu
    if phi > 1e7:
        # 精度が十分大きければ二項分布と同じ。ベータ関数の差は桁落ちするので二項分布で計算する
        ll = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(mu) + (n - k) * math.log(1 - mu)
        return float(ll), k - n * mu, -n * dmu
    a, b = mu * phi, (1 - mu) * phi
    ll = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
          + betaln(k + a, n - k + b) - betaln(a, b))
    # scipy.special.polygammaはスカラー1回ごとの呼び出しが重いので、4点をまとめてzeta(2, x)で計算する
    args = np.array([k + a, n - k + b, a, b])
    d1 = psi(args)
    d2 = zeta(2, args)
    l_mu = phi * (d1[0] - d1[1] - d1[2] + d1[3])
    l_mumu = phi ** 2 * (d2[0] + d2[1] - d2[2] - d2[3])
    return float(ll), float(l_mu * dmu), float(l_mumu * dmu ** 2 + l_mu * dmu * (1 - 2 * mu))


def _laplace(m_eta: float, s2: float, k: float, n: float, phi: float) -> tuple[float | None, float, float]:
    """etaの事前分布N(m_eta, s2)と観測から、同じ事後分布を与える正規の擬似観測
    (値, 分散)と、ラプラス近似の対数周辺尤度を返す。対数尤度がその点で凸でなく
    擬似観測で表せないときは、値をNoneにする(状態は更新せず、尤度だけ数える)。"""
    eta = m_eta
    for _ in range(50):
        _, g1, g2 = _local_loglik(eta, k, n, phi)
        G1 = g1 - (eta - m_eta) / s2
        G2 = g2 - 1 / s2
        if G2 >= 0:
            G2 = -1 / s2
        step = G1 / G2
        step = max(min(step, 2.0), -2.0)
        eta -= step
        if abs(step) < 1e-8:
            break
    ll, _, g2 = _local_loglik(eta, k, n, phi)
    v = -1 / min(g2 - 1 / s2, -1e-12)
    logml = ll - (eta - m_eta) ** 2 / (2 * s2) + 0.5 * math.log(v / s2)
    if g2 >= -1e-12:
        return None, math.inf, logml
    R = -1 / g2
    y_pseudo = R * (eta / v - m_eta / s2)
    return y_pseudo, R, logml


@dataclass
class GlmFilterResult:
    loglik: float
    mean: np.ndarray
    cov: np.ndarray
    state_levels: list[str]


def run_filter(obs: GlmObservations, params: GlmParams, until: float | None = None,
               state_levels: list[str] | None = None, trace: list | None = None,
               contrib: list | None = None) -> GlmFilterResult:
    """traceにリストを渡すと、観測のある時刻ごとに[t, 予測平均, 予測分散, 更新後平均, 更新後分散]を
    積む(平滑化用)。contribにリストを渡すと、観測ごとに(添字, 期待される占有率, 更新によるzの変化)を積む。
    傾きがなければ予測でzの平均は動かないので、zの変化は観測ごとの更新の和にちょうど分かれる。"""
    levels = state_levels if state_levels is not None else sorted(
        {lv for lv in obs.level if lv != ss.REFERENCE_LEVEL})
    idx = {lv: 2 + i for i, lv in enumerate(levels)}
    dyn = params.dynamics()
    k_state = 2 + len(levels) + (1 if params.q_nat > 0 else 0)
    nat_idx = [k_state - 1] if params.q_nat > 0 else []
    g_nat = params.national_vector(k_state)
    m = np.zeros(k_state)
    C = np.eye(k_state) * params.prior_var
    C[1, 1] = 0.0 if params.slope == "none" else PRIOR_VAR_SLOPE
    if nat_idx:
        C[-1, -1] = 0.0
    if SANGIIN in idx:
        m[idx[SANGIIN]] = params.san_offset
        C[idx[SANGIIN], idx[SANGIIN]] = params.san_prior_var
    t_prev = obs.t[0] if len(obs.t) else 0.0
    loglik = 0.0
    seen: set[str] = set()
    n_seen = 0
    comps = [(1.0, m, C)]
    # 自治体の地盤u_mは状態ベクトルに入れず、自治体ごとに平均・分散と、状態との共分散(cc)を持つ。
    # 自治体どうしの共分散だけを捨てる近似で、模擬データでは地盤を状態に入れた厳密解とほぼ一致する
    # (状態との共分散まで捨てると、国政選挙の予測の誤差が1割ほど大きくなる)。
    rows: dict[tuple[str, str], int] = {}
    if params.tau2:
        for i in range(len(obs.t)):
            lv = obs.level[i]
            if lv not in ss.NATIONAL_LEVELS and lv in params.tau2:
                rows.setdefault((lv, obs.events[i].name), len(rows))
    um = np.zeros(len(rows))
    uv = np.array([params.tau2[g] for g, _ in rows])
    cc = np.zeros((len(rows), k_state))
    if trace is not None and len(obs.t):
        trace.append([t_prev, m, C, m, C])
    for i in range(len(obs.t)):
        t = obs.t[i]
        if until is not None and t > until:
            break
        dt = t - t_prev
        if dt > 0:
            comps = ss._split(ss._collapse(comps), dt, dyn, nat_idx)
            t_prev = t
            if trace is not None:
                mp, Cp = ss._collapse(comps)
                trace.append([t, mp, Cp, mp, Cp])
        lv = obs.level[i]
        h = np.zeros(k_state)
        j = idx.get(lv)
        if j is not None:
            h[j] = 1.0
        # 得票率の観測は国政選挙と同じく正規分布で、地方の負荷も自治体の地盤も持たない
        national = lv in ss.NATIONAL_LEVELS or lv == NATIONAL_VOTE_BY
        if lv == NATIONAL_VOTE_BY:
            h += params.vote_load * g_nat
            y_nat, sd_nat = _logit(obs.y[i]), math.sqrt(params.vote_var)
        elif national:
            h += (params.san_load if lv == SANGIIN else 1.0) * g_nat
            y_nat, sd_nat = national_logit(obs.k[i], obs.n[i], params.sigma_national)
        elif lv in NATIONAL_BY_LEVELS:
            # 国政の補選は国政の党勢を1つの選挙区で測る。負荷は地方のgammaではなく国政選挙と同じにし、
            # 選挙区ごとの偏りと小選挙区の議席と得票の関係はalphaと二項分布が受け持つ
            h += (params.san_load if lv == SANGIIN_BY else 1.0) * g_nat
        else:
            h[0] = params.ell
        r = None if national else rows.get((lv, obs.events[i].name))
        if r is not None and uv[r] <= 0:
            cc[r] = 0.0
        elif r is not None:
            # 自治体どうしの共分散を捨てているので、(状態, u_m)の共分散が半正定値から外れうる。
            # 状態との共分散をu_mの分散と矛盾しない大きさに縮め、更新後の分散が負にならないようにする
            C_now = ss._collapse(comps)[1]
            qf = float(cc[r] @ np.linalg.pinv(C_now) @ cc[r])
            if qf > 0.99 * uv[r]:
                cc[r] *= math.sqrt(0.99 * uv[r] / qf)
        if contrib is not None:
            m_before = ss._collapse(comps)[0]
            expected = _expit(float(h @ m_before) + (um[r] if r is not None else 0.0))
        new = []
        u_post = []
        for w, mc, Cc in comps:
            if w <= 0:
                continue
            cov_core = Cc @ h
            m_eta = float(h @ mc)
            s2 = float(h @ cov_core)
            if r is not None:
                cov_core = cov_core + cc[r]
                m_eta += um[r]
                # 自治体どうしの共分散を捨てているので、(状態, u_m)の共分散が半正定値から外れうる。
                # 相関が1を超えないように抑える
                hc = float(h @ cc[r])
                lim = math.sqrt(max(s2, 0.0) * uv[r])
                s2_raw = max(s2, 0.0) + uv[r]
                s2 = max(s2_raw + 2 * max(min(hc, 0.999 * lim), -0.999 * lim), 1e-6 * s2_raw, 1e-12)
            if national:
                R, y_p = sd_nat ** 2, y_nat
                S = s2 + R
                logml = -0.5 * (math.log(2 * math.pi * S) + (y_p - m_eta) ** 2 / S)
            else:
                y_p, R, logml = _laplace(m_eta, s2, obs.k[i], obs.n[i], params.phi.get(lv, math.inf))
                if y_p is None:
                    new.append((math.log(w) + logml, mc, Cc))
                    u_post.append((um, uv, cc))
                    continue
                S = s2 + R
            v = y_p - m_eta
            K = cov_core / S
            new.append((math.log(w) + logml, mc + K * v, _psd_project(Cc - np.outer(K, cov_core))))
            if len(rows):
                wv = cc @ h
                if r is not None:
                    wv[r] += uv[r]
                u_post.append((um + wv * (v / S), np.maximum(uv - wv * wv / S, 0.0), cc - np.outer(wv, K)))
        top = max(lp for lp, _, _ in new)
        ll_i = top + math.log(sum(math.exp(lp - top) for lp, _, _ in new))
        comps = [(math.exp(lp - ll_i), m2, C2) for lp, m2, C2 in new]
        if len(rows):
            if len(u_post) == 1:
                um, uv, cc = u_post[0]
            else:
                ws = [w for w, _, _ in comps]
                um_new = sum(w_ * a for w_, (a, _, _) in zip(ws, u_post))
                uv = sum(w_ * (b + (a - um_new) ** 2) for w_, (a, b, _) in zip(ws, u_post))
                cc = sum(w_ * c_ for w_, (_, _, c_) in zip(ws, u_post))
                um = um_new
        if trace is not None:
            trace[-1][3:5] = ss._collapse(comps)
        if contrib is not None:
            contrib.append((i, expected, float(g_nat @ (ss._collapse(comps)[0] - m_before))))
        if n_seen >= 2 and lv in seen:
            loglik += ll_i
        seen.add(lv)
        n_seen += 1
    m, C = ss._collapse(comps)
    if until is not None and until > t_prev:
        m, C = ss._collapse(ss._split((m, C), until - t_prev, dyn, nat_idx))
    return GlmFilterResult(loglik, m, C, levels)


def fit_params(obs: GlmObservations, ell: float | None = None, until: float | None = None, slope: str = "none",
               jumps: bool = False, muni: bool = False, san: dict | None = None,
               sigma_national: float | None = None, prior_var: float = PRIOR_VAR_Z,
               national_dev: bool = False, starts: int = 3, maxiter: int = 300,
               phi_mode: str = "none", q_x: float | None = None) -> tuple[GlmParams, float]:
    """最尤推定。ellを与えなければ地方の負荷も推定する。尤度は局所解を持つことがあるので、
    q_xの初期値を変えたstarts通りの初期値から始め、最も尤度の高い解を採る。

    精度phiを置くのはn*>1の観測を持つ階層だけ(1回の試行では過分散を識別できない)。
    phi_modeは"level"なら階層ごと、"shared"なら全階層で共通、"none"なら置かず二項分布とする。
    q_xを与えるとランダムウォークの分散をその値に固定する(全党共通の値はpooled_fitで決める)。"""
    local = sorted({lv for lv in obs.level if lv not in ss.NATIONAL_LEVELS})
    levels = sorted({lv for lv in obs.level if lv != ss.REFERENCE_LEVEL})
    dispersed = sorted({lv for lv, n in zip(obs.level, obs.n) if lv in local and n > 1.0 + 1e-9})
    P = {"level": len(dispersed), "shared": min(len(dispersed), 1), "none": 0}[phi_mode]
    # 精度phiは1/phiの対数で持ち、下限を二項分布(1/phi=0)に十分近くとる
    z0 = [math.log(1e-3)] + [math.log(1 / 20.0)] * P
    bounds = [(-25, 1)] + [(-20.0, math.log(10.0))] * P
    if jumps:
        z0 += [math.log(0.05), math.log(0.1)]
        bounds += [(math.log(0.1 / 12), 0.0), (math.log(1e-3), math.log(4.0))]
    councils = [lv for lv in local if lv in COUNCIL_LEVELS]
    if muni:
        z0 += [math.log(0.5)] * len(councils)
        bounds += [(math.log(1e-4), math.log(25.0))] * len(councils)
    if ell is None:
        z0 += [math.log(0.5)]
        bounds += [(math.log(0.05), math.log(3.0))]
    votes = NATIONAL_VOTE_BY in obs.level
    if votes:
        z0 += [math.log(0.1)]
        bounds += [(math.log(1e-4), math.log(10.0))]
    if national_dev:
        z0 += [math.log(0.01)]
        bounds += [(-25, math.log(10.0))]

    def unpack(z):
        k = 1 + P
        jump_rate = jump_var = 0.0
        if jumps:
            jump_rate, jump_var = math.exp(z[k]), math.exp(z[k + 1])
            k += 2
        tau2 = {}
        if muni:
            tau2 = {g: math.exp(z[k + i]) for i, g in enumerate(councils)}
            k += len(councils)
        ell_v = ell
        if ell is None:
            ell_v = math.exp(z[k])
            k += 1
        vote_var = 0.0
        if votes:
            vote_var = math.exp(z[k])
            k += 1
        return GlmParams(vote_var=vote_var, q_x=math.exp(z[0]), phi={lv: math.exp(-z[1 + (i if P > 1 else 0)]) for i, lv in enumerate(dispersed)} if P else {},
                         ell=ell_v, slope=slope, jump_rate=jump_rate, jump_var=jump_var, tau2=tau2,
                         sigma_national=sigma_national, prior_var=prior_var,
                         q_nat=math.exp(z[k]) if national_dev else 0.0, **(san or {}))

    def nll(z):
        return -run_filter(obs, unpack(z), until=until, state_levels=levels).loglik

    best = None
    if q_x is not None:
        bounds[0] = (math.log(q_x), math.log(q_x))
    for q0 in [q_x] if q_x is not None else [1e-3, 1e-1, 1e-5][:max(starts, 1)]:
        z_init = np.array(z0)
        z_init[0] = math.log(q0)
        res = minimize(nll, z_init, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x), -best.fun


def pooled_loading(obs_by_party: dict[str, GlmObservations], grid=(0.3, 0.45, 0.6), **fit_kw) -> tuple[float, dict]:
    """全政党に共通の負荷ellを、格子上の対数尤度の合計を2次式で補間した最大点として決める。
    最大が格子の端なら端の値を返す。"""
    total = {g: 0.0 for g in grid}
    for obs in obs_by_party.values():
        for g in grid:
            total[g] += fit_params(obs, g, **fit_kw)[1]
    gs = sorted(grid)
    best = max(gs, key=lambda g: total[g])
    i = gs.index(best)
    if 0 < i < len(gs) - 1:
        x = np.log(gs[i - 1:i + 2])
        a, b, _ = np.polyfit(x, [total[g] for g in gs[i - 1:i + 2]], 2)
        if a < 0:
            best = float(np.exp(min(max(-b / (2 * a), x[0]), x[2])))
    return best, total


@dataclass
class GlmNowcast:
    x: float
    x_lo: float
    x_hi: float
    z: float
    z_se: float
    D: float
    alpha: dict[str, float]


def smooth(obs: GlmObservations, params: GlmParams, times: list[float],
           state_levels: list[str] | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """全期間の観測を使った、各時刻の状態の平均と分散(固定区間平滑化、RTS)。
    地方選挙は前向きのフィルタでの線形化(ラプラス近似の擬似観測)のまま扱い、跳びの混合は
    畳んだ1成分で近似する。自治体の効果は状態の外にあるので平滑化しない。"""
    trace: list = []
    run_filter(obs, params, state_levels=state_levels, trace=trace)
    k_state = len(trace[0][1])
    return ss.rts_smooth(trace, params.dynamics(), [k_state - 1] if params.q_nat > 0 else [], times)


def _subset(obs: GlmObservations, keep: np.ndarray) -> GlmObservations:
    idx = np.flatnonzero(keep)
    return GlmObservations(t=obs.t[idx], level=[obs.level[i] for i in idx], k=obs.k[idx], n=obs.n[idx],
                           y=obs.y[idx], events=[obs.events[i] for i in idx])


def national_interpolation_check(obs_by_party: dict[str, GlmObservations], seat_vote: list, seat_vote_san: list,
                                 workers: int | None = None, local: bool = True,
                                 drop_levels: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """国政選挙を1回ずつ除き、残りの全期間のデータ(前後両側)で平滑化してその結果を再現できるかを見る。
    除いた選挙の議席と得票の点は、負荷、参院選のつなぎ、全党共通のqの計算からも除く。qは、
    その選挙を除く対象になる政党(その選挙が最初の国政選挙でない政党)の対数尤度の合計で決める。
    local=Falseなら地方選挙を使わない(国政選挙だけの補間)。drop_levelsの階層の選挙も使わず、
    負荷の地方の占有率も残りの議会から求める。"""
    from concurrent.futures import ProcessPoolExecutor

    obs_by_party = {p: _subset(o, np.array([lv not in drop_levels for lv in o.level]))
                    for p, o in obs_by_party.items()}
    national = {p: np.array([lv in ss.NATIONAL_LEVELS for lv in o.level]) for p, o in obs_by_party.items()}
    held = {p: {o.events[i].vote_date: i for i in np.flatnonzero(national[p])[1:]} for p, o in obs_by_party.items()}
    results: dict[str, list[dict]] = {p: [] for p in obs_by_party}
    with ProcessPoolExecutor(workers) as ex:
        for d in sorted(set().union(*held.values())):
            shu = [p for p in seat_vote if p.vote_date != d]
            san = [p for p in seat_vote_san if p.vote_date != d]
            subs, ells, kw = {}, {}, {}
            for party, o in obs_by_party.items():
                if d not in held[party]:
                    continue
                ell = party_loading(o, shu, san, party, date.max)
                if ell is None:
                    continue
                keep = np.ones(len(o.t), bool)
                keep[held[party][d]] = False
                if not local:
                    keep &= national[party]
                subs[party], ells[party] = _subset(o, keep), ell
                kw[party] = {"muni": True, "san": sangiin_link(shu, san)}
            if not subs:
                continue
            q, params_by_party = pooled_fit(ex, subs, ells, kw)
            for party, params in params_by_party.items():
                o, sub, i = obs_by_party[party], subs[party], held[party][d]
                e = o.events[i]
                levels = [lv for lv in sorted({lv for lv in o.level if lv != ss.REFERENCE_LEVEL})
                          if lv in sub.level or lv == e.level]
                (m, C), = smooth(sub, params, [o.t[i]], state_levels=levels)
                h = (params.san_load if e.level == SANGIIN else 1.0) * params.national_vector(len(m))
                if e.level in levels:
                    h[2 + levels.index(e.level)] = 1.0
                y_logit, sd_nat = national_logit(o.k[i], o.n[i], params.sigma_national)
                mu, s_tot = float(h @ m), math.hypot(math.sqrt(max(float(h @ C @ h), 0.0)), sd_nat)
                results[party].append({"vote_date": d.isoformat(), "level": e.level, "y": float(o.y[i]),
                                       "pred": _expit(mu), "lo": _expit(mu - 2 * s_tot),
                                       "hi": _expit(mu + 2 * s_tot), "z": (y_logit - mu) / s_tot, "q": q})
    return results


def national_online_check(obs_by_party: dict[str, GlmObservations], seat_vote: list, seat_vote_san: list,
                          workers: int | None = None,
                          drop_levels: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """各国政選挙を、その投票日より前のデータだけで推定したモデルで予測する(月次の本番と同じ
    オンラインの使い方)。負荷、参院選のつなぎ、全党共通のqもその日より前のデータで決め直す。
    投票日が同じ地方選挙も使わない。結果の形はnational_interpolation_checkと同じ。"""
    from concurrent.futures import ProcessPoolExecutor

    obs_by_party = {p: _subset(o, np.array([lv not in drop_levels for lv in o.level]))
                    for p, o in obs_by_party.items()}
    national = {p: np.array([lv in ss.NATIONAL_LEVELS for lv in o.level]) for p, o in obs_by_party.items()}
    held = {p: {o.events[i].vote_date: i for i in np.flatnonzero(national[p])[1:]} for p, o in obs_by_party.items()}
    results: dict[str, list[dict]] = {p: [] for p in obs_by_party}
    with ProcessPoolExecutor(workers) as ex:
        for d in sorted(set().union(*held.values())):
            subs, ells, kw = {}, {}, {}
            for party, o in obs_by_party.items():
                if d not in held[party]:
                    continue
                ell = party_loading(o, seat_vote, seat_vote_san, party, d)
                if ell is None:
                    continue
                subs[party], ells[party] = _subset(o, np.array([e.vote_date < d for e in o.events])), ell
                kw[party] = {"muni": True, "san": sangiin_link(seat_vote, seat_vote_san, d)}
            if not subs:
                continue
            q, params_by_party = pooled_fit(ex, subs, ells, kw)
            for party, params in params_by_party.items():
                o, sub, i = obs_by_party[party], subs[party], held[party][d]
                e = o.events[i]
                mu, s = predict_national(sub, params, o.t[i], e.level)
                y_logit, sd_nat = national_logit(o.k[i], o.n[i], params.sigma_national)
                s_tot = math.hypot(s, sd_nat)
                results[party].append({"vote_date": d.isoformat(), "level": e.level, "y": float(o.y[i]),
                                       "pred": _expit(mu), "lo": _expit(mu - 2 * s_tot),
                                       "hi": _expit(mu + 2 * s_tot), "z": (y_logit - mu) / s_tot, "q": q})
    return results


def interpolation_summary(results: dict[str, list[dict]]) -> dict:
    """補間の結果の要約。誤差は占有率の尺度。"""
    rows = [r for rs in results.values() for r in rs]
    err = np.array([r["pred"] - r["y"] for r in rows])
    return {"n": len(rows), "rmse": float(np.sqrt(np.mean(err ** 2))), "median_abs": float(np.median(np.abs(err))),
            "coverage": float(np.mean([r["lo"] <= r["y"] <= r["hi"] for r in rows])),
            "z_sd": float(np.std([r["z"] for r in rows])),
            "by_party": {p: float(np.mean([abs(r["pred"] - r["y"]) for r in rs])) for p, rs in results.items() if rs}}


def monthly_contributions(obs: GlmObservations, params: GlmParams, start: date, end: date) -> list[dict]:
    """start<日付<=endの選挙が、党勢の推定値zをそれぞれどれだけ動かしたか。動きは、その選挙での
    期待される占有率と実際の占有率の差から生じる。"""
    contrib: list = []
    run_filter(obs, params, until=ss._t(end), contrib=contrib)
    out = []
    for i, expected, dz in contrib:
        e = obs.events[i]
        if start < e.vote_date <= end:
            out.append({"vote_date": e.vote_date.isoformat(), "level": e.level, "name": e.name,
                        "n": float(obs.n[i]), "actual": float(obs.y[i]), "expected": expected, "dz": dz})
    return out


def filtered_series(obs: GlmObservations, params: GlmParams, ends: list[date]) -> list[tuple[float, float]]:
    """各月末の、その時点までのデータだけによる党勢zの平均と標準偏差(フィルタの値)。
    傾きがないので、最後の観測から月末までは平均が動かず、分散だけがq_x*経過時間ぶん増える。"""
    trace: list = []
    run_filter(obs, params, trace=trace)
    times = [r[0] for r in trace]
    out = []
    for end in ends:
        te = ss._t(end)
        j = bisect.bisect_right(times, te) - 1
        t, _, _, m, C = trace[max(j, 0)]
        g = params.national_vector(len(m))
        out.append((float(g @ m), math.sqrt(max(float(g @ C @ g) + params.q_x * max(te - t, 0.0), 0.0))))
    return out


def level_movement(obs: GlmObservations, params: GlmParams, ends: list[date]) -> list[dict[str, float]]:
    """月(ends[j-1]<日付<=ends[j])ごとに、選挙の階層別の、党勢の推定値zの動きの絶対値の合計。"""
    contrib: list = []
    run_filter(obs, params, contrib=contrib)
    out: list[dict[str, float]] = [{} for _ in ends]
    for i, _, dz in contrib:
        j = bisect.bisect_left(ends, obs.events[i].vote_date)
        if j < len(ends):
            lv = obs.level[i]
            out[j][lv] = out[j].get(lv, 0.0) + abs(dz)
    return out


def nowcast(obs: GlmObservations, params: GlmParams, tau: date) -> GlmNowcast:
    r = run_filter(obs, params, until=ss._t(tau))
    g = params.national_vector(len(r.mean))
    z, se = float(g @ r.mean), math.sqrt(max(float(g @ r.cov @ g), 0.0))
    return GlmNowcast(x=_expit(z), x_lo=_expit(z - 2 * se), x_hi=_expit(z + 2 * se), z=z, z_se=se,
                      D=float(r.mean[1]), alpha={lv: float(r.mean[2 + i]) for i, lv in enumerate(r.state_levels)})


def predict_national(obs: GlmObservations, params: GlmParams, until: float, level: str) -> tuple[float, float]:
    """時刻untilまでの観測から、国政選挙(level)のlogit占有率の予測の平均と標準偏差。"""
    levels = sorted({lv for lv in obs.level if lv != ss.REFERENCE_LEVEL})
    r = run_filter(obs, params, until=until, state_levels=levels)
    h = (params.san_load if level == SANGIIN else 1.0) * params.national_vector(len(r.mean))
    if level != ss.REFERENCE_LEVEL and level in r.state_levels:
        h[2 + r.state_levels.index(level)] = 1.0
    return float(h @ r.mean), math.sqrt(float(h @ r.cov @ h))


def national_forecast_check(obs: GlmObservations, seat_vote: list, since: date = date(2022, 1, 1),
                            slope: str = "none", jumps: bool = False, muni: bool = False,
                            seat_vote_san: list | None = None, ell=None,
                            **fit_kw) -> list[dict]:
    """各国政選挙の前日までのデータだけで、パラメータを決め直して結果を予測する。
    ellは固定値、選挙の日付から値を返す関数、Noneならその政党のデータで推定。
    比較の基準は同じ院の前回の結果。予測はlogitの空間で行い、占有率に戻して返す。"""
    results = []
    last: dict[str, float] = {}
    for i, e in enumerate(obs.events):
        if e.level not in ss.NATIONAL_LEVELS:
            continue
        y_logit, sd_nat = national_logit(obs.k[i], obs.n[i], fit_kw.get("sigma_national"))
        if e.vote_date >= since and e.level in last:
            until = obs.t[i] - 1 / 30
            san = sangiin_link(seat_vote, seat_vote_san, e.vote_date) if seat_vote_san else None
            ell_e = ell(e.vote_date) if callable(ell) else ell
            params, _ = fit_params(obs, ell_e, until=until, slope=slope, jumps=jumps, muni=muni, san=san, **fit_kw)
            m, s = predict_national(obs, params, until, e.level)
            s_tot = math.hypot(s, sd_nat)
            results.append({
                "vote_date": e.vote_date.isoformat(), "level": e.level, "y": float(obs.y[i]),
                "pred": _expit(m), "lo": _expit(m - 2 * s_tot), "hi": _expit(m + 2 * s_tot),
                "z": (y_logit - m) / s_tot, "naive": last[e.level], "ell": params.ell,
            })
        last[e.level] = float(obs.y[i])
    return results


PARAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "trend_glm_params.json"
# 保存したパラメータがどの観測の組み合わせで推定したものか。v3は国政・地方議会の補選(議席)を加えた形
MODEL = "v3"


def save_params(path: Path, params: dict[str, GlmParams], meta: dict,
                national_dev: dict[str, GlmParams] | None = None) -> None:
    body = {"meta": meta, "parties": {p: asdict(v) for p, v in params.items()}}
    if national_dev:
        body["national_dev"] = {p: asdict(v) for p, v in national_dev.items()}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1))


def load_params(path: Path) -> tuple[dict[str, GlmParams], dict, dict[str, GlmParams]]:
    body = json.loads(path.read_text())
    params = {p: GlmParams(**v) for p, v in body["parties"].items()}
    dev = {p: GlmParams(**v) for p, v in body.get("national_dev", {}).items()}
    return params, body["meta"], dev


def latest_national(events: list[Event]) -> str | None:
    dates = [e.vote_date for e in events if e.level in ss.NATIONAL_LEVELS]
    return max(dates).isoformat() if dates else None


def refit_reason(events: list[Event], path: Path = PARAMS_PATH) -> str | None:
    """パラメータを推定し直すべき理由。保存したときより新しい国政選挙がデータにあれば推定し直す。"""
    if not path.exists():
        return "保存したパラメータがない"
    meta = json.loads(path.read_text())["meta"]
    if meta.get("model") != MODEL:
        return f"保存したパラメータが{MODEL}の観測で推定したものでない"
    saved = meta.get("last_national")
    now = latest_national(events)
    if now is not None and (saved is None or now > saved):
        return f"保存後の国政選挙({now})がある"
    return None


def _fit_one(job: tuple) -> GlmParams:
    obs, ell, fit_kw = job
    return fit_params(obs, ell, **fit_kw)[0]


def _fit_q(job: tuple) -> tuple[GlmParams, float]:
    obs, ell, fit_kw, q = job
    return fit_params(obs, ell, q_x=q, **fit_kw)


def pooled_fit(ex, obs_by_party: dict[str, GlmObservations], ells: dict[str, float], fit_kw: dict[str, dict],
               grid=(0.005, 0.01, 0.02, 0.04)) -> tuple[float, dict[str, GlmParams]]:
    """全政党に共通のランダムウォークの分散qを、対数尤度の合計の最大点として決め、その値での
    各党のパラメータを返す。格子の上で合計を求め、最大点と両隣の3点をlog qの2次式で補間する。
    最大が格子の端なら、fit_paramsのqの範囲に達するまで外側へ倍々に広げる。exはmapを持つExecutor。"""
    parties = list(ells)
    fits: dict[float, dict[str, tuple[GlmParams, float]]] = {}

    def run(qs):
        qs = [q for q in qs if q not in fits]
        out = list(ex.map(_fit_q, [(obs_by_party[p], ells[p], fit_kw[p], q) for q in qs for p in parties]))
        for j, q in enumerate(qs):
            fits[q] = {p: out[j * len(parties) + i] for i, p in enumerate(parties)}

    def total(q):
        return sum(ll for _, ll in fits[q].values())

    gs = sorted(grid)
    run(gs)
    while True:
        i = max(range(len(gs)), key=lambda j: total(gs[j]))
        if i == 0 and gs[0] / 2 > math.exp(-25):
            gs.insert(0, gs[0] / 2)
            run(gs[:1])
        elif i == len(gs) - 1 and gs[-1] * 2 < math.e:
            gs.append(gs[-1] * 2)
            run(gs[-1:])
        else:
            break
    q = gs[i]
    if 0 < i < len(gs) - 1:
        x = np.log(gs[i - 1:i + 2])
        a, b, _ = np.polyfit(x, [total(g) for g in gs[i - 1:i + 2]], 2)
        if a < 0:
            q = float(np.exp(min(max(-b / (2 * a), x[0]), x[2])))
    run([q])
    return q, {p: fits[q][p][0] for p in parties}


def estimate_all(obs_by_party: dict[str, GlmObservations], ells: dict[str, float], fit_kw: dict[str, dict],
                 workers: int | None = None, national_dev: bool = False) -> tuple[dict, dict]:
    """全党のパラメータを推定する。ランダムウォークの分散は全党共通の値をpooled_fitで決める。
    fit_kwは政党ごとのfit_paramsの引数(参院のつなぎが政党ごとに違う)。"""
    from concurrent.futures import ProcessPoolExecutor

    parties = list(ells)
    with ProcessPoolExecutor(workers) as ex:
        q, params = pooled_fit(ex, obs_by_party, ells, fit_kw)
        dev = {}
        if national_dev:
            dev = dict(zip(parties, ex.map(_fit_one, [(obs_by_party[p], ells[p],
                                                       {**fit_kw[p], "national_dev": True, "q_x": q})
                                                      for p in parties])))
    return params, dev


def month_facts(result: dict, top: int = 10) -> dict:
    """記事の事実データに載せる党勢指数v3の節。各党の最新月の値(その月までの選挙による値)と
    前月からの変化、その月の選挙のうち党勢を占有率の尺度で大きく動かしたもの。dzの党ごとの和は、
    最新月のxのlogitの前月からの変化にちょうど等しい。"""
    parties, impacts = {}, []
    for p, r in result["parties"].items():
        s = r["series"]
        cur, prev = s[-1], (s[-2] if len(s) > 1 else None)
        x = cur.get("x_f", cur["x"])
        parties[p] = {"x": x, "x_lo": cur.get("x_f_lo", cur["x_lo"]), "x_hi": cur.get("x_f_hi", cur["x_hi"]),
                      "change": x - prev.get("x_f", prev["x"]) if prev else None}
        for c in r.get("contributions", []):
            # 占有率の尺度での動きは、dzが小さければx(1-x)dzで近似できる
            impacts.append({"party": p, **c, "dx": x * (1 - x) * c["dz"]})
    impacts.sort(key=lambda c: -abs(c["dx"]))
    return {"tau": result["tau"], "parties": parties, "impacts": impacts[:top]}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="logitの状態とベータ二項分布の観測による党勢の現在推定(党勢指数v3)")
    ap.add_argument("--refit", action="store_true",
                    help="パラメータを必ず推定し直して保存する。指定しなくても、保存後の国政選挙が"
                         "データにあれば推定し直す。それ以外は保存済みのパラメータでフィルタと平滑化だけを行う")
    ap.add_argument("--national-dev", action="store_true",
                    help="--refitと併せて、診断用の国政だけの成分z_Nのパラメータも推定する")
    ap.add_argument("--validate", action="store_true",
                    help="国政選挙を1回ずつ除いた補間の検証も行い、結果を別のファイルに書く(8コアで30分ほどかかる)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    run(refit=args.refit, national_dev=args.national_dev, validate=args.validate, workers=args.workers)


def run(refit: bool = False, national_dev: bool = False, validate: bool = False,
        workers: int | None = None) -> dict:
    """月次の党勢指数v3を計算して保存する。v3はv2の観測に国政・地方議会の補選の議席を加えたもの
    (補選の得票率は、国政選挙の補間を良くしなかったので使わない)。保存後の国政選挙がデータにあれば
    パラメータを推定し直す。戻り値は出力の中身(result)、保存先(path)、検証の保存先(validation_path)、
    推定し直した理由(refit_reason、推定し直さなければNone)。"""
    from .by_elections import load_events_with_by_elections
    from .trend import OUTPUT_DIR, month_ends

    events, _ = load_events_with_by_elections()
    events = [e for e in events if e.level != NATIONAL_VOTE_BY]
    # 追う党と各党の始まりは、補選を除いた選挙で決める(補選だけに出た党は追わない)
    regular = [e for e in events if e.level not in BY_LEVELS]
    first = first_appearance(regular)
    last = regular[-1].vote_date
    tau = month_ends((last.year, last.month), (last.year, last.month))[-1]
    obs_by_party = {p: glm_observations(events, p, first[p], filter_levels=False) for p in ss.tracked_parties(regular)}
    reason = "--refitの指定" if refit else refit_reason(events)
    shu = ss.load_seat_vote_points()
    san = ss.load_seat_vote_points(house="sangiin")
    nat = latest_national(events)
    if reason and PARAMS_PATH.exists() and nat not in {p.vote_date.isoformat() for p in shu + san}:
        # 負荷γ_pは直近の国政選挙の得票率から決めるので、総務省の集計が出るまで推定し直さない
        print(f"{nat}の国政選挙の得票率がまだないので、保存済みのパラメータを使う", flush=True)
        reason = None
    if reason:
        print("パラメータを推定し直す:", reason, flush=True)
        ells = {}
        for party, obs in obs_by_party.items():
            ell = party_loading(obs, shu, san, party, tau)
            if ell is None:
                print(party, "国政選挙の結果がなく負荷を決められない", flush=True)
                continue
            ells[party] = ell
        fit_kw = {p: {"muni": True, "san": sangiin_link(shu, san)} for p in ells}
        params_by_party, dev_by_party = estimate_all(obs_by_party, ells, fit_kw, workers, national_dev)
        meta = {"model": MODEL, "estimated_at": tau.isoformat(), "last_event": last.isoformat(),
                "last_national": latest_national(events)}
        save_params(PARAMS_PATH, params_by_party, meta, dev_by_party)
        print(PARAMS_PATH, flush=True)
    else:
        params_by_party, meta, dev_by_party = load_params(PARAMS_PATH)
    obs_by_party = {p: obs_by_party[p] for p in params_by_party}
    out = {"tau": tau.isoformat(), "params_estimated_at": meta["estimated_at"], "parties": {}}
    prev = month_ends((tau.year, tau.month), (tau.year, tau.month))[0].replace(day=1) - timedelta(days=1)
    for party, params in params_by_party.items():
        obs = obs_by_party[party]
        # 過去の月は、その後の選挙も使う平滑化の値で描く。最新の月はフィルタの値と同じになる
        ends = [m for m in month_ends((2019, 1), (tau.year, tau.month)) if m >= first[party]]
        times = [ss._t(m) for m in ends]
        # x_fはその月までのデータだけによる値。月ごとの選挙の寄与の和はx_fのlogitの変化に等しい
        series = []
        for m_end, (m, C), (zf, sf) in zip(ends, smooth(obs, params, times), filtered_series(obs, params, ends)):
            g = params.national_vector(len(m))
            z, se = float(g @ m), math.sqrt(max(float(g @ C @ g), 0.0))
            series.append({"tau": m_end.isoformat(), "x": _expit(z),
                           "x_lo": _expit(z - 2 * se), "x_hi": _expit(z + 2 * se),
                           "x_f": _expit(zf), "x_f_lo": _expit(zf - 2 * sf), "x_f_hi": _expit(zf + 2 * sf)})
        nc = nowcast(obs, params, tau)
        out["parties"][party] = {
            "x": nc.x, "x_lo": nc.x_lo, "x_hi": nc.x_hi, "alpha": nc.alpha,
            "params": {"ell": params.ell, "q_x": params.q_x, "phi": params.phi, "tau2": params.tau2,
                       "san_load": params.san_load, "san_offset": params.san_offset},
            "loglik": run_filter(obs, params).loglik, "series": series,
            "contributions": monthly_contributions(obs, params, prev, tau),
            "movement": [{"tau": m_end.isoformat(), **mv} for m_end, mv in zip(ends, level_movement(obs, params, ends))],
            "national": [{"vote_date": obs.events[i].vote_date.isoformat(), "level": obs.level[i], "y": float(obs.y[i])}
                         for i in range(len(obs.t)) if obs.level[i] in ss.NATIONAL_LEVELS],
        }
        if party in dev_by_party:
            pn = dev_by_party[party]
            dev = [{"tau": m_end.isoformat(), "z_N": float(m[-1]), "z_N_se": math.sqrt(max(float(C[-1, -1]), 0.0))}
                   for m_end, (m, C) in zip(ends, smooth(obs, pn, times))]
            out["parties"][party]["national_dev"] = {"q_nat": pn.q_nat, "series": dev}
        print(party, f"x={nc.x:.3f} [{nc.x_lo:.3f}, {nc.x_hi:.3f}]", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"state_space_glm_{tau.isoformat()}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(path)
    vpath = None
    if validate:
        checks = national_interpolation_check(obs_by_party, shu, san, workers)
        vpath = OUTPUT_DIR / f"state_space_glm_validation_{tau.isoformat()}.json"
        vpath.write_text(json.dumps({"tau": tau.isoformat(), "summary": interpolation_summary(checks),
                                     "parties": checks}, ensure_ascii=False, indent=1))
        print(vpath)
    return {"result": out, "path": path, "validation_path": vpath, "refit_reason": reason}


if __name__ == "__main__":
    main()
