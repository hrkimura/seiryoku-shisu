"""党勢指数$C_p(t)$と政党支持率$S_p(t)$の連動を、見せかけの相関を避けて検証する。

手順(政党ごと):
1. 両系列の和分次数をADF検定とKPSS検定の組み合わせで判定する。
2. ともにI(1)ならEngle-Granger法で共和分を検定する。
3. 判定結果に応じて、水準または差分のどちらの相関を主に見るかを決める。
   - I(0)同士 -> 水準
   - ともにI(1)かつ共和分あり -> 水準(長期関係、$C_p-\\beta S_p$の$\\beta$も記録)
   - それ以外 -> 差分
4. 十分な長さがあれば$\\mathrm{Corr}(\\Delta S_p(t), \\Delta C_p(t+k))$をkごとに計算する。

$C_p(t)$は直近12か月の後方窓で集計したフロー指標なので、$\\Delta C_p(t)$は
「月tに入った選挙」と「月t-12に窓から抜けた選挙」の差を反映する。月次の
世論の変化と1対1に対応する量ではない点に注意して読む。

標本が短いうちは検定を行わず、理由を添えて"insufficient"を返す。支持率は
2026年10月から積み上げ始めたため(fetch/miharu.py参照)、当面はこの状態が続く。
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

RESULT_PATH = Path(__file__).resolve().parents[2] / "data" / "support_analysis.json"

ALPHA = 0.05
# ADF・KPSSは数十点ないと検出力がほとんど無く、判定がほぼ常に「非定常」へ偏る。
MIN_OBS_UNIT_ROOT = 24
MIN_OBS_CORR = 12
MAX_LAG = 6
MIN_OBS_LAG = 24

YM = tuple[int, int]


def _prev(ym: YM) -> YM:
    y, m = ym
    return (y, m - 1) if m > 1 else (y - 1, 12)


def contiguous_pairs(
    cp_history: list[dict], support_monthly: dict[YM, dict[str, float]], party: str,
) -> list[tuple[YM, float, float]]:
    """$C_p$と$S_p$がともにある月のうち、最新月で終わる連続した区間を返す。

    差分・単位根検定は等間隔の系列を前提にするため、欠測月をまたいでつながない。
    $C_p$は共有(0-1)なので%に直して$S_p$と単位を揃える。
    """
    cp = {(h["year"], h["month"]): h["C_p"] for h in cp_history}
    common = sorted(
        ym for ym in support_monthly
        if ym in cp and party in support_monthly[ym]
    )
    if not common:
        return []
    run = [common[-1]]
    for ym in reversed(common[:-1]):
        if ym != _prev(run[-1]):
            break
        run.append(ym)
    run.reverse()
    return [(ym, 100.0 * cp[ym].get(party, 0.0), support_monthly[ym][party]) for ym in run]


def _is_constant(x: np.ndarray) -> bool:
    return bool(np.allclose(x, x[0]))


def _stationarity(x: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    if _is_constant(x):
        return {"verdict": "constant"}
    with warnings.catch_warnings():
        # KPSSはp値が表の範囲外だと警告を出し、端の値(0.01/0.1)に丸める。判定には十分。
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(x, regression="c", autolag="AIC")[1])
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
    if adf_p < ALPHA and kpss_p >= ALPHA:
        verdict = "stationary"
    elif adf_p >= ALPHA and kpss_p < ALPHA:
        verdict = "nonstationary"
    else:
        verdict = "ambiguous"
    return {"verdict": verdict, "adf_p": adf_p, "kpss_p": kpss_p}


def integration_order(x: np.ndarray) -> dict:
    """和分次数を0か1で返す。判定できなければorderはNone。

    水準で2つの検定の結論が割れた場合は、差分が定常ならI(1)として扱う。
    I(0)の系列を差分しても相関が見せかけになることはないが、逆にI(1)の系列を
    水準で相関させると見せかけの相関を拾うので、迷ったら差分側に倒す。
    """
    level = _stationarity(x)
    if level["verdict"] == "stationary":
        return {"order": 0, "level": level}
    diff = _stationarity(np.diff(x))
    order = 1 if diff["verdict"] == "stationary" else None
    return {"order": order, "level": level, "diff": diff}


def cointegration(c: np.ndarray, s: np.ndarray) -> dict:
    """Engle-Granger法。$C_p(t) = a + \\beta S_p(t) + u(t)$の残差uの定常性を見る。"""
    from statsmodels.tsa.stattools import coint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(coint(c, s, trend="c")[1])
    beta, a = np.polyfit(s, c, 1)
    return {"p": p, "cointegrated": p < ALPHA, "beta": float(beta), "intercept": float(a)}


def _corr(x: np.ndarray, y: np.ndarray, min_n: int) -> dict | None:
    from scipy.stats import pearsonr

    n = len(x)
    if n < min_n or _is_constant(x) or _is_constant(y):
        return None
    r, p = pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": n}


def lag_correlations(ds: np.ndarray, dc: np.ndarray, max_lag: int = MAX_LAG) -> dict[int, dict]:
    """$\\mathrm{Corr}(\\Delta S(t), \\Delta C(t+k))$。k>0は支持率が先行、k<0は党勢指数が先行。"""
    out: dict[int, dict] = {}
    n = len(ds)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x, y = ds[: n - k], dc[k:]
        else:
            x, y = ds[-k:], dc[: n + k]
        c = _corr(x, y, MIN_OBS_LAG)
        if c is not None:
            out[k] = c
    return out


def analyze_party(
    pairs: list[tuple[YM, float, float]], provisional_start: YM | None = None,
) -> dict:
    n = len(pairs)
    result: dict = {
        "n_months": n,
        "start": f"{pairs[0][0][0]}-{pairs[0][0][1]:02d}" if pairs else None,
        "end": f"{pairs[-1][0][0]}-{pairs[-1][0][1]:02d}" if pairs else None,
        "n_provisional_months": (
            sum(1 for ym, _, _ in pairs if ym >= provisional_start) if provisional_start else None
        ),
    }
    c = np.array([p[1] for p in pairs])
    s = np.array([p[2] for p in pairs])
    dc, ds = np.diff(c), np.diff(s)

    result["level_corr"] = _corr(c, s, MIN_OBS_CORR) if n else None
    result["diff_corr"] = _corr(dc, ds, MIN_OBS_CORR) if n > 1 else None

    if n < MIN_OBS_UNIT_ROOT:
        result["status"] = "insufficient"
        result["reason"] = f"単位根検定には{MIN_OBS_UNIT_ROOT}か月以上必要(現在{n}か月)"
        result["primary"] = None
        return result

    ic, is_ = integration_order(c), integration_order(s)
    result["integration"] = {"C_p": ic, "S_p": is_}
    oc, os_ = ic["order"], is_["order"]

    if oc is None or os_ is None:
        result["status"] = "undetermined"
        result["reason"] = "和分次数を0か1に判定できない系列がある"
        result["primary"] = None
    elif oc == 0 and os_ == 0:
        result["status"] = "I(0)-I(0)"
        result["primary"] = "level"
    elif oc == 1 and os_ == 1:
        co = cointegration(c, s)
        result["cointegration"] = co
        if co["cointegrated"]:
            result["status"] = "I(1)-I(1) cointegrated"
            result["primary"] = "level"
        else:
            result["status"] = "I(1)-I(1) not cointegrated"
            result["primary"] = "difference"
    else:
        result["status"] = "mixed"
        result["primary"] = "difference"

    if len(ds) >= MIN_OBS_LAG + MAX_LAG:
        lags = lag_correlations(ds, dc)
        result["lag_corr"] = {str(k): v for k, v in lags.items()}
        if lags:
            best = max(lags, key=lambda k: abs(lags[k]["r"]))
            result["lag_peak"] = best
    return result


def analyze_all(
    cp_history: list[dict],
    support_monthly: dict[YM, dict[str, float]],
    provisional_start: YM | None = None,
) -> dict[str, dict]:
    """$C_p$側にも$S_p$側にも現れる政党すべてについて分析する。

    「無党派」と「無所属」は対応する量ではないので対象にしない。
    """
    cp_parties = {p for h in cp_history for p in h["C_p"]}
    s_parties = {p for v in support_monthly.values() for p in v}
    parties = sorted((cp_parties & s_parties) - {"無所属", "無党派"})
    return {
        p: analyze_party(contiguous_pairs(cp_history, support_monthly, p), provisional_start)
        for p in parties
    }


def save_results(results: dict[str, dict]) -> None:
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _renormalized(S_p: dict[str, float]) -> dict[str, float]:
    """無党派を除いて合計100%に直す。党勢指数v2は政党の間の占有率なので、同じ尺度に揃える。"""
    S = {p: v for p, v in S_p.items() if p != "無党派"}
    total = sum(S.values())
    return {p: v / total * 100 for p, v in S.items()} if total else {}


def _pearson_with_ci(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Pearsonの相関と、Fisherのz変換による95%信頼区間(n>3で定義される)。"""
    from scipy.stats import norm, pearsonr

    n = len(x)
    if n < 3 or _is_constant(x) or _is_constant(y):
        return None
    r, p = pearsonr(x, y)
    out = {"r": float(r), "p": float(p), "n": n, "ci95": None}
    if n > 3 and abs(r) < 1:
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        q = norm.ppf(0.975)
        out["ci95"] = [math.tanh(z - q * se), math.tanh(z + q * se)]
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> dict | None:
    from scipy.stats import spearmanr

    n = len(x)
    if n < 3 or _is_constant(x) or _is_constant(y):
        return None
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": n}


# 差をとる期間(か月)。v2の水準の誤差は隣の月どうしでほぼ共通なので、期間を延ばすと、党勢の
# 本当の変化が大きくなる一方で両端の誤差はあまり増えず、相関を検出しやすくなる
DIFF_HORIZONS = (1, 3, 6)


def trend_support_diffs(result: dict, history: list[dict], horizon: int = 1) -> dict:
    """党勢指数v2のhorizonか月の差と、同じ期間の政党支持率の差を、党×期間の点として集める。

    期間は最新の月末から過去へhorizonか月ずつ、重ならないように区切る(重ねると点が独立でなくなる)。
    支持率は月末の値にそろえる(support_history.value_atの補間)。v2の値はmonth_factsと同じく
    各月末までの選挙による値(x_f)を使う。単位はどちらも%ポイント。同じ期間の各党の点は独立では
    ない(占有率の和は一定)ので、相関の信頼区間は目安として読む。
    """
    from datetime import date

    from .support_history import value_at

    cache: dict[str, dict | None] = {}

    def support(tau: str) -> dict | None:
        if tau not in cache:
            v = value_at(history, date.fromisoformat(tau))
            cache[tau] = {**v, "S_p": _renormalized(v["S_p"])} if v else None
        return cache[tau]

    points = []
    for party, r in result["parties"].items():
        s = r["series"]
        for i in range(len(s) - 1, horizon - 1, -horizon):
            prev, cur = s[i - horizon], s[i]
            a, b = support(prev["tau"]), support(cur["tau"])
            if not a or not b or party not in a["S_p"] or party not in b["S_p"]:
                continue
            points.append({
                "party": party, "from": prev["tau"], "to": cur["tau"],
                "dx": 100 * (cur.get("x_f", cur["x"]) - prev.get("x_f", prev["x"])),
                "ds": b["S_p"][party] - a["S_p"][party],
            })
    x = np.array([p["ds"] for p in points])
    y = np.array([p["dx"] for p in points])
    points.sort(key=lambda p: (p["to"], p["party"]))
    return {
        "horizon": horizon,
        "points": points,
        "n": len(points),
        "periods": sorted({(p["from"], p["to"]) for p in points}),
        "alignment": {tau: {k: v[k] for k in ("method", "dates", "offset_days")}
                      for tau, v in sorted(cache.items()) if v},
        "pearson": _pearson_with_ci(x, y),
        "spearman": _spearman(x, y),
    }
