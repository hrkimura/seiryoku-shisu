"""党勢指数v4: v3の状態空間モデル(state_space_glm)に、各社の世論調査の政党支持率を観測として加える。

地方選挙の結果は届くまでに数か月遅れるので、v3の直近の値は予測に頼る。支持率はほぼ毎週出るので、
その間を埋める観測になる。支持率は議席と水準が違い(党の性格による差)、選挙直後の山のように
議席に残らない動きも持つので、水準差alpha_調査・負荷b・減衰するブームw・調査会社の癖hを挟んで
国政の党勢につなぐ(GlmParamsのpoll_*)。v3のパラメータは保存済みの値に固定し、
世論調査の部分のパラメータだけをデータから推定する。
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import state_space as ss
from . import state_space_glm as glm
from .fetch.wikipedia_polls import HISTORY_PATH, Poll, from_json, party_total
from .lineage import ALLIANCE_MEMBERS, BRANCH_OF, LINEAGE_EVENTS
from .trend import Event

POLL = glm.POLL


def load_polls(path: Path = HISTORY_PATH) -> list[Poll]:
    return from_json(json.loads(path.read_text(encoding="utf-8")))


def _alliance_periods() -> dict[str, tuple[str, str]]:
    """合流した党の名義が支持率の表にあった期間(合流の日, 分裂の日)を、合流した党ごとに返す。"""
    out: dict[str, list[str | None]] = {}
    for ev in LINEAGE_EVENTS:
        if ev.kind == "合流":
            for a in ev.after:
                out.setdefault(a, [None, None])[0] = ev.date
        elif ev.kind == "分裂":
            for b in ev.before:
                out.setdefault(b, [None, None])[1] = ev.date
    return {k: (v[0] or "0000-00-00", v[1] or "9999-12-31") for k, v in out.items()}


def party_share(poll: Poll, party: str) -> float | None:
    """支持政党のある人の中での、その党(同じ系譜のブランチを含む)の比率。表に列がなければNone。

    合流した党(中道改革連合)の名義があった期間は、出身政党の支持者の一部が合流した名義を答えるので、
    出身政党の値は党勢を表さない。その期間の調査は出身政党の観測に使わない。分裂後に残った少しの
    支持は、合流した党の列にあっても出身政党には足さない(どちらの党か分けられない)。"""
    if party not in poll.shares:
        return None
    for alliance, members in ALLIANCE_MEMBERS.items():
        if party not in members:
            continue
        if any(party in c["parties"] for c in poll.combined):
            return None
        start, end = _alliance_periods().get(alliance, ("0000-00-00", "9999-12-31"))
        if alliance in poll.shares and start <= poll.end < end:
            return None
    value = poll.shares[party] + sum(v for p, v in poll.shares.items() if BRANCH_OF.get(p) == party)
    total = party_total(poll)
    if total <= 0:
        return None
    return value / total


def sample_sizes(polls: list[Poll]) -> dict[str, float]:
    """調査会社ごとのサンプル数の中央値。サンプル数が載っていない行の代わりに使う。キー""は全社の中央値。"""
    by_firm: dict[str, list[int]] = {}
    for p in polls:
        if p.n:
            by_firm.setdefault(p.firm, []).append(p.n)
    out = {f: float(statistics.median(v)) for f, v in by_firm.items()}
    out[""] = float(statistics.median([n for v in by_firm.values() for n in v]))
    return out


def poll_events(polls: list[Poll], party: str, start: date, until: date | None = None) -> list[Event]:
    """世論調査を観測のイベントにする。nは支持政党のある回答者の数、shares[party]はその中の比率、
    nameは調査会社。調査の日付は調査期間の最終日とする。"""
    sizes = sample_sizes(polls)
    out = []
    for p in polls:
        end = date.fromisoformat(p.end)
        if end < start or (until is not None and end > until):
            continue
        s = party_share(p, party)
        if s is None:
            continue
        n = (p.n or sizes.get(p.firm, sizes[""])) * party_total(p) / 100
        out.append(Event(vote_date=end, level=POLL, P=0.0, n=n, shares={party: s}, name=p.firm))
    return out


def with_polls(obs: glm.GlmObservations, events: list[Event]) -> glm.GlmObservations:
    """選挙の観測に世論調査の観測を加え、時刻順に並べ直す。同じ時刻なら選挙を先にする。"""
    t = np.concatenate([obs.t, [ss._t(e.vote_date) for e in events]])
    y = np.concatenate([obs.y, [e.shares[next(iter(e.shares))] for e in events]])
    n = np.concatenate([obs.n, [e.n for e in events]])
    order = np.argsort(t, kind="stable")
    level, evs = obs.level + [e.level for e in events], obs.events + events
    return glm.GlmObservations(t=t[order], level=[level[i] for i in order], k=(y * n)[order], n=n[order],
                               y=y[order], events=[evs[i] for i in order])


def firms_of(obs: glm.GlmObservations) -> list[str]:
    return sorted({e.name for e, lv in zip(obs.events, obs.level) if lv == POLL})


# 推定する世論調査のパラメータ。どれも正なので対数で持つ。
POLL_PARAMS = ("poll_load", "poll_decay", "poll_boom_var", "poll_house_var", "poll_extra_var")
_BOUNDS = {
    "poll_load": (math.log(0.05), math.log(5.0)),
    # 減衰の時定数が1週間から20年の範囲
    "poll_decay": (math.log(1 / 240), math.log(4.0)),
    "poll_boom_var": (math.log(1e-5), math.log(4.0)),
    "poll_house_var": (math.log(1e-5), math.log(4.0)),
    "poll_extra_var": (math.log(1e-6), math.log(4.0)),
}
_INIT = {"poll_load": 1.0, "poll_decay": 0.3, "poll_boom_var": 0.05, "poll_house_var": 0.02, "poll_extra_var": 0.01}


def fit_poll_params(obs: glm.GlmObservations, base: glm.GlmParams, until: float | None = None,
                    maxiter: int = 200) -> tuple[glm.GlmParams, float]:
    """v3のパラメータbaseを固定し、世論調査の部分のパラメータを最尤推定する。
    尤度はすべての観測(選挙と世論調査)の1期先予測によるもの。"""
    firms = firms_of(obs) if until is None else sorted(
        {e.name for e, lv, t in zip(obs.events, obs.level, obs.t) if lv == POLL and t <= until})
    levels = sorted({lv for lv in obs.level if lv != ss.REFERENCE_LEVEL})

    def unpack(z):
        return replace(base, poll_firms=firms, **{k: math.exp(v) for k, v in zip(POLL_PARAMS, z)})

    def nll(z):
        # 探索の途中で分散が極端な値に行くと、共分散が数値的に壊れることがある。その点は尤度が
        # 非常に低いものとして扱い、探索を続ける
        try:
            ll = glm.run_filter(obs, unpack(z), until=until, state_levels=levels).loglik
        except (np.linalg.LinAlgError, ValueError, OverflowError, ZeroDivisionError):
            return 1e12
        return -ll if math.isfinite(ll) else 1e12

    z0 = np.array([math.log(_INIT[k]) for k in POLL_PARAMS])
    res = minimize(nll, z0, method="L-BFGS-B", bounds=[_BOUNDS[k] for k in POLL_PARAMS],
                   options={"maxiter": maxiter})
    return unpack(res.x), -res.fun


PARAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "trend_glm_poll_params.json"


def load_inputs(polls_path: Path = HISTORY_PATH):
    """v3と同じ選挙の観測と、保存済みのv3のパラメータ、世論調査を読む。"""
    from .by_elections import BY_LEVELS, NATIONAL_VOTE_BY, load_events_with_by_elections
    from .trend import first_appearance

    events, _ = load_events_with_by_elections()
    events = [e for e in events if e.level != NATIONAL_VOTE_BY]
    first = first_appearance([e for e in events if e.level not in BY_LEVELS])
    base, _, _ = glm.load_params(glm.PARAMS_PATH)
    polls = load_polls(polls_path)
    obs = {p: glm.glm_observations(events, p, first[p], filter_levels=False) for p in base}
    return obs, base, polls, first


def _fit_job(job):
    import time

    party, obs, base, until = job
    t0 = time.time()
    fitted, ll = fit_poll_params(obs, base, until=until)
    print(f"{party} 推定 {time.time() - t0:.0f}秒", flush=True)
    return party, fitted, ll


def _param_row(f: glm.GlmParams, ll: float) -> dict:
    return {**{k: getattr(f, k) for k in POLL_PARAMS}, "loglik": ll, "firms": f.poll_firms}


def fit_all(obs_by_party: dict, base: dict, polls: list[Poll], first: dict, until: date | None = None,
            workers: int | None = None, checkpoint: Path | None = None) -> dict[str, dict]:
    """全党の世論調査のパラメータを推定する。1党に数十分かかるので、checkpointを渡すと党ごとに
    終わった時点で書き出し、すでに書き出してある党は推定し直さない。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    done: dict[str, dict] = {}
    if checkpoint is not None and checkpoint.exists():
        done = json.loads(checkpoint.read_text())
    jobs = []
    for party, obs in obs_by_party.items():
        if party in done:
            continue
        merged = with_polls(obs, poll_events(polls, party, first[party], until))
        jobs.append((party, merged, base[party], None if until is None else ss._t(until)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_fit_job, j) for j in jobs]):
            p, f, ll = fut.result()
            done[p] = _param_row(f, ll)
            if checkpoint is not None:
                checkpoint.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    return done


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    obs_by_party, base, polls, first = load_inputs()
    print(f"読み込み {time.time() - t0:.0f}秒", flush=True)
    checkpoint = PARAMS_PATH.with_suffix(".partial.json")
    fitted = fit_all(obs_by_party, base, polls, first, workers=args.workers, checkpoint=checkpoint)
    body = {"meta": {"polls_last": polls[-1].end, "estimated_at": date.today().isoformat()}, "parties": fitted}
    PARAMS_PATH.write_text(json.dumps(body, ensure_ascii=False, indent=1))
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(body, ensure_ascii=False, indent=1), flush=True)
    print(f"合計 {time.time() - t0:.0f}秒")




def poll_refit_reason(base_meta: dict, path: Path = PARAMS_PATH) -> str | None:
    """世論調査のパラメータを推定し直す理由。世論調査のパラメータはv3のパラメータを固定して
    推定したものなので、v3を推定し直したら(国政選挙のあと)こちらも推定し直す。"""
    if not path.exists():
        return "保存済みの世論調査のパラメータがない"
    meta = json.loads(path.read_text()).get("meta", {})
    if meta.get("base_estimated_at") != base_meta.get("estimated_at"):
        return f"v3のパラメータが推定し直された({base_meta.get('estimated_at')})"
    return None


def attach_polls(obs_by_party: dict, base: dict, base_meta: dict, first: dict, tau: date,
                 workers: int | None = None, path: Path = PARAMS_PATH,
                 polls_path: Path = HISTORY_PATH) -> tuple[dict, dict, dict]:
    """v3の観測とパラメータに、tauまでの世論調査を加える(党勢指数v4)。世論調査のパラメータは
    保存済みの値を使い、v3を推定し直した後の最初の実行でだけ推定し直す(8コアで1〜2時間)。
    世論調査のパラメータがない党(推定の後に追い始めた党)は、世論調査なしのv3のまま。
    戻り値は(パラメータ, 観測, 記録)。"""
    polls = [p for p in load_polls(polls_path) if date.fromisoformat(p.end) <= tau]
    reason = poll_refit_reason(base_meta, path)
    if reason:
        print("世論調査のパラメータを推定し直す:", reason, flush=True)
        # 途中の結果は、どのv3のパラメータに対するものかをファイル名で区別する
        checkpoint = path.with_name(f"{path.stem}.partial_{base_meta.get('estimated_at')}.json")
        fitted = fit_all(obs_by_party, base, polls, first, workers=workers, checkpoint=checkpoint)
        body = {"meta": {"polls_last": polls[-1].end, "estimated_at": date.today().isoformat(),
                         "base_estimated_at": base_meta.get("estimated_at")}, "parties": fitted}
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1))
        checkpoint.unlink(missing_ok=True)
    with_poll = load_poll_params(base, path)
    params = {p: with_poll.get(p, b) for p, b in base.items()}
    obs = {p: with_polls(o, poll_events(polls, p, first[p])) if p in with_poll else o
           for p, o in obs_by_party.items()}
    info = {"n": len(polls), "last": polls[-1].end if polls else None, "refit_reason": reason,
            "parties_without_polls": sorted(p for p in base if p not in with_poll)}
    return params, obs, info


def _params_from_row(base: glm.GlmParams, row: dict) -> glm.GlmParams:
    return replace(base, poll_firms=row["firms"], **{k: row[k] for k in POLL_PARAMS})


def load_poll_params(base: dict[str, glm.GlmParams], path: Path = PARAMS_PATH) -> dict[str, glm.GlmParams]:
    body = json.loads(path.read_text())
    return {p: _params_from_row(base[p], v) for p, v in body["parties"].items() if p in base}


def national_dates(obs_by_party: dict) -> dict[str, list[date]]:
    """党ごとの、予測の対象にする国政選挙の投票日(党の最初の国政選挙は除く)。"""
    return {p: [o.events[i].vote_date for i, lv in enumerate(o.level) if lv in ss.NATIONAL_LEVELS][1:]
            for p, o in obs_by_party.items()}


def _fit_oos_job(job):
    key, obs, base, until = job
    fitted, ll = fit_poll_params(obs, base, until=until)
    print(f"{key} 推定済み", flush=True)
    return key, fitted, ll


def fit_out_of_sample(obs_by_party: dict, base: dict, polls: list[Poll], first: dict, checkpoint: Path,
                      workers: int | None = None) -> dict[str, dict]:
    """予測する国政選挙ごとに、その投票日の前日までのデータで世論調査のパラメータを推定し直す。
    v3のパラメータbaseは全期間の値のままで、v3とv4に共通なので比較には効かない。
    鍵は「投票日|党」。全部で数時間かかるので、1件終わるごとにcheckpointへ書き出し、
    書き出し済みの鍵は推定し直さない。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    done: dict[str, dict] = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    jobs = []
    for party, dates in national_dates(obs_by_party).items():
        for d in dates:
            key = f"{d.isoformat()}|{party}"
            if key in done:
                continue
            until = d.fromordinal(d.toordinal() - 1)
            merged = with_polls(obs_by_party[party], poll_events(polls, party, first[party], until))
            jobs.append((key, merged, base[party], ss._t(until)))
    # 新しい選挙ほどデータが多く時間がかかるので先に流し、終わりぎわに長い推定が1つだけ残るのを避ける
    jobs.sort(key=lambda j: j[0], reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_fit_oos_job, j) for j in jobs]):
            key, f, ll = fut.result()
            done[key] = _param_row(f, ll)
            checkpoint.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    return done


def _months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, min(d.day, 28))


def _predict_job(job):
    party, d, level, i_true, sub, params, k, n = job
    mu, s = glm.predict_national(sub, params, ss._t(d), level)
    y_logit, sd_nat = glm.national_logit(k, n, params.sigma_national)
    s_tot = math.hypot(s, sd_nat)
    return {"party": party, "vote_date": d.isoformat(), "level": level, "y": (k + 0.5) / (n + 1.0),
            "pred": glm._expit(mu), "lo": glm._expit(mu - 2 * s_tot), "hi": glm._expit(mu + 2 * s_tot),
            "z": (y_logit - mu) / s_tot, "sd": s_tot,
            "logpdf": -0.5 * math.log(2 * math.pi * s_tot ** 2) - 0.5 * ((y_logit - mu) / s_tot) ** 2}


def lagged_online_check(obs_by_party: dict, base: dict, poll_params: dict, polls: list[Poll], first: dict,
                        lags: tuple[int, ...] = (0, 3, 6, 12), workers: int | None = None,
                        out_of_sample: bool = False) -> list[dict]:
    """各国政選挙の結果を、その投票日より前のデータで予測する。地方選挙(補選を含む)は投票日の
    lagか月前より前のものだけを使い、結果が届くまでの遅れを再現する。v3は世論調査なし、v4は
    投票日の前日までの世論調査を加える。v3のパラメータは全期間で推定した保存済みの値。
    v4の世論調査のパラメータは、out_of_sampleなら poll_params[「投票日|党」](fit_out_of_sampleの
    結果をGlmParamsにしたもの)、そうでなければ全期間の値 poll_params[党]。"""
    from concurrent.futures import ProcessPoolExecutor

    jobs, meta = [], []
    for party, o in obs_by_party.items():
        if not out_of_sample and party not in poll_params:
            continue
        merged = with_polls(o, poll_events(polls, party, first[party]))
        nat = [i for i, lv in enumerate(o.level) if lv in ss.NATIONAL_LEVELS][1:]
        for i in nat:
            d, level = o.events[i].vote_date, o.level[i]
            pp = poll_params.get(f"{d.isoformat()}|{party}") if out_of_sample else poll_params[party]
            if pp is None:
                continue
            for lag in lags:
                cut = _months_before(d, lag)
                for model, src, params in (("v3", o, base[party]), ("v4", merged, pp)):
                    keep = np.array([
                        e.vote_date < d and (lv in ss.NATIONAL_LEVELS or lv == POLL or e.vote_date < cut)
                        for e, lv in zip(src.events, src.level)])
                    jobs.append((party, d, level, i, glm._subset(src, keep), params, o.k[i], o.n[i]))
                    meta.append({"model": model, "lag": lag})
    with ProcessPoolExecutor(workers) as ex:
        return [{**m, **r} for m, r in zip(meta, ex.map(_predict_job, jobs, chunksize=4))]


def summarize_check(rows: list[dict]) -> list[dict]:
    out = []
    for key in sorted({(r["model"], r["lag"]) for r in rows}, key=lambda k: (k[1], k[0])):
        rs = [r for r in rows if (r["model"], r["lag"]) == key]
        err = np.array([r["pred"] - r["y"] for r in rs])
        out.append({"model": key[0], "lag": key[1], "n": len(rs), "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "coverage": float(np.mean([r["lo"] <= r["y"] <= r["hi"] for r in rs])),
                    "mean_sd_logit": float(np.mean([r["sd"] for r in rs])),
                    "mean_logpdf": float(np.mean([r["logpdf"] for r in rs]))})
    return out


def council_lag_profile(events, today: date, months: int = 24) -> list[dict]:
    """直近の月ごとの地方議会選挙の件数と、4年前(統一地方選の周期)の同じ月の件数。
    結果が届いていない月ほど比が小さくなるので、データの遅れが読める。"""
    from collections import Counter

    c = Counter((e.vote_date.year, e.vote_date.month) for e in events if e.level in glm.COUNCIL_LEVELS)
    out = []
    for k in range(months):
        d = _months_before(today, k)
        out.append({"month": f"{d.year}-{d.month:02d}", "count": c[(d.year, d.month)],
                    "count_4y_ago": c[(d.year - 4, d.month)]})
    return out


def check_main(argv: list[str] | None = None) -> None:
    import argparse

    from .by_elections import NATIONAL_VOTE_BY, load_events_with_by_elections
    from .trend import OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-of-sample", action="store_true",
                    help="v4の世論調査のパラメータを、予測する選挙の前日までのデータで推定し直す(数時間)")
    args = ap.parse_args(argv)
    obs_by_party, base, polls, first = load_inputs()
    events, _ = load_events_with_by_elections()
    profile = council_lag_profile([e for e in events if e.level != NATIONAL_VOTE_BY], date.today())
    (OUTPUT_DIR / "poll").mkdir(parents=True, exist_ok=True)
    if args.out_of_sample:
        rows_oos = fit_out_of_sample(obs_by_party, base, polls, first,
                                     OUTPUT_DIR / "poll" / "out_of_sample_params.json", workers=args.workers)
        params = {k: _params_from_row(base[k.split("|")[1]], v) for k, v in rows_oos.items()}
    else:
        params = load_poll_params(base)
    rows = lagged_online_check(obs_by_party, base, params, polls, first, workers=args.workers,
                               out_of_sample=args.out_of_sample)
    out = {"summary": summarize_check(rows), "council_lag_profile": profile, "rows": rows}
    name = "lagged_online_check_oos" if args.out_of_sample else "lagged_online_check"
    path = OUTPUT_DIR / "poll" / f"{name}_{date.today().isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"summary": out["summary"], "council_lag_profile": profile}, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["check"]:
        check_main(sys.argv[2:])
    else:
        main()
