"""党勢指数v4で、地方選挙が国政の党勢の推定に何を足すかの検証。v1の「地方選は国政選挙の前哨戦か」
(precursor.py)の後継で、国政選挙のあとパラメータを推定し直した月に月次実行が回す。

2026-11-15の結果: 支持率があれば、地方選挙を加えても国政選挙の予測・補間の外れは誤差の範囲でしか
変わらず、国政選挙の結果がその後の地方選挙の予測を良くすることもない。地方選挙を観測に残すのは、
支持率が使えない時期(世論調査の表に載る前の新党、合流していた期間の出身政党)を埋めるためである。

3つの検証はどれも国政選挙(各党の最初の国政選挙を除く)ごとの党×選挙を単位とし、比べる2つの形の
差を党×選挙で対にして、平均と標準誤差で判定する。
- online: 投票日の前日までのデータによる予測(state_space_poll.lagged_online_check)
- interpolation: その国政選挙だけを除き、前後のデータで平滑化して当てる
- carryover: 国政選挙の結果を入れた場合と除いた場合の、その後CARRYOVER_DAYS日の地方議会選挙の予測
パラメータは全期間で推定した保存済みの値に固定する(interpolationとcarryoverは常に、onlineは
out_of_sampleでなければ)。比べる形のどちらにも同じ値を使うので、差の比較には効かない。
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from . import state_space as ss
from . import state_space_glm as glm
from . import state_space_poll as sp

# 地方選挙の結果が届くまでの遅れ(月)。LAG_ALLは地方選挙をすべて外すのと同じで、国政の補選も外れる
LAGS = (0, 6)
LAG_ALL = 1200
# 国政選挙の結果がどれだけ持ち越されるかを見る期間。v1の検証(1か月)より長くとり、統一地方選などの
# 大きな選挙の前の数か月を含める
CARRYOVER_DAYS = 92


def _local_levels() -> set[str]:
    from .by_elections import COUNCIL_BY_LEVELS

    return set(glm.COUNCIL_LEVELS) | set(COUNCIL_BY_LEVELS)


def paired(rows: list[dict], a: str, b: str, key: str = "arm") -> dict:
    """形aと形bを党×選挙で対にした比較。se_diffは二乗誤差の差(負ならaが当たる)、logpdf_diffは
    予測の対数尤度の差(正ならaが当たる)。tは差の平均を標準誤差で割った値。"""
    R = {(r[key], r["party"], r["vote_date"]): r for r in rows}
    keys = sorted({(r["party"], r["vote_date"]) for r in rows if (a, r["party"], r["vote_date"]) in R
                   and (b, r["party"], r["vote_date"]) in R})
    out = {"a": a, "b": b, "n": len(keys)}
    for name, f in (("se_diff", lambda x: (x["pred"] - x["y"]) ** 2), ("logpdf_diff", lambda x: x.get("logpdf"))):
        if any(f(R[(a,) + k]) is None for k in keys):
            continue
        d = np.array([f(R[(a,) + k]) - f(R[(b,) + k]) for k in keys])
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        out[name] = {"mean": float(d.mean()), "se": se, "t": float(d.mean() / se) if se > 0 else float("nan"),
                     "a_better": int((d < 0).sum() if name == "se_diff" else (d > 0).sum())}
    return out


def summarize(rows: list[dict], key: str = "arm") -> list[dict]:
    out = []
    for arm in dict.fromkeys(r[key] for r in rows):
        rs = [r for r in rows if r[key] == arm]
        err = np.array([r["pred"] - r["y"] for r in rs])
        row = {key: arm, "n": len(rs), "rmse": float(np.sqrt(np.mean(err ** 2)))}
        if all("lo" in r for r in rs):
            row["coverage"] = float(np.mean([r["lo"] <= r["y"] <= r["hi"] for r in rs]))
        if all("logpdf" in r for r in rs):
            row["mean_logpdf"] = float(np.mean([r["logpdf"] for r in rs]))
        out.append(row)
    return out


def online(obs, base, poll_params, polls, first, workers=None, out_of_sample=False) -> dict:
    rows = sp.lagged_online_check(obs, base, poll_params, polls, first, lags=LAGS + (LAG_ALL,),
                                  workers=workers, out_of_sample=out_of_sample)
    names = {("v3", LAG_ALL): "国政のみ", ("v4", LAG_ALL): "国政+世論調査"}
    for r in rows:
        r["arm"] = names.get((r["model"], r["lag"]),
                             f"{'国政+地方' if r['model'] == 'v3' else 'v4'}(遅れ{r['lag']}か月)")
    return {"summary": summarize(rows),
            "paired": [paired(rows, "v4(遅れ0か月)", "国政+世論調査"), paired(rows, "v4(遅れ6か月)", "国政+世論調査"),
                       paired(rows, "国政+地方(遅れ0か月)", "国政+世論調査"),
                       paired(rows, "国政+地方(遅れ6か月)", "国政+世論調査"),
                       paired(rows, "国政+地方(遅れ0か月)", "国政のみ")],
            "rows": rows}


def _interp_job(job):
    arm, party, o, params, i, use_local, local = job
    keep = np.ones(len(o.t), bool)
    keep[i] = False
    if not use_local:
        keep &= np.array([lv not in local for lv in o.level])
    sub = glm._subset(o, keep)
    e = o.events[i]
    levels = [lv for lv in sorted({lv for lv in o.level if lv != ss.REFERENCE_LEVEL})
              if lv in sub.level or lv == e.level]
    (m, C), = glm.smooth(sub, params, [o.t[i]], state_levels=levels)
    h = (params.san_load if e.level == glm.SANGIIN else 1.0) * params.national_vector(len(m))
    if e.level in levels:
        h[2 + levels.index(e.level)] = 1.0
    y_logit, sd_nat = glm.national_logit(o.k[i], o.n[i], params.sigma_national)
    mu = float(h @ m)
    s = math.hypot(math.sqrt(max(float(h @ C @ h), 0.0)), sd_nat)
    return {"arm": arm, "party": party, "vote_date": e.vote_date.isoformat(), "level": e.level,
            "y": float(o.y[i]), "pred": glm._expit(mu), "lo": glm._expit(mu - 2 * s), "hi": glm._expit(mu + 2 * s),
            "logpdf": -0.5 * math.log(2 * math.pi * s * s) - 0.5 * ((y_logit - mu) / s) ** 2}


def _national_pairs(o, merged):
    """観測oとそれに世論調査を足したmergedで、同じ国政選挙(各党の最初を除く)の添字の組。"""
    nat = [i for i, lv in enumerate(o.level) if lv in ss.NATIONAL_LEVELS][1:]
    natm = [i for i, lv in enumerate(merged.level) if lv in ss.NATIONAL_LEVELS][1:]
    return list(zip(nat, natm))


def interpolation(obs, base, poll_params, polls, first, workers=None) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    local = _local_levels()
    jobs = []
    for party, o in obs.items():
        if party not in poll_params:
            continue
        merged = sp.with_polls(o, sp.poll_events(polls, party, first[party]))
        for i, j in _national_pairs(o, merged):
            jobs += [("国政のみ", party, o, base[party], i, False, local),
                     ("国政+地方", party, o, base[party], i, True, local),
                     ("国政+世論調査", party, merged, poll_params[party], j, False, local),
                     ("v4", party, merged, poll_params[party], j, True, local)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_interp_job, jobs, chunksize=2))
    return {"summary": summarize(rows),
            "paired": [paired(rows, "国政+地方", "国政のみ"), paired(rows, "v4", "国政+世論調査"),
                       paired(rows, "国政+世論調査", "国政+地方")],
            "rows": rows}


def _carry_job(job):
    model, party, obs, params, i_nat = job
    d = obs.events[i_nat].vote_date
    end = d + timedelta(days=CARRYOVER_DAYS)
    levels = sorted({lv for lv in obs.level if lv != ss.REFERENCE_LEVEL})
    preds = {}
    for tag, keep in (("with", np.ones(len(obs.t), bool)), ("without", np.arange(len(obs.t)) != i_nat)):
        sub = glm._subset(obs, keep)
        contrib: list = []
        glm.run_filter(sub, params, until=ss._t(end), state_levels=levels, contrib=contrib)
        preds[tag] = {(sub.events[k].vote_date, sub.events[k].name, sub.level[k]): (e, float(sub.y[k]), float(sub.n[k]))
                      for k, e, _, _ in contrib
                      if sub.level[k] in glm.COUNCIL_LEVELS and d < sub.events[k].vote_date <= end}
    ks = sorted(set(preds["with"]) & set(preds["without"]))
    if not ks:
        return None
    n = np.array([preds["with"][k][2] for k in ks])
    y = np.array([preds["with"][k][1] for k in ks]) / n
    ew = np.array([preds["with"][k][0] for k in ks])
    eo = np.array([preds["without"][k][0] for k in ks])
    return {"model": model, "party": party, "vote_date": d.isoformat(), "n_local": len(ks),
            "se_with": float(np.sum(n * (y - ew) ** 2) / n.sum()),
            "se_without": float(np.sum(n * (y - eo) ** 2) / n.sum())}


def carryover(obs, base, poll_params, polls, first, workers=None) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for party, o in obs.items():
        if party not in poll_params:
            continue
        merged = sp.with_polls(o, sp.poll_events(polls, party, first[party]))
        for i, j in _national_pairs(o, merged):
            jobs += [("v3", party, o, base[party], i), ("v4", party, merged, poll_params[party], j)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_carry_job, jobs) if r]
    summary = []
    for model in ("v3", "v4"):
        d = np.array([r["se_with"] - r["se_without"] for r in rows if r["model"] == model])
        if len(d) < 2:
            continue
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        summary.append({"model": model, "n": len(d), "se_diff": float(d.mean()), "se": se,
                        "t": float(d.mean() / se) if se > 0 else float("nan"),
                        "with_better": int((d < 0).sum())})
    return {"days": CARRYOVER_DAYS, "summary": summary, "rows": rows}


def run(workers: int | None = None, out_of_sample: bool = False, out_dir: Path | None = None) -> Path:
    """3つの検証を回して output/trend/poll/local_value[_oos]_YYYY-MM-DD.json に書く。out_of_sampleなら、
    予測の世論調査のパラメータに、選挙ごとにその前日までで推定したもの(state_space_poll check
    --out-of-sampleの途中結果)を使う。"""
    from .trend import OUTPUT_DIR

    obs, base, polls, first = sp.load_inputs()
    poll_params = sp.load_poll_params(base)
    online_params = poll_params
    if out_of_sample:
        rows = json.loads((OUTPUT_DIR / "poll" / "out_of_sample_params.json").read_text())
        online_params = {k: sp._params_from_row(base[k.split("|")[1]], v) for k, v in rows.items()}
    body = {"date": date.today().isoformat(), "out_of_sample": out_of_sample,
            "online": online(obs, base, online_params, polls, first, workers, out_of_sample),
            "interpolation": interpolation(obs, base, poll_params, polls, first, workers),
            "carryover": carryover(obs, base, poll_params, polls, first, workers)}
    out_dir = out_dir or OUTPUT_DIR / "poll"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"local_value{'_oos' if out_of_sample else ''}_{body['date']}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1))
    return path


def status_lines(path: Path) -> list[str]:
    """月次実行の報告に出す要約。"""
    body = json.loads(path.read_text())
    lines = [f"[参考] 地方選挙の役割の検証を回し直した({path.name})。二乗誤差の差の t(負なら前者が当たる):"]
    for sec in ("online", "interpolation"):
        for p in body[sec]["paired"]:
            s = p.get("se_diff")
            if s:
                lines.append(f"[参考]   {'予測' if sec == 'online' else '補間'} {p['a']} 対 {p['b']}: t={s['t']:+.1f}"
                             f"(前者が近い {s['a_better']}/{p['n']}件)")
    for s in body["carryover"]["summary"]:
        lines.append(f"[参考]   持ち越し {s['model']}: 国政選挙を入れると地方選挙の予測の二乗誤差の差 t={s['t']:+.1f}")
    return lines


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="党勢指数v4で、地方選挙が国政の党勢の推定に何を足すかの検証")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-of-sample", action="store_true",
                    help="予測の世論調査のパラメータに、選挙ごとにその前日までで推定したものを使う")
    args = ap.parse_args(argv)
    path = run(args.workers, args.out_of_sample)
    print("\n".join(status_lines(path)))
    print(path)


if __name__ == "__main__":
    main()
