"""補欠選挙を党勢指数v2の観測にする。

国政の補選は総選挙と通常選挙の間に国政レベルの党勢を直接測る数少ない機会で、議会の補選は
通常選挙の間を埋める。どちらも改選数が1〜2の事実上の小選挙区で、議席と得票の関係が通常選挙と
違うので、通常選挙とは別の階層として扱う。

1議席の選挙の議席は得票をほぼ1ビットに粗くしたものなので、国政の補選は得票率でも観測にできる
(NATIONAL_VOTE_BY)。票は届出政党にだけ付け、候補を立てなかった党には観測を置かない。
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from .trend import Event

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NATIONAL_PATH = DATA_DIR / "national_by_elections.json"
GIKAI_PATH = DATA_DIR / "gikai_by_elections.json"

SHUGIIN_BY = "衆院補選"
SANGIIN_BY = "参院補選"
PREF_COUNCIL_BY = "都道府県議会補選"
MUNI_COUNCIL_BY = "市区町村議会補選"
NATIONAL_VOTE_BY = "国政補選得票"
NATIONAL_BY_LEVELS = (SHUGIIN_BY, SANGIIN_BY)
COUNCIL_BY_LEVELS = (PREF_COUNCIL_BY, MUNI_COUNCIL_BY)
BY_LEVELS = NATIONAL_BY_LEVELS + COUNCIL_BY_LEVELS + (NATIONAL_VOTE_BY,)


def _completed_by_elections(history: list, min_year: int) -> list:
    """未定・無投票を除いた完了済みの補欠選挙。無投票を除くのは通常選挙と同じ扱い
    (municipal_registry._all_completed_gikai)。"""
    return [
        row for row in history
        if "補欠" in row.name
        and row.vote_date != "未定"
        and row.vote_date[:4].isdigit() and int(row.vote_date[:4]) >= min_year
        and row.turnout not in ("-%", "")
        and row.detail_url
    ]


def load_gikai_cache() -> dict:
    if GIKAI_PATH.exists():
        return json.loads(GIKAI_PATH.read_text(encoding="utf-8"))
    return {"elections": {}, "done_ids": [], "errors": {}}


def save_gikai_cache(state: dict) -> None:
    GIKAI_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def build_gikai_cache(jichitai_ids: list[int] | None = None, min_year: int = 2018,
                      checkpoint_every: int = 20, refresh: bool = False) -> dict:
    """議会の補欠選挙の当選者の党派別議席を集め、data/gikai_by_elections.jsonへ
    チェックポイント保存する(中断・再開可能)。選挙履歴は通常選挙の取得でキャッシュ済み。
    refreshなら取得済みの自治体も読み直し、新しい補選を加える(月次実行用)。選挙履歴のページは
    毎月5日のrefresh_cliがキャッシュを取り直しているので、通信は新しい補選の詳細ページだけになる。"""
    from . import municipal_registry
    from .fetch import go2senkyo

    ids = jichitai_ids if jichitai_ids is not None else list(map(int, municipal_registry.load_gikai_term_chain_cache()["chains"]))
    state = load_gikai_cache()
    done = set() if refresh else set(state["done_ids"])
    for i, jid in enumerate(ids):
        if jid in done:
            continue
        try:
            rows = []
            for row in _completed_by_elections(go2senkyo.jurisdiction_history(jid, "gikai"), min_year):
                seats = municipal_registry._tally_seats(go2senkyo.parse_candidates(row.detail_url))
                if seats:
                    rows.append({"vote_date": row.vote_date.replace("/", "-"), "name": row.name, "seats": seats})
            if rows:
                state["elections"][str(jid)] = rows
            else:
                state["elections"].pop(str(jid), None)
            if jid not in state["done_ids"]:
                state["done_ids"].append(jid)
            state["errors"].pop(str(jid), None)
            done.add(jid)
        except Exception as e:
            # 通信の失敗はdone_idsに入れず、次の実行で取り直す
            state["errors"][str(jid)] = str(e)
        if (i + 1) % checkpoint_every == 0:
            save_gikai_cache(state)
            print(f"checkpoint: {i + 1}/{len(ids)}", flush=True)
    save_gikai_cache(state)
    return state


def _allocation(r: dict, c: dict, national_parties: set[str], endorsed: bool) -> dict[str, float]:
    """候補cの票(または議席)の党ごとの割合。届出が無所属の候補は、endorsedなら支援した党で等分する
    (national_endorsementsの規則)。"""
    from .index import _restrict_to_national
    from .lineage import canonicalize

    party = _restrict_to_national(canonicalize(c["party"]), national_parties)
    if party == "無所属" and endorsed:
        from .national_endorsements import resolve

        bs = resolve(r["vote_date"], r["district"], c["name"], c.get("rec", ""))
        if bs:
            # 国政政党でない推薦元の取り分は無所属に残す(index._effective_party_sharesと同じ)
            out: dict[str, float] = {}
            for b in bs:
                q = _restrict_to_national(canonicalize(b), national_parties)
                out[q] = out.get(q, 0.0) + 1.0 / len(bs)
            return out
    return {party: 1.0}


def national_events(national_parties: set[str], endorsed: bool = False) -> list[Event]:
    data = json.loads(NATIONAL_PATH.read_text(encoding="utf-8"))
    events = []
    for r in data["elections"]:
        winner = next(c for c in r["candidates"] if c["name"] == r["winner"])
        events.append(Event(date.fromisoformat(r["vote_date"]), SHUGIIN_BY if r["house"] == "衆議院" else SANGIIN_BY,
                            0.0, 1.0, _allocation(r, winner, national_parties, endorsed), r["district"]))
    return events


def national_vote_events(national_parties: set[str], endorsed: bool = False) -> list[Event]:
    """国政の補選の得票率。sharesは有効投票に対する届出政党ごとの得票率で、無所属と国政政党以外の
    候補の票は分母にだけ入る。衆参の補選は件数が少ないので1つの階層にまとめる。"""
    data = json.loads(NATIONAL_PATH.read_text(encoding="utf-8"))
    events = []
    for r in data["elections"]:
        total = sum(c["votes"] for c in r["candidates"])
        votes: dict[str, float] = {}
        for c in r["candidates"]:
            for p, w in _allocation(r, c, national_parties, endorsed).items():
                votes[p] = votes.get(p, 0.0) + w * c["votes"]
        shares = {p: v / total for p, v in votes.items() if p != "無所属"}
        events.append(Event(date.fromisoformat(r["vote_date"]), NATIONAL_VOTE_BY, 0.0, 1.0, shares, r["district"]))
    return events


def endorse_general_elections(events: list[Event], national_parties: set[str]) -> list[Event]:
    """衆院選・参院選のイベントの無所属の議席を、支援した党に等分し直す。合流した名義(中道改革連合など)の
    支援は、イベントの議席と同じく元の党で等分する。"""
    from .index import _restrict_to_national
    from .lineage import ALLIANCE_MEMBERS, BRANCH_OF, canonicalize
    from .national_endorsements import apply, general_seat_deltas

    out = []
    for e in events:
        if e.level not in ("衆院選", "参院選"):
            out.append(e)
            continue
        delta: dict[str, float] = {}
        for p, d in general_seat_deltas(e.vote_date.isoformat()).items():
            q = _restrict_to_national(canonicalize(p), national_parties) if p != "無所属" else p
            members = ALLIANCE_MEMBERS.get(q, (q,))
            for m in members:
                m = BRANCH_OF.get(m, m)
                delta[m] = delta.get(m, 0.0) + d / len(members)
        seats = apply({p: v * e.n for p, v in e.shares.items()}, delta)
        out.append(Event(e.vote_date, e.level, e.P, e.n, {p: v / e.n for p, v in seats.items()}, e.name))
    return out


def gikai_events(national_parties: set[str]) -> list[Event]:
    """議会の補選。nameは通常選挙と同じ議会名にする(自治体の地盤u_mを共有させる場合に使える)。"""
    from . import municipal_registry, turnover
    from .fetch import population
    from .index import _phi

    turnover._ensure_governor_ids_loaded()
    gov_id_to_name = {jid: name for name, jid in turnover._GOVERNOR_JICHITAI_IDS.items()}
    pref_pop = population.prefecture_population_by_name()
    muni_pop = population.municipal_population_by_name()
    events = []
    for jid_str, rows in load_gikai_cache()["elections"].items():
        jid = int(jid_str)
        if jid in gov_id_to_name:
            name, level = gov_id_to_name[jid], PREF_COUNCIL_BY
            size = pref_pop.get(name)
        else:
            try:
                name = municipal_registry.jurisdiction_name(jid)
            except Exception:
                # 首長の選挙履歴のページが無い(404)自治体は名前を引けず、人口も対応づけられない
                continue
            level = MUNI_COUNCIL_BY
            size = muni_pop.get(name) if name else None
        if size is None:
            continue
        for r in rows:
            n = sum(r["seats"].values())
            events.append(Event(date.fromisoformat(r["vote_date"]), level, float(size), float(n),
                                _phi(r["seats"], national_parties), name))
    return events


def load_events_with_by_elections(endorse_by: bool = False, endorse_general: bool = False
                                  ) -> tuple[list[Event], date | None]:
    """trend.load_eventsのイベントに補欠選挙を加える。月の境界はload_eventsと同じく、
    直近の完全に終わった月までにそろえる。endorse_by・endorse_generalなら、それぞれ国政の補選・
    通常選挙の無所属の候補を支援した党に等分する。"""
    from .monthly_index import _current_national_parties
    from .trend import load_events, merge_branches, split_alliances

    events, cutoff = load_events()
    parties = _current_national_parties()
    last = max(e.vote_date for e in events)
    last_month_end = date(last.year + (last.month == 12), last.month % 12 + 1, 1)
    if endorse_general:
        events = endorse_general_elections(events, parties)
    extra = [e for e in national_events(parties, endorse_by) + national_vote_events(parties, endorse_by)
             + gikai_events(parties) if e.vote_date < last_month_end]
    return sorted(events + merge_branches(split_alliances(extra)), key=lambda e: e.vote_date), cutoff


# 図で動きを階層ごとに分けるときに、補選を合算する通常選挙の階層
PARENT_LEVEL = {SHUGIIN_BY: "衆院選", SANGIIN_BY: "参院選", PREF_COUNCIL_BY: "都道府県議会",
                MUNI_COUNCIL_BY: "市区町村議会"}

BACKTEST_DIR = Path(__file__).resolve().parents[2] / "output" / "by_elections"
VARIANTS = {
    "なし": frozenset(BY_LEVELS),
    "国政補選": frozenset(BY_LEVELS) - set(NATIONAL_BY_LEVELS),
    "国政補選得票": frozenset(BY_LEVELS) - {NATIONAL_VOTE_BY},
    "議会補選": frozenset(BY_LEVELS) - set(COUNCIL_BY_LEVELS),
    "両方": frozenset({NATIONAL_VOTE_BY}),
}


def paired_difference(a: dict[str, list[dict]], b: dict[str, list[dict]]) -> dict:
    """国政選挙の補間の二乗誤差の差(b-a)の、件ごとの平均と標準誤差。同じ選挙の件どうしは
    相関するので、標準誤差は選挙ごとに差を合計してから求める(trend_model.texの補間の節)。"""
    import numpy as np

    sq = lambda rs: {(p, r["vote_date"]): (r["pred"] - r["y"]) ** 2 for p, rows in rs.items() for r in rows}
    sa, sb = sq(a), sq(b)
    keys = sorted(set(sa) & set(sb))
    by_date: dict[str, float] = {}
    count: dict[str, int] = {}
    for k in keys:
        by_date[k[1]] = by_date.get(k[1], 0.0) + sb[k] - sa[k]
        count[k[1]] = count.get(k[1], 0) + 1
    n, G = len(keys), len(by_date)
    mean = sum(by_date.values()) / n
    resid = np.array([by_date[d] - count[d] * mean for d in by_date])
    se = math.sqrt(G / (G - 1) * float(np.sum(resid ** 2))) / n
    return {"n": n, "elections": G, "mean": mean, "se": se,
            "rmse_a": float(np.sqrt(np.mean([sa[k] for k in keys]))),
            "rmse_b": float(np.sqrt(np.mean([sb[k] for k in keys]))),
            "by_election": {d: v for d, v in sorted(by_date.items())}}


def backtest(variants: list[str] | None = None, workers: int | None = None,
             endorse_by: bool = False, endorse_general: bool = False, online: bool = False) -> dict:
    """補選を使うかどうかの形ごとに国政選挙の補間を行い、結果を保存する。保存済みの形は計算し直さない
    (1つの形で8コア30分ほどかかる)。endorse_generalなら、通常選挙の観測と議席と得票の曲線の両方で
    無所属の当選者を支援した党に等分する。このとき補間で当てる値も変わるので、二乗誤差は推薦を
    使わない形とそのままは比べられない。onlineなら補間ではなく、投票日より前のデータだけでの予測で比べる。"""
    from . import state_space as ss
    from . import state_space_glm as glm
    from .trend import first_appearance

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    events, _ = load_events_with_by_elections(endorse_by, endorse_general)
    first = first_appearance([e for e in events if e.level not in BY_LEVELS])
    parties = ss.tracked_parties([e for e in events if e.level not in BY_LEVELS])
    obs = {p: glm.glm_observations(events, p, first[p], filter_levels=False) for p in parties}
    shu = ss.load_seat_vote_points(endorsed=endorse_general)
    san = ss.load_seat_vote_points(house="sangiin", endorsed=endorse_general)
    results = {}
    for name in variants or list(VARIANTS):
        suffix = ("_推薦全部" if endorse_general else ("_推薦補選" if endorse_by else "")) + ("_オンライン" if online else "")
        path = BACKTEST_DIR / f"backtest_{name}{suffix}.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))["parties"]
            continue
        print("補間:", name, flush=True)
        check = glm.national_online_check if online else glm.national_interpolation_check
        checks = check(obs, shu, san, workers, drop_levels=VARIANTS[name])
        path.write_text(json.dumps({"drop_levels": sorted(VARIANTS[name]), "summary": glm.interpolation_summary(checks),
                                    "parties": checks}, ensure_ascii=False, indent=1), encoding="utf-8")
        results[name] = checks
    return results


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="補欠選挙の取得と、補選を使うかどうかの補間の比較")
    ap.add_argument("command", choices=["crawl", "backtest"])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    if args.command == "crawl":
        build_gikai_cache()
        return
    results = backtest(workers=args.workers)
    from . import state_space_glm as glm
    for name, r in results.items():
        s = glm.interpolation_summary(r)
        print(f"{name}: n={s['n']} rmse={s['rmse']:.4f} median_abs={s['median_abs']:.4f} coverage={s['coverage']:.2f}")
    for name in results:
        if name == "なし":
            continue
        d = paired_difference(results["なし"], results[name])
        print(f"{name} - なし: 二乗誤差の差の平均 {d['mean']:+.5f} (標準誤差 {d['se']:.5f}, {d['n']}件, {d['elections']}選挙)")


if __name__ == "__main__":
    main()
