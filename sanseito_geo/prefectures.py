"""都道府県の違い(市区町村の特徴をそろえた後の上乗せ)を、自民党の減り方・候補者・地方組織で説明する。"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import analysis as an
import elections as el

ROOT = Path(__file__).resolve().parents[1]
OUT = an.OUT
COVARS = ["elderly", "secondary", "unemp", "log_density", "manuf", "fshare22", "d_fshare", "share22", "log_pop22"]
ELECTION_DAY = "2025-07-20"


def prefecture_effects(t: pd.DataFrame, covars: list[str] = COVARS) -> pd.Series:
    """市区町村の特徴をそろえた重み付き回帰の、都道府県ごとの定数。有効投票で重みをつけた平均を0にそろえる。"""
    d = t.dropna(subset=covars + ["d_share"])
    fe = pd.get_dummies(d["pref"], prefix="p", dtype=float)
    m = sm.WLS(d["d_share"], pd.concat([d[covars], fe], axis=1), weights=d["valid25"]).fit()
    pe = m.params[[c for c in m.params.index if c.startswith("p_")]]
    pe.index = [int(c[2:]) for c in pe.index]
    w = d.groupby("pref")["valid25"].sum().reindex(pe.index)
    return pe - np.average(pe, weights=w)


def councillors(chains: dict, pref_starts: list[int], cutoff: str = ELECTION_DAY) -> pd.DataFrame:
    """各議会の cutoff 以前の最後の選挙の議席から、都道府県ごとの参政党の議席と総議席。

    pref_starts は都道府県(知事)の自治体IDを昇順に並べたもので、IDがその間にある議会をその都道府県に数える。
    """
    rows = []
    for jid, terms in chains.items():
        pre = [x for x in terms if x["vote_date"] <= cutoff]
        if not pre:
            continue
        last = max(pre, key=lambda x: x["vote_date"])
        pref = bisect.bisect_right(pref_starts, int(jid))
        rows.append({"pref": pref, "sansei": last["seats"].get("参政党", 0), "seats": sum(last["seats"].values())})
    return pd.DataFrame(rows).groupby("pref")[["sansei", "seats"]].sum()


def district_share(path: Path, party: str = "参政党") -> pd.Series:
    """選挙区の都道府県ごとの得票率(%)。「合計」の行は除いて分母を作る。"""
    d = el.pref_party_district(path)
    d = d[d["party"] != "合計"]
    return d[d["party"] == party].groupby("pref")["votes"].sum() / d.groupby("pref")["votes"].sum() * 100


def prefecture_table(t: pd.DataFrame, dist25: pd.Series, council: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("pref")
    p = pd.DataFrame({"pr25": g["votes25"].sum() / g["valid25"].sum() * 100,
                      "d_ldp": g.apply(lambda x: np.average(x["d_ldp"], weights=x["valid25"]), include_groups=False),
                      "pop": g["pop25"].sum()})
    p["cand_pull"] = dist25.reindex(p.index) - p["pr25"]
    p["council_per_m"] = council["sansei"].reindex(p.index).fillna(0) / p["pop"] * 1e6
    return p


def explain(p: pd.DataFrame, effects: pd.Series, xs: list[str]) -> tuple[pd.DataFrame, float]:
    """都道府県の上乗せを xs で回帰(頑健な標準誤差)。係数・p値と決定係数。"""
    m = sm.OLS(effects.reindex(p.index), sm.add_constant(p[xs])).fit(cov_type="HC1")
    r = pd.DataFrame({"coef": m.params[xs], "lo": m.conf_int().loc[xs, 0], "hi": m.conf_int().loc[xs, 1],
                      "p": m.pvalues[xs]})
    return r, float(m.rsquared)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from seiryoku import turnover as tv

    tv._ensure_governor_ids_loaded()
    starts = list(tv._GOVERNOR_JICHITAI_IDS.values())
    chains = json.load(open(ROOT / "data" / "gikai_term_chains.json"))["chains"]
    t = an.prepare(pd.read_csv(OUT / "units.csv", dtype={"codes": str}))
    pe = prefecture_effects(t)
    p = prefecture_table(t, district_share(el.RAW / "sangiin_2025_pref_party_district.xlsx"), councillors(chains, starts))
    p["effect"] = pe
    p.to_csv(OUT / "pref_table.csv")
    r, r2 = explain(p, pe, ["d_ldp", "cand_pull", "council_per_m"])
    r.to_csv(OUT / "pref_explain.csv")
    print(r.round(3).to_string(), "\nR2", round(r2, 2), "\n効果のばらつき(標準偏差)", round(pe.std(), 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
