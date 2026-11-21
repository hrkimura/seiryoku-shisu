"""追加の確認: 国籍・在留資格別、観光、都市と農村、個人単位の調査。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import analysis as an
import tourism as tr

OUT = an.OUT


def joint_effects(t: pd.DataFrame, cols: list[str], controls: list[str]) -> pd.DataFrame:
    """複数の増加を同時に入れた重み付き最小二乗。各列が1ポイント増えたときの伸びと95%区間。"""
    d = an.complete(t, ["d_share", "valid25"] + cols + controls)
    x = np.column_stack([d[cols].to_numpy(float), an.design(d, controls)])
    b, v = an.wls(d["d_share"].to_numpy(float), x, d["valid25"].to_numpy(float), d["pref"].to_numpy())
    se = np.sqrt(np.diag(v))[: len(cols)]
    return pd.DataFrame({"group": cols, "coef": b[: len(cols)], "lo": b[: len(cols)] - 1.96 * se,
                         "hi": b[: len(cols)] + 1.96 * se,
                         "mean_increase": np.average(d[cols], weights=d["valid25"], axis=0), "n": len(d)})


def holm(p: np.ndarray) -> np.ndarray:
    """Holm法で補正したp値。"""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def with_holm(r: pd.DataFrame) -> pd.DataFrame:
    """joint_effects の結果に、正規近似のp値とHolm補正後のp値を足す。"""
    from scipy import stats

    se = (r["hi"] - r["lo"]) / (2 * 1.96)
    p = 2 * stats.norm.sf(np.abs(r["coef"] / se))
    return r.assign(p=p, p_holm=holm(p))


def prefecture_table(t: pd.DataFrame, nights_base: pd.DataFrame, nights_now: pd.DataFrame) -> pd.DataFrame:
    """都道府県ごとの参政党の伸びと、人口あたりの外国人延べ宿泊者数(2024年)とその2019年からの増加。"""
    s = t.groupby("pref")[["votes22", "valid22", "votes25", "valid25", "for22", "pop22", "for25", "pop25"]].sum()
    s["d_share"] = s["votes25"] / s["valid25"] * 100 - s["votes22"] / s["valid22"] * 100
    s["share22"] = s["votes22"] / s["valid22"] * 100
    s["d_fshare"] = s["for25"] / s["pop25"] * 100 - s["for22"] / s["pop22"] * 100
    now = nights_now.set_index("pref")["foreign_nights"]
    base = nights_base.set_index("pref")["foreign_nights"]
    s["fn_now"] = now / s["pop25"]
    s["d_fn"] = s["fn_now"] - base / s["pop22"]
    s["log_fn_now"] = np.log(s["fn_now"])
    return s.reset_index()


def prefecture_effect(p: pd.DataFrame, x: str, controls: list[str]) -> tuple[float, float, float]:
    """都道府県(47)の重み付き回帰。頑健な標準誤差(HC1)。"""
    m = sm.WLS(p["d_share"], sm.add_constant(p[[x] + controls]), weights=p["valid25"]).fit(cov_type="HC1")
    lo, hi = m.conf_int().loc[x]
    return float(m.params[x]), float(lo), float(hi)


def density_split(t: pd.DataFrame) -> pd.DataFrame:
    """人口密度で3つに分けた、自分の単位での増加の効果。"""
    q = t["log_density"].quantile([1 / 3, 2 / 3]).tolist()
    parts = {"農村(密度の下位3分の1)": t[t["log_density"] < q[0]],
             "中間": t[(t["log_density"] >= q[0]) & (t["log_density"] < q[1])],
             "都市(密度の上位3分の1)": t[t["log_density"] >= q[1]]}
    rows = []
    ctrl = an.CONTROLS + an.REGION
    for name, d in parts.items():
        for method, fit in (("最小二乗", an.ols(d, "d_share", "d_fshare", ctrl)),
                            ("操作変数", an.tsls(d, "d_share", "d_fshare", "pred_fshare", ctrl))):
            rows.append({"group": name, "method": method, "coef": fit.coef, "lo": fit.ci[0], "hi": fit.ci[1],
                         "n": fit.n, "first_stage_F": fit.first_stage_f})
    return pd.DataFrame(rows)


def reiwa_overlap(t: pd.DataFrame) -> pd.DataFrame:
    """2022年のれいわの得票率が1ポイント高い地域での、参政党の伸びの違い。地域の特徴をそろえる前と後。"""
    rows = []
    for name, ctrl in (("統制なし", []), ("地域の特徴をそろえる", ["share22", "log_pop22"] + an.REGION)):
        fit = an.ols(t, "d_share", "reiwa22", ctrl)
        rows.append({"spec": name, "coef": fit.coef, "lo": fit.ci[0], "hi": fit.ci[1], "n": fit.n})
    return pd.DataFrame(rows)


def level_effect(t: pd.DataFrame) -> pd.DataFrame:
    """2025年の外国人の割合が1ポイント高い地域での、2025年の参政党の得票率の違い(ある時点の多さどうしの比較)。"""
    rows = []
    for name, ctrl in (("都道府県の違いをそろえる", ["log_pop22"]), ("地域の特徴もそろえる", ["log_pop22"] + an.REGION)):
        fit = an.ols(t, "share25", "fshare25", ctrl)
        rows.append({"spec": name, "coef": fit.coef, "lo": fit.ci[0], "hi": fit.ci[1], "n": fit.n})
    return pd.DataFrame(rows)


def vote_distribution(t: pd.DataFrame, thresholds=(1, 3, 5)) -> pd.DataFrame:
    """外国人の割合(2025年)が閾値未満の地域が、参政党の票と人口に占める割合(%)。"""
    return pd.DataFrame([{"below": th,
                          "votes_pct": t.loc[t["fshare25"] < th, "votes25"].sum() / t["votes25"].sum() * 100,
                          "pop_pct": t.loc[t["fshare25"] < th, "pop25"].sum() / t["pop25"].sum() * 100}
                         for th in thresholds])


def main() -> None:
    t = an.prepare(pd.read_csv(OUT / "units.csv", dtype={"codes": str}))
    ch = pd.read_csv(OUT / "composition_changes.csv")
    t = pd.concat([t, ch], axis=1)
    nat = [c for c in ch if c.startswith("国籍:")]
    st = [c for c in ch if c.startswith("資格:")]
    comp = pd.concat([with_holm(joint_effects(t, nat, an.CONTROLS + an.REGION)),
                      with_holm(joint_effects(t, st, an.CONTROLS + an.REGION))])
    comp.to_csv(OUT / "composition_effects.csv", index=False)
    p = prefecture_table(t, tr.pref_nights(tr.RAW / "2019.xlsx"), tr.pref_nights(tr.RAW / "2024.xlsx"))
    rows = []
    for x, ctrl in (("log_fn_now", []), ("d_fn", []), ("log_fn_now", ["share22", "d_fshare"])):
        c, lo, hi = prefecture_effect(p, x, ctrl)
        rows.append({"x": x, "controls": "+".join(ctrl), "coef": c, "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv(OUT / "tourism_effects.csv", index=False)
    density_split(t).to_csv(OUT / "density_split.csv", index=False)
    reiwa_overlap(t).to_csv(OUT / "reiwa_overlap.csv", index=False)
    level_effect(t).to_csv(OUT / "level_effect.csv", index=False)
    vote_distribution(t).to_csv(OUT / "vote_distribution.csv", index=False)
    for f in ("composition_effects", "tourism_effects", "density_split", "reiwa_overlap", "level_effect",
              "vote_distribution"):
        print(pd.read_csv(OUT / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
