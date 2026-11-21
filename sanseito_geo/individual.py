"""東大谷口研・朝日2026年有権者調査の個票で、住む都道府県の外国人の割合・増え方と、
外国人への態度・参政党への投票の関係を見る(年代・性別を統制したロジット、都道府県で群をなす標準誤差)。

個票は再配布禁止なので、結果の係数だけを出力する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sanseito_profile"))

from profile_utas import DATA, age_group  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
LIKERT = [1, 2, 3, 4, 5]


def outcomes(r: pd.DataFrame) -> pd.DataFrame:
    """外国人労働者の受け入れ推進に反対(Q35_8が4・5)、在留外国人で文化が脅かされる(Q36_6が4・5)、
    2025年参院選比例で参政党(Q4=6、投票した人のみ)。"""
    o = pd.DataFrame(index=r.index)
    for name, col in (("受け入れ推進に反対", "Q35_8"), ("文化が脅かされる", "Q36_6")):
        v = r[col].where(r[col].isin(LIKERT))
        o[name] = v.isin([4, 5]).astype(float).where(v.notna())
    o["参政党に投票"] = (r["Q4"] == 6).astype(float).where(r["Q4"].isin(range(1, 13)))
    return o


def prefecture_exposure(units: pd.DataFrame) -> pd.DataFrame:
    g = units.groupby("pref")[["for22", "pop22", "for25", "pop25"]].sum()
    g["外国人の割合(2025年)"] = g["for25"] / g["pop25"] * 100
    g["外国人の割合の増加"] = g["外国人の割合(2025年)"] - g["for22"] / g["pop22"] * 100
    return g[["外国人の割合(2025年)", "外国人の割合の増加"]]


def fit_all(r: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    d = r[["Prefecture", "Q39", "Q40"]].copy()
    d["age_g"] = age_group(r["Q40"])
    d["male"] = r["Q39"].map({1: 1, 2: 0})
    d = d.join(outcomes(r)).merge(exposure, left_on="Prefecture", right_index=True, how="left")
    rows = []
    for y in ("受け入れ推進に反対", "文化が脅かされる", "参政党に投票"):
        for x in exposure.columns:
            e = d.rename(columns={y: "y", x: "x"}).dropna(subset=["y", "x", "age_g", "male"])
            m = smf.logit("y ~ x + C(age_g) + male", data=e).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": e["Prefecture"]})
            lo, hi = m.conf_int().loc["x"]
            rows.append({"outcome": y, "exposure": x, "log_odds": m.params["x"], "lo": lo, "hi": hi, "n": len(e)})
    return pd.DataFrame(rows)


def main() -> None:
    res = fit_all(pd.read_csv(DATA), prefecture_exposure(pd.read_csv(OUT / "units.csv")))
    res.to_csv(OUT / "individual_check.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
