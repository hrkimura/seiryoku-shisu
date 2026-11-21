"""分析単位(市区町村、政令市は区)ごとに、参政党の得票率・外国人の割合・シフトシェアの予測値をまとめる。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import elections as el
import foreign as fo
import units as un

RAW = un.RAW
OUT = Path(__file__).resolve().parent / "output"


def designated(path: Path) -> set[str]:
    d = pd.read_excel(path, header=None, skiprows=7, dtype={0: str}).iloc[:, [0, 2]]
    d.columns = ["code", "name"]
    d = d[d["name"].notna()]
    d["name"] = d["name"].astype(str).map(un.norm)
    return un.designated_cities(d)


def juki_pair(year: str) -> pd.DataFrame:
    """その年の1月1日の、総人口と外国人住民の数。"""
    tot = un.load_juki(RAW / f"juki/{year}_total.xlsx").rename(columns={"pop": f"pop{year[2:]}"})
    fr = un.load_juki(RAW / f"juki/{year}_foreign.xlsx").rename(columns={"pop": f"for{year[2:]}"})
    return tot[["pref", "name", f"pop{year[2:]}"]].merge(fr[["pref", "name", f"for{year[2:]}"]],
                                                       on=["pref", "name"], how="left").fillna(0.0)


def shift_share(base: pd.DataFrame, n22: pd.Series, n25: pd.Series) -> pd.Series:
    """国籍ごとの全国の増加数を、2015年に各地域が抱えていた国籍別の人数の割合で配分した、予測増加数。"""
    groups = fo.GROUPS + [fo.OTHER]
    national_base = base[groups].sum()
    alloc = base[groups] / national_base
    return (alloc * (n25 - n22)[groups]).sum(axis=1)


def covariates(path: Path, cities: set[str]) -> pd.DataFrame:
    """SSDSE-A(市区町村の統計)から地域の特徴。政令市は区の値がないので、区には市の値を当てる。"""
    d = pd.read_csv(path, encoding="cp932", header=None, skiprows=3, dtype={0: str}).copy()
    d["pref"] = d[0].str[1:3].astype(int)
    d["name"] = d[2].astype(str).map(un.norm)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame({"pref": d["pref"], "name": d["name"], "cov_pop": num(3)})
    out["elderly"] = num(15) / num(3) * 100
    out["manuf"] = num(62) / num(57) * 100
    out["secondary"] = num(112) / num(102) * 100
    out["unemp"] = num(105) / (num(102) + num(105)) * 100
    dens = num(3) / num(37)
    out["log_density"] = np.log(dens.where(dens > 0))
    return out


COVARS = ["elderly", "manuf", "secondary", "unemp", "log_density"]


def attach_covariates(units: pd.DataFrame, cov: pd.DataFrame, cities: set[str]) -> pd.DataFrame:
    """単位ごとに地域の特徴を付ける。「A+B」の組(3つだけ)は構成する市区町村の単純平均。"""
    lookup = {(p, n): r for p, n, r in zip(cov["pref"], cov["name"], cov[COVARS].to_dict("records"))}
    rows = []
    for p, unit in zip(units["pref"], units["unit"]):
        recs = []
        for m in unit.split("+"):
            key = (p, un.city_of(m, cities) or m)
            if key in lookup:
                recs.append(lookup[key])
        rows.append({k: np.mean([r[k] for r in recs]) if recs else np.nan for k in COVARS})
    return pd.concat([units.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build() -> pd.DataFrame:
    cities = designated(RAW / "juki/2025_total.xlsx")
    s22 = el.shares(el.load_year("2022"))
    s25 = el.shares(el.fill_blanks(el.load_year("2025"), el.pref_party_totals(RAW / "sangiin_2025_pref_party.xlsx")))
    u25 = un.to_units(s25)
    groups = un.composite_groups(u25)
    u25 = un.regroup(u25, groups, ["valid", "votes"])
    u22 = un.regroup(un.to_units(s22), groups, ["valid", "votes"])
    l22 = el.load_year("2022")
    l25 = el.fill_blanks(el.load_year("2025"), el.pref_party_totals(RAW / "sangiin_2025_pref_party.xlsx"))
    r22 = un.regroup(un.to_units(el.shares(l22, "れいわ新選組")), groups, ["valid", "votes"])
    ldp22 = un.regroup(un.to_units(el.shares(l22, "自由民主党")), groups, ["valid", "votes"])
    ldp25 = un.regroup(un.to_units(el.shares(l25, "自由民主党")), groups, ["valid", "votes"])

    juki = juki_pair("2022").merge(juki_pair("2025"), on=["pref", "name"], how="outer")
    codes = un.load_juki(RAW / "juki/2025_total.xlsx")[["pref", "name", "code"]]
    base = fo.base_2015(RAW / "zairyu/2015_06.xlsx", un.load_juki(RAW / "juki/2025_total.xlsx"))
    base["pred"] = shift_share(base, fo.national_2022(RAW / "zairyu/2022_06_national.xlsx"),
                               fo.national_2025(RAW / "zairyu/2025_06_national.xlsx"))
    base = base.rename(columns={"総数": "for15"})[["pref", "name", "for15", "pred"]]
    m = juki.merge(base, on=["pref", "name"], how="left").rename(columns={"name": "unit"})
    m = un.regroup(m, groups, ["pop22", "for22", "pop25", "for25", "for15", "pred"])

    t = u22.rename(columns={"valid": "valid22", "votes": "votes22"}).merge(
        u25.rename(columns={"valid": "valid25", "votes": "votes25"}), on=["pref", "unit"]).merge(
        m, on=["pref", "unit"], how="left")
    t = t.merge(r22.rename(columns={"votes": "reiwa_votes22"})[["pref", "unit", "reiwa_votes22"]], on=["pref", "unit"])
    t["reiwa22"] = t["reiwa_votes22"] / t["valid22"] * 100
    for name, u in (("ldp22", ldp22), ("ldp25", ldp25)):
        t = t.merge(u.assign(**{name: u["votes"] / u["valid"] * 100})[["pref", "unit", name]], on=["pref", "unit"])
    t["d_ldp"] = t["ldp25"] - t["ldp22"]
    t["share22"] = t["votes22"] / t["valid22"] * 100
    t["share25"] = t["votes25"] / t["valid25"] * 100
    t["d_share"] = t["share25"] - t["share22"]
    t["fshare22"] = t["for22"] / t["pop22"] * 100
    t["fshare25"] = t["for25"] / t["pop25"] * 100
    t["d_fshare"] = t["fshare25"] - t["fshare22"]
    t["pred_fshare"] = t["pred"] / t["pop22"] * 100
    code_of = {(p, n): c for p, n, c in zip(codes["pref"], codes["name"], codes["code"])}
    t["codes"] = ["+".join(code_of[(p, m)][:5] for m in u.split("+")) for p, u in zip(t["pref"], t["unit"])]
    return attach_covariates(t, covariates(RAW / "SSDSE-A-2026.csv", cities), cities)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t = build()
    t.to_csv(OUT / "units.csv", index=False)
    print(len(t), t[["share22", "share25", "d_share", "fshare22", "d_fshare", "pred_fshare"]].describe().round(2))


if __name__ == "__main__":
    main()
