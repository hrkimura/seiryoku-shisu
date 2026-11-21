"""開票区名と住民基本台帳の市区町村を、共通の分析単位に揃える。

単位は市区町村で、政令市は区。ただし浜松市は2024年に区を再編したので、両年を比べられるよう市全体にまとめる。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

import elections as el

RAW = el.RAW
# 同じ自治体の表記ゆれ。住民基本台帳の表記に合わせる
VARIANTS = {"鯵ヶ沢町": "鰺ヶ沢町", "梼原町": "檮原町", "須惠町": "須恵町"}
MERGED_CITIES = {"浜松市"}


def norm(s: str) -> str:
    s = re.sub(r"\s", "", unicodedata.normalize("NFKC", s))
    return VARIANTS.get(s, s)


def strip_gun(name: str) -> str:
    """住民基本台帳の「余市郡余市町」を「余市町」にする。「蒲郡市」のような市名は変えない。"""
    m = re.match(r"^(.+?郡)(.+[町村])$", name)
    return m.group(2) if m else name


def load_juki(path: Path) -> pd.DataFrame:
    """住民基本台帳の市区町村別の表から、団体コード・都道府県・名前・人口(計)。

    郡の行と政令市の市全体の行は除く(区の行を使う)。浜松市だけは逆に区の行を除いて市全体の行を使う。
    """
    d = pd.read_excel(path, header=None, skiprows=7, dtype={0: str})
    d = d.iloc[:, [0, 2, 5]]
    d.columns = ["code", "name", "pop"]
    d = d[d["name"].notna() & (d["name"].astype(str) != "-") & d["code"].str.fullmatch(r"\d{6}", na=False)].copy()
    d["name"] = d["name"].astype(str).map(norm)
    d = d[~d["name"].str.endswith("郡")]
    d["name"] = d["name"].map(lambda n: norm(strip_gun(n)))
    d["pref"] = d["code"].str[:2].astype(int)
    cities = designated_cities(d)
    whole = d["name"].isin(cities - MERGED_CITIES)
    merged_wards = d["name"].map(lambda n: city_of(n, MERGED_CITIES) not in (None, n))
    d = d[~whole & ~merged_wards]
    return d[["pref", "code", "name", "pop"]].reset_index(drop=True)


def designated_cities(juki: pd.DataFrame) -> set[str]:
    """区を持つ市(政令市)の名前。"""
    names = set(juki["name"])
    return {n for n in names if n.endswith("市") and any(m != n and m.startswith(n) and m.endswith("区") for m in names)}


def city_of(area: str, cities: set[str]) -> str | None:
    for c in cities:
        if area.startswith(c):
            return c
    return None


# 2022年の宮城県の表は仙台市の区を市名なしで書いている
BARE_WARDS = {4: "仙台市"}


def area_to_muni(pref: int, area: str) -> str:
    """開票区名を分析単位の名前にする(「第1」などの分割と浜松市の区を市にまとめる)。"""
    a = norm(area)
    a = re.sub(r"第\d+$", "", a)
    if pref in BARE_WARDS and a.endswith("区") and not a.startswith(BARE_WARDS[pref]):
        a = BARE_WARDS[pref] + a
    return city_of(a, MERGED_CITIES) or a


def to_units(shares: pd.DataFrame) -> pd.DataFrame:
    """開票区の得票を市区町村ごとに足し合わせる。「A+B」の開票区は構成する市区町村の組として残す。"""
    rows = []
    for r in shares.itertuples():
        munis = sorted({area_to_muni(r.pref, a) for a in r.area.split("+")})
        rows.append({"pref": r.pref, "unit": "+".join(munis), "valid": r.valid, "votes": r.votes})
    t = pd.DataFrame(rows)
    return t.groupby(["pref", "unit"], as_index=False)[["valid", "votes"]].sum()


def regroup(t: pd.DataFrame, groups: dict[tuple[int, str], str], cols: list[str]) -> pd.DataFrame:
    """市区町村単位の表を、組(「A+B」)にまとめ直す。groups は (都道府県, 市区町村) → 組の名前。"""
    t = t.copy()
    t["unit"] = [groups.get((p, u), u) for p, u in zip(t["pref"], t["unit"])]
    return t.groupby(["pref", "unit"], as_index=False)[cols].sum()


def composite_groups(units: pd.DataFrame) -> dict[tuple[int, str], str]:
    out = {}
    for p, u in zip(units["pref"], units["unit"]):
        if "+" in u:
            for m in u.split("+"):
                out[(p, m)] = u
    return out
