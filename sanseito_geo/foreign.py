"""在留外国人統計(国籍別)から、シフトシェア操作変数の材料を作る。

- 2015年6月末の市区町村別・国籍別の人数(各地域にもともと住んでいた外国人の国籍構成)
- 2022年6月末と2025年6月末の、全国の国籍別の人数(国籍ごとの全国での増え方)
国籍の区分は、2015年の市区町村別の表にある区分に合わせる。
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

import units as un

# 2015年以降に市制施行・改称した自治体の旧名
RENAMED_SINCE_2015 = {"富谷町": "富谷市", "篠山市": "丹波篠山市", "那珂川町": "那珂川市"}
GROUPS = ["中国", "台湾", "韓国・朝鮮", "フィリピン", "ブラジル", "ベトナム", "ペルー", "米国"]
OTHER = "その他"
# 2022年・2025年の全国の表では韓国と朝鮮が別の行
NATIONAL_PARTS = {"韓国・朝鮮": ["韓国", "朝鮮"]}


def _n(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


def group_national(by_nat: pd.Series) -> pd.Series:
    """全国の国籍別の人数を、GROUPS と「その他」にまとめる。"""
    out = {}
    for g in GROUPS:
        out[g] = sum(by_nat.get(p, 0.0) for p in NATIONAL_PARTS.get(g, [g]))
    out[OTHER] = by_nat["総数"] - sum(out.values())
    return pd.Series(out)


def national_2022(path: Path) -> pd.Series:
    d = pd.read_excel(path, header=None, skiprows=6)
    s = pd.to_numeric(d[1], errors="coerce")
    s.index = d[0].map(_n)
    return group_national(s[~s.index.duplicated()].dropna())


def national_2025(path: Path) -> pd.Series:
    d = pd.read_excel(path, header=None, skiprows=4)
    t = d[d[3].map(_n) == "総数"]
    s = pd.to_numeric(t[4], errors="coerce")
    s.index = t[2].map(_n)
    return group_national(s[~s.index.duplicated()].dropna())


def base_2015(path: Path, juki: pd.DataFrame) -> pd.DataFrame:
    """2015年6月末の市区町村別・国籍別の表を、住民基本台帳の分析単位(pref, name)に対応させる。

    表は1列目に都道府県、2列目に市・郡・政令市、3列目に郡の町村か政令市の区を書く形。
    """
    d = pd.read_excel(path, header=None)
    header = d.iloc[3].map(lambda v: _n(v) if isinstance(v, str) else v)
    cols = {g: int(header[header == g].index[0]) for g in GROUPS + ["総数"]}
    names = {(p, n) for p, n in zip(juki["pref"], juki["name"])}
    pref, parent, rows = None, None, []
    for i in range(4, len(d)):
        c0, c1, c2 = d.iat[i, 0], d.iat[i, 1], d.iat[i, 2]
        if isinstance(c0, str) and _n(c0) in un.el.PREF_NAMES:
            pref = un.el.PREF_NAMES.index(_n(c0)) + 1
        if pref is None:
            continue
        if isinstance(c1, str):
            parent = un.norm(c1)
            cand = [parent]
        elif isinstance(c2, str):
            child = un.norm(c2)
            cand = [parent + child, child] if parent and parent.endswith("市") else [child]
        else:
            continue
        cand = [RENAMED_SINCE_2015.get(c, c) for c in cand]
        name = next((c for c in cand if (pref, c) in names), None)
        if name is None:
            # 「三宅島三宅村」のように前に島名などが付いた名前
            name = next((n for p, n in names if p == pref and cand[0].endswith(n) and len(n) >= 3), None)
        if name is None:
            continue
        rec = {"pref": pref, "name": name}
        for g, j in cols.items():
            rec[g] = pd.to_numeric(d.iat[i, j], errors="coerce")
        rows.append(rec)
    t = pd.DataFrame(rows).fillna(0.0)
    t = t.groupby(["pref", "name"], as_index=False).sum()
    t[OTHER] = t["総数"] - t[GROUPS].sum(axis=1)
    return t
