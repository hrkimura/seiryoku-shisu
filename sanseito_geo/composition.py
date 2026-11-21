"""国籍別・在留資格別の外国人の増え方(2022年6月末→2025年6月末)を、分析単位ごとに計算する。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw" / "zairyu"

# 2022年6月末の市区町村別の表に列がある国籍。それ以外(朝鮮・ミャンマーなど)は「その他」
NATIONS = ["中国", "ベトナム", "韓国", "フィリピン", "ブラジル", "ネパール", "インドネシア", "米国", "タイ", "台湾"]
OTHER = "その他"
# 市全体で分析し、表に市全体の行がないときは区の行を足す市(浜松市)
WARD_CITIES = {"22130"}

STATUS_GROUPS = {
    "技能実習": ["技能実習"],
    "特定技能": ["特定技能"],
    "留学": ["留学"],
    "専門・技術": ["教授", "芸術", "宗教", "報道", "高度専門職", "経営・管理", "法律・会計業務", "医療", "研究",
               "教育", "技術・人文知識・国際業務", "企業内転勤", "介護", "興行", "技能"],
    "永住・定住・配偶者": ["永住者", "日本人の配偶者等", "永住者の配偶者等", "定住者"],
    "特別永住者": ["特別永住者"],
    "家族滞在": ["家族滞在"],
}


def _n(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


def label(s: str) -> str:
    """「01_023：中国」「16：技術・人文知識・国際業務」の番号を外す。"""
    return re.sub(r"^[\d_]+[:：]", "", _n(s))


def status_group(name: str) -> str:
    n = label(name)
    # 「永住者の配偶者等」を「永住者」より先に見るため、長い名前から当てる
    pairs = sorted(((p, g) for g, ps in STATUS_GROUPS.items() for p in ps), key=lambda x: -len(x[0]))
    for p, g in pairs:
        if n.startswith(p):
            return g
    return OTHER


def nation_group(name: str) -> str:
    n = label(name)
    return n if n in NATIONS else OTHER


def wide_2022(path: Path, mapper) -> pd.DataFrame:
    """2022年6月末の市区町村別の表(列が国籍か在留資格)を、市区町村コード×グループの表にする。"""
    d = pd.read_excel(path, header=None, dtype={0: str})
    header = d.iloc[1]
    body = d.iloc[3:]
    body = body[body[0].astype(str).str.fullmatch(r"\d{5}", na=False)]
    cols = {j: mapper(v) for j, v in header.items() if isinstance(v, str) and j >= 3}
    out = pd.DataFrame(index=body[0].to_numpy())
    for j, g in cols.items():
        v = pd.to_numeric(body[j], errors="coerce").fillna(0.0).to_numpy()
        out[g] = out[g] + v if g in out else v
    out.index.name = "code"
    return out


def long_2025(path: Path, key: str, mapper) -> pd.DataFrame:
    """2025年6月末の取り出した表(市区町村コード×国籍×在留資格)を、市区町村コード×グループの表にする。"""
    d = pd.read_csv(path, dtype={"市区町村コード": str})
    d["g"] = d[key].map(mapper)
    out = d.pivot_table(index="市区町村コード", columns="g", values="在留外国人数", aggfunc="sum", fill_value=0)
    out.index.name = "code"
    return out.astype(float)


def per_unit(table: pd.DataFrame, unit_codes: list[str]) -> pd.DataFrame:
    """単位(「01101+01102」のようなコードの組)ごとに足す。

    区を再編した浜松市(22130)は、2025年の表に市全体の行がないので、区の行を足す。
    それ以外のコードが表にない単位は欠損にする(2022年の在留資格別の表は外国人の少ない町村を省いている)。
    """
    rows = []
    for codes in unit_codes:
        parts, missing = [], False
        for c in codes.split("+"):
            if c in table.index:
                parts.append(table.loc[[c]])
            elif c in WARD_CITIES:
                parts.append(table[(table.index.str[:4] == c[:4]) & (table.index != c)])
            else:
                missing = True
        rows.append(pd.Series(float("nan"), index=table.columns) if missing else pd.concat(parts).sum())
    return pd.DataFrame(rows).reset_index(drop=True)


def changes(unit_codes: list[str], pop: pd.Series) -> pd.DataFrame:
    """国籍別・在留資格別の、人口(2022年1月)に対する増加(ポイント)。列名は「国籍:中国」「資格:技能実習」。"""
    n22 = per_unit(wide_2022(RAW / "2022_06.xlsx", nation_group), unit_codes)
    s22 = per_unit(wide_2022(RAW / "2022_06_status.xlsx", status_group), unit_codes)
    src = RAW / "2025_06_by_nat_status.csv"
    n25 = per_unit(long_2025(src, "国籍・地域", nation_group), unit_codes)
    s25 = per_unit(long_2025(src, "在留資格", status_group), unit_codes)
    out = {}
    for prefix, a, b in (("国籍", n22, n25), ("資格", s22, s25)):
        for g in sorted(set(a.columns) | set(b.columns)):
            inc = b.get(g, 0.0) - a.get(g, 0.0)
            out[f"{prefix}:{g}"] = (inc.to_numpy() if hasattr(inc, "to_numpy") else inc) / pop.to_numpy() * 100
    return pd.DataFrame(out)
