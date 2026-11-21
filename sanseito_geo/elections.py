"""参院選比例の開票区別・政党別得票数(総務省のExcel)を、開票区×政党の長い表に読み込む。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

# 2025年の総務省の表には、名前だけあって数字が空欄の開票区がある。その印に使う政党名
BLANK = "(空欄)"
RAW = Path(__file__).resolve().parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "output"


def to_number(v) -> float:
    """'21,066.619' のような文字列も数値にする。空欄は NaN。"""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return float("nan")
    s = v.replace(",", "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def norm_party(s: str) -> str:
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", s))


def parse_sheet(d: pd.DataFrame) -> pd.DataFrame:
    """1都道府県分のシートから、開票区名・政党・得票総数の長い表を返す。

    政党名の行の、政党名と同じ列に「得票総数」(按分の小数を含む)がある、という両年に共通の形を使う。
    """
    head = d.iloc[:15]
    party_row = next(i for i in range(len(head))
                     if any(isinstance(v, str) and "参政党" in norm_party(v) for v in head.iloc[i]))
    parties = {j: norm_party(v) for j, v in d.iloc[party_row].items() if isinstance(v, str) and v.strip()}
    label_row = next(i for i in range(len(head)) if head.iloc[i].astype(str).str.contains("開票区名").any())
    rows = []
    for i in range(max(party_row, label_row) + 1, len(d)):
        c0, c1 = d.iat[i, 0], d.iat[i, 1]
        name = c0 if isinstance(c0, str) and c0.strip() else c1 if isinstance(c1, str) and c1.strip() else None
        if name is None:
            continue
        name = re.sub(r"\s", "", unicodedata.normalize("NFKC", name))
        # 2025年の表には郡・振興局ごとの小計の行が混ざっている
        if name.endswith(("計", "郡", "振興局", "支庁", "事務所")) or name in ("得票総数",):
            continue
        got = False
        for j, p in parties.items():
            v = to_number(d.iat[i, j])
            if v == v:
                rows.append({"area": name, "party": p, "votes": v})
                got = True
        if not got:
            rows.append({"area": name, "party": BLANK, "votes": float("nan")})
    return pd.DataFrame(rows)


def drop_city_totals(t: pd.DataFrame) -> pd.DataFrame:
    """政令市の区や「薩摩川内市第1」のように開票区が分かれた市は、市の合計行もあるので落とす。"""
    areas = set(t["area"])
    cities = {a for a in areas
              if any(b != a and b.startswith(a) and (b.endswith("区") or re.search(r"第\d+$", b)) for b in areas)}
    return t[~t["area"].isin(cities)]


def load_year(year: str) -> pd.DataFrame:
    frames = []
    for f in sorted((RAW / f"sangiin_{year}").glob("*.xlsx")):
        t = drop_city_totals(parse_sheet(pd.read_excel(f, header=None)))
        t.insert(0, "pref", int(f.stem))
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def shares(long: pd.DataFrame, party: str = "参政党") -> pd.DataFrame:
    """開票区ごとの有効投票(全政党の得票の合計)と、指定した党の得票率(%)。"""
    total = long.groupby(["pref", "area"])["votes"].sum().rename("valid")
    p = long[long["party"] == party].groupby(["pref", "area"])["votes"].sum().rename("votes")
    out = pd.concat([total, p], axis=1).fillna({"votes": 0.0}).reset_index()
    out["share"] = out["votes"] / out["valid"] * 100
    return out


PREF_NAMES = ["北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県",
              "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
              "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
              "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県",
              "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]


def pref_party_totals(path: Path) -> pd.DataFrame:
    """総務省「都道府県別党派別得票数(比例代表)」から、都道府県×政党の得票総数。

    政党が4つずつ横に並ぶ表が縦に積まれていて、各政党の「得票総数」は政党名と同じ列にある。
    """
    d = pd.read_excel(path, header=None)
    rows, parties = [], {}
    for i in range(len(d)):
        first = d.iat[i, 0]
        names = {j: norm_party(v) for j, v in d.iloc[i, 1:].items() if isinstance(v, str) and v.strip()
                 and "得票" not in v and "政党等" not in v and "名簿" not in v and "除く" not in v}
        if not isinstance(first, str) and names:
            parties = names
            continue
        if isinstance(first, str) and re.sub(r"\s", "", first) in PREF_NAMES:
            pref = PREF_NAMES.index(re.sub(r"\s", "", first)) + 1
            for j, p in parties.items():
                rows.append({"pref": pref, "party": p, "votes": to_number(d.iat[i, j])})
    return pd.DataFrame(rows)


def fill_blanks(long: pd.DataFrame, pref_totals: pd.DataFrame) -> pd.DataFrame:
    """数字が空欄の開票区を、都道府県の合計から他の開票区を引いた値で埋める。

    空欄が同じ都道府県に複数あるときは個別には分けられないので、「A+B」という1つの単位にまとめる。
    """
    blanks = long[long["party"] == BLANK]
    filled = long[long["party"] != BLANK]
    extra = []
    for pref, g in blanks.groupby("pref"):
        name = "+".join(sorted(g["area"]))
        have = filled[filled["pref"] == pref].groupby("party")["votes"].sum()
        tot = pref_totals[(pref_totals["pref"] == pref) & (pref_totals["party"] != "合計")].set_index("party")["votes"]
        for party, v in (tot - have.reindex(tot.index, fill_value=0.0)).items():
            extra.append({"pref": pref, "area": name, "party": party, "votes": v})
    return pd.concat([filled, pd.DataFrame(extra)], ignore_index=True)
