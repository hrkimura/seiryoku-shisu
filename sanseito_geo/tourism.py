"""観光庁「宿泊旅行統計調査」(年の確定値)の第2表から、都道府県別の延べ宿泊者数と外国人延べ宿泊者数。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw" / "kanko"
# 第2表(年計)の列: 1 が延べ宿泊者数、16 が外国人延べ宿泊者数
COL_TOTAL, COL_FOREIGN = 1, 16


def pref_nights(path: Path) -> pd.DataFrame:
    return parse_table(pd.read_excel(path, sheet_name="第2表(年計)", header=None))


def parse_table(d: pd.DataFrame) -> pd.DataFrame:
    """第2表の「01北海道」のような行から、都道府県番号・延べ宿泊者数・外国人延べ宿泊者数。"""
    rows = []
    for i in range(len(d)):
        m = re.match(r"^(\d{2})\D", unicodedata.normalize("NFKC", str(d.iat[i, 0])).strip())
        if m and 1 <= int(m.group(1)) <= 47:
            rows.append({"pref": int(m.group(1)), "nights": float(d.iat[i, COL_TOTAL]),
                         "foreign_nights": float(d.iat[i, COL_FOREIGN])})
    return pd.DataFrame(rows).drop_duplicates("pref")
