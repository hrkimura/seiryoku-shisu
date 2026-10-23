"""国政選挙の無所属の候補を、推薦・支持・支援した党に等分する(党勢指数v2の比較用)。

分類の規則:
- 数える: 党の組織(本部・都道府県組織・支部)が推薦・支持・支援(自主支援・友情支援を含む)を表明したもの
- 数えない: 自主投票、「〜の一部」、議員個人の応援、当選後の追加公認
- 届出が無所属の候補(国政政党でない団体の候補を含む)にだけ使い、公認候補への推薦は使わない
- 数えた党で等分する。いまの国政政党でない推薦元の取り分は無所属に残す(首長選の
  index._effective_party_sharesと同じ)

選挙結果テンプレートの推薦の欄を自動で読み、欄が空で本文に記述がある候補は人手で拾った結果を
data/national_endorsements_manual.jsonに原文と出典ごと置いて、自動取得より優先する。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANUAL_PATH = DATA_DIR / "national_endorsements_manual.json"


def backers(text: str) -> list[str]:
    """推薦の欄の文言から、規則で数える党を出現順に返す。"""
    from .fetch.wikipedia_governor import extract_parties
    from .lineage import BRANCH_OF

    t = re.sub(r"[^・、,，\s]*の一部", "", text)
    t = re.sub(r"[^・、,，\s]*追加公認", "", t)
    out = []
    for p in extract_parties(t):
        p = BRANCH_OF.get(p, p)
        if p not in out:
            out.append(p)
    return out


def _manual() -> dict:
    return json.loads(MANUAL_PATH.read_text(encoding="utf-8")) if MANUAL_PATH.exists() else {}


def key(vote_date: str, district: str, name: str) -> str:
    return f"{vote_date}|{district}|{name}"


def resolve(vote_date: str, district: str, name: str, rec: str) -> list[str]:
    m = _manual().get(key(vote_date, district, name))
    return list(m["backers"]) if m is not None else backers(rec)


def general_seat_deltas(vote_date: str) -> dict[str, float]:
    """通常選挙の無所属の当選者を支援した党に等分したときの、党ごとの議席の増減。"""
    from .fetch.wikipedia_districts import CACHE_PATH

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8")).get(vote_date)
    if data is None:
        raise KeyError(f"{vote_date}の無所属の当選者を取得していない")
    delta: dict[str, float] = {}
    for r in data["independents"]:
        bs = resolve(vote_date, r["district"], r["name"], r["rec"])
        if not bs:
            continue
        delta["無所属"] = delta.get("無所属", 0.0) - 1.0
        for b in bs:
            delta[b] = delta.get(b, 0.0) + 1.0 / len(bs)
    return delta


def apply(seats: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    out = dict(seats)
    for p, d in delta.items():
        out[p] = out.get(p, 0.0) + d
    return {p: v for p, v in out.items() if v > 1e-9}
