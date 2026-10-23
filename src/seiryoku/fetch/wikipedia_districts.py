"""Wikipediaの選挙区記事から、国政選挙の無所属の候補とその推薦・支持・支援を取り出す。

総務省の集計は党派別の議席数だけで、候補者ごとの推薦を持たない。選挙区の記事には選挙ごとの
選挙結果テンプレートがあり、届出党派と推薦の欄が同じ書式で並ぶので、そこから機械的に取る。
推薦の欄が空で本文に支援の記述がある候補は、人手で拾った結果を別のファイルに置き、自動取得より
優先する。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .wikipedia_governor import _fetch_article, _strip_markup

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CACHE_PATH = DATA_DIR / "national_independents_wikipedia.json"

# (院, 選挙回, 投票日, 選挙の記事)
ELECTIONS = [
    ("衆議院", 48, "2017-10-22", "第48回衆議院議員総選挙"),
    ("衆議院", 49, "2021-10-31", "第49回衆議院議員総選挙"),
    ("衆議院", 50, "2024-10-27", "第50回衆議院議員総選挙"),
    ("衆議院", 51, "2026-02-08", "第51回衆議院議員総選挙"),
    ("参議院", 24, "2016-07-10", "第24回参議院議員通常選挙"),
    ("参議院", 25, "2019-07-21", "第25回参議院議員通常選挙"),
    ("参議院", 26, "2022-07-10", "第26回参議院議員通常選挙"),
    ("参議院", 27, "2025-07-20", "第27回参議院議員通常選挙"),
]

PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県",
    "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県",
    "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]
_FULL = {p if p == "北海道" else p[:-1]: p for p in PREFECTURES}
SANGIIN_DISTRICTS = [p for p in PREFECTURES if p not in ("鳥取県", "島根県", "徳島県", "高知県")] + [
    "鳥取県・島根県", "徳島県・高知県"]

_API_RAW = "https://ja.wikipedia.org/w/index.php?action=raw&title="


def shugiin_districts(election_wikitext: str) -> list[str]:
    """衆院選の記事の小選挙区当選者テンプレートから、選挙区の記事名を順に返す。
    第49回までの形式(|北海道1区=...)と、都道府県ごとに当選者を並べる形式の両方を読む。"""
    m = re.search(r"\{\{衆議院小選挙区当選者\(第49回まで\)(.*?)\n\}\}", election_wikitext, re.S)
    if m:
        keys = re.findall(r"\|([^|=\n]+?)(\d+)区=", m.group(1))
        return [f"{_FULL[p]}第{n}区" for p, n in keys]
    m = re.search(r"\{\{衆院小選挙区当選者\n(.*?)\n\}\}", election_wikitext, re.S)
    out, pref, i = [], None, 0
    for line in m.group(1).split("\n"):
        h = re.match(r"\|(.+?)=\s*$", line)
        if h:
            pref, i = _FULL[h.group(1)], 0
        elif re.match(r"[0-9A-Fa-f]{3,6}:", line.strip()):
            i += 1
            out.append(f"{pref}第{i}区")
    return out


def _templates(wikitext: str, kind: str) -> list[str]:
    out, start = [], 0
    while (i := wikitext.find("{{選挙結果 (" + kind + ")", start)) >= 0:
        depth, j = 0, i
        while j < len(wikitext):
            if wikitext.startswith("{{", j):
                depth, j = depth + 1, j + 2
            elif wikitext.startswith("}}", j):
                depth, j = depth - 1, j + 2
                if depth == 0:
                    break
            else:
                j += 1
        out.append(wikitext[i:j])
        start = j
    return out


def _clean(s: str) -> str:
    return re.sub(r"\{\{Cite.*$", "", _strip_markup(re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", s)), flags=re.S).strip()


def _number(s: str) -> float:
    s = re.sub(r"[^\d.]", "", s)
    return float(s) if s else 0.0


def parse_shugiin(wikitext: str, kai: int) -> list[dict]:
    """小選挙区の記事から、第kai回の候補(氏名・届出党派・得票・推薦の欄)を得票順に返す。"""
    for t in _templates(wikitext, "衆議院小選挙区"):
        if not re.search(r"\|\s*選挙回\s*=\s*%d\s*(\||\n)" % kai, t):
            continue
        rows = []
        for line in t.split("\n")[2:]:
            line = re.sub(r"\{\{nowrap\|([^}]*)\}\}", r"\1", line)
            line = re.sub(r"\[\[([^|\]]*)\|([^\]]*)\]\]", r"\2", line)
            line = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", line)
            p = line.split("|")
            if len(p) >= 7:
                rows.append({"name": _strip_markup(p[2]), "party": _strip_markup(p[4]),
                             "votes": _number(p[6]), "rec": _clean(p[7]) if len(p) > 7 else ""})
        return sorted(rows, key=lambda r: -r["votes"])
    return []


def parse_sangiin(wikitext: str, kai: int) -> tuple[int, list[dict]]:
    """選挙区の記事から、第kai回の通常選挙の(定数, 候補)を得票順に返す。補欠選挙の表は除く。"""
    for t in _templates(wikitext, "参議院選挙区"):
        if "補欠年" in t or not re.search(r"\|\s*選挙回\s*=\s*%d\s*(\||\n)" % kai, t):
            continue
        # [[記事|表示名]]の中の「|」の直後には「=」が来ないので、そこでは切れない
        field = lambda k: (m.group(1) if (m := re.search(r"\|\s*" + k + r"\s*=(.*?)(?=\|[^|=\n\[\]{}]*=|\n|\}\}$)", t)) else "")
        rows = []
        for n in range(1, 40):
            if not re.search(r"\|%d氏名=" % n, t):
                break
            rows.append({"name": _strip_markup(field(f"{n}氏名")), "party": _strip_markup(field(f"{n}所属")),
                         "votes": _number(field(f"{n}得票")), "rec": _clean(field(f"{n}推薦"))})
        seats = int(_number(field("定数")) or 1)
        return seats, sorted(rows, key=lambda r: -r["votes"])
    return 0, []


def _raw(title: str) -> str | None:
    article = _fetch_article(title)
    return None if article is None else article[1]


def collect(elections=ELECTIONS) -> dict:
    """各選挙の当選者のうち、届出が無所属の候補と推薦の欄を集めてdata/に保存する。記事の取得は
    util.fetchのキャッシュに残るので、途中で止まっても続きから取り直せる。"""
    from .util import fetch

    out = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    for house, kai, vote_date, title in elections:
        if vote_date in out:
            continue
        rows, missing = [], []
        if house == "衆議院":
            districts = shugiin_districts(fetch(_API_RAW + title).decode("utf-8"))
        else:
            districts = [f"{d}選挙区" for d in SANGIIN_DISTRICTS]
        for d in districts:
            text = _raw(d)
            if text is None:
                missing.append(d)
                continue
            if house == "衆議院":
                seats, cands = 1, parse_shugiin(text, kai)
            else:
                seats, cands = parse_sangiin(text, kai)
            if not cands:
                missing.append(d)
            for c in cands[:seats]:
                if c["party"].startswith("無所属"):
                    rows.append({"district": d, **c})
        out[vote_date] = {"house": house, "kai": kai, "districts": len(districts), "missing": missing,
                          "independents": rows}
        CACHE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(vote_date, house, "無所属の当選", len(rows), "取れなかった選挙区", missing, flush=True)
    return out
