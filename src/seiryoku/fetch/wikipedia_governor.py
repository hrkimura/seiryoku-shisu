"""Wikipediaの知事選挙記事から、当選者の推薦・支持政党を取り出す。

全国首長名簿(jichisoken.py)は党本部レベルの推薦・支持しか列に記録せず、
自民党・日本維新の会の県連推薦はどこにも残らない。また新しい年版の公開は
選挙から最大1年9か月ほど遅れる。そこで知事選挙に限り、記事の選挙結果欄
「当選者推薦・支持」を補助の情報源として足し合わせる(endorsements.py)。

取得結果は記事の版(revid)と元の文言ごとdata/に保存し、後から検証できるようにする。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

from .util import fetch

CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "governor_endorsements_wikipedia.json"
# 選挙結果欄が無い記事は、本文の記述から人手で拾った結果をここに置き、自動取得より優先する。
MANUAL_PATH = Path(__file__).resolve().parents[3] / "data" / "governor_endorsements_manual.json"

_API = (
    "https://ja.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=ids%7Ccontent"
    "&rvslots=main&format=json&formatversion=2&redirects=1&titles="
)

# 長い名称を先に置く(「国民民主党」より先に「NHKから国民を守る党」を消費させる等)。
# 対象外の政党も名前として残し、按分の分母に入れる(index._effective_party_shares側で
# 政党要件を満たさない政党は無所属・諸派にまとめられる)。
_PARTY_PATTERNS: list[tuple[str, str]] = [
    (r"NHKから国民を守る党|NHK党|N国党", "NHKから国民を守る党"),
    (r"自由民主党|自民党|自民", "自由民主党"),
    (r"立憲民主党|立憲|立民", "立憲民主党"),
    (r"国民民主党|国民民主|国民(?!を)", "国民民主党"),
    (r"公明党|公明", "公明党"),
    (r"日本共産党|共産党|共産", "日本共産党"),
    (r"社会民主党|社民党|社民", "社会民主党"),
    (r"日本維新の会|維新の会|維新", "日本維新の会"),
    (r"参政党", "参政党"),
    (r"れいわ新選組|いのちの党", "いのちの党"),
    (r"中道改革連合", "中道改革連合"),
    (r"チームみらい", "チームみらい"),
    (r"日本保守党", "日本保守党"),
    (r"希望の党", "希望の党"),
    (r"民進党", "民進党"),
    (r"自由党", "自由党"),
    (r"沖縄社会大衆党|社大党|社大", "沖縄社会大衆党"),
    (r"新党大地", "新党大地"),
]


def article_title(prefecture: str, vote_date: str) -> str:
    return f"{vote_date[:4]}年{prefecture}知事選挙"


def _strip_markup(text: str) -> str:
    t = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", text, flags=re.S)
    t = re.sub(r"\{\{Efn2?\|", "", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\{\{[^|{}]*\|", "", t)
    t = re.sub(r"<[^>]+>|'''|''|\}\}", "", t)
    return t.strip()


# 項目は1行ずつのことも1行に並ぶこともあるので、次の「|項目名=」の手前で切る。
# [[リンク|表示名]]や{{small|...}}の中の「|」は直後に「=」が来ないので、ここでは切れない。
_NEXT_FIELD = r"(?=\|[^|=\n\]\}]*=|\n\}\})"


def parse_winner(wikitext: str) -> dict | None:
    """選挙結果テンプレートの「当選者名」「当選者推薦・支持」を返す。項目が無ければNone。"""
    m = re.search(r"\|\s*当選者名\s*=(.*?)" + _NEXT_FIELD, wikitext, re.S)
    e = re.search(r"\|\s*当選者推薦・支持\s*=(.*?)" + _NEXT_FIELD, wikitext, re.S)
    if m is None or e is None:
        return None
    return {"winner": _strip_markup(m.group(1)), "text": _strip_markup(e.group(1))}


def extract_parties(text: str) -> list[str]:
    """推薦・支持の文言から政党名を出現順に取り出す。「自主投票」は推薦・支持ではないので除く。"""
    t = re.sub(r"[^、。,，\n]*自主投票", "", text)
    found: list[tuple[int, str]] = []
    for pattern, name in _PARTY_PATTERNS:
        for m in re.finditer(pattern, t):
            found.append((m.start(), name))
        t = re.sub(pattern, lambda m: "　" * len(m.group(0)), t)
    out: list[str] = []
    for _, name in sorted(found):
        if name not in out:
            out.append(name)
    return out


def _load_manual() -> dict:
    if MANUAL_PATH.exists():
        return json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    return {}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_article(title: str) -> tuple[int, str] | None:
    data = json.loads(fetch(_API + quote(title)))
    page = data["query"]["pages"][0]
    if page.get("missing"):
        return None
    rev = page["revisions"][0]
    return rev["revid"], rev["slots"]["main"]["content"]


def lookup(prefecture: str, vote_date: str, cache: dict | None = None) -> dict | None:
    """(都道府県, 投票日)の知事選挙について、保存済みの結果を返す。無ければ取得して保存する。

    記事が無い・項目を読み取れない場合も結果として保存する(毎月取りに行かないため)。
    通信に失敗した場合は保存せずNoneを返す。
    """
    key = f"{prefecture}|{vote_date}"
    manual = _load_manual()
    if key in manual:
        return manual[key]
    own_cache = cache is None
    cache = _load_cache() if own_cache else cache
    if key in cache:
        return cache[key]
    title = article_title(prefecture, vote_date)
    try:
        article = _fetch_article(title)
    except Exception:
        return None
    if article is None:
        entry = {"title": title, "status": "no_article"}
    else:
        revid, wikitext = article
        parsed = parse_winner(wikitext)
        if parsed is None:
            entry = {"title": title, "revid": revid, "status": "no_field"}
        else:
            entry = {
                "title": title, "revid": revid, "status": "ok",
                "winner": parsed["winner"], "text": parsed["text"],
                "parties": extract_parties(parsed["text"]),
            }
    cache[key] = entry
    if own_cache:
        _save_cache(cache)
    return entry


def lookup_many(elections: list[tuple[str, str]]) -> dict[str, dict]:
    """複数の選挙をまとめて引き、キャッシュの保存を1回で済ませる。"""
    cache = _load_cache()
    for pref, vote_date in elections:
        lookup(pref, vote_date, cache=cache)
    _save_cache(cache)
    return cache
