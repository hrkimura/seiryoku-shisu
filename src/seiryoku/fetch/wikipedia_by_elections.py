"""Wikipediaの「YYYY年日本の補欠選挙」の記事から、国政の補欠選挙と再選挙の結果を読み取る。

記事は「== 4月の補欠選挙 ==」のような節ごとに投票日を書き、選挙区ごとの「=== 衆議院X ===」の下に
選挙結果テンプレートを置く。選挙区が1つだけの年は見出しがないことがあるので、そのときは節の
「選挙区：衆議院[[X]]」の行で補う。通常選挙と同時の合併選挙は通常選挙の議席に含まれるので読まない。
"""
from __future__ import annotations

import re
from datetime import date
from urllib.parse import quote

from .wikipedia_districts import _templates, sangiin_rows, shugiin_rows
from .wikipedia_governor import _strip_markup

_API_RAW = "https://ja.wikipedia.org/w/index.php?action=raw&title="
_ARTICLE_URL = "https://ja.wikipedia.org/wiki/"


class ParseError(ValueError):
    pass


def article_title(year: int) -> str:
    return f"{year}年日本の補欠選挙"


def _sections(wikitext: str) -> list[tuple[str, str]]:
    """2段目の見出し(== … ==)ごとに(見出し, 本文)を返す。"""
    parts = re.split(r"^==([^=].*?)==\s*$", wikitext, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def _vote_date(section: str) -> date | None:
    m = re.search(r"^\*\s*投票日\s*[：:]\s*(.*)$", section, flags=re.M)
    if not m:
        return None
    m = re.search(r"(\d{4})年.*?(\d{1,2})月(\d{1,2})日", _strip_markup(m.group(1)))
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _district(heading: str) -> tuple[str, str] | None:
    """「衆議院島根県第1区」「参議院広島県選挙区（再選挙）」から(院, 選挙区)を返す。"""
    m = re.fullmatch(r"(衆議院|参議院)(.+?)(?:（再選挙）)?", _strip_markup(heading).strip())
    return (m.group(1), m.group(2)) if m else None


def parse_year(wikitext: str, year: int) -> list[dict]:
    """記事の補欠選挙と再選挙を、data/national_by_elections.json と同じ形で返す。結果の表が
    まだない選挙区は返さない。読み取れない形のときはParseErrorを送出する。"""
    source = _ARTICLE_URL + quote(article_title(year))
    out = []
    for heading, body in _sections(wikitext):
        if not re.search(r"補欠|再選挙", heading) or "合併" in heading:
            continue
        vote_date = _vote_date(body)
        tables = [t for t in _templates(body, "衆議院小選挙区補欠")] + [
            t for t in _templates(body, "参議院選挙区") if "補欠年" in t]
        if not tables:
            continue
        if vote_date is None:
            raise ParseError(f"{year}年「{heading}」の投票日が読めない")
        listed = [(h, d) for h, d in re.findall(r"選挙区\s*[：:]\s*(衆議院|参議院)\s*\[\[([^\]|]+)", body)]
        for t in sorted(tables, key=body.find):
            pos = body.find(t)
            house = "衆議院" if "衆議院小選挙区補欠" in t else "参議院"
            headings = re.findall(r"^===([^=].*?)===\s*$", body[:pos], flags=re.M)
            found = [h for h in headings if (d := _district(h)) and d[0] == house]
            if found:
                district = _district(found[-1])[1]
                kind = "再選挙" if "再選挙" in found[-1] else "補欠選挙"
            elif len([d for h, d in listed if h == house]) == 1:
                district = next(d for h, d in listed if h == house)
                kind = "補欠選挙"
            else:
                raise ParseError(f"{year}年「{heading}」の表の選挙区が分からない")
            if house == "衆議院":
                candidates = shugiin_rows(t)
            else:
                seats, candidates = sangiin_rows(t)
                if seats != 1:
                    raise ParseError(f"{year}年 {district} の定数が1でない({seats})")
            if not candidates or not candidates[0]["votes"]:
                raise ParseError(f"{year}年 {district} の候補か得票が読めない")
            out.append({
                "vote_date": vote_date.isoformat(), "house": house, "district": district, "kind": kind,
                "winner": candidates[0]["name"], "party": candidates[0]["party"], "source": source,
                "candidates": candidates,
            })
    return out


def fetch_wikitext(year: int, force: bool = True) -> str | None:
    """記事の本文。記事がまだなければNone。"""
    from .util import fetch

    try:
        return fetch(_API_RAW + quote(article_title(year)), force=force).decode("utf-8")
    except Exception as e:
        if "404" in str(e):
            return None
        raise


def sections_without_results(wikitext: str, today: date) -> list[str]:
    """投票日を過ぎたのに結果の表がまだない節。記事の更新待ちか、書式が変わって読めていない。"""
    out = []
    for heading, body in _sections(wikitext):
        if not re.search(r"補欠|再選挙", heading) or "合併" in heading:
            continue
        vote_date = _vote_date(body)
        has_table = _templates(body, "衆議院小選挙区補欠") or any("補欠年" in t for t in _templates(body, "参議院選挙区"))
        if vote_date is not None and vote_date < today and not has_table:
            out.append(f"{heading}(投票日 {vote_date.isoformat()})")
    return out
