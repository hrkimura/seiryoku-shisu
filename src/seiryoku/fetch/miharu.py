"""三春充希氏(未来社会プロジェクト)による各社世論調査の政党支持率の平均値を取得する。

出典はnote記事「世論の動向【継続更新】」。過去の推移はグラフ画像でしか公開されて
おらず、数値として読めるのは記事本文に載っている「現在の平均値」だけである。
そのため過去分は遡れず、実行のたびに現在値を1件ずつ積み上げて時系列を作る
(support_history.py)。
"""
from __future__ import annotations

import html
import json
import re

from ..lineage import canonicalize
from .util import fetch

NOTE_KEY = "nf05620cd5f69"
SOURCE_URL = f"https://note.com/miraisyakai/n/{NOTE_KEY}"
_API_URL = f"https://note.com/api/v3/notes/{NOTE_KEY}"

# 記事中の略称 -> 正式名称。「無党派」は政党ではないが、支持率の分母の一部として残す。
ABBREVIATIONS = {
    "無党派": "無党派",
    "自民": "自由民主党",
    "立憲": "立憲民主党",
    "維新": "日本維新の会",
    "国民": "国民民主党",
    "公明": "公明党",
    "共産": "日本共産党",
    "参政": "参政党",
    "みらい": "チームみらい",
    "保守": "日本保守党",
    "中道": "中道改革連合",
    "社民": "社会民主党",
    "いのち": "いのちの党",
    "れいわ": "いのちの党",
}

_SECTION_RE = re.compile(r"政党支持率の平均（(\d{4})年(\d{1,2})月(\d{1,2})日更新）")
_ROW_RE = re.compile(r"^\s*(\S+?)[\s　]+(\d+(?:\.\d+)?)\s*$")


class ParseError(RuntimeError):
    """記事の構成が想定と変わり、支持率を読み取れなかった場合の例外。"""


def _body_to_text(body_html: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", body_html)
    t = re.sub(r"</(p|h\d|blockquote|figure)>", "\n", t)
    return html.unescape(re.sub(r"<[^>]+>", "", t))


def parse_support(body_html: str) -> dict:
    """記事本文のHTMLから、政党支持率の更新日と政党ごとの値(%)を取り出す。

    戻り値: {"updated": "YYYY-MM-DD", "S_p": {正式名称: 値}, "unknown_labels": [...]}
    未知の略称(新党の登場など)は落とさず略称のまま残し、unknown_labelsで知らせる。
    """
    text = _body_to_text(body_html)
    m = _SECTION_RE.search(text)
    if m is None:
        raise ParseError("「政党支持率の平均（…更新）」の見出しが見つかりません")
    updated = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 見出しの後ろには「10％未満拡大」の小見出しを挟んで数値の段落が続くので、
    # 見出し以降で最初に現れる「ラベル 数値」の連続行のかたまりを採る。
    S_p: dict[str, float] = {}
    unknown: list[str] = []
    started = False
    for line in text[m.end():].splitlines():
        row = _ROW_RE.match(line)
        if row is None:
            if started:
                break
            continue
        started = True
        label, value = row.group(1), float(row.group(2))
        name = ABBREVIATIONS.get(label)
        if name is None:
            unknown.append(label)
            name = label
        else:
            name = canonicalize(name) if name != "無党派" else name
        S_p[name] = S_p.get(name, 0.0) + value
    if not S_p:
        raise ParseError("政党支持率の数値行が見つかりません")
    return {"updated": updated, "S_p": S_p, "unknown_labels": unknown}


def fetch_support() -> dict:
    """noteのAPIから記事本文を取得して支持率を返す(キャッシュは使わず毎回取得する)。"""
    raw = fetch(_API_URL, force=True)
    body = json.loads(raw)["data"]["body"]
    result = parse_support(body)
    result["source"] = SOURCE_URL
    return result
