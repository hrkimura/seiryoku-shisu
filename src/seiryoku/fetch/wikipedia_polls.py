"""英語版Wikipediaの「Opinion polling for the ... Japanese general election」から、各社の政党支持率
(Party identification)の表を取り出す。

複数社の表がそろうのは2017年以降で、それ以前の選挙の記事にはNHKの値しかないので対象にしない。
表には誤差の列がなく、サンプル数だけが載っている(載っていない行や「~1,000」の概数もある)。

見出しは多段で、合流した党の列が出身政党の列にまたがる(中道改革連合が立憲・公明の2列に
またがり、値も2列結合のセルに1つだけ入る)ことがある。ウィキテキストではなく描画済みのHTMLを
読み、rowspan・colspanを展開した格子で列の位置を決める。結合セルの値は両党に配らず、
合計の観測としてcombinedに入れる。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote

from bs4 import BeautifulSoup, Tag

from .util import fetch

# 古い順。最新のページ以外は内容が固まっているのでキャッシュを使う。
PAGES = [
    "Opinion polling for the 2021 Japanese general election",
    "Opinion polling for the 2024 Japanese general election",
    "Opinion polling for the 2026 Japanese general election",
    "Opinion polling for the next Japanese general election",
]
LIVE_PAGES = {"Opinion polling for the next Japanese general election"}

# 列見出しのリンク先(英語版の記事名) -> 本プロジェクトの政党名
PARTY_LINKS = {
    "Liberal Democratic Party (Japan)": "自由民主党",
    "Constitutional Democratic Party of Japan": "立憲民主党",
    "Japan Innovation Party": "日本維新の会",
    "Nippon Ishin no Kai": "日本維新の会",
    "Komeito": "公明党",
    "Kōmeitō": "公明党",
    "New Komeito Party": "公明党",
    "Japanese Communist Party": "日本共産党",
    "Democratic Party For the People": "国民民主党",
    "Democratic Party for the People": "国民民主党",
    # 2026-08にいのちの党へ改称(lineage.LINEAGE_EVENTS)
    "Reiwa Shinsengumi": "いのちの党",
    "Sanseitō": "参政党",
    "Social Democratic Party (Japan)": "社会民主党",
    "NHK Party": "NHK党",
    "Conservative Party of Japan": "日本保守党",
    "Team Mirai": "チームみらい",
    "Centrist Reform Alliance": "中道改革連合",
    "Kibō no Tō": "希望の党",
    "Democratic Party (Japan, 2016)": "民進党",
    "Liberal Party (Japan, 2016)": "自由党",
    "Party for Japanese Kokoro": "日本のこころ",
    "Democratic Reform Party (Japan, 2026)": "民主改革の会",
    "Party of Life (Japan)": "いのちの党",
}
OTHER_LABELS = {"others": "その他", "oth.": "その他"}
NO_PARTY_LABELS = {"no party", "none", "no party id"}
UNDECIDED_PREFIXES = ("und", "dk", "don't know")

# 同じ調査の表記ゆれ。調査方式が変わった社(毎日新聞は2020年ごろから社会調査研究センターとの
# ネット・携帯調査に移った)は、癖が変わるので別の調査として残す。
FIRM_ALIASES = {
    "Nikkei & TV Tokyo": "Nikkei/TV Tokyo",
    "Nikkei": "Nikkei/TV Tokyo",
    "Nikkei-TV Tokyo": "Nikkei/TV Tokyo",
    "Yomiuri Shimbun": "Yomiuri/NNN",
    "Yomiuri-NNN": "Yomiuri/NNN",
    "Asahi Shimbun": "Asahi",
    "Mainichi & SSRC": "Mainichi/SSRC",
    "Mainichi Shimbun": "Mainichi",
    "Mainchi Shimbun": "Mainichi",
    "FNN-Sankei": "Sankei/FNN",
    "go2senkyo & JX": "Senkyo.com/JX",
    "go2senkyo": "Senkyo.com/JX",
    "Jiji News": "Jiji Press",
}
_ARCHIVED = re.compile(r"\s*Archived \d{4}-\d{2}-\d{2} at the Wayback Machine.*$")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE_TOKEN = re.compile(r"(\d{1,2})(?:\s+([A-Za-z]{3,9})\.?)?(?:\s+(\d{4}))?")
_FOOTNOTE = re.compile(r"\[[^\]]*\]")


@dataclass
class Poll:
    start: str
    end: str
    firm: str
    n: int | None
    n_approx: bool
    shares: dict[str, float]
    # 結合セル: 複数の党の合計だけが載っているもの
    combined: list[dict] = field(default_factory=list)
    no_party: float | None = None
    undecided: float | None = None
    page: str = ""


def api_url(title: str) -> str:
    return ("https://en.wikipedia.org/w/api.php?action=parse&prop=text&format=json&formatversion=2&page="
            + quote(title.replace(" ", "_")))


def _text(cell: Tag) -> str:
    for sup in cell.find_all("sup"):
        sup.decompose()
    return _FOOTNOTE.sub("", cell.get_text(" ", strip=True)).strip()


def normalize_firm(firm: str) -> str:
    f = _ARCHIVED.sub("", firm).strip()
    return FIRM_ALIASES.get(f, f)


def party_total(poll: Poll) -> float:
    """政党(その他を含む)の値の合計。無党派・わからないの列は社によって欠けたり入力ミスがあるので、
    支持政党のある人の中での比率は、この合計を分母にして出す。"""
    return sum(poll.shares.values()) + sum(c["value"] for c in poll.combined)


def parse_dates(text: str, year_hint: int | None) -> tuple[date, date] | None:
    """「6–7 Feb 2026」「28 Feb – 1 Mar」「31 Dec 2025 – 2 Jan 2026」のような調査期間を(開始, 終了)にする。"""
    toks = []
    for d, mon, y in _DATE_TOKEN.findall(text.replace("–", " - ").replace("—", " - ")):
        m = _MONTHS.get(mon[:3].lower()) if mon else None
        if mon and m is None:
            continue
        toks.append((int(d), m, int(y) if y else None))
    if not toks:
        return None
    end_d, end_m, end_y = toks[-1]
    if end_m is None:
        return None
    end_y = end_y or year_hint
    if end_y is None:
        return None
    start_d, start_m, start_y = toks[0]
    start_m = start_m or end_m
    start_y = start_y or (end_y - 1 if start_m > end_m else end_y)
    try:
        return date(start_y, start_m, start_d), date(end_y, end_m, end_d)
    except ValueError:
        return None


def _number(text: str) -> float | None:
    t = text.replace(",", "").strip().strip("'")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%?", t)
    return float(m.group(1)) if m else None


def _sample(text: str) -> tuple[int | None, bool]:
    t = text.replace(",", "").strip()
    m = re.fullmatch(r"(~|≈|c\.\s*)?(\d+)", t)
    if not m:
        return None, False
    return int(m.group(2)), bool(m.group(1))


def _grid(table: Tag) -> list[list[tuple[Tag, int]]]:
    """rowspan・colspanを展開し、各マスに(元のセル, そのセルのcolspan)を置いた格子を返す。"""
    grid: dict[tuple[int, int], tuple[Tag, int]] = {}
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            while (r, c) in grid:
                c += 1
            rs = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = (cell, cs)
            c += cs
    width = max((c for _, c in grid), default=-1) + 1
    return [[grid.get((r, c), (None, 0)) for c in range(width)] for r in range(len(rows))]


def _link_title(cell: Tag) -> str | None:
    a = cell.find("a", href=re.compile(r"^/wiki/"))
    if a is None:
        return None
    return unquote(a["href"][len("/wiki/"):]).replace("_", " ")


def _column_labels(grid, n_header: int) -> list[str | None]:
    """列ごとに、いちばん下の段の意味のある見出し(政党名・その他・無党派など)を返す。"""
    labels: list[str | None] = []
    width = len(grid[0]) if grid else 0
    for c in range(width):
        label = None
        for r in range(n_header):
            cell, cs = grid[r][c]
            # 表の途中の時期の区切り(「中道改革連合の分裂後」など)は全列にまたがる見出しの帯になる
            if cell is None or cs * 2 > width:
                continue
            title = _link_title(cell)
            text = _text(cell).lower()
            if title in PARTY_LINKS:
                label = PARTY_LINKS[title]
            elif title and text:
                label = f"?{title}"
            elif text in OTHER_LABELS:
                label = OTHER_LABELS[text]
            elif text in NO_PARTY_LABELS:
                label = "無党派"
            elif text.startswith(UNDECIDED_PREFIXES):
                label = "わからない"
            elif "date" in text:
                label = "date"
            elif "firm" in text or "source" in text:
                label = "firm"
            elif "sample" in text:
                label = "n"
            elif text == "lead":
                label = "lead"
        labels.append(label)
    return labels


def parse_table(table: Tag, year_hint: int | None, page: str) -> tuple[list[Poll], list[str]]:
    grid = _grid(table)
    if not grid:
        return [], []
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    n_header = 0
    for tr in rows:
        if tr.find("td", recursive=False) is not None:
            break
        n_header += 1
    labels = _column_labels(grid, n_header)
    unknown = sorted({l[1:] for l in labels if l and l.startswith("?")})
    if "date" not in labels or "firm" not in labels:
        return [], unknown
    i_date, i_firm = labels.index("date"), labels.index("firm")
    i_n = labels.index("n") if "n" in labels else None
    polls = []
    for r in range(n_header, len(grid)):
        row = grid[r]
        firm = normalize_firm(_text(row[i_firm][0])) if row[i_firm][0] is not None else ""
        if not firm or "election" in firm.lower() or "result" in firm.lower():
            continue
        dates = parse_dates(_text(row[i_date][0]), year_hint) if row[i_date][0] is not None else None
        if dates is None:
            continue
        n, approx = _sample(_text(row[i_n][0])) if i_n is not None and row[i_n][0] is not None else (None, False)
        poll = Poll(start=dates[0].isoformat(), end=dates[1].isoformat(), firm=firm, n=n, n_approx=approx,
                    shares={}, page=page)
        seen: set[int] = set()
        for c, label in enumerate(labels):
            cell, cs = row[c]
            if cell is None or label in (None, "date", "firm", "n", "lead") or label.startswith("?"):
                continue
            if id(cell) in seen:
                continue
            seen.add(id(cell))
            value = _number(_text(cell))
            if value is None:
                continue
            span = [l for l in labels[c:c + cs]]
            if cs > 1:
                if all(l and l not in ("無党派", "わからない", "lead") and not l.startswith("?") for l in span):
                    poll.combined.append({"parties": span, "value": value})
                continue
            if label == "無党派":
                poll.no_party = value
            elif label == "わからない":
                poll.undecided = value
            else:
                poll.shares[label] = value
        if poll.shares:
            polls.append(poll)
    return polls, unknown


def parse_page(html: str, page: str) -> tuple[list[Poll], list[str]]:
    """ページのうち「Party identification」の節にある表だけを読む。"""
    soup = BeautifulSoup(html, "html.parser")
    in_section, year = False, None
    polls, unknown = [], set()
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name == "h2":
            in_section = _text(el).lower().startswith("party identification")
            year = None
        elif el.name in ("h3", "h4"):
            m = re.match(r"(\d{4})", _text(el))
            if m and el.name == "h3":
                year = int(m.group(1))
        elif in_section and "wikitable" in (el.get("class") or []):
            p, u = parse_table(el, year, page)
            polls += p
            unknown |= set(u)
    return polls, sorted(unknown)


def fetch_page(title: str) -> str:
    raw = fetch(api_url(title), force=title in LIVE_PAGES)
    return json.loads(raw)["parse"]["text"]


def dedupe(polls: list[Poll]) -> list[Poll]:
    """ページの境目で同じ調査が2つのページに載ることがあるので、(会社, 終了日)で1件にする。後のページを優先。"""
    out: dict[tuple[str, str], Poll] = {}
    for p in polls:
        out[(p.firm, p.end)] = p
    return sorted(out.values(), key=lambda p: (p.end, p.firm))


def fetch_all(pages: list[str] = PAGES) -> tuple[list[Poll], list[str]]:
    polls, unknown = [], set()
    for title in pages:
        p, u = parse_page(fetch_page(title), title)
        polls += p
        unknown |= set(u)
    return dedupe(polls), sorted(unknown)


def to_json(polls: list[Poll]) -> list[dict]:
    return [asdict(p) for p in polls]


def from_json(rows: list[dict]) -> list[Poll]:
    return [Poll(**r) for r in rows]


HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "poll_history.json"


def update_history(path: Path = HISTORY_PATH, pages: list[str] = PAGES) -> dict:
    """全ページを読み直して保存する。過去の行もWikipedia側で訂正されることがあるので、読めた行は
    読み直した値で置き換える。保存済みで今回読めなかった行は残し、件数をmissingで返す。
    総選挙のあと「next」のページの行は「Opinion polling for the 20xx Japanese general election」へ
    移るので、PAGESにそのページを足すまでは、その分がmissingになる。"""
    polls, unknown = fetch_all(pages)
    if not polls:
        raise RuntimeError("世論調査の表を1件も読めませんでした(ページの構成が変わった可能性)")
    saved = from_json(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else []
    fetched = {(p.firm, p.end) for p in polls}
    missing = [p for p in saved if (p.firm, p.end) not in fetched]
    merged = dedupe(missing + polls)
    path.write_text(json.dumps(to_json(merged), ensure_ascii=False, indent=1), encoding="utf-8")
    return {"polls": len(merged), "first": merged[0].end, "last": merged[-1].end, "unknown_parties": unknown,
            "missing": len(missing)}


if __name__ == "__main__":
    print(json.dumps(update_history(), ensure_ascii=False))
