"""国会(衆議院・参議院)の過去の会派別議員数を取得する。

参議院は「会派別所属議員数の変遷」という単一の通史ページに1947年〜の全召集日ぶんが
載っているため、そのまま時系列を取得できる。衆議院には参議院のような単一の通史
ページが無く(年度ごとに別ページへ分散し、URLパターンも年によって不統一)、全期間の
再現は諦め、総務省が総選挙のたびに公表する「届出政党等別...当選人数」ファイルを
選挙ごとに集める方式にした。この結果ファイルは第45回(2009年)以降はExcel
(xls/xlsx)で機械可読だが、第44回(2005年)以前はPDF(文章埋め込み型の表)で
本モジュールでは対象外にしている([[seiryoku_shisu_design]]参照、必要になれば
local_history_pdf.pyと同様の正規表現抽出を追加する)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .local_parties import _label_to_iso_date
from .util import fetch

SANGIIN_HISTORY_URL = (
    "https://www.sangiin.go.jp/japanese/san60/s60_shiryou/giinsuu_kaiha.htm"
)
SHUGIIN_ICHIRAN_URL = "https://www.soumu.go.jp/senkyo/senkyo_s/data/shugiin/ichiran.html"
_SHUGIIN_RESULT_LINK_TEXT = "届出政党等別男女別新前元別当選人数"
_SHUGIIN_NON_PARTY_CELLS = {
    "",
    "男", "女", "計", "新", "前", "元", "区分",
    "小計", "合計", "政党等所属",  # 政党ではなく複数政党の再集計列
}
# 合計の表の「政党等所属」は、党派別の表に列挙されない政治団体の当選者(2024年の日本保守党の
# 小選挙区の1議席など)。落とすと総議席が定数に足りなくなるので、諸派として数える。
_SHUGIIN_OTHERS_LABEL = "諸派"

SANGIIN_ICHIRAN_URL = "https://www.soumu.go.jp/senkyo/senkyo_s/data/sangiin/ichiran.html"
_SANGIIN_RESULT_LINK_TEXT = "党派別男女別新前元別当選人数"
_SANGIIN_NON_PARTY_CELLS = {
    "",
    "男", "女", "計", "新", "現", "元", "区分",
    "小計", "合計",
}


@dataclass
class Snapshot:
    session: str
    date: str  # 元号表記のまま(例: "R7.8.1")
    seats: dict[str, int]


def fetch_sangiin_history() -> list[Snapshot]:
    html = fetch(SANGIIN_HISTORY_URL).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    snapshots: list[Snapshot] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for i in range(1, len(rows) - 1, 2):
            header = [c.get_text(strip=True) for c in rows[i].find_all(["td", "th"])]
            data = [c.get_text(strip=True) for c in rows[i + 1].find_all(["td", "th"])]
            if len(header) < 3 or not data:
                continue
            session, conv_date = header[0], header[1]
            party_names = header[2 : 2 + len(data)]
            seats = {}
            for name, val in zip(party_names, data):
                if not name or not val or not val.isdigit():
                    continue
                seats[name] = seats.get(name, 0) + int(val)
            if seats:
                snapshots.append(Snapshot(session=session, date=conv_date, seats=seats))
    return snapshots


def _discover_shugiin_elections() -> list[tuple[int, str]]:
    """(選挙回次, 選挙結果ページURL) を返す(総務省の一覧ページから機械的に発見)。"""
    html = fetch(SHUGIIN_ICHIRAN_URL).decode("shift_jis", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    elections: list[tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        m = re.match(r"第(\d+)回衆議院議員総選挙", a.get_text(strip=True))
        href = a["href"]
        if m and re.search(r"/shugiin\d+/index\.html$", href):
            elections.append((int(m.group(1)), "https://www.soumu.go.jp" + href))
    return sorted(elections)


def _shugiin_result_file_url(index_url: str) -> str | None:
    html = fetch(index_url).decode("shift_jis", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if _SHUGIIN_RESULT_LINK_TEXT in a.get_text(strip=True):
            return urljoin(index_url, a["href"])
    return None


def _shugiin_election_date(index_url: str) -> str:
    """選挙結果ページの<title>(例: 「...平成21年8月30日執行 衆議院議員総選挙...」)から
    投票日をISO日付で返す。"""
    html = fetch(index_url).decode("shift_jis", errors="ignore")
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if not m:
        raise ValueError(f"titleタグが見つかりません: {index_url}")
    return _label_to_iso_date(m.group(1))


def _norm(s: object) -> str:
    """全角スペース等も含めて空白を除去する(セル内で「小　　計」のように
    政党名と関係ない空白が挿入されているため、単純なstrip()では不十分)。"""
    return re.sub(r"\s+", "", s) if isinstance(s, str) else ""


def _header_blocks(rows: list[tuple], header_indices: list[int]) -> list[tuple[int, dict[int, list[str]]]]:
    """見出し行を(最後の見出し行の位置, {列: 見出しの断片})のブロックにまとめる。
    長い党名(2019年参院選の「労働の解放をめざす/労働者党」、2021年衆院選の「ＮＨＫと裁判してる党
    弁護士法72条違反で」)は見出しが2行に折り返され、続きの行も見出しに見える。続きの行を
    別のブロックとすると、元の見出しのブロックが空になり同じブロックの党ごと抜ける。"""
    blocks: list[tuple[int, dict[int, list[str]]]] = []
    prev = None
    for idx in header_indices:
        cells = {i: v for i, v in enumerate(rows[idx]) if i >= 2 and isinstance(v, str) and _norm(v)}
        if prev is not None and idx == prev + 1:
            for i, v in cells.items():
                blocks[-1][1].setdefault(i, []).append(v)
            blocks[-1] = (idx, blocks[-1][1])
        else:
            blocks.append((idx, {i: [v] for i, v in cells.items()}))
        prev = idx
    return blocks


def _parse_shugiin_result_sheet(rows: list[tuple]) -> dict[str, int]:
    """当選人数シート1枚から{政党: 当選者数}を返す。

    1シートに政党名の見出し行が複数回繰り返されることがある(6政党ずつ×複数ブロック)
    ため、見出し行を1つだけと仮定せず全て見つけ、それぞれのブロック内で完結させて
    (次の見出し行の手前までに区切って)「計」行を探す——ブロックをまたいで最後の
    「計」行を探すと、別ブロックの値を政党に紐付けてしまうバグになる。
    また、「小計」「合計」「政党等所属」は政党ではなく複数政党の再集計列(無所属を
    含む全体の合計等)なので政党名として扱わない(「無所属」はここでは除外しない
    正真正銘の1カテゴリ)。
    """

    def _looks_like_party_name(c: object) -> bool:
        # タイトル行(例:「(1)　届出政党等別男女別新前元別当選人数...」)は長い説明文が
        # 1セルだけ入っているため、短い政党名(「無所属」1つだけの見出し行もある)と
        # 文字数で区別する。
        n = _norm(c)
        return bool(n) and n not in _SHUGIIN_NON_PARTY_CELLS and len(n) <= 12

    def _qualifying_cells(row: tuple) -> int:
        return sum(1 for c in row[2:] if isinstance(c, str) and _looks_like_party_name(c))

    header_indices = [i for i, row in enumerate(rows) if row and _qualifying_cells(row) >= 1]
    if not header_indices:
        raise ValueError("政党名の見出し行が見つかりません")

    seats: dict[str, int] = {}
    for block_i, (header_idx, header) in enumerate(blocks := _header_blocks(rows, header_indices)):
        block_end = blocks[block_i + 1][0] if block_i + 1 < len(blocks) else len(rows)
        party_cols = []
        for i, pieces in sorted(header.items()):
            name = "".join(_norm(p) for p in pieces)
            if name == "政党等所属":
                party_cols.append((i, _SHUGIIN_OTHERS_LABEL))
            elif any(_looks_like_party_name(p) for p in pieces):
                party_cols.append((i, name))
        # 小選挙区・比例代表・合計の3ブロックそれぞれに「計」行があり、
        # このヘッダーに対応する範囲内(次の見出しの手前まで)での最後の行を使う。
        total_rows = [row for row in rows[header_idx + 1 : block_end] if row and len(row) > 1 and _norm(row[1]) == "計"]
        if not total_rows:
            continue
        total_row = total_rows[-1]
        for col, party in party_cols:
            val_col = col + 2
            if val_col < len(total_row) and isinstance(total_row[val_col], (int, float)):
                seats[party] = seats.get(party, 0) + int(total_row[val_col])
    if not seats:
        raise ValueError("「計」の行が見つかりません")
    return seats


def fetch_shugiin_history(min_election: int = 45) -> list[Snapshot]:
    """衆議院の総選挙ごとの党派別当選者数を返す(総務省の選挙結果ファイルより)。

    total務省は総選挙のたびに党派別当選者数のファイルを公表しており、第45回
    (2009年)以降はExcel(xls/xlsx)で機械可読。min_election未満(PDF配布の年)は
    対象外にする(モジュールdocstring参照)。
    """
    from .local_history import _sheets_as_rows  # xls/xlsx両対応の読み込みを再利用

    snapshots: list[Snapshot] = []
    for number, index_url in _discover_shugiin_elections():
        if number < min_election:
            continue
        file_url = _shugiin_result_file_url(index_url)
        if file_url is None:
            continue
        date = _shugiin_election_date(index_url)
        seats: dict[str, int] = {}
        for rows in _sheets_as_rows(file_url).values():
            try:
                sheet_seats = _parse_shugiin_result_sheet(rows)
            except ValueError:
                continue
            for party, n in sheet_seats.items():
                seats[party] = seats.get(party, 0) + n
        if seats:
            snapshots.append(Snapshot(session=f"第{number}回", date=date, seats=seats))
    return snapshots


def _discover_sangiin_elections() -> list[tuple[int, str]]:
    """(選挙回次, 選挙結果ページURL) を返す(総務省の一覧ページから機械的に発見)。"""
    html = fetch(SANGIIN_ICHIRAN_URL).decode("shift_jis", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    elections: list[tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        m = re.match(r"第(\d+)回参議院議員通常選挙結果", a.get_text(strip=True))
        href = a["href"]
        if m and re.search(r"/sangiin\d+/index\.html$", href):
            elections.append((int(m.group(1)), "https://www.soumu.go.jp" + href))
    return sorted(elections)


def _sangiin_result_file_url(index_url: str) -> str | None:
    """結果ファイルへのリンクを絶対URLで返す。

    古い回(例: 第21回)はルート相対でなくディレクトリ相対のhref(例: "xls/....xls")
    を使うため、単純な"https://www.soumu.go.jp"+hrefの連結では壊れる
    (2026-09に発見)。urljoinでindex_url自身を基準に解決する。
    """
    html = fetch(index_url).decode("shift_jis", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if _SANGIIN_RESULT_LINK_TEXT in a.get_text(strip=True):
            return urljoin(index_url, a["href"])
    return None


def _sangiin_election_date(index_url: str) -> str:
    """選挙結果ページの<title>から投票日をISO日付で返す(_shugiin_election_dateと同じ)。"""
    html = fetch(index_url).decode("shift_jis", errors="ignore")
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if not m:
        raise ValueError(f"titleタグが見つかりません: {index_url}")
    return _label_to_iso_date(m.group(1))


def _strip_by_election_suffix(value: object) -> int:
    """「21(1)」のような値から、合併実施された補欠選挙ぶんの外書き「(1)」を除き、
    通常選挙(半数改選)そのものの当選者数だけを返す。空文字は0。"""
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return 0
    m = re.match(r"\d+", value)
    return int(m.group()) if m else 0


def _parse_sangiin_result_sheet(rows: list[tuple]) -> dict[str, int]:
    """当選人数シート1枚から{政党: その回の改選当選者数}を返す。

    総務省のこのシートは6政党ずつのブロックが横に並んでおり、各ブロックは
    「選挙区・比例代表・合計」の3つの区分をそれぞれ「新・現・元・計」の行で持つ
    (shugiinの届出政党等別ファイルとは異なるレイアウト)。区分名(「選挙区」
    「比例代表」「合計」)は1セルにまとまっている年度と、1文字ずつ複数行に
    分割されている年度があり(2026-09に発見、後者は各区分内の「新・現・元・計」
    のどの行に何文字目が来るかも年度によってばらばら)、区分名のテキストでは
    確実に判別できない。その代わり、集計順が常に「(比例代表・選挙区のいずれか
    2区分)→合計(全体)」で、各区分の最後の行が「計」であることを利用し、
    ブロック内で「計」とラベル付けされた行のうち\\textbf{最後}のもの
    (=合計区分の計、選挙区+比例代表を合算した、その回の改選当選者数)を使う
    (shugiin側のtotal_rows[-1]と同じ考え方)。
    """

    _NUMERIC_RE = re.compile(r"^\d+(\(\d+\))?$")

    def _looks_like_party_name(c: object) -> bool:
        n = _norm(c)
        if not n or n in _SANGIIN_NON_PARTY_CELLS or len(n) > 12:
            return False
        # 当選者数の値(「23」「7(1)」等)が政党名の見出しと誤認されないように、
        # 数字だけのセルは除外する(2026-09に発見、実際の値セルで再現)。
        return not _NUMERIC_RE.match(n)

    def _qualifying_cells(row: tuple) -> int:
        return sum(1 for c in row[2::3] if isinstance(c, str) and _looks_like_party_name(c))

    header_indices = [i for i, row in enumerate(rows) if row and _qualifying_cells(row) >= 1]
    if not header_indices:
        raise ValueError("政党名の見出し行が見つかりません")

    seats: dict[str, int] = {}
    for block_i, (header_idx, header) in enumerate(blocks := _header_blocks(rows, header_indices)):
        block_end = blocks[block_i + 1][0] if block_i + 1 < len(blocks) else len(rows)
        party_cols = [
            (i, "".join(_norm(p) for p in pieces)) for i, pieces in sorted(header.items())
            if any(_looks_like_party_name(p) for p in pieces)
        ]
        total_rows = [
            row
            for row in rows[header_idx + 1 : block_end]
            if row and len(row) > 1 and _norm(row[1]) == "計"
        ]
        if not total_rows:
            continue
        total_row = total_rows[-1]
        for col, party in party_cols:
            val_col = col + 2  # 男・女・計の3列組のうち「計」列
            if val_col < len(total_row):
                seats[party] = seats.get(party, 0) + _strip_by_election_suffix(total_row[val_col])
    if not seats:
        raise ValueError("「計」の行が見つかりません")
    return seats


def fetch_sangiin_election_history(min_election: int = 21) -> list[Snapshot]:
    """参議院の半数改選ごとの党派別当選者数を返す(総務省の選挙結果ファイルより)。

    sangiin.go.jpの「会派別所属議員数の変遷」(fetch_sangiin_history)は選挙直後の
    参議院\\textbf{全体}(改選されなかった残り半分も含む)のスナップショットであり、
    その回の選挙で実際に決まった議席数ではない(2026-09に発見)。$C_p(T)$の
    「その選挙イベントで決まったこと」という定義に合わせるため、総務省が
    選挙ごとに公表する当選者数ファイルから、その回の改選当選者数だけを取得する。
    min_election=21(2007年)より前は結果ページの構成が異なる可能性があるため対象外。
    """
    from .local_history import _sheets_as_rows

    snapshots: list[Snapshot] = []
    for number, index_url in _discover_sangiin_elections():
        if number < min_election:
            continue
        file_url = _sangiin_result_file_url(index_url)
        if file_url is None:
            continue
        date = _sangiin_election_date(index_url)
        seats: dict[str, int] = {}
        for rows in _sheets_as_rows(file_url).values():
            try:
                sheet_seats = _parse_sangiin_result_sheet(rows)
            except ValueError:
                continue
            for party, n in sheet_seats.items():
                seats[party] = seats.get(party, 0) + n
        if seats:
            snapshots.append(Snapshot(session=f"第{number}回", date=date, seats=seats))
    return snapshots
