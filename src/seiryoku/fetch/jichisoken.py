"""地方自治総合研究所「全国首長名簿」の推薦・支持政党データ。

名簿本体(PDF)ではなく、同じ「全国首長名簿」ページで配布されているExcel版
(都道府県知事・市区町村長それぞれの当該年度の選挙結果に、党派ごとの推薦(〇)・
支持(△)マークが付いた表)を使う。この表は年版ごとに直近1年間(5月〜翌4月)に
行われた選挙ぶんしか収録していないため、1年版だけでは対象期間に選挙が無かった
自治体の推薦・支持情報が取れない。そこで公開されている年版(2026年9月時点で
2018〜2025年の8年ぶん)を全て遡って合算し、同じ自治体が複数年版に登場する
場合は最新の年版(=直近の選挙)を優先する。それでも一度も選挙が無かった期間の
自治体は取れない(その場合は従来通り形式上の届出政党を使う、
[[seiryoku_shisu_design]]参照)。

年版ごとにシート名・ヘッダ行の位置・列名の表記が微妙に異なることを確認済みなので、
可能な限り位置に依存しない検出(シート名に「知事」「首長」を含む、ヘッダ行は
「自治体名」列を含む行を探す、政党列は列名で検索する)を行う。それでも解析できない
年版はスキップし、警告を出す(黙って無視しない)。

各エントリのEndorsement.vote_dateは、そのエントリが記録している年版の期間ではなく
当選者自身の実際の当選年をそのまま示している(実データで確認済み——例: 2023年版の
シートに「2019年」当選の知事がそのまま載っている)。そのため、ある選挙(前任者の
当選など)を過去に遡って特定したい場合は、`*_endorsements()`が返す「最新の既知の
状態」ではなく`*_endorsements_by_year()`と`endorsement_for_vote_year()`を使い、
その選挙の投票年と一致するEndorsementを年版を問わず探す([[seiryoku_shisu_design]]の
「前任者側の推薦・支持政党を当選時点の年版で判定」参照)。
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import openpyxl
from bs4 import BeautifulSoup

from .util import fetch

INDEX_URL = "https://jichisoken.jp/jichitaisenkyo/"
CORRECTIONS_PATH = Path(__file__).resolve().parents[3] / "data" / "meibo_corrections.json"

# 社大/大(沖縄独自の地域政党)は国政政党ではないため対象外とし、
# 無所属・諸派側の集計に委ねる([[seiryoku_shisu_design]]の「国政政党のみ追跡」方針)。
_PARTY_COLUMNS = {
    "自": "自由民主党",
    "立民": "立憲民主党",
    "立": "立憲民主党",
    "国民": "国民民主党",
    "国": "国民民主党",
    "公": "公明党",
    "共": "日本共産党",
    "社": "社会民主党",
    "維": "日本維新の会",
}

# 政党列は党本部レベルの推薦・支持しか記録せず、県本部・県連など地方組織独自の
# 推薦・支持は「その他1/2」欄に「公明党県本部推薦」のような文言で書かれる(年版の凡例)。
# 推薦・支持はどのレベルのものでも数える方針なので、この欄も読む。
_OTHER_COLUMNS = ("その他1", "その他2")
_OTHER_NOTE_KEYWORDS = re.compile(r"推薦|支持|支援|応援")
_OTHER_NOTE_PARTIES: list[tuple[str, str]] = [
    (r"自由民主党|自民党|自民", "自由民主党"),
    (r"立憲民主党|立憲", "立憲民主党"),
    (r"国民民主党|国民民主", "国民民主党"),
    (r"公明党|公明", "公明党"),
    (r"日本共産党|共産党", "日本共産党"),
    (r"社会民主党|社民党|社民", "社会民主党"),
    (r"日本維新の会|維新", "日本維新の会"),
    (r"参政党", "参政党"),
    (r"れいわ新選組", "いのちの党"),
]

_DATE_COLUMN_CANDIDATES = ("選挙執行日", "選挙執行月日", "選挙執行年")


@dataclass
class Endorsement:
    name: str  # 自治体名(市区町村)または都道府県名
    formal_party: str  # 届出政党(多くは「無所属」)
    endorsing_parties: list[str]  # 推薦・支持を表明した国政政党(複数可、無ければ空)
    vote_date: str
    year: str  # データの年版(古い年版を新しい年版で上書きするために使う)
    # 同じ選挙の落選者を推薦・支持した政党。他の情報源と突き合わせるときに、
    # 名簿と矛盾する推薦(名簿では別の候補を推していた政党)を除くのに使う。
    rival_endorsing_parties: list[str] = field(default_factory=list)


def discover_years() -> list[tuple[str, str]]:
    """(年版, ページURL) を新しい順ではなく古い順に返す(合算時に新しい年版で
    上書きするため)。"""
    # 目次だけはキャッシュを使わず毎回取り直す。キャッシュのままだと新しい年版が
    # 公開されても見つからない。取得に失敗したら手元の目次で続ける。retries=0とwait=False
    # は、オフライン時にutil.fetchのWAFクールダウン(11分)を待たないため。wait=Falseが
    # ないと、1回目の失敗で始まったクールダウンを、同じ実行中の2回目以降の呼び出しが待つ。
    try:
        raw = fetch(INDEX_URL, force=True, retries=0, wait=False)
    except Exception:
        raw = fetch(INDEX_URL)
    html = raw.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    years: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = __import__("re").search(r"/jichitaisenkyo/(\d{4})/?$", href)
        if m:
            url = href if href.startswith("http") else "https://jichisoken.jp" + href
            years[m.group(1)] = url
    return sorted(years.items())


def _discover_xlsx_urls(year_page_url: str) -> tuple[str | None, str | None]:
    """年版ページから (都道府県エクセルURL, 市区町村エクセルURL) を返す。"""
    html = fetch(year_page_url).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    pref_url = muni_url = None
    for a in soup.find_all("a", href=True):
        if not a["href"].lower().endswith(".xlsx"):
            continue
        text = a.get_text(strip=True)
        href = a["href"] if a["href"].startswith("http") else "https://jichisoken.jp" + a["href"]
        if "都道府県" in text:
            pref_url = href
        elif "市区町村" in text:
            muni_url = href
    return pref_url, muni_url


def _find_header_row(rows: list[tuple]) -> int | None:
    for i, row in enumerate(rows[:5]):
        if row and "自治体名" in row:
            return i
    return None


def _find_sheet(sheetnames: list[str], marker: str) -> str | None:
    return next((s for s in sheetnames if marker in s), None)


def _load_workbook(url: str):
    return openpyxl.load_workbook(BytesIO(fetch(url)), data_only=True, read_only=True)


def _endorsing_parties(row: list, header: list) -> list[str]:
    parties = []
    for col_name, party in _PARTY_COLUMNS.items():
        if col_name not in header:
            continue
        idx = header.index(col_name)
        if idx < len(row) and row[idx] not in (None, ""):
            if party not in parties:
                parties.append(party)
    for col_name in _OTHER_COLUMNS:
        if col_name not in header:
            continue
        idx = header.index(col_name)
        note = row[idx] if idx < len(row) else None
        if not isinstance(note, str) or not _OTHER_NOTE_KEYWORDS.search(note):
            continue
        for pattern, party in _OTHER_NOTE_PARTIES:
            if re.search(pattern, note):
                if party not in parties:
                    parties.append(party)
                break
    return parties


def _date_column_index(header: list) -> int | None:
    for cand in _DATE_COLUMN_CANDIDATES:
        if cand in header:
            return header.index(cand)
    return None


def _parse_governor_sheet(wb, year: str) -> dict[str, Endorsement]:
    sheet_name = _find_sheet(wb.sheetnames, "知事")
    if sheet_name is None:
        raise ValueError(f"{year}年版: 知事シートが見つかりません({wb.sheetnames})")
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header_idx = _find_header_row(rows)
    if header_idx is None:
        raise ValueError(f"{year}年版: ヘッダ行(自治体名列)が見つかりません")
    header = list(rows[header_idx])

    name_idx = header.index("自治体名")
    date_idx = _date_column_index(header)
    party_idx = header.index("党派") if "党派" in header else None

    result: dict[str, Endorsement] = {}
    rivals: dict[str, list[str]] = {}
    current_pref: str | None = None
    is_winner_row = False
    for row in rows[header_idx + 1 :]:
        row = list(row)
        if name_idx < len(row) and row[name_idx] is not None:
            current_pref = row[name_idx]
            is_winner_row = True  # 各都道府県ブロックの先頭行(得票数最多)が当選者
        else:
            is_winner_row = False
        if current_pref is None:
            continue
        if not is_winner_row:
            _add_unique(rivals.setdefault(current_pref, []), _endorsing_parties(row, header))
            continue
        result[current_pref] = Endorsement(
            name=current_pref,
            formal_party=(row[party_idx] if party_idx is not None else None) or "無所属",
            endorsing_parties=_endorsing_parties(row, header),
            vote_date=str(row[date_idx]) if date_idx is not None and row[date_idx] else "",
            year=year,
        )
    _attach_rivals(result, rivals)
    return result


def _add_unique(target: list[str], items: list[str]) -> None:
    for p in items:
        if p not in target:
            target.append(p)


def _attach_rivals(result: dict[str, Endorsement], rivals: dict[str, list[str]]) -> None:
    for name, parties in rivals.items():
        if name in result:
            result[name].rival_endorsing_parties = parties


def _parse_municipal_sheet(wb, year: str) -> dict[str, Endorsement]:
    sheet_name = _find_sheet(wb.sheetnames, "首長")
    if sheet_name is None:
        raise ValueError(f"{year}年版: 首長シートが見つかりません({wb.sheetnames})")
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header_idx = _find_header_row(rows)
    if header_idx is None:
        raise ValueError(f"{year}年版: ヘッダ行(自治体名列)が見つかりません")
    header = list(rows[header_idx])

    name_idx = header.index("自治体名")
    date_idx = _date_column_index(header)
    party_idx = header.index("党派") if "党派" in header else None
    elected_idx = header.index("当落") if "当落" in header else None

    result: dict[str, Endorsement] = {}
    rivals: dict[str, list[str]] = {}
    current_name: str | None = None
    is_winner_row = False
    for row in rows[header_idx + 1 :]:
        row = list(row)
        if name_idx < len(row) and row[name_idx]:
            current_name = row[name_idx]
        if elected_idx is not None:
            # 「当落」列がある年版: 「当」の行だけを採用する。
            is_winner_row = elected_idx < len(row) and row[elected_idx] == "当"
        else:
            # 無い年版: 各自治体ブロックの先頭行(得票数最多)を当選者とみなす。
            is_winner_row = name_idx < len(row) and row[name_idx] is not None
        if current_name is None:
            continue
        if not is_winner_row:
            _add_unique(rivals.setdefault(current_name, []), _endorsing_parties(row, header))
            continue
        result[current_name] = Endorsement(
            name=current_name,
            formal_party=(row[party_idx] if party_idx is not None else None) or "無所属",
            endorsing_parties=_endorsing_parties(row, header),
            vote_date=str(row[date_idx]) if date_idx is not None and row[date_idx] else "",
            year=year,
        )
    _attach_rivals(result, rivals)
    return result


def governor_endorsements_by_year(years_back: int = 8) -> dict[str, dict[str, Endorsement]]:
    """{年版: {都道府県名: Endorsement}}。年版を1つに合算せず個別に保持する
    (前任者が実際に選挙で選ばれた年のデータを後から引けるようにするため、
    [[seiryoku_shisu_design]]の「前任者の年版判定」参照)。
    """
    result: dict[str, dict[str, Endorsement]] = {}
    for year, page_url in discover_years()[-years_back:]:
        pref_url, _ = _discover_xlsx_urls(page_url)
        if pref_url is None:
            continue
        try:
            wb = _load_workbook(pref_url)
            result[year] = _parse_governor_sheet(wb, year)
        except Exception as e:
            warnings.warn(f"jichisoken {year}年版(知事)の解析に失敗: {e}")
    apply_corrections(result, load_corrections())
    return result


def load_corrections(path: Path = CORRECTIONS_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def apply_corrections(by_year: dict[str, dict[str, Endorsement]], corrections: dict) -> None:
    """明らかな誤記(推薦の印が隣の候補者の行にずれている等)を人手の修正で置き換える。

    キーは「自治体名|当選年」。修正の根拠はdata/meibo_corrections.jsonに出典とともに残す。
    """
    for key, fix in corrections.items():
        name, vote_year = key.split("|")
        for entries in by_year.values():
            entry = entries.get(name)
            if entry is not None and extract_vote_year(entry.vote_date) == vote_year:
                entry.endorsing_parties = list(fix["endorsing_parties"])
                entry.rival_endorsing_parties = list(fix["rival_endorsing_parties"])


def municipal_head_endorsements_by_year(years_back: int = 8) -> dict[str, dict[str, Endorsement]]:
    """{年版: {市区町村名: Endorsement}}。governor_endorsements_by_yearと同じ理由で
    年版ごとに分けて保持する。
    """
    result: dict[str, dict[str, Endorsement]] = {}
    for year, page_url in discover_years()[-years_back:]:
        _, muni_url = _discover_xlsx_urls(page_url)
        if muni_url is None:
            continue
        try:
            wb = _load_workbook(muni_url)
            result[year] = _parse_municipal_sheet(wb, year)
        except Exception as e:
            warnings.warn(f"jichisoken {year}年版(市区町村)の解析に失敗: {e}")
    return result


def merge_latest(by_year: dict[str, dict[str, Endorsement]]) -> dict[str, Endorsement]:
    """年版を古い順に合算し、複数年版に登場する自治体は新しい年版(=直近の選挙)で
    上書きする({自治体名: Endorsement}、「最新の既知の状態」だけが必要な場合に使う)。
    """
    result: dict[str, Endorsement] = {}
    for year in sorted(by_year):
        result.update(by_year[year])
    return result


def governor_endorsements(years_back: int = 8) -> dict[str, Endorsement]:
    """{都道府県名: Endorsement}。公開されている年版を古い順に合算し、複数年版に
    登場する都道府県は新しい年版(=直近の選挙)で上書きする。
    """
    return merge_latest(governor_endorsements_by_year(years_back))


def municipal_head_endorsements(years_back: int = 8) -> dict[str, Endorsement]:
    """{市区町村名: Endorsement}。公開されている年版を古い順に合算し、複数年版に
    登場する市区町村は新しい年版(=直近の選挙)で上書きする。
    """
    return merge_latest(municipal_head_endorsements_by_year(years_back))


def extract_vote_year(vote_date: str | None) -> str | None:
    """Endorsement.vote_dateやgo2senkyoのvote_dateから先頭の4桁年を取り出す。
    年版によって表記("2019年"/"2020-09-08 00:00:00"/"2021/04/11"等)が
    まちまちなため、フォーマットに依存せず年だけを比較する([[seiryoku_shisu_design]]参照)。
    """
    if not vote_date:
        return None
    m = re.search(r"(\d{4})", vote_date)
    return m.group(1) if m else None


def endorsement_for_vote_year(
    by_year: dict[str, dict[str, Endorsement]], name: str, vote_date: str | None
) -> Endorsement | None:
    """nameが実際に選ばれた選挙(vote_date)と同じ年のEndorsementを、年版を問わず
    全年版から探して返す。「都道府県知事」シートは各都道府県の現職の実際の当選年を
    そのまま記録している(年版自体が示す期間の選挙だけとは限らない)ことを実データで
    確認済みなので、年版のラベルではなくEndorsement自身のvote_dateの年で照合する。
    見つからなければNone(呼び出し側は形式上の届出政党のみで判定する)。
    """
    target = extract_vote_year(vote_date)
    if target is None:
        return None
    for entries in by_year.values():
        entry = entries.get(name)
        if entry is not None and extract_vote_year(entry.vote_date) == target:
            return entry
    return None
