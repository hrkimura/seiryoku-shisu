"""総務省の参院選結果ページから、比例代表の開票区別・政党別得票数(都道府県ごとのExcel)を取得する。

取得済みのファイルは飛ばすので、通信が切れたら同じコマンドで続きから再開できる。
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.soumu.go.jp"
RAW = Path(__file__).resolve().parent / "data" / "raw"
ELECTIONS = {26: "2022", 27: "2025"}
UA = {"User-Agent": "Mozilla/5.0"}


def page_url(n: int, pref: int) -> str:
    return f"{BASE}/senkyo/senkyo_s/data/sangiin{n}/sangiin{n}_8_{pref:02d}.html"


def party_xlsx_link(html: str) -> str | None:
    """都道府県ページのHTMLから「政党別」のExcelへのリンクを返す。"""
    m = re.search(r'href="(/main_content/\d+\.xlsx?)"[^>]*>\s*政党別\s*<', html)
    return BASE + m.group(1) if m else None


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_all(retries: int = 3) -> list[str]:
    failed = []
    for n, year in ELECTIONS.items():
        out = RAW / f"sangiin_{year}"
        out.mkdir(parents=True, exist_ok=True)
        for pref in range(1, 48):
            dest = out / f"{pref:02d}.xlsx"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            for attempt in range(retries):
                try:
                    html = get(page_url(n, pref)).decode("cp932", errors="replace")
                    link = party_xlsx_link(html)
                    if link is None:
                        raise RuntimeError("政党別のリンクがない")
                    tmp = dest.with_suffix(".part")
                    tmp.write_bytes(get(link))
                    tmp.rename(dest)
                    break
                except Exception as e:  # noqa: BLE001 通信の失敗は種類を問わず再試行する
                    if attempt == retries - 1:
                        failed.append(f"{year} {pref:02d}: {e}")
                    time.sleep(5 * (attempt + 1))
            time.sleep(1)
    return failed


def fetch_offices(retries: int = 3) -> list[str]:
    """国土数値情報 P05(市町村役場等、2022年)を都道府県ごとに取得する。取得済みは飛ばす。"""
    out = RAW / "p05"
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for pref in range(1, 48):
        dest = out / f"P05-22_{pref:02d}.zip"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://nlftp.mlit.go.jp/ksj/gml/data/P05/P05-22/P05-22_{pref:02d}_GML.zip"
        for attempt in range(retries):
            try:
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(get(url))
                tmp.rename(dest)
                break
            except Exception as e:  # noqa: BLE001 通信の失敗は種類を問わず再試行する
                if attempt == retries - 1:
                    failed.append(f"P05 {pref:02d}: {e}")
                time.sleep(5 * (attempt + 1))
        time.sleep(1)
    return failed


def main() -> int:
    failed = fetch_all() + fetch_offices()
    for f in failed:
        print("失敗", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
