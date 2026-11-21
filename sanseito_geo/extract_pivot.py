"""在留外国人統計のテーブルデータ(Excel のデータモデル形式、2023年12月以降)から、
市区町村×国籍・地域×在留資格の人数を取り出して CSV にする。

pbixray が必要(pip install pbixray)。プロジェクトの環境とは別の環境で実行してよい。
"""
from __future__ import annotations

import sys
from pathlib import Path

from pbixray import PBIXRay

RAW = Path(__file__).resolve().parent / "data" / "raw" / "zairyu"


def extract(src: Path, dest: Path) -> None:
    m = PBIXRay(str(src))
    table = list(m.tables)[0]
    d = m.get_table(table)
    out = d.groupby(["市区町村コード", "国籍・地域", "在留資格"], as_index=False)["在留外国人数"].sum()
    out.to_csv(dest, index=False)


def main() -> int:
    for name in sys.argv[1:]:
        extract(RAW / f"{name}.xlsx", RAW / f"{name}_by_nat_status.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
