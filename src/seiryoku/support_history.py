"""政党支持率$S_p(t)$のスナップショットをdata/に積み上げ、月次系列に変換する。

cp_history.pyと違い、こちらは追記方式である。取得元が現在値しか公開しておらず
(fetch/miharu.py参照)、過去の値を後から取り直せないため、一度記録した値は消さない。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "support_history.json"


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def record_snapshot(snapshot: dict, fetched: str | None = None) -> bool:
    """取得結果を1件追記する。同じ更新日の値が既にあれば何もしない。

    戻り値は新しく追記したかどうか。
    """
    history = load_history()
    if any(h["updated"] == snapshot["updated"] for h in history):
        return False
    history.append({
        "updated": snapshot["updated"],
        "fetched": fetched or date.today().isoformat(),
        "S_p": snapshot["S_p"],
        "source": snapshot.get("source"),
    })
    history.sort(key=lambda h: h["updated"])
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def monthly_series(history: list[dict]) -> dict[tuple[int, int], dict[str, float]]:
    """月tの$S_p(t)$として、更新日が月t内にある最後のスナップショットを採る。

    $C_p(t)$は月tの末までの12か月窓なので、月末に最も近い値を対応させる。
    スナップショットのない月は欠測として含めない(補間はしない)。
    """
    by_month: dict[tuple[int, int], dict] = {}
    for h in sorted(history, key=lambda h: h["updated"]):
        y, m, _ = (int(x) for x in h["updated"].split("-"))
        by_month[(y, m)] = h["S_p"]
    return by_month
