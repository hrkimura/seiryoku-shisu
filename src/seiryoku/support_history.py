"""政党支持率$S_p(t)$のスナップショットをdata/に積み上げ、月次系列に変換する。

cp_history.pyと違い、こちらは追記方式である。取得元が現在値しか公開しておらず
(fetch/miharu.py参照)、過去の値を後から取り直せないため、一度記録した値は消さない。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
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


def _nearest_month_end(d: date) -> date:
    this_end = date(d.year + d.month // 12, d.month % 12 + 1, 1) - timedelta(days=1)
    prev_end = date(d.year, d.month, 1) - timedelta(days=1)
    return this_end if (this_end - d) < (d - prev_end) else prev_end


def value_at(history: list[dict], day: date) -> dict | None:
    """月末dayの$S_p$を、更新日の前後の記録から求める。

    前後に記録があれば更新日の間隔で線形補間する(両方にある政党だけ)。片側にしかなければ
    最も近い記録を使うが、その記録の更新日に最も近い月末がdayである場合に限る。そうでない
    ときに使うと、別の月末の値として数えるべき記録を二重に使うことになる。
    戻り値は{"S_p", "method"("exact"/"interpolated"/"nearest"), "dates"(使った更新日), "offset_days"}。
    """
    snaps = sorted(history, key=lambda h: h["updated"])
    if not snaps:
        return None
    days = [date.fromisoformat(h["updated"]) for h in snaps]
    before = [i for i, d in enumerate(days) if d <= day]
    after = [i for i, d in enumerate(days) if d >= day]
    if before and after and days[before[-1]] == day:
        i = before[-1]
        return {"S_p": dict(snaps[i]["S_p"]), "method": "exact", "dates": [snaps[i]["updated"]], "offset_days": 0}
    if before and after:
        i, j = before[-1], after[0]
        w = (day - days[i]).days / (days[j] - days[i]).days
        a, b = snaps[i]["S_p"], snaps[j]["S_p"]
        return {
            "S_p": {p: (1 - w) * a[p] + w * b[p] for p in a if p in b},
            "method": "interpolated",
            "dates": [snaps[i]["updated"], snaps[j]["updated"]],
            "offset_days": min((day - days[i]).days, (days[j] - day).days),
        }
    i = before[-1] if before else after[0]
    if _nearest_month_end(days[i]) != day:
        return None
    return {"S_p": dict(snaps[i]["S_p"]), "method": "nearest", "dates": [snaps[i]["updated"]],
            "offset_days": abs((days[i] - day).days)}
