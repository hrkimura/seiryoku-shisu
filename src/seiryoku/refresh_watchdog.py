"""refresh_cli.pyの更新クロールが未完了なら自動で再開させる、日次のウォッチドッグ。

2026-09にユーザー指示「10日ぐらい、11時あたりに終わったかどうかチェックし、
終わってなければ自動実行」を受けて追加。判断は3通り:

- 既に完了済み(refresh_cli.is_done())なら何もしない。
- 既に実行中(refresh_cli.is_running())なら何もしない(二重起動を防ぐ)。
- どちらでもなければrefresh_cli.main()を呼んで再開する。ただし、いつまでも
  終わらない場合に無限に自動実行し続けるのを避けるため、試行回数を
  WATCHDOG_STATE_PATHに記録し、10回を超えたら諦めて手動確認を促す。回数は月ごとに数え直す
  (休止や再起動で止まるのは毎月のことなので、月をまたいで積み上げると、いずれ再開しなくなる)。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import refresh_cli

WATCHDOG_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "refresh_watchdog_state.json"
MAX_ATTEMPTS = 10


def _load_state() -> dict:
    if WATCHDOG_STATE_PATH.exists():
        return json.loads(WATCHDOG_STATE_PATH.read_text(encoding="utf-8"))
    return {"attempts": 0}


def _save_state(state: dict) -> None:
    WATCHDOG_STATE_PATH.parent.mkdir(exist_ok=True)
    WATCHDOG_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main(today: date | None = None) -> None:
    from .monthly_article import notify

    today = today or date.today()
    if refresh_cli.is_done():
        print("更新クロールは既に完了済みです。何もしません。")
        return
    if refresh_cli.is_running():
        print("更新クロールは既に実行中です。何もしません。")
        return

    cycle = f"{today.year:04d}-{today.month:02d}"
    state = _load_state()
    if state.get("cycle") != cycle:
        state = {"cycle": cycle, "attempts": 0}
    attempts = state.get("attempts", 0) + 1
    state["attempts"] = attempts
    _save_state(state)

    if attempts > MAX_ATTEMPTS:
        print(f"{MAX_ATTEMPTS}回試行しても完了しませんでした。自動再実行は打ち切ります。手動で確認してください。")
        # 月次実行はクロールの完了を待つので、ここで止まると今月の下書きはできない
        notify("更新クロールが終わりません",
               f"今月は{MAX_ATTEMPTS}回再開しても完了せず、月次実行と下書きが止まっています。"
               "journalctl --user -u seiryoku-shisu-refresh-watchdog で確かめてください")
        return

    print(f"更新クロールが未完了のため再開します(試行{attempts}/{MAX_ATTEMPTS}回目)。")
    refresh_cli.main()


if __name__ == "__main__":
    main()
