"""政党支持率$S_p(t)$を取得して、新しい更新日の値なら追記する(毎日のタイマーから実行する)。

取得元は不定期に更新され、前の値は上書きされて残らない。月次実行の1回だけでは更新を
取りこぼすうえ、月末とのずれも大きくなるので、月次実行とは別に毎日取りに行く。
通信に失敗した日は終了コード1で終わるだけで、翌日の実行がそのまま取り直す。
"""
from __future__ import annotations

import sys

from . import support_history
from .fetch import miharu


def main() -> int:
    try:
        snapshot = miharu.fetch_support()
    except Exception as e:
        print(f"[取得失敗] {type(e).__name__}: {e}")
        return 1
    if support_history.record_snapshot(snapshot):
        print(f"{snapshot['updated']}更新の値を追記しました")
    else:
        print(f"{snapshot['updated']}更新の値は記録済みです")
    if snapshot["unknown_labels"]:
        print(f"[要確認] 略称表にない政党名: {', '.join(snapshot['unknown_labels'])}"
              "(fetch/miharu.pyのABBREVIATIONSに追加してください)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
