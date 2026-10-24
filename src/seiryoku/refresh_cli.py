"""選挙結果キャッシュ(首長・議会それぞれのterm chain)の更新クロール。

turnover.build_term_chain_cache()・municipal_registry.build_gikai_term_chain_cache()は
「一度処理した自治体は二度と再取得しない」設計(中断・再開可能にするための恒久
done_ids)なので、新しい首長・議会選挙が実施されても自動では反映されない
(2026-09、沖縄県知事選で発覚)。

この対応として、対象の約1788自治体を毎回強制的に(util.fetchのキャッシュを
無視して)確認し直すrefresh_and_rebuild_*_term_chains()を用意した。当初は
go2senkyo.comのレート制限(4秒間隔)だけから首長・議会あわせて約4時間で
終わる想定だったが、実際には約40件に1件の頻度でWAFチャレンジが発生し、
そのたびに11分のクールダウンが挟まるため、実効速度は間隔だけから期待される
5分の1程度に落ち込むことが判明した(2026-09にユーザーと調査)。間隔を
8秒に広げてブロック頻度を下げる対策をしたが、それでも数時間〜十数時間かかる
可能性がある。記事生成(cli.py、月次)とは別のsystemdタイマーで、記事生成より
数日前倒しで実行する。

ロック・完了マーカー・再試行回数の管理により、この関数は「未完了なら再開、
実行中なら何もしない、完了済みなら何もしない」という自己管理型の設計になって
いる(2026-09にユーザー指示「10日ぐらい、終わっていなければ自動実行」を
受けて追加)。ウォッチドッグ(refresh_watchdog.py)から毎日呼び出す想定。
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from . import municipal_registry, turnover

LOCK_PATH = Path(__file__).resolve().parents[2] / "data" / "refresh_cli.lock"
DONE_MARKER_PATH = Path(__file__).resolve().parents[2] / "data" / "refresh_cli_done.json"
# 首長・議会それぞれの「今回のサイクルで完走済みか」を覚えておくマーカー。
# refresh_stale_term_chains()自体の中断・再開はチェックポイントで対応済みだが、
# 「首長側は既に完走し、議会側の途中でプロセスが落ちた」場合、進捗ファイルが
# 無い(=完走済みで綺麗に消えている)首長側を、議会側の再開のたびに毎回
# フルスキャンし直してしまう不具合があった(2026-09に発覚——約10時間分の
# 首長側スキャンが無駄に再実行された)。このマーカーで、既に完走した方の
# フェーズをサイクル内でスキップできるようにする。両方完了したら削除し、
# 次回サイクルでは両方とも改めてスキャンする。
HEAD_DONE_MARKER_PATH = Path(__file__).resolve().parents[2] / "data" / "refresh_cli_head_done.json"
GIKAI_DONE_MARKER_PATH = Path(__file__).resolve().parents[2] / "data" / "refresh_cli_gikai_done.json"


def is_running() -> bool:
    """LOCK_PATHに記録されたPIDがまだ生きていればTrue(実行中とみなす)。"""
    if not LOCK_PATH.exists():
        return False
    try:
        pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 生きてはいるが別ユーザー所有、実行中として扱う
    return True


def is_done() -> bool:
    """前回、首長・議会とも最後まで完走していればTrue。"""
    return DONE_MARKER_PATH.exists()


def done_at() -> date | None:
    """最後に完走した日。完了マーカーは次のサイクルの開始まで残るので、今月の分かどうかはこれで見る。"""
    if not DONE_MARKER_PATH.exists():
        return None
    return date.fromtimestamp(DONE_MARKER_PATH.stat().st_mtime)


def _clear_done_marker() -> None:
    if DONE_MARKER_PATH.exists():
        DONE_MARKER_PATH.unlink()


def main() -> None:
    LOCK_PATH.parent.mkdir(exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    _clear_done_marker()
    try:
        if HEAD_DONE_MARKER_PATH.exists():
            print("首長の選挙結果キャッシュは今回のサイクルで確認済みです。スキップします。")
            head_stats = json.loads(HEAD_DONE_MARKER_PATH.read_text(encoding="utf-8"))
        else:
            print("首長の選挙結果キャッシュを確認中...")
            head_stats = turnover.refresh_and_rebuild_term_chains()
            print(f"  確認: {head_stats['checked']}件, 更新対象: {len(head_stats['stale'])}件 {head_stats['stale']}")
            HEAD_DONE_MARKER_PATH.write_text(json.dumps(head_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        if GIKAI_DONE_MARKER_PATH.exists():
            print("議会の選挙結果キャッシュは今回のサイクルで確認済みです。スキップします。")
            gikai_stats = json.loads(GIKAI_DONE_MARKER_PATH.read_text(encoding="utf-8"))
        else:
            print("議会の選挙結果キャッシュを確認中...")
            gikai_stats = municipal_registry.refresh_and_rebuild_gikai_term_chains()
            print(f"  確認: {gikai_stats['checked']}件, 更新対象: {len(gikai_stats['stale'])}件 {gikai_stats['stale']}")
            GIKAI_DONE_MARKER_PATH.write_text(json.dumps(gikai_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        DONE_MARKER_PATH.write_text(
            json.dumps(
                {
                    "head_checked": head_stats["checked"],
                    "head_stale": head_stats["stale"],
                    "gikai_checked": gikai_stats["checked"],
                    "gikai_stale": gikai_stats["stale"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # 次回サイクルでは両方とも改めてスキャンするため、サイクル内スキップ用の
        # マーカーはここでリセットする。
        if HEAD_DONE_MARKER_PATH.exists():
            HEAD_DONE_MARKER_PATH.unlink()
        if GIKAI_DONE_MARKER_PATH.exists():
            GIKAI_DONE_MARKER_PATH.unlink()
        print("完了しました。")
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()


if __name__ == "__main__":
    main()
