"""共通のダウンロード・キャッシュユーティリティ。"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_UA = "seiryoku-shisu-bot/0.1 (+https://github.com/; personal research project)"

# ホストごとの最小リクエスト間隔(秒)。go2senkyo.comはボット検知(AWS WAF)で
# 短時間に連続アクセスするとチャレンジ応答(HTTP 202・本文空)を返すことを確認済み
# ([[seiryoku_shisu_design]]参照)なので、他のホストより間隔を空ける。
# 4.0秒では約1788件中40件に1件程度の頻度でWAFチャレンジが発生し、そのたびに
# 11分のクールダウンが挟まるため、体感の実効速度は間隔だけから期待される
# 5分の1程度に落ち込んでいた(2026-09、更新クロールの実行速度をユーザーと
# 調査して発覚)。間隔を広げてブロック頻度自体を下げられるか試すため8.0秒に
# 変更した。
_MIN_INTERVAL = {
    "go2senkyo.com": 8.0,
    "ja.wikipedia.org": 1.0,
}
_DEFAULT_INTERVAL = 0.2
_last_request_time: dict[str, float] = {}


class WafChallengeError(RuntimeError):
    """WAF等のボット対策によりチャレンジ応答・接続拒否が返された場合の例外。"""


# 実測で「WAFのブロックは約10分待てば解消する」ことを確認済み([[seiryoku_shisu_design]]
# 参照)。ブロックされた直後に短い間隔で再試行しても無駄なので、この時間だけ待つ。
_WAF_COOLDOWN_SECONDS = 11 * 60

# ホストごとの「いつまでブロックされている可能性が高いか」の共有状態。1回のブロック検知
# につき1回だけこの時刻まで待てばよく、同じブロック期間中に発生した別のfetch呼び出し
# (例: 同じ自治体のhead/gikai両方が同時にブロックされているケース)が、それぞれ独立に
# フルの11分を待ち直して待ち時間が積み重なるのを防ぐ。
_blocked_until: dict[str, float] = {}


def _throttle(host: str) -> None:
    interval = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    last = _last_request_time.get(host, 0.0)
    wait = interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _last_request_time[host] = time.time()


def _wait_out_shared_block(host: str) -> None:
    until = _blocked_until.get(host, 0.0)
    remaining = until - time.time()
    if remaining > 0:
        # WAFブロックのクールダウン待ちは無言だと「なぜ遅いか」が全く分からなく
        # なるため、必ず出力する(2026-09、更新クロールが想定の5倍遅い原因を
        # 調査した際にユーザー指摘——このprintが無く、実際に起きていても
        # ログに一切残らなかった)。
        print(f"[waf] {host}への次のアクセスまで{remaining:.0f}秒待機します", flush=True)
        time.sleep(remaining)


def fetch(url: str, *, force: bool = False, retries: int = 1, wait: bool = True) -> bytes:
    """URLをダウンロードし、cache/にキャッシュする。二回目以降はキャッシュを返す。

    WAFのチャレンジ応答(HTTP 202・x-amzn-waf-actionヘッダ)、および接続そのものが
    拒否される場合(WAFがより強くブロックしていると思われるConnectionError等)の
    両方を検知した場合は、約10分のクールダウンを1回挟んで指定回数まで再試行し、
    それでも解消しなければ例外を送出する。
    """
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    path = CACHE_DIR / f"{key}{suffix}"
    if path.exists() and not force:
        return path.read_bytes()

    host = urlparse(url).hostname or ""
    if not wait and _blocked_until.get(host, 0.0) > time.time():
        raise WafChallengeError(f"{host}はクールダウン中のため取得しません: {url}")
    _wait_out_shared_block(host)
    for attempt in range(retries + 1):
        _throttle(host)
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[waf] {host}への接続エラーを検知、{_WAF_COOLDOWN_SECONDS}秒のクールダウンに入ります: {e}", flush=True)
            _blocked_until[host] = time.time() + _WAF_COOLDOWN_SECONDS
            if attempt < retries:
                _wait_out_shared_block(host)
                continue
            raise WafChallengeError(f"接続エラーが解消しませんでした: {url}: {e}") from e
        if resp.headers.get("x-amzn-waf-action") == "challenge":
            print(f"[waf] {host}のWAFチャレンジ応答を検知、{_WAF_COOLDOWN_SECONDS}秒のクールダウンに入ります", flush=True)
            _blocked_until[host] = time.time() + _WAF_COOLDOWN_SECONDS
            if attempt < retries:
                _wait_out_shared_block(host)
                continue
            raise WafChallengeError(f"WAFチャレンジが解消しませんでした: {url}")
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return resp.content
    raise WafChallengeError(f"WAFチャレンジが解消しませんでした: {url}")
