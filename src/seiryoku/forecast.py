"""TimesFM-3を使った翌月予測(design_document.tex \\S7.2「採用ルール」)。

バックテスト(2026-09)の結論:
- 国政選挙が絡まない月は、ゼロ予測を含むどの手法も追加の情報を持たない
  (ランダムウォークで近似できる)。よって予測自体を出さない。
- 国政選挙が絡む月は、個別政党の有意性ではなくプールした検定結果を優先し、
  バックテスト対象の6党全てにTimesFM-3を使う。

したがって`forecast_next_month`は、来月が国政選挙に絡まないなら`None`を返す
(ゼロという「予測値」を返すのではなく、そもそも予測しない——2026-09に
ユーザー指摘「ゼロ予測するぐらいなら何も出さないほうがいい」)。
"""
from __future__ import annotations

import json
from pathlib import Path

from . import national_elections

FORECAST_HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "forecast_history.json"

# バックテストで安定した時系列を持つと確認された6党のみを対象にする
# (中道改革連合・チームみらい・参政党・いのちの党は履歴不足等で対象外、
# note記事「予測とサプライズについて」参照)。
ADOPTED_PARTIES = [
    "自由民主党", "公明党", "立憲民主党", "日本維新の会", "国民民主党", "日本共産党",
]


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def should_forecast(year: int, month: int, election_months: list[tuple[int, int]]) -> bool:
    """(year, month)の翌月が「国政選挙が絡む月」かどうか。"""
    return national_elections.is_election_linked_month(*_next_month(year, month), election_months)


def _party_diff_series(history: list[dict], party: str) -> list[float]:
    """historyの月次差分を返す(先頭月は差分が定義できないため含めない)。
    欠測月(その政党の値が無い月)は0.0として扱う——バックテストで使った
    windowed12_dataと同じ穴埋め方針。"""
    values = [h["C_p"].get(party, 0.0) for h in history]
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _covariate_value_for_month(year: int, month: int, election_months: list[tuple[int, int]]) -> float:
    """バックテストで使ったcovariate(国政選挙が絡む月は非ゼロ)と同じ定義。
    月(year, month)自体が選挙月なら+1、その直前12か月に選挙があってその選挙が
    ちょうど窓から抜けるタイミングなら-1(両方に該当する場合は加算)。"""
    election_set = set(election_months)
    val = 0.0
    if (year, month) in election_set:
        val += 1.0
    if (year - 1, month) in election_set:  # 12か月前(=1年前の同じ月)が選挙月なら窓から抜ける
        val -= 1.0
    return val


def _covariate_series(history: list[dict], election_months: list[tuple[int, int]]) -> list[float]:
    """historyの各月の差分(_party_diff_seriesと同じ並び、先頭月は含めない)に
    対応するcovariateの列。"""
    months = [(h["year"], h["month"]) for h in history]
    return [_covariate_value_for_month(y, m, election_months) for y, m in months[1:]]


def forecast_next_month(
    history: list[dict],
    election_months: list[tuple[int, int]] | None = None,
    min_context: int = 12,
) -> dict[str, float] | None:
    """来月がADOPTED_PARTIESの予測対象月でなければNone。対象月なら
    {政党: 来月の差分の予測値}を返す。

    timesfmパッケージが無い環境でもimportエラーで落ちないよう、実際に使う
    タイミングまでimportを遅延する(データ取得・チャート生成はtimesfm無しでも
    動く必要があるため)。
    """
    if not history:
        return None
    history = sorted(history, key=lambda h: (h["year"], h["month"]))
    if election_months is None:
        election_months = national_elections.load_election_months()

    last = history[-1]
    if not should_forecast(last["year"], last["month"], election_months):
        return None
    if len(history) <= min_context:
        return None

    import numpy as np
    import timesfm

    model = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
    # past_future_covariatesはcontext_len+horizon分の長さが必要(将来分も既知の
    # covariateとして渡す設計のため)。予測対象月自身が選挙に絡むかどうかは
    # should_forecast()で確定済みなので、その月ぶんの値を1つ追加する。
    next_year, next_month = _next_month(last["year"], last["month"])
    future_cov = _covariate_value_for_month(next_year, next_month, election_months)
    cov_full = np.array(_covariate_series(history, election_months) + [future_cov], dtype=np.float32)

    result: dict[str, float] = {}
    for party in ADOPTED_PARTIES:
        diffs = _party_diff_series(history, party)
        ctx = np.array(diffs, dtype=np.float32)
        out = model.predict(context=ctx, horizon=1, past_future_covariates=cov_full)
        result[party] = float(out.forecast[0])
    return result


# 国政に強い/地方に強いという分類(design_document.tex \S7.3・note記事
# 「地方重視率」参照)。確定値期間・速報値期間のどちらで見ても全党一致する
# 安定した軸であることを確認済みなので定数として持つ(2026-09にユーザー提案の
# 「機械的な方向の目安」)。首長重視率(議会型・首長型)の軸は速報値期間を
# 混ぜると逆転する政党があり不安定なため、ここでは使わない。
NATIONAL_FOCUSED_PARTIES = {"自由民主党", "立憲民主党", "日本維新の会", "国民民主党"}
LOCAL_FOCUSED_PARTIES = {"公明党", "日本共産党"}


def direction_hint(
    year: int, month: int, election_months: list[tuple[int, int]] | None = None,
) -> dict[str, int] | None:
    """(year, month)が国政選挙が絡む月でなければNone。絡む月なら
    {政党: +1 or -1}で「上がる/下がる」の機械的な目安を返す。

    国政に強い政党は選挙が窓に参入する月(+1)で上がり退出する月(-1)で下がる、
    地方に強い政党はその逆、という符号ルールで、実際に66件(バックテスト
    対象6党×11イベント月)中53件(80%)が的中した(design_document.tex
    \\S7.3、二項検定で$p=3.6\\times10^{-7}$)。TimesFM-3の個別予測はこの
    機械的な効果を上回る「真の勢い」を検出できなかったため、2026-09に
    こちらへ切り替えた(forecast_next_month参照、コード自体は残してある)。
    """
    if election_months is None:
        election_months = national_elections.load_election_months()
    next_year, next_month = _next_month(year, month)
    covariate = _covariate_value_for_month(next_year, next_month, election_months)
    if covariate == 0:
        return None
    sign = 1 if covariate > 0 else -1
    hints: dict[str, int] = {}
    for party in ADOPTED_PARTIES:
        if party in NATIONAL_FOCUSED_PARTIES:
            hints[party] = sign
        elif party in LOCAL_FOCUSED_PARTIES:
            hints[party] = -sign
    return hints


def load_forecast_history() -> dict[str, dict[str, float]]:
    if FORECAST_HISTORY_PATH.exists():
        return json.loads(FORECAST_HISTORY_PATH.read_text(encoding="utf-8"))
    return {}


def record_forecast(year: int, month: int, forecast_values: dict[str, float]) -> dict[str, dict[str, float]]:
    """forecast_next_month()の結果を、予測対象の年月をキーにして保存する
    (翌月の実際の差分と答え合わせするため、draft.py参照)。"""
    history = load_forecast_history()
    history[f"{year:04d}-{month:02d}"] = forecast_values
    FORECAST_HISTORY_PATH.parent.mkdir(exist_ok=True)
    FORECAST_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history
