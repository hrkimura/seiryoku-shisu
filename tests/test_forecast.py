import importlib.util

import pytest

from seiryoku import forecast

_TIMESFM_AVAILABLE = importlib.util.find_spec("timesfm") is not None


def test_should_forecast_true_when_next_month_is_election_linked():
    elections = [(2022, 7)]
    assert forecast.should_forecast(2022, 6, elections) is True  # 翌月7月が選挙月
    assert forecast.should_forecast(2023, 6, elections) is True  # 翌月7月が窓抜け月


def test_should_forecast_false_otherwise():
    elections = [(2022, 7)]
    assert forecast.should_forecast(2022, 7, elections) is False  # 翌月8月は絡まない
    assert forecast.should_forecast(2022, 1, elections) is False


def test_should_forecast_handles_december_rollover():
    elections = [(2023, 1)]
    assert forecast.should_forecast(2022, 12, elections) is True  # 翌月2023-01


def test_party_diff_series_computes_month_over_month_change():
    history = [
        {"C_p": {"自由民主党": 0.5}},
        {"C_p": {"自由民主党": 0.6}},
        {"C_p": {}},  # 欠測は0.0扱い
    ]
    assert forecast._party_diff_series(history, "自由民主党") == [
        pytest.approx(0.1), pytest.approx(-0.6),
    ]


def test_covariate_series_marks_election_month_and_window_exit():
    history = [
        {"year": 2022, "month": 6}, {"year": 2022, "month": 7}, {"year": 2022, "month": 8},
        {"year": 2023, "month": 6}, {"year": 2023, "month": 7}, {"year": 2023, "month": 8},
    ]
    elections = [(2022, 7)]
    cov = forecast._covariate_series(history, elections)
    # covはhistory[1:]に対応: 2022-07(+1), 2022-08(0), 2023-06(0), 2023-07(-1), 2023-08(0)
    assert cov == [1.0, 0.0, 0.0, -1.0, 0.0]


def test_forecast_next_month_returns_none_when_not_election_linked():
    history = [{"year": 2022, "month": m, "C_p": {}} for m in range(1, 14)]
    assert forecast.forecast_next_month(history, election_months=[]) is None


def test_forecast_next_month_returns_none_when_history_too_short():
    history = [{"year": 2022, "month": m, "C_p": {}} for m in range(1, 6)]
    # 翌月(2022-06)を選挙月にしても、min_context(既定12)未満ならNone
    assert forecast.forecast_next_month(history, election_months=[(2022, 6)]) is None


def test_forecast_next_month_returns_none_for_empty_history():
    assert forecast.forecast_next_month([]) is None


def test_direction_hint_returns_none_when_next_month_not_election_linked():
    assert forecast.direction_hint(2022, 1, election_months=[]) is None


def test_direction_hint_national_focused_rises_when_election_enters_window():
    elections = [(2022, 7)]
    hints = forecast.direction_hint(2022, 6, elections)  # 翌月7月が選挙月(+1)
    assert hints["自由民主党"] == 1  # 国政型は選挙が窓に入ると上がる
    assert hints["公明党"] == -1  # 地方型はその逆


def test_direction_hint_national_focused_falls_when_election_exits_window():
    elections = [(2022, 7)]
    hints = forecast.direction_hint(2023, 6, elections)  # 翌月7月が窓抜け月(-1)
    assert hints["自由民主党"] == -1
    assert hints["日本共産党"] == 1


def test_direction_hint_covers_only_adopted_parties():
    hints = forecast.direction_hint(2022, 6, [(2022, 7)])
    assert set(hints.keys()) == set(forecast.ADOPTED_PARTIES)


def test_direction_hint_handles_december_rollover():
    elections = [(2023, 1)]
    hints = forecast.direction_hint(2022, 12, elections)  # 翌月2023-01が選挙月
    assert hints is not None
    assert hints["自由民主党"] == 1


@pytest.mark.skipif(not _TIMESFM_AVAILABLE, reason="timesfmがインストールされていない環境ではスキップ")
def test_forecast_next_month_calls_timesfm_when_election_linked_and_context_is_enough():
    # 最終月を2022-12にし、その翌月(2023-01、選挙月そのもの)がshould_forecastの
    # 対象になるようにする。min_context(既定12)を超えるよう13か月分用意する。
    history = [{"year": 2021, "month": 12, "C_p": {"自由民主党": 0.5}}]
    history += [
        {"year": 2022, "month": m, "C_p": {"自由民主党": 0.5 + 0.01 * m}}
        for m in range(1, 13)
    ]
    result = forecast.forecast_next_month(history, election_months=[(2023, 1)])
    assert result is not None
    assert set(result.keys()) == set(forecast.ADOPTED_PARTIES)
