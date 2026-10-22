from seiryoku import charts, national_elections
from seiryoku.fetch import diet_history, jichisoken
from seiryoku.fetch.diet_history import Snapshot


def _fixture_history():
    return [
        {"year": 2022, "month": m, "C_p": {"自由民主党": 0.6, "公明党": 0.4},
         "n_events": 1, "level_weight_share": {"衆院選": 1.0} if m == 7 else {"市区町村議会": 1.0}}
        for m in range(1, 13)
    ]


def test_generate_main_chart_writes_a_png(tmp_path, monkeypatch):
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history",
                         lambda: [Snapshot(session="a", date="2022-07-10", seats={})])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}})

    out_path = tmp_path / "chart.png"
    result = charts.generate_main_chart(_fixture_history(), out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_main_chart_adds_national_focus_panel_when_given_the_extra_series(tmp_path, monkeypatch):
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history",
                         lambda: [Snapshot(session="a", date="2022-07-10", seats={})])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}})

    history = _fixture_history()
    national_history = [
        {"year": h["year"], "month": h["month"], "C_p": {"自由民主党": 0.7, "公明党": 0.3}}
        for h in history
    ]
    local_history = [
        {"year": h["year"], "month": h["month"], "C_p": {"自由民主党": 0.4, "公明党": 0.6}}
        for h in history
    ]

    out_path = tmp_path / "chart.png"
    result = charts.generate_main_chart(
        history, out_path=out_path, national_history=national_history, local_history=local_history,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_main_chart_adds_executive_focus_panel_when_given_the_extra_series(tmp_path, monkeypatch):
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history",
                         lambda: [Snapshot(session="a", date="2022-07-10", seats={})])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}})

    history = _fixture_history()
    executive_history = [
        {"year": h["year"], "month": h["month"], "C_p": {"自由民主党": 0.7, "公明党": 0.3}}
        for h in history
    ]
    local_assembly_history = [
        {"year": h["year"], "month": h["month"], "C_p": {"自由民主党": 0.4, "公明党": 0.6}}
        for h in history
    ]

    out_path = tmp_path / "chart.png"
    result = charts.generate_main_chart(
        history, out_path=out_path,
        executive_history=executive_history, local_assembly_history=local_assembly_history,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_main_chart_adds_both_focus_panels_together(tmp_path, monkeypatch):
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history",
                         lambda: [Snapshot(session="a", date="2022-07-10", seats={})])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}})

    history = _fixture_history()

    def _series(a, b):
        return [{"year": h["year"], "month": h["month"], "C_p": {"自由民主党": a, "公明党": b}} for h in history]

    out_path = tmp_path / "chart.png"
    result = charts.generate_main_chart(
        history, out_path=out_path,
        national_history=_series(0.7, 0.3), local_history=_series(0.4, 0.6),
        executive_history=_series(0.6, 0.4), local_assembly_history=_series(0.3, 0.7),
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_executive_focus_chart_writes_a_png(tmp_path):
    executive_history = [
        {"year": 2022 + i, "month": 1, "C_p": {"自由民主党": 0.2, "公明党": 0.5 + 0.01 * i}}
        for i in range(4)
    ]
    local_assembly_history = [
        {"year": 2022 + i, "month": 1, "C_p": {"自由民主党": 0.5, "公明党": 0.1}}
        for i in range(4)
    ]

    out_path = tmp_path / "executive_focus.png"
    result = charts.generate_executive_focus_chart(
        executive_history, local_assembly_history, out_path=out_path,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_party_matrix_chart_writes_a_png(tmp_path):
    out_path = tmp_path / "party_matrix.png"
    result = charts.generate_party_matrix_chart(out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_national_local_scatter_writes_a_png(tmp_path):
    national_history = [
        {"year": 2022 + i, "month": 1, "C_p": {"自由民主党": 0.5 + 0.01 * i, "公明党": 0.1}}
        for i in range(4)
    ]
    local_history = [
        {"year": 2022 + i, "month": 1, "C_p": {"自由民主党": 0.2, "公明党": 0.15 + 0.01 * i}}
        for i in range(4)
    ]

    out_path = tmp_path / "scatter.png"
    result = charts.generate_national_local_scatter(national_history, local_history, out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_national_local_scatter_skips_parties_with_too_few_shared_months(tmp_path):
    national_history = [{"year": 2022, "month": 1, "C_p": {"自由民主党": 0.5, "新党": 0.1}}]
    local_history = [{"year": 2022, "month": 1, "C_p": {"自由民主党": 0.2, "新党": 0.2}}]

    out_path = tmp_path / "scatter.png"
    charts.generate_national_local_scatter(national_history, local_history, out_path=out_path)

    assert out_path.exists()


def test_provisional_start_derives_may_of_latest_jichisoken_version(monkeypatch):
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2023": {}, "2025": {}})
    assert charts._provisional_start() == (2025, 5)


def test_provisional_start_returns_none_when_no_data(monkeypatch):
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    assert charts._provisional_start() is None


def test_check_provisional_start_update_flags_change_only_once(monkeypatch, tmp_path):
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2023": {}, "2025": {}})
    monkeypatch.setattr(charts, "PROVISIONAL_START_STATE_PATH", tmp_path / "state.json")

    value, changed = charts.check_provisional_start_update()
    assert value == (2025, 5)
    assert changed is True

    # 同じ値のまま2回目を実行すると、もう変化したとは報告されない
    value, changed = charts.check_provisional_start_update()
    assert value == (2025, 5)
    assert changed is False


def test_check_provisional_start_update_detects_new_version(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(charts, "PROVISIONAL_START_STATE_PATH", state_path)
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}})
    charts.check_provisional_start_update()

    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {"2025": {}, "2026": {}})
    value, changed = charts.check_provisional_start_update()
    assert value == (2026, 5)
    assert changed is True


def test_unified_local_election_months_every_four_years_in_april():
    months = charts._unified_local_election_months(2018, 2027)
    assert (2019, 4) in months
    assert (2023, 4) in months
    assert (2027, 4) in months
    assert (2021, 4) not in months


def _trend_result():
    taus = [f"{y}-{m:02d}-28" for y in (2024, 2025) for m in range(1, 13)]
    parties = {}
    for k, p in enumerate(["自由民主党", "参政党"]):
        x = 0.5 if k == 0 else 0.03
        parties[p] = {
            "series": [{"tau": t, "x": x, "x_lo": x / 2, "x_hi": min(2 * x, 0.99)} for t in taus],
            "movement": [{"tau": t, "市区町村議会": 0.1, "衆院選": 0.3 if t == "2024-10-28" else 0.0} for t in taus],
            "national": [{"vote_date": "2024-10-27", "level": "衆院選", "y": x},
                         {"vote_date": "2025-07-20", "level": "参院選", "y": x}],
            "alpha": {"参院選": -0.2}, "params": {"san_load": 0.73},
        }
    return {"tau": taus[-1], "parties": parties}


def test_generate_trend_charts_write_pngs(tmp_path, monkeypatch):
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history",
                         lambda: [Snapshot(session="a", date="2025-07-20", seats={})])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    r = _trend_result()
    v = {"parties": {"参政党": [{"vote_date": "2025-07-20", "level": "参院選", "y": 0.03,
                                "pred": 0.04, "lo": 0.02, "hi": 0.08}]}}
    a = charts.generate_trend_chart(r, out_path=tmp_path / "a.png")
    b = charts.generate_trend_by_party_chart(r, v, out_path=tmp_path / "b.png")
    assert a.stat().st_size > 0 and b.stat().st_size > 0
    c = charts.generate_trend_support_scatter(
        r, {"updated": "2026-10-06", "S_p": {"無党派": 40.0, "自由民主党": 30.0, "参政党": 4.0, "日本保守党": 1.0}},
        out_path=tmp_path / "c.png")
    assert c.stat().st_size > 0


def test_generate_trend_support_diff_scatter_writes_png(tmp_path):
    pts = [{"party": p, "from": "2026-09-30", "to": "2026-10-31", "dx": dx, "ds": ds}
           for p, dx, ds in [("自由民主党", 3.0, 1.5), ("参政党", -1.0, -0.8), ("立憲民主党", 0.5, 0.0)]]
    diffs = {"horizon": 3, "points": pts, "n": 3, "periods": [("2026-09-30", "2026-10-31")],
             "pearson": {"r": 0.9, "p": 0.1, "n": 3, "ci95": None}, "spearman": {"rho": 1.0, "p": 0.0, "n": 3}}
    out = charts.generate_trend_support_diff_scatter(diffs, out_path=tmp_path / "d.png")
    assert out.stat().st_size > 0


def test_generate_trend_model_diagram_writes_a_png(tmp_path):
    out = charts.generate_trend_model_diagram(out_path=tmp_path / "m.png")
    assert out.stat().st_size > 0
