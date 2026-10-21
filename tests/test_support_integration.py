"""結合テスト: 実際のnote記事から支持率を取得し、実データの$C_p$履歴と組み合わせて
分析まで通す(ネットワーク越し)。書き込み先だけ一時ディレクトリに差し替える。"""
from seiryoku import cli, cp_history, support_analysis, support_history


def test_support_pipeline_with_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(support_history, "HISTORY_PATH", tmp_path / "support_history.json")
    monkeypatch.setattr(support_analysis, "RESULT_PATH", tmp_path / "support_analysis.json")

    out = cli._update_support_analysis(cp_history.load_history())

    assert out["fetch_error"] is None
    assert out["appended"] is True
    snap = support_history.load_history()[0]
    assert "自由民主党" in snap["S_p"] and "無党派" in snap["S_p"]
    assert 90 <= sum(snap["S_p"].values()) <= 110
    assert out["unknown_labels"] == []
    assert "自由民主党" in out["by_party"]
    assert (tmp_path / "support_analysis.json").exists()


def test_support_fetch_with_real_note(tmp_path, monkeypatch):
    from seiryoku import support_fetch

    monkeypatch.setattr(support_history, "HISTORY_PATH", tmp_path / "support_history.json")
    assert support_fetch.main() == 0
    assert support_fetch.main() == 0
    assert len(support_history.load_history()) == 1


def test_trend_support_diffs_with_real_v2_result(tmp_path):
    import json
    from datetime import date

    from seiryoku import charts
    from seiryoku.cli import OUTPUT_DIR

    path = sorted(OUTPUT_DIR.glob("trend/state_space_glm_2*.json"))[-1]
    result = json.loads(path.read_text())
    history = support_history.load_history()
    first = date.fromisoformat(history[0]["updated"])
    for h in support_analysis.DIFF_HORIZONS:
        d = support_analysis.trend_support_diffs(result, history, h)
        # 最初の記録より前の月末で使えるのは、その記録に最も近い月末だけ
        assert all(tau >= support_history._nearest_month_end(first).isoformat() for tau in d["alignment"])
        for p in d["points"]:
            assert p["party"] != "無党派"
        if d["points"]:
            assert charts.generate_trend_support_diff_scatter(d, out_path=tmp_path / f"d{h}.png").stat().st_size > 0
