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
