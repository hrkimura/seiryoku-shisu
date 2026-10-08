"""結合テスト: モックを使わず、cli.run()を実データ(ネットワーク越し、util.fetchの
キャッシュ経由)で通しで実行し、各構成要素が正しく組み合わさることを確認する。
"""
from datetime import date
from pathlib import Path

from seiryoku import coverage_history
from seiryoku.cli import run


def test_run_produces_consistent_indices():
    result = run()

    assert result["C_p"], "C_pが空です"

    c_p_total = sum(result["C_p"].values())
    assert 0.99 <= c_p_total <= 1.01, f"C_pの合計が1にならない: {c_p_total}"

    assert all(v >= 0 for v in result["C_p"].values())

    # 無所属・諸派は必ず1つの勢力として含まれる([[seiryoku_shisu_design]]の設計方針)
    assert "無所属" in result["C_p"]

    assert result["C_p_coverage"]["known_jurisdictions"] > 0
    assert result["municipal_registry_coverage"]["total_jichitai"] == 1788

    turnover_share = result["W_p_turnover"]["W_p"]
    turnover_total = sum(turnover_share.values())
    assert 0.99 <= turnover_total <= 1.01 or turnover_total == 0, (
        f"W_p^turnoverの合計が1にならない: {turnover_total}"
    )
    assert all(v >= 0 for v in turnover_share.values())

    # net_share(W_p - L_pをsum sqrt(P)で正規化)は1イベントごとにゼロサムなので、
    # 全政党で合計すると常に0
    net_total = sum(result["W_p_turnover"]["net_share"].values())
    assert abs(net_total) < 1e-6, f"net_shareの合計が0にならない: {net_total}"

    # run()のたびにその日のカバレッジが記録される(note記事のグラフ用)
    # 月次実行の図はデータ月ごとのディレクトリに出力され、記事用のnotes/images/は触らない
    image_dir = Path(result["monthly_index"]["image_dir"])
    assert "notes" not in image_dir.parts
    for name in ("cp_and_level_share.png", "national_vs_local.png", "executive_focus.png"):
        assert (image_dir / name).exists()

    # 前月差の分解は、無所属を除いて正規化した割合の差なので、全政党の合計はほぼ0になる
    att = result["monthly_index"]["attribution"]
    assert abs(sum(att["total"].values())) < 1e-9
    for p in att["total"]:
        assert abs(att["total"][p] - att["entered"].get(p, 0.0) - att["exited"].get(p, 0.0)) < 1e-9

    today = date.today().isoformat()
    history = coverage_history.load_history()
    assert any(h["date"] == today for h in history)
