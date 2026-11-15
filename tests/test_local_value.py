import json
import math

import pytest

from seiryoku import local_value, monthly_article


def _row(arm, party, d, y, pred, logpdf=None):
    r = {"arm": arm, "party": party, "vote_date": d, "y": y, "pred": pred, "lo": y - 0.1, "hi": y + 0.1}
    if logpdf is not None:
        r["logpdf"] = logpdf
    return r


def test_paired_compares_the_same_party_and_election():
    rows = [_row("A", "p", "2024", 0.5, 0.5, -1.0), _row("B", "p", "2024", 0.5, 0.6, -2.0),
            _row("A", "q", "2024", 0.2, 0.3, -1.5), _row("B", "q", "2024", 0.2, 0.2, -1.0),
            _row("A", "r", "2024", 0.1, 0.1, -0.5), _row("B", "r", "2024", 0.1, 0.3, -3.0),
            # 片方にしかない党×選挙は比べない
            _row("A", "s", "2024", 0.1, 0.9, -9.0)]
    out = local_value.paired(rows, "A", "B")
    assert out["n"] == 3
    d = [0.0 - 0.01, 0.01 - 0.0, 0.0 - 0.04]
    assert out["se_diff"]["mean"] == pytest.approx(sum(d) / 3)
    assert out["se_diff"]["a_better"] == 2
    assert out["logpdf_diff"]["a_better"] == 2
    sd = math.sqrt(sum((x - sum(d) / 3) ** 2 for x in d) / 2)
    assert out["se_diff"]["t"] == pytest.approx((sum(d) / 3) / (sd / math.sqrt(3)))


def test_paired_skips_logpdf_when_a_row_lacks_it():
    rows = [_row("A", "p", "1", 0.5, 0.5), _row("B", "p", "1", 0.5, 0.6),
            _row("A", "q", "1", 0.5, 0.4), _row("B", "q", "1", 0.5, 0.6)]
    out = local_value.paired(rows, "A", "B")
    assert "se_diff" in out and "logpdf_diff" not in out


def test_summarize_keeps_the_order_of_first_appearance():
    rows = [_row("B", "p", "1", 0.5, 0.6, -1.0), _row("A", "p", "1", 0.5, 0.5, -2.0)]
    out = local_value.summarize(rows)
    assert [r["arm"] for r in out] == ["B", "A"]
    assert out[0]["rmse"] == pytest.approx(0.1)
    assert out[0]["coverage"] == 1.0 and out[1]["mean_logpdf"] == -2.0


def test_status_lines_are_reference_only(tmp_path):
    p = {"a": "v4", "b": "国政+世論調査", "n": 38, "se_diff": {"t": 0.3, "a_better": 13}}
    body = {"online": {"paired": [p]}, "interpolation": {"paired": [p]},
            "carryover": {"summary": [{"model": "v4", "t": 1.8}]}}
    path = tmp_path / "local_value_2026-11-15.json"
    path.write_text(json.dumps(body, ensure_ascii=False))
    lines = local_value.status_lines(path)
    assert len(lines) == 4 and all(l.startswith("[参考]") for l in lines)
    # 検証の結果だけでは、記事にしない月にClaudeを呼ばない
    warnings = monthly_article.collect_warnings("", [], None, False, False, None, local_value=lines)
    assert monthly_article.decide({"parties": {}}, 2026, 12, warnings)["needs_claude"] is False


def test_failure_of_the_check_needs_attention():
    warnings = monthly_article.collect_warnings("", [], None, False, False, None,
                                                local_value=["[要確認] 地方選挙の役割の検証に失敗した: X"])
    assert "[要確認] 地方選挙の役割の検証に失敗した: X" in warnings
