import pytest

from seiryoku import attribution
from seiryoku.monthly_index import MonthEntries


def _me():
    # 2窓(window_months=2)で、2024-01が外れ、2024-03が入る。
    entries = {
        (2024, 1): [("市区町村長", 100.0, 1.0, {"自由民主党": 1.0})],
        (2024, 2): [("市区町村長", 100.0, 1.0, {"公明党": 1.0})],
        (2024, 3): [
            ("市区町村長", 100.0, 1.0, {"日本共産党": 1.0}),
            ("市区町村長", 400.0, 1.0, {"無所属": 1.0}),
        ],
    }
    labels = {ym: [{"name": f"{ym}-{i}", "level": e[0], "vote_date": None} for i, e in enumerate(es)]
              for ym, es in entries.items()}
    return MonthEntries(entries=entries, labels=labels, months=sorted(entries))


def test_total_change_equals_entered_plus_exited():
    d = attribution.decompose(_me(), (2024, 3), window_months=2)
    for p in d["total"]:
        assert d["total"][p] == pytest.approx(d["entered"].get(p, 0.0) + d["exited"].get(p, 0.0))
    assert d["exiting_month"] == {"year": 2024, "month": 1}
    assert d["total"]["自由民主党"] == pytest.approx(-0.5)
    assert d["total"]["日本共産党"] == pytest.approx(0.5)


def test_independent_winner_does_not_move_normalized_shares():
    d = attribution.decompose(_me(), (2024, 3), window_months=2)
    top = {r["name"]: r for r in d["top_entered"]}
    assert all(abs(v) < 1e-12 for v in top["(2024, 3)-1"]["effect"].values())
    assert d["top_entered"][0]["name"] == "(2024, 3)-0"


def test_exited_election_effect_is_measured_by_removing_it_alone():
    d = attribution.decompose(_me(), (2024, 3), window_months=2)
    exited = d["top_exited"][0]
    assert exited["name"] == "(2024, 1)-0"
    assert exited["effect"]["自由民主党"] == pytest.approx(-0.5)
