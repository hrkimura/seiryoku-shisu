import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_2024 as c  # noqa: E402
from profile_utas import wilson  # noqa: E402


def test_range_helper():
    assert c._r(6, 9) == frozenset({6, 7, 8, 9})


def test_share_with_and_without_valid_set():
    p, lo, hi, n = c.share(pd.Series([1, 2, 4, 5, 6, 99]), c.AGREE, c._r(1, 5))
    assert n == 4 and p == pytest.approx(50.0)
    wl, wh = wilson(2, 4)
    assert lo == pytest.approx(wl * 100) and hi == pytest.approx(wh * 100)
    p, _, _, n = c.share(pd.Series([1, 0, 0, 99]), frozenset({1}), None)
    assert n == 3 and p == pytest.approx(100 / 3)
    p, _, _, n = c.share(pd.Series([99, 99]), frozenset({1}), None)
    assert n == 0 and math.isnan(p)


def test_compare_groups_and_columns():
    d25 = pd.DataFrame({it.col25: [1, 4, 6, 1, 1] for it in c.ITEMS})
    d25["Q3"] = [c.SANSEI_2024, c.SANSEI_2024, c.SANSEI_2024, 1, 2]
    d26 = pd.DataFrame({it.col26: [2, 2, 5] for it in c.ITEMS})
    d26["Q4"] = [c.SANSEI_2025, 1, 13]
    t = c.compare(d25, d26)
    assert list(t.columns) == ["item", "group", "pct", "lo", "hi", "n"]
    g = t[t.item == "防衛力を強化すべき"].set_index("group")
    assert g.loc["2024年から参政党(29人)", "pct"] == pytest.approx(50.0)
    assert g.loc["2024年から参政党(29人)", "n"] == 2
    assert g.loc["2024年に他党", "pct"] == pytest.approx(100.0) and g.loc["2024年に他党", "n"] == 2
    assert g.loc["2025年に参政党", "pct"] == pytest.approx(100.0) and g.loc["2025年に他党", "n"] == 1
    assert len(t) == 4 * len(c.ITEMS)
