import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prefectures as pf  # noqa: E402


def test_councillors_uses_last_election_before_cutoff_and_prefecture_ranges():
    chains = {"1": [{"vote_date": "2023-04-09", "seats": {"参政党": 1, "自由民主党": 9}},
                    {"vote_date": "2025-09-01", "seats": {"参政党": 5, "自由民主党": 5}}],
              "5": [{"vote_date": "2024-01-01", "seats": {"自由民主党": 3}}],
              "10": [{"vote_date": "2025-07-20", "seats": {"参政党": 2, "無所属": 2}},
                     {"vote_date": "2021-07-20", "seats": {"参政党": 0, "無所属": 4}}],
              "12": [{"vote_date": "2026-01-01", "seats": {"参政党": 9}}]}
    c = pf.councillors(chains, [1, 10], cutoff="2025-07-20")
    assert c.loc[1, "sansei"] == 1 and c.loc[1, "seats"] == 13
    assert c.loc[2, "sansei"] == 2 and c.loc[2, "seats"] == 4
    assert list(c.index) == [1, 2]


def test_prefecture_effects_recover_centered_constants():
    rng = np.random.default_rng(0)
    n = 300
    t = pd.DataFrame({"pref": np.repeat([1, 2, 3], 100), "valid25": 1.0, "x": rng.normal(size=n)})
    t["d_share"] = 2.0 * t["x"] + t["pref"].map({1: 1.0, 2: 2.0, 3: 6.0}) + rng.normal(scale=0.01, size=n)
    pe = pf.prefecture_effects(t, ["x"])
    assert pe.loc[1] == pytest.approx(-2.0, abs=0.02) and pe.loc[3] == pytest.approx(3.0, abs=0.02)
    assert np.average(pe, weights=[100, 100, 100]) == pytest.approx(0.0, abs=1e-9)


def test_prefecture_table_and_explain():
    t = pd.DataFrame({"pref": [1, 1, 2, 3], "votes25": [10.0, 10.0, 30.0, 5.0], "valid25": [100.0, 100.0, 100.0, 100.0],
                      "d_ldp": [-2.0, -4.0, -1.0, 0.0], "pop25": [1e6, 1e6, 1e6, 2e6]})
    dist = pd.Series({1: 15.0, 2: 20.0, 3: 5.0})
    council = pd.DataFrame({"sansei": [4, 1]}, index=[1, 2])
    p = pf.prefecture_table(t, dist, council)
    assert p.loc[1, "pr25"] == pytest.approx(10.0) and p.loc[1, "cand_pull"] == pytest.approx(5.0)
    assert p.loc[1, "d_ldp"] == pytest.approx(-3.0) and p.loc[2, "cand_pull"] == pytest.approx(-10.0)
    assert p.loc[1, "council_per_m"] == pytest.approx(2.0) and p.loc[3, "council_per_m"] == 0
    rng = np.random.default_rng(1)
    q = pd.DataFrame({"a": rng.normal(size=47), "b": rng.normal(size=47)}, index=range(1, 48))
    eff = 0.5 * q["a"] - 0.2 * q["b"] + rng.normal(scale=0.01, size=47)
    r, r2 = pf.explain(q, eff, ["a", "b"])
    assert r.loc["a", "coef"] == pytest.approx(0.5, abs=0.01) and r.loc["b", "coef"] == pytest.approx(-0.2, abs=0.01)
    assert r2 > 0.99 and (r["lo"] < r["coef"]).all() and (r["p"] < 0.01).all()


def test_district_share_excludes_total_row(monkeypatch):
    d = pd.DataFrame({"pref": [1, 1, 1, 2, 2], "party": ["参政党", "自由民主党", "合計", "参政党", "合計"],
                      "votes": [10.0, 30.0, 40.0, 5.0, 5.0]})
    monkeypatch.setattr(pf.el, "pref_party_district", lambda path: d)
    s = pf.district_share(Path("x"))
    assert s.loc[1] == pytest.approx(25.0) and s.loc[2] == pytest.approx(100.0)
