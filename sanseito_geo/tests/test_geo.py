import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
import dataset  # noqa: E402
import elections as el  # noqa: E402
import foreign as fo  # noqa: E402
import geo  # noqa: E402
import units as un  # noqa: E402


def test_to_number_handles_commas_blanks_and_numbers():
    assert el.to_number("21,066.619") == pytest.approx(21066.619)
    assert el.to_number("1,947       ") == 1947.0
    assert el.to_number(5) == 5.0
    assert np.isnan(el.to_number(""))
    assert np.isnan(el.to_number("  "))
    assert np.isnan(el.to_number(None))
    assert np.isnan(el.to_number("abc"))


def _sheet(rows):
    """parse_sheet 用の小さな表。0行目が政党名と前置き、1行目が見出し、2行目から開票区。"""
    width = 7
    head = [[None] * width for _ in range(3)]
    head[0][0] = "令和7年執行"
    head[1][2] = "参政 党"
    head[1][5] = "自由民主党"
    head[2][0] = "開票区名"
    body = []
    for c0, c1, a, b in rows:
        r = [None] * width
        r[0], r[1], r[2], r[5] = c0, c1, a, b
        body.append(r)
    return pd.DataFrame(head + body)


def test_parse_sheet_reads_parties_skips_subtotals_and_marks_blanks():
    d = _sheet([("広島市", None, "30", "70"), (None, "広島市中区", "10", "20"), ("安芸郡", None, "5", "5"),
                ("県計", None, "99", "99"), (None, None, None, None), ("葛飾区", None, None, None), ("府中 町", None, 3, "1,000"),
                ("宗谷総合振興局", None, "1", "1"), ("日高支庁", None, "1", "1"), ("新城設楽事務所", None, "1", "1"),
                ("得票総数", None, "1", "1")])
    t = el.parse_sheet(d)
    assert list(t.columns) == ["area", "party", "votes"]
    assert set(t["party"]) == {"参政党", "自由民主党", el.BLANK}
    assert t[(t.area == "広島市中区") & (t.party == "参政党")]["votes"].item() == 10
    assert t[(t.area == "府中町") & (t.party == "自由民主党")]["votes"].item() == 1000
    assert set(t.area) == {"広島市", "広島市中区", "葛飾区", "府中町"}
    assert t[t.party == el.BLANK]["area"].tolist() == ["葛飾区"]
    assert t[t.party == el.BLANK]["votes"].isna().all()


def test_drop_city_totals_keeps_wards_and_splits():
    t = pd.DataFrame({"area": ["広島市", "広島市中区", "薩摩川内市", "薩摩川内市第1", "府中市", "府中町"],
                      "party": "p", "votes": 1.0})
    kept = set(el.drop_city_totals(t)["area"])
    assert kept == {"広島市中区", "薩摩川内市第1", "府中市", "府中町"}


def test_fill_blanks_uses_prefecture_residual_and_joins_names():
    long = pd.DataFrame({"pref": [1, 1, 1, 1, 2], "area": ["A", "A", "B", "C", "D"],
                         "party": ["x", "y", el.BLANK, el.BLANK, "x"], "votes": [10, 5, np.nan, np.nan, 4]})
    tot = pd.DataFrame({"pref": [1, 1, 1, 2, 2], "party": ["x", "y", "合計", "x", "合計"],
                        "votes": [30, 9, 39, 4, 4]})
    tot = pd.concat([tot, pd.DataFrame({"pref": [1], "party": ["z"], "votes": [7]})], ignore_index=True)
    out = el.fill_blanks(long, tot)
    assert list(out.columns) == ["pref", "area", "party", "votes"]
    assert list(out.index) == list(range(len(out)))
    bc = out[out.area == "B+C"].set_index("party")["votes"]
    assert bc["x"] == 20 and bc["y"] == 4 and bc["z"] == 7 and "合計" not in bc.index
    assert (out[out.area == "B+C"]["pref"] == 1).all()
    assert el.BLANK not in set(out.party)
    assert out[out.area == "D"]["votes"].tolist() == [4]


def test_shares_sums_all_parties_and_fills_missing_party():
    long = pd.DataFrame({"pref": [1, 1, 1], "area": ["A", "A", "B"], "party": ["参政党", "自由民主党", "自由民主党"],
                         "votes": [1.0, 3.0, 2.0]})
    s = el.shares(long).set_index("area")
    assert s.loc["A", "valid"] == 4 and s.loc["A", "share"] == pytest.approx(25.0)
    assert s.loc["B", "votes"] == 0 and s.loc["B", "share"] == 0


def test_norm_and_strip_gun():
    assert un.norm(" 鯵ヶ沢町 ") == "鰺ヶ沢町"
    assert un.norm("須惠町") == "須恵町"
    assert un.strip_gun("余市郡余市町") == "余市町"
    assert un.strip_gun("蒲郡市") == "蒲郡市"
    assert un.strip_gun("郡山市") == "郡山市"


def test_area_to_muni_handles_splits_bare_wards_and_hamamatsu():
    assert un.area_to_muni(46, "薩摩川内市第１") == "薩摩川内市"
    assert un.area_to_muni(4, "青葉区") == "仙台市青葉区"
    assert un.area_to_muni(4, "仙台市青葉区") == "仙台市青葉区"
    assert un.area_to_muni(13, "北区") == "北区"
    assert un.area_to_muni(22, "浜松市中央区") == "浜松市"


def test_units_regroup_and_composites():
    s = pd.DataFrame({"pref": [46, 46, 23, 23, 4, 46], "area": ["薩摩川内市第1", "薩摩川内市第2", "豊橋市+豊田市", "岡崎市",
                                                          "青葉区", "鹿屋市第1+指宿市"],
                      "valid": [10.0, 5.0, 100.0, 50.0, 7.0, 1.0], "votes": [1.0, 2.0, 10.0, 5.0, 1.0, 0.0]})
    u = un.to_units(s).set_index("unit")
    assert u.loc["薩摩川内市", "valid"] == 15 and u.loc["薩摩川内市", "votes"] == 3
    assert "仙台市青葉区" in u.index and "指宿市+鹿屋市" in u.index
    groups = un.composite_groups(u.reset_index())
    assert groups[(23, "豊橋市")] == "豊橋市+豊田市" and groups[(23, "豊田市")] == "豊橋市+豊田市"
    assert groups[(46, "鹿屋市")] == "指宿市+鹿屋市" and len(groups) == 4
    other = pd.DataFrame({"pref": [23, 23, 23], "unit": ["豊橋市", "豊田市", "岡崎市"], "pop": [1.0, 2.0, 4.0]})
    r = un.regroup(other, groups, ["pop"]).set_index("unit")["pop"]
    assert r["豊橋市+豊田市"] == 3 and r["岡崎市"] == 4


def test_designated_cities_and_city_of():
    j = pd.DataFrame({"name": ["札幌市", "札幌市中央区", "府中市", "北区"]})
    c = un.designated_cities(j)
    assert c == {"札幌市"}
    assert un.city_of("札幌市北区", c) == "札幌市" and un.city_of("府中市", c) is None


def test_group_national_combines_korea_and_other():
    s = pd.Series({"総数": 100.0, "中国": 30.0, "韓国": 10.0, "朝鮮": 2.0, "ベトナム": 20.0})
    g = fo.group_national(s)
    assert g["韓国・朝鮮"] == 12 and g["中国"] == 30 and g["台湾"] == 0
    assert g[fo.OTHER] == 100 - 30 - 12 - 20


def test_shift_share_allocates_national_growth_by_base_shares():
    groups = fo.GROUPS + [fo.OTHER]
    base = pd.DataFrame(0.0, index=[0, 1], columns=groups)
    base.loc[0, "中国"], base.loc[1, "中国"] = 3.0, 1.0
    base.loc[0, "ベトナム"], base.loc[1, "ベトナム"] = 0.0, 2.0
    n22 = pd.Series(5.0, index=groups)
    n25 = pd.Series(5.0, index=groups)
    n25["中国"], n25["ベトナム"] = 45.0, 15.0
    p = dataset.shift_share(base, n22, n25)
    assert p.tolist() == pytest.approx([30.0, 20.0])


def test_distances_and_neighbors():
    lon = np.array([139.0, 139.0, 140.0])
    lat = np.array([35.0, 35.1, 35.0])
    d = geo.distances_km(lon, lat)
    assert d[0, 1] == pytest.approx(11.12, abs=0.05)
    assert d[0, 0] == 0 and d[0, 2] == pytest.approx(d[2, 0])
    assert d[0, 2] == pytest.approx(91.1, abs=0.2)
    t = pd.DataFrame({"inc": [1.0, 4.0, 9.0], "pop": [100.0, 200.0, 300.0]})
    nb = geo.neighbor_change(t, d, d[0, 1], "inc", "pop")
    assert nb[0] == pytest.approx(2.0) and nb[1] == pytest.approx(1.0)
    assert np.isnan(nb[2])
    small = pd.DataFrame({"inc": [0.1, 0.2, 0.0], "pop": [0.5, 0.5, 1.0]})
    assert geo.neighbor_change(small, d, 20.0, "inc", "pop")[0] == pytest.approx(40.0)


def test_unit_location_averages_and_falls_back_to_same_city_prefix():
    office = {"01101": (141.0, 43.0), "01102": (141.2, 43.2), "22131": (137.0, 34.0), "22132": (137.2, 34.2)}
    assert geo.unit_location("01101+01102", office) == pytest.approx((141.1, 43.1))
    assert geo.unit_location("01101+22131", office) == pytest.approx((139.0, 38.5))
    assert geo.unit_location("22130", office) == pytest.approx((137.1, 34.1))
    assert all(np.isnan(geo.unit_location("99999", office)))


def _sim(n=600, seed=0):
    rng = np.random.default_rng(seed)
    pref = np.repeat(np.arange(20), n // 20)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = z + u
    c = rng.normal(size=n)
    y = 2.0 * d + 0.5 * c + 3.0 * u + rng.normal(size=n) + pref * 0.1
    return pd.DataFrame({"pref": pref, "y": y, "d": d, "z": z, "c": c, "w": rng.uniform(1, 3, size=n)})


def test_ols_matches_closed_form_and_tsls_recovers_effect():
    t = _sim()
    x = np.column_stack([t["d"], analysis.design(t, ["c"])])
    w = t["w"].to_numpy()
    b = np.linalg.solve(x.T @ (x * w[:, None]), x.T @ (w * t["y"].to_numpy()))
    fit = analysis.ols(t, "y", "d", ["c"], weight="w")
    assert fit.coef == pytest.approx(b[0])
    assert fit.coef > 2.5
    iv = analysis.tsls(t, "y", "d", "z", ["c"], weight="w")
    assert iv.coef == pytest.approx(2.0, abs=0.3)
    assert iv.first_stage_f > 100 and 0 < iv.se < 0.5
    assert iv.ci[0] < 2.0 < iv.ci[1]


def test_complete_drops_rows_with_missing_used_columns():
    t = _sim(60)
    t.loc[0, "c"] = np.nan
    assert analysis.ols(t, "y", "d", ["c"], weight="w").n == 59
    assert analysis.ols(t, "y", "d", [], weight="w").n == 60


def test_cluster_se_and_first_stage_match_statsmodels():
    import statsmodels.api as sm

    t = _sim(400, seed=3)
    x = np.column_stack([t["d"], analysis.design(t, ["c"])])
    ref = sm.WLS(t["y"], x, weights=t["w"]).fit(cov_type="cluster", cov_kwds={"groups": t["pref"]})
    fit = analysis.ols(t, "y", "d", ["c"], weight="w")
    assert fit.se == pytest.approx(np.asarray(ref.bse)[0], rel=1e-6)
    zx = np.column_stack([t["z"], analysis.design(t, ["c"])])
    fs = sm.WLS(t["d"], zx, weights=t["w"]).fit(cov_type="cluster", cov_kwds={"groups": t["pref"]})
    iv = analysis.tsls(t, "y", "d", "z", ["c"], weight="w")
    assert iv.first_stage_f == pytest.approx(np.asarray(fs.tvalues)[0] ** 2, rel=1e-6)


def test_tsls_matches_weighted_closed_form():
    t = _sim(400, seed=4)
    xc = analysis.design(t, ["c"])
    X = np.column_stack([t["d"], xc])
    Z = np.column_stack([t["z"], xc])
    W = t["w"].to_numpy()
    zwz = np.linalg.pinv(Z.T @ (Z * W[:, None]))
    xwz = (X * W[:, None]).T @ Z
    b = np.linalg.pinv(xwz @ zwz @ xwz.T) @ (xwz @ zwz @ ((Z * W[:, None]).T @ t["y"].to_numpy()))
    assert analysis.tsls(t, "y", "d", "z", ["c"], weight="w").coef == pytest.approx(b[0], rel=1e-8)


def test_unused_missing_columns_do_not_drop_rows():
    t = _sim(60)
    t["unused"] = np.nan
    assert analysis.ols(t, "y", "d", ["c"], weight="w").n == 60
    assert analysis.tsls(t, "y", "d", "z", ["c"], weight="w").n == 60
    t.loc[1, "z"] = np.nan
    assert analysis.tsls(t, "y", "d", "z", ["c"], weight="w").n == 59
