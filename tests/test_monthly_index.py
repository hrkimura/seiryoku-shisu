import pytest

from seiryoku import monthly_index, municipal_registry, turnover
from seiryoku.fetch import diet_history, jichisoken, population
from seiryoku.fetch.diet_history import Snapshot
from seiryoku.fetch.jichisoken import Endorsement


def test_ym_parses_iso_date():
    assert monthly_index._ym("2023-04-09") == (2023, 4)
    assert monthly_index._ym(None) is None
    assert monthly_index._ym("") is None


def test_months_back_crosses_year_boundary():
    assert monthly_index._months_back((2024, 2), 3) == [(2024, 2), (2024, 1), (2023, 12)]
    assert monthly_index._months_back((2024, 6), 1) == [(2024, 6)]


def test_snapshots_from_windows_sums_entries_across_the_window():
    # (level_type, P_e, n_e, shares)。1月・3月とも同じN_ell(=n_e*P_e=16)の
    # 単一イベントなので、階層間の重みはどちらも同じsqrt(16)=4になる。
    entries_by_month = {
        (2024, 1): [("衆院選", 16.0, 1.0, {"自由民主党": 1.0})],
        (2024, 2): [],
        (2024, 3): [("市区町村議会", 16.0, 1.0, {"公明党": 1.0})],
    }
    months = [(2024, 1), (2024, 2), (2024, 3)]

    single = monthly_index._snapshots_from_windows(entries_by_month, months, window_months=1)
    by_month = {(s.year, s.month): s for s in single}
    assert (2024, 2) not in by_month  # その月単独では選挙が無い
    assert by_month[(2024, 1)].C_p == {"自由民主党": 1.0}
    assert by_month[(2024, 1)].level_weight_share["衆院選"] == 1.0
    assert by_month[(2024, 3)].C_p == {"公明党": 1.0}

    windowed = monthly_index._snapshots_from_windows(entries_by_month, months, window_months=3)
    by_month_w = {(s.year, s.month): s for s in windowed}
    # 2024-02は単独では選挙が無いが、3か月ウィンドウなら1月ぶんを含む
    assert by_month_w[(2024, 2)].C_p == {"自由民主党": 1.0}
    assert by_month_w[(2024, 2)].n_events == 1
    # 2024-03は1月・3月の両方を含む(衆院選・市区町村議会の階層重みが同じなので自民・公明が半々)
    assert by_month_w[(2024, 3)].C_p == {"自由民主党": 0.5, "公明党": 0.5}
    assert by_month_w[(2024, 3)].n_events == 2
    # 重みシェアの内訳も同じウィンドウで合算される(衆院選・市区町村議会が半々)
    assert by_month_w[(2024, 3)].level_weight_share["衆院選"] == 0.5
    assert by_month_w[(2024, 3)].level_weight_share["市区町村議会"] == 0.5


def test_level_aggregate_is_invariant_to_how_a_level_is_partitioned():
    # 同じ総人口(P=100、n_e=1)を1自治体にまとめても、2自治体(60+40)に分割
    # しても、階層内のphiと重みシェアは変わらない——階層内はP_eだけの加重平均
    # という定義の核心(2026-09に再設計、notes/new_index_formula.tex参照)。
    one_unit = [("市区町村議会", 100.0, 1.0, {"自由民主党": 0.7, "公明党": 0.3})]
    two_units = [
        ("市区町村議会", 60.0, 1.0, {"自由民主党": 0.7, "公明党": 0.3}),
        ("市区町村議会", 40.0, 1.0, {"自由民主党": 0.7, "公明党": 0.3}),
    ]
    c_p_one, share_one = monthly_index._level_aggregate(one_unit)
    c_p_two, share_two = monthly_index._level_aggregate(two_units)
    assert c_p_one == c_p_two
    assert share_one == share_two


def test_level_aggregate_combines_levels_via_sqrt_of_level_totals():
    # 都道府県知事(N_ell=n_e*P_e=1*100=100)と市区町村議会(N_ell=1*400=400)の
    # 2階層。階層の重みはsqrt(100)=10とsqrt(400)=20なので、10:20の比で合成される。
    entries = [
        ("都道府県知事", 100.0, 1.0, {"自由民主党": 1.0}),
        ("市区町村議会", 400.0, 1.0, {"公明党": 1.0}),
    ]
    c_p, share = monthly_index._level_aggregate(entries)
    assert c_p["自由民主党"] == pytest.approx(10 / 30)
    assert c_p["公明党"] == pytest.approx(20 / 30)
    assert share["都道府県知事"] == pytest.approx(1 / 3)
    assert share["市区町村議会"] == pytest.approx(2 / 3)


def test_level_aggregate_weights_events_within_a_level_by_population_only():
    # 階層内はP_eだけの加重平均であることの確認(60:40の比のまま、n_eは無視)。
    entries = [
        ("市区町村議会", 60.0, 1.0, {"自由民主党": 1.0}),
        ("市区町村議会", 40.0, 1.0, {"公明党": 1.0}),
    ]
    c_p, _ = monthly_index._level_aggregate(entries)
    assert c_p["自由民主党"] == pytest.approx(0.6)
    assert c_p["公明党"] == pytest.approx(0.4)


def test_level_aggregate_within_level_share_ignores_n_e():
    # n_eP_eではなくP_eだけで階層内シェアを決めることの確認(2026-09に
    # ユーザー指摘: 議員定数n_eは人口にほぼ比例しないため、n_eP_eを使うと
    # 大自治体の影響が過大になる)。同じ人口(P=50ずつ)でも議席数(n_e)が
    # 大きく異なる2自治体を混ぜても、階層内シェアは人口比の50:50のままになる。
    entries = [
        ("市区町村議会", 50.0, 1.0, {"自由民主党": 1.0}),   # 小さな議会(n_e=1)
        ("市区町村議会", 50.0, 100.0, {"公明党": 1.0}),  # 同じ人口だが議席数100の議会
    ]
    c_p, _ = monthly_index._level_aggregate(entries)
    assert c_p["自由民主党"] == pytest.approx(0.5)
    assert c_p["公明党"] == pytest.approx(0.5)


def test_level_aggregate_cross_level_weight_uses_n_e():
    # 一方、階層をまたぐ重みW_ellにはn_eが効くことの確認。人口は同じでも
    # 議席数が大きい階層の方がN_ell=n_e*P_eが大きくなり、W_ellも大きくなる。
    entries = [
        ("都道府県知事", 100.0, 1.0, {"自由民主党": 1.0}),
        ("都道府県議会", 100.0, 4.0, {"公明党": 1.0}),
    ]
    c_p, share = monthly_index._level_aggregate(entries)
    # N_ell: 知事=100, 議会=400。W_ell: sqrt(100)=10, sqrt(400)=20。10:20の比。
    assert c_p["自由民主党"] == pytest.approx(10 / 30)
    assert c_p["公明党"] == pytest.approx(20 / 30)
    assert share["都道府県知事"] == pytest.approx(1 / 3)
    assert share["都道府県議会"] == pytest.approx(2 / 3)


def test_level_aggregate_empty_when_no_entries():
    assert monthly_index._level_aggregate([]) == ({}, {})


def test_month_range_crosses_year_boundary():
    assert monthly_index._month_range((2023, 11), (2024, 2)) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_terms_in_month_returns_exact_month_matches():
    chain = [
        {"vote_date": "2023-04-09", "seats": {"A": 1}},
        {"vote_date": "2019-04-07", "seats": {"B": 1}},
    ]
    parsed = monthly_index._parsed_chain(chain)
    assert monthly_index._terms_in_month(parsed, (2023, 4)) == [{"vote_date": "2023-04-09", "seats": {"A": 1}}]
    assert monthly_index._terms_in_month(parsed, (2023, 5)) == []
    assert monthly_index._terms_in_month(parsed, (2020, 1)) == []


def test_current_national_parties_uses_strict_requirement(monkeypatch):
    from seiryoku.fetch import diet

    monkeypatch.setattr(
        diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5, "小政党": 1, "無所属": 1}}
    )
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {"小政党": 0.5})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    assert monthly_index._current_national_parties() == {"自由民主党"}


def test_build_month_snapshots_only_includes_months_with_an_actual_election(monkeypatch):
    """フロー指標なので、選挙が無かった月は結果に出てこない(2026-09の設計変更)。"""
    from seiryoku.fetch import diet

    monkeypatch.setattr(
        diet_history,
        "fetch_sangiin_election_history",
        lambda: [Snapshot(session="第26回", date="2022-07-10", seats={"自由民主党": 1, "公明党": 0})],
    )
    monkeypatch.setattr(
        diet_history,
        "fetch_shugiin_history",
        lambda: [Snapshot(session="48", date="2017-10-22", seats={"自由民主党": 1, "無所属": 1, "公明党": 0}),
                 Snapshot(session="49", date="2021-10-31", seats={"自由民主党": 1, "無所属": 1, "公明党": 0})],
    )
    # 政党要件は「直近の国政選挙時点」を全期間に固定で適用する(2026-09にユーザー指示、簡略化)
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5, "公明党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {"石川県": 100})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {
            "chains": {
                "100": [{"vote_date": "2022-03-13", "party": "無所属"}],  # 知事選(2022年3月)
                "1": [{"vote_date": "2019-04-07", "party": "自由民主党"}],  # 市長選(2019年4月)
            }
        },
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市" if jid == 1 else None)
    monkeypatch.setattr(municipal_registry, "load_gikai_term_chain_cache", lambda: {"chains": {}})
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {"石川県": 10000})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(
        jichisoken,
        "governor_endorsements_by_year",
        lambda: {"2022": {"石川県": Endorsement("石川県", "無所属", ["公明党"], "2022-03-13", "2022")}},
    )
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken,
        "endorsement_for_vote_year",
        lambda by_year, name, vote_date: by_year.get(vote_date[:4], {}).get(name),
    )

    snapshots = monthly_index.build_month_snapshots(min_year=2019, window_months=1)
    by_month = {(s.year, s.month): s for s in snapshots}

    # 選挙のあった月だけが結果に含まれる(市長選・知事選・衆院選)
    assert (2019, 4) in by_month  # 市長選(A市、自由民主党)
    assert (2022, 3) in by_month  # 知事選(石川県、無所属→公明党推薦)
    assert (2021, 10) in by_month  # 衆院選

    # 選挙が無い月(例: 2020年1月)は含まれない
    assert (2020, 1) not in by_month

    muni_snapshot = by_month[(2019, 4)]
    assert muni_snapshot.C_p == {"自由民主党": 1.0}
    assert muni_snapshot.n_events == 1

    gov_snapshot = by_month[(2022, 3)]
    assert gov_snapshot.C_p == {"公明党": 1.0}
    assert gov_snapshot.n_events == 1


def test_build_month_snapshots_weights_diet_by_seat_count(monkeypatch):
    """国会だけ実際の議席数n_eをsqrt(n_e * P_e)として掛ける(2026-09の設計変更、
    design_document.tex \\S2.2参照)。首長は常にn_e=1なのでsqrt(P_e)のまま。

    知事選(自民、P=10000、n_e=1、重みsqrt(10000)=100)と同じ12か月ウィンドウに
    参院選(公明、n_e=9議席、P=10000、重みsqrt(9*10000)=300)が入る設定にし、
    n_eを考慮しない旧式なら50:50になるところが25:75になることを確認する。
    """
    from seiryoku.fetch import diet

    monkeypatch.setattr(
        diet_history,
        "fetch_sangiin_election_history",
        lambda: [Snapshot(session="第26回", date="2022-08-03", seats={"公明党": 9})],
    )
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"参議院": {"公明党": 9, "自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {"石川県": 100})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"100": [{"vote_date": "2022-01-15", "party": "自由民主党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: None)
    monkeypatch.setattr(municipal_registry, "load_gikai_term_chain_cache", lambda: {"chains": {}})
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {"石川県": 10000})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(min_year=2022, window_months=12)
    by_month = {(s.year, s.month): s for s in snapshots}

    sangiin_snapshot = by_month[(2022, 8)]
    assert sangiin_snapshot.C_p == {"自由民主党": 0.25, "公明党": 0.75}
    assert sangiin_snapshot.n_events == 2


def test_build_month_snapshots_includes_assembly_elections_without_jichisoken_correction(monkeypatch):
    """都道府県議会・市区町村議会(gikai)も国会と同じく届出政党そのままでphi_p(e)を
    計算し、jichisoken補正は試みない(2026年9月にユーザー指摘、design_document.tex
    \\S2.1参照)。市長選(自民、P=10000、n_e=1、重みsqrt(10000)=100)と同じ窓に
    市議会選(自民6・無所属4、n_e=10議席、P=10000、重みsqrt(10*10000)≒316.2)が
    入る設定で、議会選のほうが大きな重みを持つことを確認する。"""
    from seiryoku.fetch import diet

    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history", lambda: [])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"1": [{"vote_date": "2022-04-10", "party": "自由民主党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市")
    monkeypatch.setattr(
        municipal_registry,
        "load_gikai_term_chain_cache",
        lambda: {
            "chains": {
                "1": [{"vote_date": "2022-04-10", "seats": {"自由民主党": 6, "無所属": 4}}],
            }
        },
    )
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(min_year=2022, window_months=1)
    by_month = {(s.year, s.month): s for s in snapshots}

    snapshot = by_month[(2022, 4)]
    assert snapshot.n_events == 2

    import math

    head_w = math.sqrt(10000)
    gikai_w = math.sqrt(10 * 10000)
    total_w = head_w + gikai_w
    expected_ldp = (head_w * 1.0 + gikai_w * 0.6) / total_w
    expected_ind = (gikai_w * 0.4) / total_w
    assert snapshot.C_p["自由民主党"] == expected_ldp
    assert snapshot.C_p["無所属"] == expected_ind


def test_build_month_snapshots_local_only_excludes_diet_elections(monkeypatch):
    """local_only=Trueは衆参の国政選挙イベントを一切含めない「地方選挙限定」の
    系列を作る(precursor.pyで「地方選は国政選挙の前哨戦か」を検証するために使う)。
    fetch_sangiin_election_history等を呼ばずに済むよう、呼ばれたら例外にして確認する。
    """
    from seiryoku.fetch import diet

    def _boom():
        raise AssertionError("local_only=Trueなのに国政選挙データを取得しようとした")

    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history", _boom)
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", _boom)
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"1": [{"vote_date": "2022-04-10", "party": "自由民主党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市")
    monkeypatch.setattr(municipal_registry, "load_gikai_term_chain_cache", lambda: {"chains": {}})
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(min_year=2022, window_months=1, local_only=True)
    by_month = {(s.year, s.month): s for s in snapshots}

    assert (2022, 4) in by_month
    assert by_month[(2022, 4)].C_p == {"自由民主党": 1.0}


def test_build_month_snapshots_national_only_excludes_local_elections(monkeypatch):
    """national_only=Trueはlocal_onlyの対称形で、知事・市区町村長・都道府県議会・
    市区町村議会を一切含めない「国政選挙限定」の系列を作る(政党ごとに国政/地方の
    強さを比較するために2026-09に追加)。同じ月に衆院選(自民)と市長選(公明)を
    重ねておき、国政選挙のシェアだけが残ることを確認する。"""
    from seiryoku.fetch import diet

    monkeypatch.setattr(
        diet_history,
        "fetch_shugiin_history",
        lambda: [Snapshot(session="49", date="2022-04-10", seats={"自由民主党": 1})],
    )
    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history", lambda: [])
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"1": [{"vote_date": "2022-04-10", "party": "公明党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市")
    monkeypatch.setattr(municipal_registry, "load_gikai_term_chain_cache", lambda: {"chains": {}})
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(min_year=2022, window_months=1, national_only=True)
    by_month = {(s.year, s.month): s for s in snapshots}

    assert (2022, 4) in by_month
    assert by_month[(2022, 4)].C_p == {"自由民主党": 1.0}
    assert by_month[(2022, 4)].n_events == 1


def test_build_month_snapshots_rejects_both_local_only_and_national_only():
    with pytest.raises(ValueError):
        monthly_index.build_month_snapshots(local_only=True, national_only=True)


def test_build_month_snapshots_rejects_only_level_with_local_or_national_only():
    with pytest.raises(ValueError):
        monthly_index.build_month_snapshots(only_level="衆院選", local_only=True)
    with pytest.raises(ValueError):
        monthly_index.build_month_snapshots(only_level="衆院選", national_only=True)
    with pytest.raises(ValueError):
        monthly_index.build_month_snapshots(assembly_only=True, executive_only=True)
    with pytest.raises(ValueError):
        monthly_index.build_month_snapshots(assembly_only=True, local_only=True)


def test_build_month_snapshots_executive_only_excludes_assemblies(monkeypatch):
    """executive_only=Trueは知事・市区町村長だけに絞る(assembly_onlyの対称形)。
    同じ月に知事選(自民)と市議会選(公明)を重ねておき、市議会側が混ざらないことを
    確認する。"""
    from seiryoku.fetch import diet

    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history", lambda: [])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {"石川県": 100})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"100": [{"vote_date": "2022-04-10", "party": "自由民主党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市")
    monkeypatch.setattr(
        municipal_registry,
        "load_gikai_term_chain_cache",
        lambda: {"chains": {"1": [{"vote_date": "2022-04-10", "seats": {"公明党": 10}}]}},
    )
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {"石川県": 10000})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(
        min_year=2022, window_months=1, executive_only=True,
    )
    by_month = {(s.year, s.month): s for s in snapshots}

    assert (2022, 4) in by_month
    assert by_month[(2022, 4)].C_p == {"自由民主党": 1.0}
    assert by_month[(2022, 4)].n_events == 1


def test_build_month_snapshots_only_level_isolates_a_single_level(monkeypatch):
    """only_level=X指定時は6階層のうちXだけに絞る(政党ごとにどの階層に強いかを
    見るために2026-09に追加)。都道府県知事(自民)と市区町村議会(公明)を同じ月に
    重ねておき、only_level="都道府県知事"なら市区町村議会側が混ざらないことを
    確認する。"""
    from seiryoku.fetch import diet

    monkeypatch.setattr(diet_history, "fetch_sangiin_election_history", lambda: [])
    monkeypatch.setattr(diet_history, "fetch_shugiin_history", lambda: [])
    monkeypatch.setattr(diet, "fetch_diet_seats", lambda: {"衆議院": {"自由民主党": 5}})
    monkeypatch.setattr(monthly_index, "_latest_shugiin_district_pct", lambda: {})
    monkeypatch.setattr(monthly_index, "_latest_sangiin_district_pct", lambda: {})
    monkeypatch.setattr(population, "national_population", lambda: 10000)
    monkeypatch.setattr(turnover, "_ensure_governor_ids_loaded", lambda: None)
    monkeypatch.setattr(turnover, "_GOVERNOR_JICHITAI_IDS", {"石川県": 100})
    monkeypatch.setattr(
        turnover,
        "load_term_chain_cache",
        lambda: {"chains": {"100": [{"vote_date": "2022-04-10", "party": "自由民主党"}]}},
    )
    monkeypatch.setattr(municipal_registry, "jurisdiction_name", lambda jid: "A市")
    monkeypatch.setattr(
        municipal_registry,
        "load_gikai_term_chain_cache",
        lambda: {"chains": {"1": [{"vote_date": "2022-04-10", "seats": {"公明党": 10}}]}},
    )
    monkeypatch.setattr(population, "prefecture_population_by_name", lambda: {"石川県": 10000})
    monkeypatch.setattr(population, "municipal_population_by_name", lambda: {"A市": 10000})
    monkeypatch.setattr(jichisoken, "governor_endorsements_by_year", lambda: {})
    monkeypatch.setattr(jichisoken, "municipal_head_endorsements_by_year", lambda: {})
    monkeypatch.setattr(
        jichisoken, "endorsement_for_vote_year", lambda by_year, name, vote_date: None
    )

    snapshots = monthly_index.build_month_snapshots(
        min_year=2022, window_months=1, only_level="都道府県知事",
    )
    by_month = {(s.year, s.month): s for s in snapshots}

    assert (2022, 4) in by_month
    assert by_month[(2022, 4)].C_p == {"自由民主党": 1.0}
    assert by_month[(2022, 4)].n_events == 1
