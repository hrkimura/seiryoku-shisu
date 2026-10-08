import json

from seiryoku.fetch import wikipedia_governor as w

_ONE_LINE = (
    "{{選挙結果 (日本の首長)\n"
    "|当選者名='''[[鈴木健太_(政治家)|鈴木健太]]''' |当選者年齢=49 |当選者所属=無所属 "
    "|当選者推薦・支持=[[日本維新の会 (2016-)|日本維新の会]]が自主的に支持\n"
    "|落選者名=[[猿田和三]] |落選者推薦・支持=[[立憲民主党 (日本 2020)|立憲民主党]]推薦\n}}"
)

_MULTI_LINE = (
    "{{選挙結果 (日本の首長)\n"
    "|当選者名='''[[花角英世]]'''\n"
    "|当選者年齢=60\n"
    "|当選者推薦・支持=（支持）[[自由民主党 (日本)|自由民主党]]・[[公明党]]\n"
    "|落選者名=池田千賀子\n}}"
)


def test_parse_winner_reads_one_line_and_multi_line_templates():
    assert w.parse_winner(_ONE_LINE) == {"winner": "鈴木健太", "text": "日本維新の会が自主的に支持"}
    assert w.parse_winner(_MULTI_LINE) == {"winner": "花角英世", "text": "（支持）自由民主党・公明党"}


def test_parse_winner_returns_none_without_result_template():
    assert w.parse_winner("本文だけの記事") is None


def test_extract_parties_handles_abbreviations_and_local_organizations():
    assert w.extract_parties("（推薦）自民県連・立民・国民県連・公明県本部") == [
        "自由民主党", "立憲民主党", "国民民主党", "公明党",
    ]


def test_extract_parties_does_not_read_nhk_party_as_dpfp_and_skips_jishu_touhyou():
    assert w.extract_parties("NHKから国民を守る党推薦") == ["NHKから国民を守る党"]
    assert w.extract_parties("自民党は自主投票") == []
    assert w.extract_parties("公明党推薦、自民党は自主投票") == ["公明党"]


def test_lookup_prefers_manual_entries(monkeypatch):
    w.MANUAL_PATH.write_text(json.dumps({"東京都|2024-07-07": {"status": "manual", "parties": ["公明党"]}}))
    assert w.lookup("東京都", "2024-07-07")["parties"] == ["公明党"]


def test_lookup_caches_results_but_not_network_failures(monkeypatch):
    assert w.lookup("秋田県", "2025-04-06") is None
    assert not w.CACHE_PATH.exists()

    monkeypatch.setattr(w, "_fetch_article", lambda title: (123, _ONE_LINE))
    entry = w.lookup("秋田県", "2025-04-06")
    assert entry["parties"] == ["日本維新の会"] and entry["revid"] == 123

    monkeypatch.setattr(w, "_fetch_article", lambda title: (_ for _ in ()).throw(AssertionError))
    assert w.lookup("秋田県", "2025-04-06")["status"] == "ok"


def test_lookup_records_missing_article(monkeypatch):
    monkeypatch.setattr(w, "_fetch_article", lambda title: None)
    assert w.lookup("山形県", "2017-01-22") == {"title": "2017年山形県知事選挙", "status": "no_article"}
