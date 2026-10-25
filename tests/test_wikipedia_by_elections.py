import json
from datetime import date

from seiryoku import by_elections
from seiryoku.fetch import wikipedia_by_elections as wbe

WIKITEXT = """'''2027年日本の補欠選挙'''
== 概要 ==
本文
== 4月の補欠選挙・再選挙 ==
=== 概要 ===
* 投票日：2027年（令和9年）[[4月25日]]
=== 衆議院甲県第1区 ===
{{選挙結果 (衆議院小選挙区補欠)
|年=2027|選挙回=51|有権者=300000|投票率=40.00
|当|[[山田太郎 (政治家)|山田太郎]]|50|[[自由民主党 (日本)|自由民主党]]|新|80000|公明党推薦
||鈴木花子|45|立憲民主党|元|60000.5|
}}
=== 参議院乙県選挙区（再選挙） ===
{{選挙結果 (参議院選挙区)
|補欠年=2027|選挙回=27|定数=1|有権者=900000|投票率=35.00
|1氏名=[[佐藤一郎]]|1年齢=55|1所属=[[無所属]]|1新旧=新|1得票=200000|1推薦=[[立憲民主党 (日本 2020)|立憲民主党]]推薦
|2氏名=田中次郎|2年齢=60|2所属=参政党|2新旧=新|2得票=150000|2推薦=
}}
== 7月の補欠（合併）選挙 ==
* 投票日：2027年（令和9年）7月25日
{{選挙結果 (参議院選挙区)
|補欠年=2027|選挙回=26|定数=1
|1氏名=合併太郎|1所属=無所属|1得票=1|1推薦=
}}
== 10月の補欠選挙 ==
* 投票日：2027年（令和9年）10月24日
* 選挙区：衆議院[[丙県第2区]]
==== 立候補をめぐる動き ====
{{選挙結果 (衆議院小選挙区補欠)
|年=2027|選挙回=51
|当|高橋三郎|40|国民民主党|新|50000|
}}
== 脚注 ==
"""


def test_parse_year_reads_headings_dates_and_candidates():
    rows = wbe.parse_year(WIKITEXT, 2027)
    assert [(r["vote_date"], r["house"], r["district"], r["kind"]) for r in rows] == [
        ("2027-04-25", "衆議院", "甲県第1区", "補欠選挙"),
        ("2027-04-25", "参議院", "乙県選挙区", "再選挙"),
        ("2027-10-24", "衆議院", "丙県第2区", "補欠選挙"),
    ]
    shu = rows[0]
    assert (shu["winner"], shu["party"]) == ("山田太郎", "自由民主党")
    assert shu["candidates"][1] == {"name": "鈴木花子", "party": "立憲民主党", "votes": 60000.5, "rec": ""}
    assert shu["candidates"][0]["rec"] == "公明党推薦"
    assert rows[1]["candidates"][0]["rec"] == "立憲民主党推薦"
    assert rows[0]["source"].startswith("https://ja.wikipedia.org/wiki/2027")


def test_sections_without_results_reports_past_sections_without_tables():
    text = "== 4月の補欠選挙 ==\n* 投票日：2027年4月25日\n=== 衆議院甲県第1区 ===\n候補者のみ\n"
    assert wbe.sections_without_results(text, date(2027, 5, 7)) == ["4月の補欠選挙(投票日 2027-04-25)"]
    assert wbe.sections_without_results(text, date(2027, 4, 1)) == []
    assert wbe.sections_without_results(WIKITEXT, date(2027, 12, 1)) == []


def test_parse_year_raises_when_district_is_unknown():
    text = "== 4月の補欠選挙 ==\n* 投票日：2027年4月25日\n{{選挙結果 (衆議院小選挙区補欠)\n|年=2027\n|当|A|1|B|新|10|\n}}\n"
    try:
        wbe.parse_year(text, 2027)
    except wbe.ParseError:
        return
    raise AssertionError("ParseErrorにならない")


def test_update_national_adds_only_new_past_elections(tmp_path, monkeypatch):
    path = tmp_path / "national.json"
    known = wbe.parse_year(WIKITEXT, 2027)[0]
    path.write_text(json.dumps({"note": "n", "elections": [known]}, ensure_ascii=False))
    monkeypatch.setattr(by_elections, "NATIONAL_PATH", path)
    res = by_elections.update_national(date(2027, 5, 7), {2026: None, 2027: WIKITEXT})
    assert [r["district"] for r in res["added"]] == ["乙県選挙区"]
    saved = json.loads(path.read_text())
    assert saved["note"] == "n" and [r["district"] for r in saved["elections"]] == ["甲県第1区", "乙県選挙区"]
    assert by_elections.update_national(date(2027, 5, 7), {2026: None, 2027: WIKITEXT})["added"] == []


def test_recorded_by_elections_match_wikipedia():
    # 記事の書式が変わって読めなくなったら、ここで気づく(通信はutil.fetchのキャッシュ経由)
    recorded = json.loads(by_elections.NATIONAL_PATH.read_text(encoding="utf-8"))["elections"]
    parsed = []
    for year in range(2019, 2025):
        parsed += wbe.parse_year(wbe.fetch_wikitext(year, force=False), year)
    key = lambda r: (r["vote_date"], r["district"])
    assert sorted(parsed, key=key) == sorted(recorded, key=key)
