from datetime import date

from bs4 import BeautifulSoup

from seiryoku.fetch import wikipedia_polls as w

# 2026年のページの表を縮めたもの: 多段の見出し、中道改革連合の結合セル、全列にまたがる時期の帯
HTML = """
<h2>Pollsters</h2>
<table class="wikitable"><tr><th>Pollster</th></tr><tr><td>NHK</td></tr></table>
<h2>Party identification</h2>
<h3>2026</h3>
<table class="wikitable">
<tr><th rowspan="3">Fieldwork date</th><th rowspan="3">Polling firm</th><th rowspan="3">Sample size</th>
<th rowspan="2"><a href="/wiki/Liberal_Democratic_Party_(Japan)">LDP</a></th>
<th colspan="2"><a href="/wiki/Centrist_Reform_Alliance">CRA</a></th>
<th rowspan="3">Others</th><th rowspan="3">No party</th><th rowspan="3">Und./no ans.</th><th rowspan="3">Lead</th></tr>
<tr><th><a href="/wiki/Constitutional_Democratic_Party_of_Japan">CDP</a></th><th><a href="/wiki/Komeito">Komei</a></th></tr>
<tr><th></th><th></th><th></th></tr>
<tr><th colspan="10"><a href="/wiki/Centrist_Reform_Alliance">After the split</a></th></tr>
<tr><td>8 Feb 2026</td><td>Election results</td><td>55.7%</td><td>36.7</td><td colspan="2">18.2</td><td>1</td><td colspan="2">–</td><td>18.5</td></tr>
<tr><td>6–7 Feb</td><td>Kyodo News<sup>[a]</sup></td><td>1,054</td><td>43.7</td><td colspan="2">12</td><td>1.9</td><td>9.8</td><td>2.6</td><td>31.7</td></tr>
<tr><td>31 Jan – 2 Feb</td><td>Nikkei &amp; TV Tokyo</td><td>~1,000</td><td>40</td><td>1.2</td><td>0.8</td><td>–</td><td>30</td><td>5</td><td>38.8</td></tr>
</table>
<h2>Voting intention (proportional vote)</h2>
<table class="wikitable"><tr><th>Fieldwork date</th><th>Polling firm</th><th><a href="/wiki/Liberal_Democratic_Party_(Japan)">LDP</a></th></tr>
<tr><td>6–7 Feb</td><td>NHK</td><td>50</td></tr></table>
"""


def test_parse_dates():
    assert w.parse_dates("6–7 Feb 2026", None) == (date(2026, 2, 6), date(2026, 2, 7))
    assert w.parse_dates("12– 13 Sep", 2026) == (date(2026, 9, 12), date(2026, 9, 13))
    assert w.parse_dates("28 Feb – 1 Mar", 2025) == (date(2025, 2, 28), date(2025, 3, 1))
    assert w.parse_dates("31 Dec 2025 – 2 Jan 2026", None) == (date(2025, 12, 31), date(2026, 1, 2))
    assert w.parse_dates("30 Dec – 2 Jan", 2026) == (date(2025, 12, 30), date(2026, 1, 2))
    assert w.parse_dates("8 Feb", None) is None


def test_normalize_firm():
    assert w.normalize_firm("ANN Archived 2018-12-03 at the Wayback Machine") == "ANN"
    assert w.normalize_firm("Nikkei & TV Tokyo") == "Nikkei/TV Tokyo"
    assert w.normalize_firm("Mainichi/SSRC") == "Mainichi/SSRC"


def test_parse_page_reads_only_party_identification():
    polls, unknown = w.parse_page(HTML, "test")
    assert unknown == []
    assert [p.firm for p in polls] == ["Kyodo News", "Nikkei/TV Tokyo"]
    kyodo, nikkei = polls
    assert (kyodo.start, kyodo.end, kyodo.n, kyodo.n_approx) == ("2026-02-06", "2026-02-07", 1054, False)
    assert kyodo.shares == {"自由民主党": 43.7, "その他": 1.9}
    assert kyodo.combined == [{"parties": ["立憲民主党", "公明党"], "value": 12.0}]
    assert (kyodo.no_party, kyodo.undecided) == (9.8, 2.6)
    assert nikkei.shares == {"自由民主党": 40.0, "立憲民主党": 1.2, "公明党": 0.8}
    assert nikkei.start == "2026-01-31" and nikkei.n == 1000 and nikkei.n_approx
    assert w.party_total(kyodo) == 43.7 + 1.9 + 12


def test_unknown_party_is_reported():
    html = HTML.replace("Komeito", "Some_New_Party")
    _, unknown = w.parse_page(html, "test")
    assert unknown == ["Some New Party"]


def test_dedupe_prefers_later_page():
    a = w.Poll("2026-02-01", "2026-02-02", "NHK", 1000, False, {"自由民主党": 30.0}, page="old")
    b = w.Poll("2026-02-01", "2026-02-02", "NHK", 1000, False, {"自由民主党": 31.0}, page="new")
    assert [p.page for p in w.dedupe([a, b])] == ["new"]


def test_update_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "fetch_page", lambda title: HTML)
    path = tmp_path / "polls.json"
    info = w.update_history(path, pages=["p1", "p2"])
    assert info == {"polls": 2, "first": "2026-02-02", "last": "2026-02-07", "unknown_parties": [], "missing": 0}
    import json
    polls = w.from_json(json.loads(path.read_text()))
    assert polls[1].combined[0]["value"] == 12.0
    # 総選挙のあとページの名前が変わって読めなくなった行は、保存済みの値を残して件数を返す
    old = w.Poll(start="2025-01-01", end="2025-01-02", firm="Old", n=1000, n_approx=False, shares={"自由民主党": 30.0})
    path.write_text(json.dumps(w.to_json([old] + polls), ensure_ascii=False))
    info = w.update_history(path, pages=["p1"])
    assert info["missing"] == 1 and info["polls"] == 3 and info["first"] == "2025-01-02"
