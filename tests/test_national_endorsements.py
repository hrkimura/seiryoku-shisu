import json

import pytest

from seiryoku import by_elections as be
from seiryoku import national_endorsements as ne
from seiryoku.trend import Event
from datetime import date


@pytest.mark.parametrize("text,expected", [
    ("立憲民主党・国民民主党推薦", ["立憲民主党", "国民民主党"]),
    ("日本共産党・社会民主党・沖縄社会大衆党支援", ["日本共産党", "社会民主党", "沖縄社会大衆党"]),
    ("立憲民主党沖縄県連・日本共産党沖縄県委員会・れいわ新選組支持", ["立憲民主党", "日本共産党", "いのちの党"]),
    ("（自主支援）立憲民主党埼玉県連・国民民主党埼玉県連・自由民主党の一部", ["立憲民主党", "国民民主党"]),
    ("", []),
])
def test_backers_follow_the_rule(text, expected):
    assert ne.backers(text) == expected


def test_manual_entry_overrides_the_template(tmp_path, monkeypatch):
    path = tmp_path / "manual.json"
    path.write_text(json.dumps({ne.key("2019-04-21", "沖縄県第3区", "屋良朝博"):
                                {"backers": ["立憲民主党", "日本共産党"], "quote": "…", "source": "…"}}))
    monkeypatch.setattr(ne, "MANUAL_PATH", path)
    assert ne.resolve("2019-04-21", "沖縄県第3区", "屋良朝博", "") == ["立憲民主党", "日本共産党"]
    assert ne.resolve("2019-04-21", "大阪府第12区", "樽床伸二", "") == []


def test_general_election_independents_move_to_their_backers(tmp_path, monkeypatch):
    from seiryoku.fetch import wikipedia_districts as wd

    cache = tmp_path / "ind.json"
    cache.write_text(json.dumps({"2019-07-21": {"independents": [
        {"district": "沖縄県選挙区", "name": "a", "rec": "立憲民主党・日本共産党推薦"},
        {"district": "X選挙区", "name": "b", "rec": ""}]}}))
    monkeypatch.setattr(wd, "CACHE_PATH", cache)
    monkeypatch.setattr(ne, "MANUAL_PATH", tmp_path / "none.json")
    assert ne.general_seat_deltas("2019-07-21") == {"無所属": -1.0, "立憲民主党": 0.5, "日本共産党": 0.5}
    e = Event(date(2019, 7, 21), "参院選", 1.2e8, 10.0, {"自由民主党": 0.6, "立憲民主党": 0.2, "無所属": 0.2}, "参院選")
    out, = be.endorse_general_elections([e], {"自由民主党", "立憲民主党", "日本共産党"})
    assert out.shares == pytest.approx({"自由民主党": 0.6, "立憲民主党": 0.25, "日本共産党": 0.05, "無所属": 0.1})
    assert sum(out.shares.values()) == pytest.approx(1.0)


def test_by_election_vote_shares_split_an_independent_between_backers(monkeypatch, tmp_path):
    monkeypatch.setattr(ne, "MANUAL_PATH", tmp_path / "none.json")
    parties = {"自由民主党", "立憲民主党", "国民民主党", "日本共産党", "公明党"}
    plain = {(e.vote_date.isoformat(), e.name): e for e in be.national_vote_events(parties)}
    endorsed = {(e.vote_date.isoformat(), e.name): e for e in be.national_vote_events(parties, endorsed=True)}
    k = ("2021-10-24", "静岡県選挙区")
    assert "立憲民主党" not in plain[k].shares
    assert endorsed[k].shares["立憲民主党"] == pytest.approx(endorsed[k].shares["国民民主党"])
    # 公認候補への推薦は使わないので、自民の得票率は変わらない
    assert endorsed[k].shares["自由民主党"] == pytest.approx(plain[k].shares["自由民主党"])
