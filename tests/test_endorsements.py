from seiryoku import endorsements
from seiryoku.fetch.jichisoken import Endorsement


def test_merge_adds_parties_found_only_in_other_sources():
    assert endorsements.merge_endorsements(["公明党"], [], ["自由民主党", "公明党"]) == ["公明党", "自由民主党"]


def test_merge_drops_parties_that_the_meibo_records_for_a_rival():
    assert endorsements.merge_endorsements([], ["立憲民主党"], ["立憲民主党", "国民民主党"]) == ["国民民主党"]


def test_governor_endorsing_parties_combines_meibo_and_wikipedia(monkeypatch):
    by_year = {"2025": {"栃木県": Endorsement(
        "栃木県", "無所属", ["公明党"], "2024年", "2025", rival_endorsing_parties=["日本共産党"],
    )}}
    monkeypatch.setattr(
        endorsements.wikipedia_governor, "lookup",
        lambda pref, date, cache=None: {"parties": ["自由民主党", "公明党", "日本共産党"]},
    )
    assert endorsements.governor_endorsing_parties(by_year, "栃木県", "2024-11-17") == ["公明党", "自由民主党"]


def test_governor_endorsing_parties_uses_wikipedia_alone_before_the_meibo_is_published(monkeypatch):
    monkeypatch.setattr(
        endorsements.wikipedia_governor, "lookup", lambda pref, date, cache=None: {"parties": ["自由民主党"]},
    )
    assert endorsements.governor_endorsing_parties({}, "沖縄県", "2026-09-13") == ["自由民主党"]


def test_governor_endorsing_parties_falls_back_to_meibo_when_wikipedia_unavailable(monkeypatch):
    by_year = {"2025": {"千葉県": Endorsement("千葉県", "無所属", ["公明党"], "2025年", "2025")}}
    monkeypatch.setattr(endorsements.wikipedia_governor, "lookup", lambda pref, date, cache=None: None)
    assert endorsements.governor_endorsing_parties(by_year, "千葉県", "2025-03-16") == ["公明党"]
