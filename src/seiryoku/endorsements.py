"""首長選挙の当選者の推薦・支持政党を、複数の情報源から足し合わせて決める。

方針: いずれかの情報源で推薦・支持(本部・地方組織のどちらでも、推薦・支持・支援の
いずれでも)が確認できればその政党を数える。情報源どうしが矛盾した場合は全国首長名簿
を優先する。ここでいう矛盾とは、他の情報源では当選者を推したとされる政党を、名簿が
同じ選挙の落選者の推薦・支持政党として記録している場合である。名簿に記載が無い
ことは矛盾とみなさない(名簿は自民党・維新の地方組織の推薦を記録しないため)。

知事選挙は名簿(jichisoken)とWikipedia(wikipedia_governor)、市区町村長選挙は
名簿のみを使う。市区町村長選挙は大半にWikipediaの記事が無いため。
"""
from __future__ import annotations

from .fetch import jichisoken, wikipedia_governor


def merge_endorsements(
    meibo: list[str], meibo_rivals: list[str], others: list[str],
) -> list[str]:
    merged = list(meibo)
    for p in others:
        if p not in merged and p not in meibo_rivals:
            merged.append(p)
    return merged


def governor_endorsing_parties(
    by_year: dict[str, dict[str, jichisoken.Endorsement]],
    prefecture: str,
    vote_date: str,
    wiki_cache: dict | None = None,
) -> list[str]:
    entry = jichisoken.endorsement_for_vote_year(by_year, prefecture, vote_date)
    meibo = entry.endorsing_parties if entry else []
    rivals = entry.rival_endorsing_parties if entry else []
    wiki = wikipedia_governor.lookup(prefecture, vote_date, cache=wiki_cache)
    others = (wiki or {}).get("parties") or []
    return merge_endorsements(meibo, rivals, others)


def municipal_endorsing_parties(
    by_year: dict[str, dict[str, jichisoken.Endorsement]], name: str, vote_date: str,
) -> list[str]:
    entry = jichisoken.endorsement_for_vote_year(by_year, name, vote_date)
    return entry.endorsing_parties if entry else []
