"""実際のWikipediaのページ(取得済みならキャッシュ)を読み、表の構成が想定どおりかを確かめる。"""
from collections import Counter

from seiryoku.fetch import wikipedia_polls as w


def test_real_pages():
    polls, unknown = w.fetch_all()
    assert unknown == []
    assert len(polls) > 1000
    assert polls[0].end < "2018-01-01"
    by_year = Counter(p.end[:4] for p in polls)
    assert all(by_year[str(y)] >= 90 for y in range(2018, 2026))
    for party in ("自由民主党", "日本共産党", "日本維新の会"):
        assert all(party in p.shares for p in polls)
    # 政党の値の合計は支持政党のある人の割合なので、0〜100に収まる
    assert all(0 < w.party_total(p) <= 100 for p in polls)
