import pytest

from seiryoku.fetch import miharu

_BODY = (
    "<h3>内閣支持率・不支持率の平均（2026年10月6日更新）</h3>"
    "<blockquote><p>内閣支持率 55.2<br>不支持率　 31.4</p></blockquote>"
    "<h3>政党支持率の平均（2026年10月6日更新）</h3>"
    "<figure><img src=\"x.png\"></figure><h3>10％未満拡大</h3>"
    "<p>　現在の平均値（％）は次の通りです。</p>"
    "<blockquote><p>無党派 37.7<br>自民　 33.2<br>いのち 0.4<br>新党　 0.2</p></blockquote>"
    "<p>　グラフの投稿は下記のSNSでも行っています。</p>"
)


def test_parse_support_reads_party_block_not_cabinet_block():
    r = miharu.parse_support(_BODY)
    assert r["updated"] == "2026-10-06"
    assert r["S_p"] == {"無党派": 37.7, "自由民主党": 33.2, "いのちの党": 0.4, "新党": 0.2}
    assert "内閣支持率" not in r["S_p"]


def test_parse_support_keeps_unknown_label_and_reports_it():
    assert miharu.parse_support(_BODY)["unknown_labels"] == ["新党"]


def test_parse_support_raises_when_heading_missing():
    with pytest.raises(miharu.ParseError):
        miharu.parse_support("<p>無関係な記事</p>")
