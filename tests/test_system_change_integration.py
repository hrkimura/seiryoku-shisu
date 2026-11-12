"""保存済みの国政選挙の結果で検知を動かす。2016年以降、衆院は総定数465で変わらず、参院は2019年に
改選121→124になった。"""
from datetime import date

from seiryoku import system_change as sc


def test_real_elections():
    tot = sc.seat_totals("shugiin")
    assert set(tot.values()) == {465}
    san = sc.seat_totals("sangiin")
    assert san[date(2016, 7, 10)] == 121 and san[date(2019, 7, 21)] == 124
    assert all(isinstance(w, str) for w in sc.check())
