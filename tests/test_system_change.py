import math
from datetime import date

import numpy as np

from seiryoku import monthly_article as ma
from seiryoku import system_change as sc
from seiryoku.state_space import SeatVotePoint


def _election(d, k, c, rng, total=465):
    votes = [0.35, 0.2, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04]
    pts = []
    for i, v in enumerate(votes):
        x = 1 / (1 + math.exp(-(k * math.log(v / (1 - v)) + c)))
        pts.append(SeatVotePoint(d, f"p{i}", rng.binomial(total, x) / total, v, total))
    return pts


def _history(k_last, seed=0):
    rng = np.random.default_rng(seed)
    ds = [date(2014, 12, 14), date(2017, 10, 22), date(2021, 10, 31), date(2024, 10, 27)]
    pts = []
    for d in ds[:-1]:
        pts += _election(d, 2.0, 0.5, rng)
    return pts + _election(ds[-1], k_last, 0.5, rng)


def test_same_curve_is_not_flagged():
    cc = sc.curve_change(_history(2.0), "shugiin")
    assert cc.p_value > sc.ALPHA and math.isclose(cc.k_before, 2.0, abs_tol=0.3)


def test_changed_amplification_is_flagged():
    cc = sc.curve_change(_history(3.0), "shugiin")
    assert cc.p_value < sc.ALPHA and cc.k_latest > cc.k_before + 0.5
    w = sc.warnings_for({"shugiin": _history(3.0)}, {})
    assert len(w) == 1 and w[0].startswith("[要確認] 選挙制度の変化: 衆院の2024-10-27")


def test_too_few_elections():
    rng = np.random.default_rng(0)
    assert sc.curve_change(_election(date(2024, 1, 1), 2, 0, rng) + _election(date(2025, 1, 1), 2, 0, rng),
                           "shugiin") is None


def test_total_change():
    assert sc.total_change({date(2016, 7, 10): 121, date(2019, 7, 21): 124}) == (date(2019, 7, 21), 121, 124)
    assert sc.total_change({date(2022, 7, 10): 124, date(2025, 7, 20): 124}) is None
    w = sc.warnings_for({}, {"sangiin": {date(2016, 7, 10): 121, date(2019, 7, 21): 124}})
    assert w == ["[要確認] 選挙制度の変化: 参院の議席の総数が変わった(2019-07-21の選挙で121→124)。"
                 "定数・区割り・小選挙区と比例の比率の改正がなかったか調べる。"]


def test_collect_warnings_includes_system_change():
    w = ma.collect_warnings("", [], None, False, False, 0, None, ["[要確認] 選挙制度の変化: x"])
    assert w == ["[要確認] 選挙制度の変化: x"]
