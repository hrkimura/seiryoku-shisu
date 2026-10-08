"""前月からの$C_p(t)$の変化を、どの選挙が動かしたかに分解する(記事の「指数への影響の考察」用)。

12か月の後方窓では、月tの値と月t-1の値の差は次の2つの和にちょうど分かれる。
- 窓から外れた効果: 月t-12の選挙を抜いたことによる変化
- 窓に入った効果: 月tの選挙を加えたことによる変化
個々の選挙の影響は、その1件だけを抜いた(外れた選挙なら、その1件だけを先に抜いた)
ときの変化で測る。$C_p$は階層ごとの平方根の重みを持つ非線形な集計なので、個々の
影響の合計は全体の変化と一致しないが、どの選挙が効いたかの順位づけには十分である。

値はすべて記事と同じく、無所属を除いて正規化した割合で計算する。
"""
from __future__ import annotations

from .draft import exclude_independent
from .monthly_index import MonthEntries, _level_aggregate, _months_back

YM = tuple[int, int]


def _c(entries: list) -> dict[str, float]:
    if not entries:
        return {}
    return exclude_independent(_level_aggregate(entries)[0])


def _diff(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {p: after.get(p, 0.0) - before.get(p, 0.0) for p in set(after) | set(before)}


def _window(me: MonthEntries, months: list[YM]) -> list:
    out: list = []
    for ym in months:
        out.extend(me.entries.get(ym, []))
    return out


def _size(d: dict[str, float]) -> float:
    return sum(abs(v) for v in d.values())


def _leave_one_out(base: list, month_entries: list, month_labels: list, top: int, removing: bool) -> list[dict]:
    """month_entriesの各選挙について、1件だけの影響を測る。

    removing=False(窓に入った選挙): base(月tの選挙を含む窓)からその1件を抜いた値との差。
    removing=True(窓から外れた選挙): base(月t-12の選挙を含む前月の窓)からその1件を
    抜いたときの変化。
    """
    c_base = _c(base)
    # 窓に入った選挙はbaseの末尾、外れた選挙はbaseの先頭に並んでいる(decompose参照)。
    start = 0 if removing else len(base) - len(month_entries)
    results = []
    for i, (entry, label) in enumerate(zip(month_entries, month_labels)):
        idx = start + i
        without = base[:idx] + base[idx + 1:]
        effect = _diff(c_base, _c(without)) if not removing else _diff(_c(without), c_base)
        results.append({**label, "shares": entry[3], "effect": effect})
    results.sort(key=lambda r: -_size(r["effect"]))
    return results[:top]


def decompose(me: MonthEntries, ym: YM, window_months: int = 12, top: int = 5) -> dict:
    current = _months_back(ym, window_months)
    exiting_month = _months_back(ym, window_months + 1)[-1]
    middle = current[1:]

    # 窓に入った選挙を末尾に置くため、古い月から順に並べる。
    mid_entries = _window(me, list(reversed(middle)))
    entering = me.entries.get(ym, [])
    exiting = me.entries.get(exiting_month, [])
    cur_entries = mid_entries + entering
    prev_entries = exiting + mid_entries

    c_prev, c_mid, c_cur = _c(prev_entries), _c(mid_entries), _c(cur_entries)

    by_level: dict[str, dict[str, float]] = {}
    for level in sorted({e[0] for e in entering}):
        subset = [e for e in entering if e[0] == level]
        by_level[level] = _diff(_c(mid_entries + subset), c_mid)

    return {
        "year": ym[0], "month": ym[1],
        "exiting_month": {"year": exiting_month[0], "month": exiting_month[1]},
        "total": _diff(c_cur, c_prev),
        "entered": _diff(c_cur, c_mid),
        "exited": _diff(c_mid, c_prev),
        "entered_by_level": by_level,
        "n_entered": len(entering),
        "n_exited": len(exiting),
        "top_entered": _leave_one_out(cur_entries, entering, me.labels.get(ym, []), top, removing=False),
        "top_exited": _leave_one_out(prev_entries, exiting, me.labels.get(exiting_month, []), top, removing=True),
    }
