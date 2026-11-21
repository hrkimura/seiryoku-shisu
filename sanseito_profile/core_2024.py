"""参政党が大きく伸びる前からの支持者(2024年衆院選の比例で参政党に投票、2025年調査)と、
2025年参院選の比例で参政党に投票した人(2026年調査)を、共通の設問で比べる。

2025年調査には外国人についての設問がないので、外国人への態度は比べられない。
気候変動の設問は2025年と2026年で文言が違う(「生活水準を犠牲に」と「経済や暮らしを見直す」)ので比べない。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from profile_utas import OUT, wilson

DATA_2025 = Path(__file__).resolve().parent / "data" / "utas" / "2025UTASV20250812.csv"
DATA_2026 = Path(__file__).resolve().parent / "data" / "utas" / "2026UTASV20260602.csv"
SANSEI_2024, VOTED_2024 = 8, set(range(1, 12))
SANSEI_2025, VOTED_2025 = 6, set(range(1, 13))
AGREE = frozenset({1, 2})


@dataclass(frozen=True)
class Common:
    label: str
    col25: str
    col26: str
    yes: frozenset[int]
    valid: frozenset[int] | None = None


def _r(a: int, b: int) -> frozenset[int]:
    return frozenset(range(a, b + 1))


ITEMS = [
    Common("18〜39歳", "Q48", "Q40", _r(1, 5)),
    Common("40〜59歳", "Q48", "Q40", _r(6, 9)),
    Common("60歳以上", "Q48", "Q40", _r(10, 15)),
    Common("男性", "Q47", "Q39", frozenset({1}), frozenset({1, 2})),
    Common("大学・大学院卒", "Q51", "Q43", frozenset({5, 6}), _r(1, 6)),
    Common("新聞", "Q57_1", "Q49_1", frozenset({1})),
    Common("テレビ", "Q57_2", "Q49_2", frozenset({1})),
    Common("X(旧Twitter)", "Q57_5", "Q49_5", frozenset({1})),
    Common("YouTube", "Q57_6", "Q49_6", frozenset({1})),
    Common("防衛力を強化すべき", "Q43_1", "Q35_1", AGREE, _r(1, 5)),
    Common("小さな政府の方が良い", "Q43_4", "Q35_4", AGREE, _r(1, 5)),
    Common("公共事業による雇用確保は必要", "Q43_5", "Q35_5", AGREE, _r(1, 5)),
    Common("財政再建より財政出動", "Q43_6", "Q35_6", AGREE, _r(1, 5)),
    Common("治安のための権利制約は当然", "Q43_10", "Q35_9", AGREE, _r(1, 5)),
    Common("同性婚を認めるべき", "Q43_11", "Q35_10", AGREE, _r(1, 5)),
    Common("選択的夫婦別姓を認めるべき", "Q43_12", "Q35_11", AGREE, _r(1, 5)),
    Common("中国は脅威", "Q44_1", "Q36_1", AGREE, _r(1, 5)),
    Common("格差より経済競争力", "Q44_2", "Q36_2", AGREE, _r(1, 5)),
    Common("原発をすぐ廃止", "Q44_3", "Q36_3", AGREE, _r(1, 5)),
]


def share(s: pd.Series, yes: frozenset[int], valid: frozenset[int] | None) -> tuple[float, float, float, int]:
    """該当する割合(%)と Wilson の95%区間、分母の人数。valid が None なら無回答(99)以外を分母にする。"""
    v = s[s.isin(valid)] if valid is not None else s[s != 99].dropna()
    k, n = int(v.isin(yes).sum()), len(v)
    lo, hi = wilson(k, n)
    return (k / n * 100 if n else float("nan"), lo * 100, hi * 100, n)


def compare(d25: pd.DataFrame, d26: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "2024年から参政党(29人)": (d25[d25["Q3"] == SANSEI_2024], "col25"),
        "2024年に他党": (d25[d25["Q3"].isin(VOTED_2024 - {SANSEI_2024})], "col25"),
        "2025年に参政党": (d26[d26["Q4"] == SANSEI_2025], "col26"),
        "2025年に他党": (d26[d26["Q4"].isin(VOTED_2025 - {SANSEI_2025})], "col26"),
    }
    rows = []
    for it in ITEMS:
        for g, (d, key) in groups.items():
            p, lo, hi, n = share(d[getattr(it, key)], it.yes, it.valid)
            rows.append({"item": it.label, "group": g, "pct": p, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


FIG_ITEMS = ["60歳以上", "YouTube", "X(旧Twitter)", "新聞", "財政再建より財政出動", "中国は脅威", "防衛力を強化すべき",
             "格差より経済競争力", "選択的夫婦別姓を認めるべき"]
FIG_GROUPS = {"2024年から参政党(29人)": "#eb6834", "2025年に参政党": "#2a78d6", "2025年に他党": "#a9a8a1"}
FIG_LABELS = {"2024年から参政党(29人)": "伸びる前から参政党(2024年衆院選、29人)", "2025年に参政党": "2025年参院選で参政党(103人)",
              "2025年に他党": "2025年参院選で他党"}


def fig_core(t: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    import figures as fg

    fg.setup()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(FIG_ITEMS) + 1.4), dpi=200)
    ax.grid(True, axis="x", color=fg.GRID, lw=0.6)
    ax.set_axisbelow(True)
    for k, (g, col) in enumerate(FIG_GROUPS.items()):
        dy = (k - 1) * 0.22
        d = t[t["group"] == g].set_index("item")
        for i, it in enumerate(FIG_ITEMS):
            r = d.loc[it]
            ax.hlines(i + dy, r["lo"], r["hi"], color=col, lw=1.4)
            ax.scatter(r["pct"], i + dy, s=30, color=col, edgecolor=fg.SURFACE, zorder=3,
                       label=FIG_LABELS[g] if i == 0 else None)
    ax.set_yticks(range(len(FIG_ITEMS)))
    ax.set_yticklabels(FIG_ITEMS, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("該当する人・賛成する人の割合(%)", fontsize=9.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fg.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    t = compare(pd.read_csv(DATA_2025), pd.read_csv(DATA_2026))
    t.to_csv(OUT / "core_2024_compare.csv", index=False)
    w = t.pivot(index="item", columns="group", values="pct").reindex([i.label for i in ITEMS])
    print(w.round(0).to_string())
    fig_core(t, OUT / "figs" / "fig_core_2024.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
