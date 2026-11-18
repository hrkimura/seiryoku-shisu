"""note用の図を描く。"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from party_distance import ATTITUDE, PARTIES, indicator_matrix, interval95, resample_within, unbiased_distance
from profile_utas import ITEMS, OUT, PARTY_Q4, indicator, load, wilson

FIG = OUT / "figs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_SANSEI = "#eb6834"
C_REIWA = "#2a78d6"

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

PROFILE_DEMO = ["男性", "18〜39歳", "40〜59歳", "60歳以上", "大学・大学院卒", "既婚(配偶者あり)",
                "生活水準が中の下以下", "新聞", "テレビ", "SNS", "動画サイト"]
DEMO_DISPLAY = {"新聞": "情報源:新聞", "テレビ": "情報源:テレビ", "SNS": "情報源:SNS", "動画サイト": "情報源:動画サイト"}


def setup() -> None:
    if Path(FONT).exists():
        fm.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    })


def distance_with_ci(df: pd.DataFrame, labels_demo: list[str], n_boot: int = 1000, seed: int = 2) -> pd.DataFrame:
    """全ペアの偏りを除いた距離と、党内で復元抽出したブートストラップの95%区間。"""
    rng = np.random.default_rng(seed)
    demo, att = indicator_matrix(df, labels_demo), indicator_matrix(df, ATTITUDE)
    party = df["Q4"].to_numpy()
    pairs = list(itertools.combinations(PARTIES, 2))
    q = df["Q4"]
    point = {"demo": [unbiased_distance(demo, q, a, b) for a, b in pairs],
             "att": [unbiased_distance(att, q, a, b) for a, b in pairs]}
    draws = {"demo": [], "att": []}
    for _ in range(n_boot):
        idx = resample_within(party, PARTIES, rng)
        pb = party[idx]
        draws["demo"].append([unbiased_distance(demo.iloc[idx], pb, a, b) for a, b in pairs])
        draws["att"].append([unbiased_distance(att.iloc[idx], pb, a, b) for a, b in pairs])
    out = pd.DataFrame({"a": [p for p, _ in pairs], "b": [r for _, r in pairs]})
    for k in ("demo", "att"):
        v = np.array(draws[k])
        out[k] = point[k]
        out[f"{k}_lo"], out[f"{k}_hi"] = zip(*(interval95(v[:, j]) for j in range(v.shape[1])))
    return out


def clip_at_zero(t: pd.DataFrame) -> pd.DataFrame:
    """誤差の分を引いた距離は、観測された差が誤差で期待される大きさより小さいと負になる。
    負の推定値は0に切り詰めて描く(分散成分の推定などでの一般的な扱い)。"""
    t = t.copy()
    cols = [c for c in t.columns if c.startswith(("demo", "att"))]
    t[cols] = t[cols].clip(lower=0)
    return t


def fig_distance(t: pd.DataFrame, path: Path) -> None:
    t = clip_at_zero(t)
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=200)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    is_s = (t["a"] == 6) | (t["b"] == 6)
    for focus, color, z in ((False, MUTED, 2), (True, C_SANSEI, 3)):
        s = t[is_s == focus]
        # 偏りを除いた距離はブートストラップ分布の中心から少しずれ、点が区間の端の外に出ることがあるので、
        # errorbar(点からの幅)ではなく区間の両端をそのまま描く
        ax.hlines(s["att"], s["demo_lo"], s["demo_hi"], color=color, lw=0.8, alpha=0.35, zorder=z)
        ax.vlines(s["demo"], s["att_lo"], s["att_hi"], color=color, lw=0.8, alpha=0.35, zorder=z)
        ax.scatter(s["demo"], s["att"], s=46 if focus else 30, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=z + 1)
    for _, r in t.iterrows():
        a, b = PARTY_Q4[r.a], PARTY_Q4[r.b]
        if 6 in (r.a, r.b):
            other = b if r.a == 6 else a
            ax.annotate(f"参政−{other}", (r.demo, r.att), xytext=(6, 4), textcoords="offset points",
                        fontsize=10.5, color=INK, weight="bold" if other in ("れいわ", "共産") else "normal")
        elif f"{a}-{b}" in ("自民-共産", "立憲-共産", "自民-国民"):
            ax.annotate(f"{a}−{b}", (r.demo, r.att), xytext=(6, -10), textcoords="offset points",
                        fontsize=9, color=INK2)
    ax.set_xlim(left=-0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("属性の違い(性別・年齢・学歴・婚姻・職業・生活水準)  ←似ている  違う→", fontsize=10.5)
    ax.set_ylabel(f"政策態度の違い({len(ATTITUDE)}項目)  ←似ている  違う→", fontsize=10.5)
    ax.set_title("投票者どうしの「属性の違い」と「政策態度の違い」(主要8党の全28ペア)", fontsize=12, loc="left")
    ax.text(0.0, -0.12, "違いは、項目ごとの該当割合の差を二乗平均した値の平方根(%ポイント)。人数が少ないと違いがなくても誤差だけで距離が出るため、\n"
            "その分を差し引いた。差し引きすぎて負になった場合は0とした(0は誤差と区別できないことを示す)。\n"
            "線は標本を入れ替えて計算し直した95%区間。橙は参政党を含むペア。データ: 東大谷口研・朝日2026年有権者調査(2025年参院選の比例投票先)",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def shares_ci(df: pd.DataFrame, mask: pd.Series, label: str) -> tuple[float, float, float]:
    it = {i.label: i for i in ITEMS}[label]
    y = indicator(df.loc[mask, it.col], it).dropna()
    k, n = int(y.sum()), len(y)
    lo, hi = wilson(k, n)
    return k / n * 100, lo * 100, hi * 100


def fig_profile(df: pd.DataFrame, path: Path) -> None:
    groups = [("参政党", df["Q4"] == 6, C_SANSEI), ("れいわ", df["Q4"] == 8, C_REIWA),
              ("その他の党の投票者", ~df["Q4"].isin([6, 8]), MUTED)]
    att = sorted(ATTITUDE, key=lambda lab: abs(shares_ci(df, df.Q4 == 6, lab)[0] - shares_ci(df, df.Q4 == 8, lab)[0]))
    panels = [("属性・情報源", PROFILE_DEMO), ("政策態度(賛成の割合。参政党とれいわの差が小さい順)", att)]
    heights = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 0.32 * sum(heights) + 2.4), dpi=200,
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    for ax, (title, labels) in zip(axes, panels):
        ys = np.arange(len(labels))[::-1]
        for gi, (name, mask, color) in enumerate(groups):
            off = (1 - gi) * 0.22
            vals = np.array([shares_ci(df, mask, lab) for lab in labels])
            hollow = gi == 2
            ax.hlines(ys + off, vals[:, 1], vals[:, 2], color=color, lw=1.4, alpha=0.55)
            ax.scatter(vals[:, 0], ys + off, s=34, color=SURFACE if hollow else color, edgecolor=color,
                       linewidth=1.6 if hollow else 1.2, zorder=3, label=name)
        ax.set_yticks(ys)
        ax.set_yticklabels([DEMO_DISPLAY.get(lab, lab) for lab in labels], fontsize=9.5)
        ax.set_xlim(0, 100)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.06), ncol=3, frameon=False, fontsize=9.5)
    axes[-1].set_xlabel("割合(%)。線は95%区間", fontsize=10)
    n_s, n_r = int((df.Q4 == 6).sum()), int((df.Q4 == 8).sum())
    axes[-1].text(0.0, -0.10, f"2025年参院選の比例で、参政党に投票した{n_s}人、れいわに投票した{n_r}人、その他の党に投票した{len(df) - n_s - n_r}人。"
                  "データ: 東大谷口研・朝日2026年有権者調査", transform=axes[-1].transAxes, fontsize=8, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def mean_ci(x: pd.Series) -> tuple[float, float, float]:
    from scipy import stats
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def fig_lca(t: pd.DataFrame, n_resp: dict[int, int], path: Path) -> None:
    """潜在クラスごとの、2025年参院選の比例で参政党に入れた人の割合。"""
    from lca import CLASS_NAMES
    t = t.sort_values("share").reset_index(drop=True)
    overall = t["n_sanseito"].sum() / t["n_voted"].sum() * 100
    total = sum(n_resp.values())
    fig, ax = plt.subplots(figsize=(8.5, 3.9), dpi=200)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(overall, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(overall + 0.4, -0.45, f"全体 {overall:.1f}%", color=INK2, fontsize=9, va="center")
    for i, r in t.iterrows():
        lo, hi = wilson(int(r.n_sanseito), int(r.n_voted))
        ax.hlines(i, lo * 100, hi * 100, color=C_SANSEI, lw=1.6, alpha=0.55)
        ax.scatter(r.share * 100, i, s=46, color=C_SANSEI, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(hi * 100 + 0.8, i, f"参政党の投票者の{r.share_of_all_sanseito * 100:.0f}%", va="center",
                fontsize=9, color=INK2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{CLASS_NAMES[int(c)]}(回答者の{n_resp[int(c)] / total * 100:.0f}%)" for c in t["cluster"]],
                       fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.8, len(t) - 0.5)
    ax.set_xlabel("2025年参院選の比例で参政党に投票した人の割合(%、そのクラスの投票者のうち)。線は95%区間", fontsize=9.5)
    ax.set_title("政策態度で分けた有権者のクラスと、参政党の投票者の割合", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_thermo(stay: pd.DataFrame, back: pd.DataFrame, path: Path) -> None:
    """2025年の参政党の投票者を2026年の投票先で分け、好き嫌い(0〜10点)の平均を比べる。"""
    from subgroups import THERMO
    labels = list(THERMO)
    fig, ax = plt.subplots(figsize=(8, 3.0), dpi=200)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    answered = lambda g: int(g[list(THERMO.values())].apply(lambda c: c.between(0, 10)).all(axis=1).sum())
    groups = [(f"2026年も参政党に投票({len(stay)}人、うち3つとも回答{answered(stay)}人)", stay, C_SANSEI, 0.15),
              (f"2026年は自民党に投票({len(back)}人、うち3つとも回答{answered(back)}人)", back, C_REIWA, -0.15)]
    for name, g, color, off in groups:
        for i, lab in enumerate(labels):
            col = THERMO[lab]
            m, lo, hi = mean_ci(g[col][g[col].between(0, 10)])
            y = len(labels) - 1 - i + off
            ax.hlines(y, lo, hi, color=color, lw=1.6, alpha=0.55)
            ax.scatter(m, y, s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=name if i == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 10)
    ax.set_xlabel("好き嫌いの平均(0=とても嫌い、10=とても好き)。線は95%区間", fontsize=9.5)
    ax.set_title("2025年に参政党に投票した人の、2026年の投票先別の好き嫌い", fontsize=12, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    d = df[df["Q4"].isin(PARTIES)]
    from party_distance import DEMO
    t = distance_with_ci(d, DEMO)
    t.to_csv(OUT / "utas2026_party_distance_ci.csv", index=False)
    fig_distance(t, FIG / "fig_distance.png")
    fig_profile(df, FIG / "fig_profile.png")
    lca_t = pd.read_csv(OUT / "utas2026_lca_sanseito.csv")
    fig_lca(lca_t, dict(zip(lca_t["cluster"], lca_t["n"])), FIG / "fig_lca.png")
    s = df[df["Q4"] == 6]
    fig_thermo(s[s["Q3"] == 5], s[s["Q3"] == 1], FIG / "fig_thermo.png")
    print("saved", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    sys.exit(main())
