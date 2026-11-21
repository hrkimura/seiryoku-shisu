"""外国人の増加と参政党の伸びの図。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis  # noqa: E402

OUT = analysis.OUT
FIG = OUT / "figs"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_SANSEI = "#eb6834"
C_IV = "#2a78d6"


def setup() -> None:
    if any("Noto Sans CJK JP" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams.update({"axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def binned(t: pd.DataFrame, x: str, y: str, w: str, bins: int = 20) -> pd.DataFrame:
    """x の重み付き分位で bins 個に分け、各群の x と y の重み付き平均。"""
    t = t.sort_values(x)
    cw = t[w].cumsum() / t[w].sum()
    g = np.minimum((cw * bins).astype(int), bins - 1)
    return t.groupby(g).apply(lambda d: pd.Series({x: np.average(d[x], weights=d[w]),
                                                   y: np.average(d[y], weights=d[w]),
                                                   "weight": d[w].sum()}), include_groups=False)


def fig_binned(t: pd.DataFrame, path: Path) -> None:
    b = binned(t, "d_fshare", "d_share", "valid25")
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter(b["d_fshare"], b["d_share"], s=40, color=C_SANSEI, edgecolor=SURFACE, zorder=3)
    avg = np.average(t["d_share"], weights=t["valid25"])
    ax.axhline(avg, color=INK2, lw=1, ls="--")
    ax.text(b["d_fshare"].max(), avg + 0.15, f"全国平均 +{avg:.1f}ポイント", ha="right", fontsize=9, color=INK2)
    ax.set_ylim(0, max(12, b["d_share"].max() + 1))
    ax.set_xlabel("外国人住民の割合の増加(2022→2025年、ポイント)", fontsize=10)
    ax.set_ylabel("参政党の得票率の伸び(ポイント)", fontsize=10)
    ax.set_title("外国人の増え方で20群に分けた、参政党の伸び", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_effects(r: pd.DataFrame, path: Path) -> None:
    r = r.reset_index(drop=True)
    labels = [f"{a}・{s}" for a, s in zip(r["target"], r["spec"])]
    rows = list(dict.fromkeys(labels))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4), dpi=200)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1)
    for method, color, dy in (("最小二乗", INK2, -0.12), ("操作変数", C_IV, 0.12)):
        for i, lab in enumerate(rows):
            m = (np.array(labels) == lab) & (r["method"] == method).to_numpy()
            if not m.any():
                continue
            e = r[m].iloc[0]
            ax.hlines(i + dy, e["lo"], e["hi"], color=color, lw=1.6)
            ax.scatter(e["coef"], i + dy, s=36, color=color, edgecolor=SURFACE, zorder=3,
                       label=method if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("外国人の割合が1ポイント増えたときの、参政党の得票率の伸びの変化(ポイント)", fontsize=9.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_groups(c: pd.DataFrame, path: Path) -> None:
    """国籍別・在留資格別の増加の効果(同時に推定)。"""
    c = c[c["mean_increase"].abs() >= 0.02].copy()
    c["label"] = c["group"].str.replace("国籍:", "国籍 ").str.replace("資格:", "在留資格 ")
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(c) + 1.4), dpi=200)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1)
    for i, r in enumerate(c.itertuples()):
        col = C_SANSEI if r.p_holm < 0.05 else INK2
        ax.hlines(i, r.lo, r.hi, color=col, lw=1.6)
        ax.scatter(r.coef, i, s=34, color=col, edgecolor=SURFACE, zorder=3)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["label"], fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("その種類の外国人が人口の1ポイント分増えたときの、参政党の伸びの違い(ポイント)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_tourism(p: pd.DataFrame, path: Path) -> None:
    names = dict(enumerate(["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉",
                            "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知", "三重",
                            "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口", "徳島",
                            "香川", "愛媛", "高知", "福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"], start=1))
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter(p["d_fn"], p["d_share"], s=24, color=C_SANSEI, edgecolor=SURFACE, zorder=3)
    for r in p.itertuples():
        if r.d_fn > 1.5 or r.d_share > 11 or r.d_share < 7.5:
            ax.annotate(names[r.pref], (r.d_fn, r.d_share), xytext=(4, 2), textcoords="offset points", fontsize=8,
                        color=INK2)
    ax.set_xlabel("人口1人あたりの外国人延べ宿泊者数の増加(2022→2025年、泊)", fontsize=9.5)
    ax.set_ylabel("参政党の得票率の伸び(ポイント)", fontsize=9.5)
    ax.set_title("都道府県の外国人宿泊者の増え方と、参政党の伸び", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


PLACES = ["倶知安町", "ニセコ町", "千歳市", "大泉町", "伊勢崎市", "太田市", "常総市", "真岡市", "川口市", "新宿区"]


def predictions(t: pd.DataFrame) -> pd.DataFrame:
    """外国人の変数を使わず、県・2022年の得票率・人口・地域の特徴から予測した伸びと、実際の伸び(全単位)。"""
    ctrl = ["share22", "log_pop22"] + analysis.REGION
    d = analysis.complete(t, ["d_share", "valid25"] + ctrl).reset_index(drop=True)
    x = analysis.design(d, ctrl)
    w = d["valid25"].to_numpy(float)
    b = np.linalg.pinv(x.T @ (x * w[:, None])) @ (x.T @ (w * d["d_share"].to_numpy(float)))
    return d.assign(pred=x @ b)


def place_predictions(t: pd.DataFrame, places: list[str]) -> pd.DataFrame:
    d = predictions(t)
    return d[d["unit"].isin(places)].set_index("unit").loc[[p for p in places if p in set(d["unit"])],
                                                          ["d_share", "pred", "fshare22", "fshare25"]]


# ラベルが重ならないよう、町ごとに手で決めた位置(点からのずれ、ポイント単位)
PLACE_OFFSET = {"倶知安町": (-60, 22), "ニセコ町": (14, -16), "千歳市": (-96, -16), "大泉町": (6, 10), "伊勢崎市": (8, 12),
                "太田市": (14, -18), "常総市": (-96, 14), "真岡市": (14, -20), "川口市": (-110, 6), "新宿区": (-96, -4)}


def fig_places(t: pd.DataFrame, path: Path) -> None:
    d = predictions(t)
    r = d[d["unit"].isin(PLACES)].set_index("unit")
    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=200)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter(d["pred"], d["d_share"], s=6, color=AXIS, alpha=0.6, linewidth=0, zorder=1, label="全国の市区町村")
    lim = (1, 21)
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=2)
    ax.annotate("予想どおり", (19.5, 19.5), xytext=(-4, 6), textcoords="offset points", rotation=45,
                rotation_mode="anchor", fontsize=9, color=INK2, ha="right")
    ax.scatter(r["pred"], r["d_share"], s=40, color=C_SANSEI, edgecolor=SURFACE, zorder=3, label="外国人の多さで話題の町")
    for u, x in r.iterrows():
        ax.annotate(f"{u}({x.fshare22:.0f}→{x.fshare25:.0f}%)", (x["pred"], x["d_share"]),
                    xytext=PLACE_OFFSET.get(u, (6, 2)), textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("外国人の変数を使わずに予想した伸び(ポイント)", fontsize=9.5)
    ax.set_ylabel("実際の伸び(ポイント)", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_drivers(t: pd.DataFrame, path: Path) -> None:
    """標準化した地域の特徴と外国人の増加を同時に入れたときの係数。"""
    cols = {"elderly": "高齢者の割合", "secondary": "第2次産業で働く人の割合", "unemp": "失業率",
            "log_density": "人口密度(対数)", "manuf": "製造業の従業者の割合", "fshare22": "外国人の割合(2022年)",
            "d_fshare": "外国人の割合の増加"}
    d = analysis.complete(t, ["d_share", "valid25", "share22", "log_pop22"] + list(cols)).copy()
    for c in cols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    x = np.column_stack([d[list(cols)].to_numpy(float), analysis.design(d, ["share22", "log_pop22"])])
    b, v = analysis.wls(d["d_share"].to_numpy(float), x, d["valid25"].to_numpy(float), d["pref"].to_numpy())
    se = np.sqrt(np.diag(v))[: len(cols)]
    order = np.argsort(-np.abs(b[: len(cols)]))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(cols) + 1.3), dpi=200)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1)
    for i, j in enumerate(order):
        col = C_SANSEI if list(cols)[j] in ("d_fshare", "fshare22") else INK2
        ax.hlines(i, b[j] - 1.96 * se[j], b[j] + 1.96 * se[j], color=col, lw=1.6)
        ax.scatter(b[j], i, s=36, color=col, edgecolor=SURFACE, zorder=3)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([list(cols.values())[j] for j in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("その特徴が標準的な差(1標準偏差)だけ大きいときの、参政党の伸びの違い(ポイント)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    t = analysis.prepare(pd.read_csv(OUT / "units.csv", dtype={"codes": str}))
    fig_binned(t, FIG / "fig_binned.png")
    fig_effects(pd.read_csv(OUT / "main_results.csv"), FIG / "fig_effects.png")
    import extra
    import tourism as tr

    fig_groups(pd.read_csv(OUT / "composition_effects.csv"), FIG / "fig_groups.png")
    fig_tourism(extra.prefecture_table(t, tr.pref_nights(tr.RAW / "2022.xlsx"), tr.pref_nights(tr.RAW / "2025.xlsx")),
                FIG / "fig_tourism.png")
    fig_places(t, FIG / "fig_places.png")
    fig_drivers(t, FIG / "fig_drivers.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
