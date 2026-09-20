"""図1(党勢指数チャート)を生成する。

2026-09のセッションで作り込んだmatplotlibスクリプト(政党別リッジライン・
選挙種別の重みシェア・首相在任期間の3段構成、太字フォント、8:2.2:0.9の
高さ比、タイトル/説明文は画像に含めずキャプションはnote本文側に書く)を、
data/cp_history.json・data/national_election_months.jsonから再生成できる
形で正式なコードに昇格したもの(月一更新の自動化)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import national_elections
from .fetch import jichisoken

OUT_PATH = Path(__file__).resolve().parents[2] / "notes" / "images" / "cp_and_level_share.png"

# 表示順・色は編集上の選択であり、データから機械的に決まるものではない
# (2026-09にユーザー指示で政党間の水準比較をしない設計にしたため、順序自体に
# 「大きい順」のような意味は無い。政党要件を満たす政党が増えたら末尾に追加する)。
PARTY_ORDER = [
    "自由民主党", "公明党", "立憲民主党", "中道改革連合", "日本維新の会",
    "国民民主党", "参政党", "チームみらい", "日本共産党", "いのちの党",
]

LEVEL_ORDER = ["衆院選", "参院選", "都道府県知事", "都道府県議会", "市区町村長", "市区町村議会"]
LEVEL_COLORS = ["#c0392b", "#e67e22", "#8e44ad", "#2980b9", "#27ae60", "#7f8c8d"]

# 首相の在任期間(発足月)。出典: 首相官邸・各内閣公式発表。更新頻度が低いため
# 定数として持ち、首相交代のたびに手動で追記する。
PM_PERIODS = [
    (2018, 1, "安倍晋三"),
    (2020, 9, "菅義偉"),
    (2021, 10, "岸田文雄"),
    (2024, 10, "石破茂"),
    (2025, 10, "高市早苗"),
]
PM_COLORS = ["#7f8c8d", "#16a085", "#2980b9", "#c0392b", "#8e44ad"]
# 内閣改造(政権交代を伴わないもの)の年月。首相交代と重なるものは含めない。
CABINET_RESHUFFLES = [(2018, 10), (2019, 9), (2022, 8), (2023, 9)]


def _ym_index(year: int, month: int, min_year: int) -> int:
    return (year - min_year) * 12 + (month - 1)


def _unified_local_election_months(min_year: int, max_year: int) -> list[tuple[int, int]]:
    """統一地方選は1947年以来4年に1度、4月に行われる(公職選挙法の統一選挙制度)。
    spike検出のようなデータ依存の推定はせず、周期から機械的に列挙する。"""
    return [(y, 4) for y in range(1947, max_year + 1, 4) if y >= min_year]


def _provisional_start() -> tuple[int, int] | None:
    """首長の推薦・支持データ(jichisoken)の最新版がどこまで収録しているかから、
    「速報値」の起点を動的に決める。全国首長名簿のNN年版は(NN-1)年5月1日から
    NN年4月30日までの選挙を収録する、という2026-09に実サイトで確認したパターンを
    使う(design_document.tex \\S2.2参照)。年版が見つからなければNoneを返し、
    呼び出し側は「速報値の塗り分けをしない」扱いにする。"""
    try:
        years = [int(y) for y in jichisoken.governor_endorsements_by_year().keys()]
    except Exception:
        return None
    if not years:
        return None
    return (max(years), 5)


def _build_focus_series(
    history: list[dict], numerator_history: list[dict], denominator_history: list[dict],
    party_order: list[str],
) -> dict[str, tuple[list[int], list[float]]]:
    """historyのインデックスに揃えた「numerator/(numerator+denominator)*100」の
    政党別系列を作る(地方重視率・首長重視率で共用、2026-09に追加)。"""
    num_by_ym = {(h["year"], h["month"]): h["C_p"] for h in numerator_history}
    den_by_ym = {(h["year"], h["month"]): h["C_p"] for h in denominator_history}
    series: dict[str, tuple[list[int], list[float]]] = {}
    for party in party_order:
        xs, ys = [], []
        for idx, h in enumerate(history):
            ym = (h["year"], h["month"])
            if ym not in num_by_ym or ym not in den_by_ym:
                continue
            num = num_by_ym[ym].get(party, 0.0)
            den = den_by_ym[ym].get(party, 0.0)
            if num + den <= 0:
                continue
            xs.append(idx)
            ys.append(num / (num + den) * 100)
        if len(xs) >= 3:
            series[party] = (xs, ys)
    return series


def _render_focus_panel(
    ax, focus_series: dict[str, tuple[list[int], list[float]]], ylabel: str,
    n_months: int, provisional_idx: int | None, events: list[tuple[int, str]],
    party_colors: dict, focus_min_gap: float,
) -> None:
    """地方重視率・首長重視率パネルの共通描画ロジック(2026-09に追加、
    generate_main_chart参照)。系列末尾に直接ラベル+リーダー線を置き、
    ラベルどうしが重ならないよう最低間隔(focus_min_gap)を確保する。"""
    label_x = n_months - 1 + n_months * 0.05
    line_ends = []
    for party, (xs, ys) in focus_series.items():
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.1, color=party_colors[party], zorder=3)
        line_ends.append([xs[-1], ys[-1], party])

    label_positions = sorted((list(e) for e in line_ends), key=lambda e: e[1])
    for i in range(1, len(label_positions)):
        if label_positions[i][1] - label_positions[i - 1][1] < focus_min_gap:
            label_positions[i][1] = label_positions[i - 1][1] + focus_min_gap
    label_y_by_party = {party: y_label for _, y_label, party in label_positions}
    for x_end, y_actual, party in line_ends:
        y_label = label_y_by_party[party]
        ax.plot(
            [x_end, label_x * 0.99], [y_actual, y_label],
            color=party_colors[party], alpha=0.5, linewidth=0.7, zorder=2,
        )
        ax.text(
            label_x, y_label, party, fontsize=9,
            color=party_colors[party], va="center", ha="left", zorder=5,
        )

    ax.axhline(50, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.0, linestyle=":", zorder=4)
    for idx, _ in events:
        ax.axvline(idx, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def generate_main_chart(
    history: list[dict],
    out_path: Path = OUT_PATH,
    reshuffle: bool = True,
    national_history: list[dict] | None = None,
    local_history: list[dict] | None = None,
    executive_history: list[dict] | None = None,
    local_assembly_history: list[dict] | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    if not history:
        raise ValueError("historyが空です(cp_history.load_history()の結果を渡してください)")

    history = sorted(history, key=lambda h: (h["year"], h["month"]))
    min_year = history[0]["year"]
    n_months = len(history)
    x = list(range(n_months))

    party_order = list(PARTY_ORDER)
    for h in history:
        for party in h["C_p"]:
            if party != "無所属" and party not in party_order:
                party_order.append(party)
    party_colors = dict(zip(party_order, plt.cm.tab10(np.linspace(0, 1, max(10, len(party_order))))))

    election_events = national_elections.compute_election_events()
    local_events = _unified_local_election_months(min_year, history[-1]["year"])
    events = []
    for (y, m), label in election_events:
        idx = _ym_index(y, m, min_year)
        if 0 <= idx < n_months:
            events.append((idx, f"{y % 100}/{m} {label}"))
    for y, m in local_events:
        idx = _ym_index(y, m, min_year)
        if 0 <= idx < n_months:
            events.append((idx, f"{y % 100}/{m} 統一地方選"))
    events.sort()

    provisional_start = _provisional_start()
    provisional_idx = (
        _ym_index(*provisional_start, min_year) if provisional_start else None
    )

    pm_idx = [(_ym_index(y, m, min_year), name) for y, m, name in PM_PERIODS]
    pm_periods = []
    for i, (idx, name) in enumerate(pm_idx):
        end = pm_idx[i + 1][0] if i + 1 < len(pm_idx) else n_months - 1
        pm_periods.append((idx, end, name))
    reshuffle_idxs = [_ym_index(y, m, min_year) for y, m in CABINET_RESHUFFLES]

    show_focus_panel = national_history is not None and local_history is not None
    show_executive_panel = executive_history is not None and local_assembly_history is not None

    focus_series = (
        _build_focus_series(history, local_history, national_history, party_order)
        if show_focus_panel else {}
    )
    executive_series = (
        _build_focus_series(history, executive_history, local_assembly_history, party_order)
        if show_executive_panel else {}
    )

    # ラベルどうしが重ならないよう最低間隔(パーセント値換算)を確保しつつ、
    # 系列本数が多いほどパネルを縦に伸ばして間隔を確保する(2026-09にユーザー
    # 指示: ラベルと線が重ならない程度にグラフの高さを広げる)。
    focus_min_gap = 6.0
    focus_height = max(2.6, 0.6 * len(focus_series))
    executive_height = max(2.6, 0.6 * len(executive_series))

    n_panels = 3 + int(show_focus_panel) + int(show_executive_panel)
    height_ratios = [8]
    fig_height = 13.4
    if show_focus_panel:
        height_ratios.append(focus_height)
        fig_height += focus_height
    if show_executive_panel:
        height_ratios.append(executive_height)
        fig_height += executive_height
    height_ratios += [2.2, 0.9]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, fig_height), dpi=260, sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.06},
    )
    axes = list(axes)
    ax1 = axes.pop(0)
    ax3 = axes.pop(-1)
    ax2 = axes.pop(-1)
    ax4 = axes.pop(0) if show_focus_panel else None
    ax5 = axes.pop(0) if show_executive_panel else None

    band = 1.0
    n_party = len(party_order)
    for i, party in enumerate(party_order):
        baseline = (n_party - 1 - i) * band
        ys = [h["C_p"].get(party, 0.0) for h in history]
        party_max = max(ys) if max(ys) > 0 else 0.01
        normalized = [baseline + (v / party_max) * band * 0.85 for v in ys]
        ax1.plot(x, normalized, color=party_colors[party], linewidth=1.6, zorder=3)
        ax1.fill_between(x, [baseline] * n_months, normalized, color=party_colors[party], alpha=0.15, zorder=2)
        ax1.axhline(baseline, color="#cccccc", linewidth=0.8, zorder=1)
        ax1.text(-3, baseline, party, fontsize=13, ha="right", va="bottom", color=party_colors[party])

    for idx, _ in events:
        ax1.axvline(idx, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    for idx, label in events:
        ax1.text(idx, n_party * band + 0.3, label, fontsize=9.5, rotation=90, ha="left", va="bottom", color="#333333")

    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax1.axvspan(provisional_idx - 0.5, n_months - 1, color="#f2c14e", alpha=0.22, zorder=0)
        ax1.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.4, zorder=1)
        ax1.text(
            provisional_idx - 0.5, n_party * band + 2.62, " 速報値 ", fontsize=12,
            ha="left", va="top", color="#5c4400", weight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff3cd", edgecolor="#b8860b", linewidth=1.3),
            zorder=6,
        )

    ax1.set_yticks([])
    ax1.set_ylim(-0.3, n_party * band + 2.7)
    right_margin = n_months * 0.13 if (show_focus_panel or show_executive_panel) else 0
    ax1.set_xlim(-18, n_months - 1 + right_margin)
    for spine in ["top", "right", "left"]:
        ax1.spines[spine].set_visible(False)

    level_data = {t: [h["level_weight_share"].get(t, 0.0) * 100 for h in history] for t in LEVEL_ORDER}
    ax2.stackplot(x, [level_data[t] for t in LEVEL_ORDER], labels=LEVEL_ORDER, colors=LEVEL_COLORS, alpha=0.85, zorder=2)
    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax2.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.0, linestyle=":", zorder=4)
    for idx, _ in events:
        ax2.axvline(idx, color="#555555", linewidth=0.8, linestyle="--", zorder=5)

    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("重みシェア (%)", fontsize=13)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    handles, labels = ax2.get_legend_handles_labels()
    # stackplotは積み上げの下から上へ順にhandlesを返すため、凡例が積み上げ図の
    # 見た目(上から下)と逆順になる。凡例を逆順にして見た目の並びに合わせる。
    ax2.legend(
        handles[::-1], labels[::-1],
        loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=11, frameon=False,
    )

    if show_focus_panel:
        _render_focus_panel(
            ax4, focus_series, "地方重視率 (%)", n_months, provisional_idx, events,
            party_colors, focus_min_gap,
        )
    if show_executive_panel:
        _render_focus_panel(
            ax5, executive_series, "首長重視率 (%)", n_months, provisional_idx, events,
            party_colors, focus_min_gap,
        )

    for (start, end, name), color in zip(pm_periods, PM_COLORS):
        lo, hi = max(start, -18), min(end, n_months - 1)
        ax3.axvspan(lo, hi, color=color, alpha=0.55, zorder=1)
        ax3.text((lo + hi) / 2, 0.5, name, ha="center", va="center", fontsize=12, color="white", weight="bold", zorder=3)
    if reshuffle:
        for idx in reshuffle_idxs:
            ax3.axvline(idx, color="white", linewidth=1.4, linestyle=(0, (2, 1)), zorder=2)
    for idx, _ in events:
        ax3.axvline(idx, color="#333333", linewidth=0.8, linestyle="--", zorder=4)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax3.spines[spine].set_visible(False)

    tick_idx = list(range(0, n_months, 12))
    if (n_months - 1) - tick_idx[-1] < 6:
        tick_idx[-1] = n_months - 1
    else:
        tick_idx.append(n_months - 1)
    xlabels = [f"{history[i]['year']}/{history[i]['month']:02d}" for i in tick_idx]
    ax3.set_xticks(tick_idx)
    ax3.set_xticklabels(xlabels, fontsize=11, rotation=30, ha="right")

    fig.tight_layout(rect=[0.08, 0.02, 0.88, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


NATIONAL_LOCAL_SCATTER_OUT_PATH = (
    Path(__file__).resolve().parents[2] / "notes" / "images" / "national_vs_local.png"
)
MIN_SHARED_MONTHS_FOR_SCATTER = 3


def generate_national_local_scatter(
    national_history: list[dict],
    local_history: list[dict],
    out_path: Path = NATIONAL_LOCAL_SCATTER_OUT_PATH,
) -> Path:
    """政党ごとに「国政限定C_p」対「地方限定C_p」を月ごとにプロットする散布図
    (monthly_index.build_month_snapshotsのnational_only=True・local_only=Trueの
    2系列を組み合わせる、2026-09にユーザー提案)。対角線より上なら地方に強く、
    下なら国政に強いことを示す。政党ごとの小図(small multiples)にし、色で
    時系列(年)を示す——1つの散布図に全党を重ねると政党数×月数で潰れるため。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    nat_by_ym = {(h["year"], h["month"]): h["C_p"] for h in national_history}
    loc_by_ym = {(h["year"], h["month"]): h["C_p"] for h in local_history}
    shared_months = sorted(set(nat_by_ym) & set(loc_by_ym))

    party_order = [p for p in PARTY_ORDER]
    for h in national_history + local_history:
        for party in h["C_p"]:
            if party != "無所属" and party not in party_order:
                party_order.append(party)

    series_by_party: dict[str, list[tuple[int, int, float, float]]] = {}
    for party in party_order:
        pts = [
            (y, m, nat_by_ym[(y, m)].get(party, 0.0) * 100, loc_by_ym[(y, m)].get(party, 0.0) * 100)
            for y, m in shared_months
        ]
        if len(pts) >= MIN_SHARED_MONTHS_FOR_SCATTER:
            series_by_party[party] = pts

    plotted_parties = [p for p in party_order if p in series_by_party]
    n = len(plotted_parties)
    ncols = min(5, n) or 1
    nrows = max(1, -(-n // ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    years = [y for h in shared_months for y in [h[0]]]
    year_min, year_max = min(years), max(years)
    cmap = plt.cm.viridis

    scatter_for_colorbar = None
    for i, party in enumerate(plotted_parties):
        ax = axes[i // ncols][i % ncols]
        pts = series_by_party[party]
        xs = [p[2] for p in pts]
        ys = [p[3] for p in pts]
        colors = [(p[0] - year_min) / max(1, year_max - year_min) for p in pts]
        ax.plot(xs, ys, color="#bbbbbb", linewidth=0.8, zorder=1)
        scatter_for_colorbar = ax.scatter(xs, ys, c=colors, cmap=cmap, s=22, zorder=2, edgecolors="none")
        axis_max = max(10.0, max(xs + ys) * 1.1)
        ax.plot([0, axis_max], [0, axis_max], color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_title(party, fontsize=11)
        ax.tick_params(labelsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel("国政限定 C_p (%)", fontsize=12)
    fig.supylabel("地方限定 C_p (%)", fontsize=12)
    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([str(year_min), str(year_max)])
        cbar.set_label("年", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


EXECUTIVE_FOCUS_OUT_PATH = (
    Path(__file__).resolve().parents[2] / "notes" / "images" / "executive_focus.png"
)


def generate_executive_focus_chart(
    executive_history: list[dict],
    local_assembly_history: list[dict],
    out_path: Path = EXECUTIVE_FOCUS_OUT_PATH,
) -> Path:
    """政党ごとに「首長重視率」=首長C_p/(首長C_p+地方議会C_p)の全期間平均を
    棒グラフで示す独立の図(monthly_index.build_month_snapshotsの
    executive_only=True・only_level={都道府県議会,市区町村議会}の2系列を
    組み合わせる、2026-09にユーザー提案)。首長選は件数が少なく月によって
    0%/100%に振れやすいため、月次の時系列ではなく全期間平均を主役にした
    ——月次の時系列は図1の方に補助パネルとして載せる(2026-09にユーザー指示、
    generate_main_chartのexecutive_history引数を参照)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    exe_by_ym = {(h["year"], h["month"]): h["C_p"] for h in executive_history}
    asm_by_ym = {(h["year"], h["month"]): h["C_p"] for h in local_assembly_history}
    shared_months = sorted(set(exe_by_ym) & set(asm_by_ym))

    party_order = [p for p in PARTY_ORDER]
    for h in executive_history + local_assembly_history:
        for party in h["C_p"]:
            if party != "無所属" and party not in party_order:
                party_order.append(party)
    party_colors = dict(zip(party_order, plt.cm.tab10(np.linspace(0, 1, max(10, len(party_order))))))

    avg_rows = []
    for party in party_order:
        exe_vals = [exe_by_ym[ym].get(party, 0.0) * 100 for ym in shared_months]
        asm_vals = [asm_by_ym[ym].get(party, 0.0) * 100 for ym in shared_months]
        if not exe_vals:
            continue
        exe_avg = sum(exe_vals) / len(exe_vals)
        asm_avg = sum(asm_vals) / len(asm_vals)
        if exe_avg + asm_avg <= 0:
            continue
        avg_rows.append((party, exe_avg / (exe_avg + asm_avg) * 100))
    avg_rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ys_pos = list(range(len(avg_rows)))
    ax.barh(
        ys_pos, [v for _, v in avg_rows], color=[party_colors[p] for p, _ in avg_rows],
        height=0.6, zorder=2,
    )
    ax.axvline(50, color="#999999", linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(ys_pos)
    ax.set_yticklabels([p for p, _ in avg_rows], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"首長重視率({len(shared_months)}か月平均, %)", fontsize=12)
    for y, (_, v) in zip(ys_pos, avg_rows):
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path
