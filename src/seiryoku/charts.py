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
# LEVEL_ORDERはlevel_weight_shareの実データキー(monthly_index.py参照)と一致させる
# 必要があるため変更できない。表示名だけをここで日本語として正式な言い方に直す。
LEVEL_DISPLAY_NAMES = {"衆院選": "衆議院", "参院選": "参議院"}

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


INLINE_LABEL_MIN_GAP = 6.5  # 政党間でこれだけ離れていればラベルが判別できる
# とみなす閾値(パーセント換算)。


def _find_big_gaps(focus_series: dict[str, tuple[list[int], list[float]]], min_width: int = 6) -> list[tuple[int, int]]:
    """どの政党もデータを持たない月(国政選挙が12か月窓に入っていない期間、
    _build_focus_series参照)のうち、min_width以上続く区間を返す。
    地方重視率パネルではこの区間が線と一切重ならない安全地帯になる。"""
    covered: set[int] = set()
    max_x = -1
    for xs, _ in focus_series.values():
        if xs:
            covered.update(xs)
            max_x = max(max_x, xs[-1])
    gaps = []
    i = 0
    while i <= max_x:
        if i not in covered:
            j = i
            while j <= max_x and j not in covered:
                j += 1
            if j - i >= min_width:
                gaps.append((i, j - 1))
            i = j
        else:
            i += 1
    return gaps


def _place_inline_labels(
    focus_series: dict[str, tuple[list[int], list[float]]], min_gap: float = INLINE_LABEL_MIN_GAP,
) -> dict[str, tuple[int, float]]:
    """各政党について、他のどの政党とも値が十分離れていて線を辿らずに
    判別できる時点(直近優先)を選び、ラベルをそこに直接置けるようにする。

    複数政党の値がグラフ右端でたまたま一致すること(特に0%)が多く、右端に
    ラベルを集めて引き出し線で結ぶ方式では、収束点そのもので線が重なって
    しまい対応関係を追えない。ラベル位置を時点ごと動かせば、値が分かれて
    いる瞬間の実際の線のすぐ隣に置けるので、この重なりを回避できる
    (2026-09にユーザー指摘・提案)。

    地方重視率のように大きな欠測区間がある系列では、区間の直前(=以降どの
    政党の線も存在しない安全地帯のすぐ隣)を最優先で使う(2026-09に
    ユーザー提案)。
    """
    gap_starts = {s for s, _ in _find_big_gaps(focus_series)}

    values_at_x: dict[int, dict[str, float]] = {}
    for party, (xs, ys) in focus_series.items():
        for x, y in zip(xs, ys):
            values_at_x.setdefault(x, {})[party] = y

    placed: list[tuple[int, float]] = []

    def _far_from_placed(x: int, y: float) -> bool:
        # ラベルの文字列自体が数か月分の横幅を持つため、x方向の除外幅は
        # 点どうしの間隔ではなく実際のラベル幅を目安にする必要がある。
        return all(abs(x - px) >= 9 or abs(y - py) >= min_gap for px, py in placed)

    positions: dict[str, tuple[int, float]] = {}
    for party, (xs, ys) in focus_series.items():
        gap_adjacent = [i for i in range(len(xs)) if (xs[i] + 1) in gap_starts]
        if gap_adjacent:
            i = max(gap_adjacent)
            positions[party] = (xs[i], ys[i])
            placed.append((xs[i], ys[i]))
            continue

        best_idx, best_gap = len(xs) - 1, -1.0
        chosen = None
        for i in range(len(xs) - 1, -1, -1):
            x, y = xs[i], ys[i]
            others = [v for p2, v in values_at_x[x].items() if p2 != party]
            gap = min((abs(y - v) for v in others), default=float("inf"))
            if gap > best_gap:
                best_idx, best_gap = i, gap
            if gap >= min_gap and _far_from_placed(x, y):
                chosen = i
                break
        i = chosen if chosen is not None else best_idx
        positions[party] = (xs[i], ys[i])
        placed.append((xs[i], ys[i]))
    return positions


def _resolve_label_text_positions(
    positions: dict[str, tuple[int, float]], min_gap: float = INLINE_LABEL_MIN_GAP, x_window: float = 16.0,
) -> dict[str, tuple[float, float]]:
    """新規政党どうしなど、系列全体が短くほぼ同じ値しか取らない場合は
    _place_inline_labelsでも時点を十分離せないことがある。x・yの両方が
    近い政党どうしをグループ化し、グループ内でだけ縦に積んで重なりを解消する
    (印そのものは実際の点のまま、ラベルの文字だけをずらす)。

    1回の統合で動いたラベルが、動かなかった別のラベルと新たに衝突すること
    があるため、位置が変化しなくなるまで繰り返す。
    """
    current: dict[str, tuple[float, float]] = {p: (float(x), float(y)) for p, (x, y) in positions.items()}
    for _ in range(4):
        updated = _resolve_label_text_positions_once(current, min_gap, x_window)
        if updated == current:
            break
        current = updated
    return current


def _resolve_label_text_positions_once(
    positions: dict[str, tuple[float, float]], min_gap: float, x_window: float,
) -> dict[str, tuple[float, float]]:
    items = list(positions.items())
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            _, (xi, yi) = items[i]
            _, (xj, yj) = items[j]
            if abs(xi - xj) < x_window and abs(yi - yj) < min_gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    label_positions: dict[str, tuple[float, float]] = {}
    for idxs in groups.values():
        members = [items[i] for i in idxs]
        if len(members) == 1:
            party, (x, y) = members[0]
            label_positions[party] = (x, y)
            continue
        # 右方向に伸ばしても衝突しないよう、グループの左端を共通の起点にする。
        anchor_x = min(x for _, (x, _) in members)
        mean_y = sum(y for _, (_, y) in members) / len(members)
        ordered = sorted(members, key=lambda m: m[1][1])
        k = len(ordered)
        for i, (party, _) in enumerate(ordered):
            label_positions[party] = (anchor_x, mean_y + (i - (k - 1) / 2) * min_gap)
    return label_positions


def _render_focus_panel(
    ax, focus_series: dict[str, tuple[list[int], list[float]]], ylabel: str,
    n_months: int, provisional_idx: int | None, events: list[tuple[int, str]],
    party_colors: dict,
) -> None:
    """地方重視率・首長重視率パネルの共通描画ロジック(2026-09に追加、
    generate_main_chart参照)。ラベルは各政党の線のすぐ隣に直接置く
    (位置の決め方は_place_inline_labels参照)。"""
    for party, (xs, ys) in focus_series.items():
        # 国政選挙が12か月窓に入っていない月は比率が定義できず欠測になる
        # (_build_focus_series参照)。月インデックスが連続する区間だけを
        # つないで描画し、欠測をまたいだ直線補間で実在しない推移を
        # 示さないようにする(2026-09にユーザー指摘)。
        seg_start = 0
        for i in range(1, len(xs) + 1):
            if i == len(xs) or xs[i] - xs[i - 1] != 1:
                ax.plot(
                    xs[seg_start:i], ys[seg_start:i], linewidth=2.0, color=party_colors[party], zorder=3,
                )
                seg_start = i

    # 分離度の良い時点を選んでも、他の系列と完全には離れないことがある
    # (新規政党どうしなど、短い期間ほぼ同値でしか存在しない場合)。ラベルの
    # 文字位置だけ_resolve_label_text_positionsでずらし、印(実際の点)との
    # 対応は細い引き出し線で明示する(2026-09にユーザー指示)。
    marker_positions = _place_inline_labels(focus_series)
    label_text_positions = _resolve_label_text_positions(marker_positions)
    for party, (x, y) in marker_positions.items():
        lx, ly = label_text_positions[party]
        ax.plot(x, y, marker="o", markersize=5, color=party_colors[party], zorder=4)
        # 右方向に伸ばすと画面外にはみ出す(右端付近の)場合だけ左向きに切り替える。
        # 閾値を緩めに取りすぎると、右向き・左向きのラベルが中央で正面衝突する
        # (2026-09に実際に発生・判明)。
        ha = "right" if lx > n_months - 8 else "left"
        dx = -n_months * 0.03 if ha == "right" else n_months * 0.03
        ax.plot(
            [x, lx + dx * 0.85], [y, ly], color=party_colors[party], alpha=0.7, linewidth=1.0, zorder=4,
        )
        ax.text(
            lx + dx, ly, party, fontsize=11, color=party_colors[party], va="center", ha=ha,
            weight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
        )

    ax.axhline(50, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax.axvspan(provisional_idx - 0.5, n_months - 1, color="#f2c14e", alpha=0.22, zorder=0)
        ax.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.0, linestyle=":", zorder=4)
    for idx, _ in events:
        ax.axvline(idx, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    # 0%ちょうどの線が下端に張り付いて見づらいので、下限にわずかな余白を持たせる
    # (2026-09にユーザー指示)。
    ax.set_ylim(-5, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_stack_labels(
    ax2, level_order: list[str], level_display: dict[str, str], level_data: dict[str, list[float]],
    max_fontsize: float = 14.0, min_fontsize: float = 7.0,
) -> None:
    """積み上げ帯グラフの各帯の中に直接ラベルを置く。

    「帯が最も太くなる1時点」だけを見て文字の中心をそこに置く方式では、
    その時点がたまたまグラフの端(例: 系列の先頭月)だと、中央揃えの文字の
    半分が軸の外(目盛りラベル領域)にはみ出す(2026-09にユーザー指摘で発覚)。
    正しくは「文字列全体の横幅ぶんの区間で、帯の太さが文字の高さを保って
    いる区間」を探す必要がある。文字の実寸はフォント・文字数から手計算せず、
    実際にAxesへ描画してwindow_extentを測り、それをデータ座標に変換して
    使う(この関数はfig.tight_layout後、Axesの最終位置が確定してから
    呼び出すこと)。
    """
    import matplotlib.patheffects as pe

    fig = ax2.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax2.transData.inverted()
    n = len(next(iter(level_data.values())))

    def _text_size_data(label: str, fontsize: float) -> tuple[float, float]:
        t = ax2.text(0, 0, label, fontsize=fontsize, weight="bold")
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((bbox.width, bbox.height))
        return abs(x1 - x0), abs(y1 - y0)

    def _draw(idx: int, mid_y: float, label: str, fontsize: float) -> None:
        ax2.text(
            idx, mid_y, label, fontsize=fontsize, color="white", weight="bold",
            ha="center", va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black", alpha=0.55)],
        )

    cum_bottom = [0.0] * n
    for level in level_order:
        shares = level_data[level]
        label = level_display.get(level, level)
        placed = False
        fontsize = max_fontsize
        while fontsize >= min_fontsize and not placed:
            text_w, text_h = _text_size_data(label, fontsize)
            half_w = text_w / 2
            half_h = text_h / 2 * 1.1
            for i in sorted(range(n), key=lambda k: -shares[k]):
                lo, hi = i - half_w, i + half_w
                if lo < 0 or hi > n - 1:
                    continue
                lo_i, hi_i = int(np.floor(lo)), int(np.ceil(hi)) + 1
                mid_y = cum_bottom[i] + shares[i] / 2
                # 帯の厚みだけでなく、区間内の全ての月で固定した縦位置(mid_y)
                # が実際にその帯の範囲内(上端・下端の間)に収まっているかを
                # 確認する。帯は厚みを保ったまま上下にずれることがあり、厚み
                # だけを見ると別の色にまたがって見える(2026-09にユーザー
                # 指摘: 市区町村長・参議院で発生)。
                fits = all(
                    cum_bottom[j] <= mid_y - half_h and mid_y + half_h <= cum_bottom[j] + shares[j]
                    for j in range(lo_i, hi_i)
                )
                if fits:
                    _draw(i, mid_y, label, fontsize)
                    placed = True
                    break
            fontsize -= 1
        if not placed:
            # 最小フォントでも収まる区間が無い(極端に細い帯)場合は、
            # 最も太い1点にとにかく置く。
            best_i, best_share = max(enumerate(shares), key=lambda iv: iv[1])
            _draw(best_i, cum_bottom[best_i] + best_share / 2, label, min_fontsize)
        cum_bottom = [b + s for b, s in zip(cum_bottom, shares)]


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

    focus_height = 3.8 if show_focus_panel else 0.0
    executive_height = 3.8 if show_executive_panel else 0.0

    n_panels = 3 + int(show_focus_panel) + int(show_executive_panel)
    # 1段目(党勢指数)の各政党の縦幅が窮屈だったため、高さ配分を広げる
    # (2026-09にユーザー指摘)。他パネルの絶対高さ(インチ)は変えないよう、
    # fig_height側にも同じ増分を足す。
    height_ratios = [10]
    fig_height = 15.4
    if show_focus_panel:
        height_ratios.append(focus_height)
        fig_height += focus_height
    if show_executive_panel:
        height_ratios.append(executive_height)
        fig_height += executive_height
    height_ratios += [2.2, 0.9]

    # ラベルは右端の専用列ではなく各系列の線に沿って直接置く方式にしたため
    # (_place_inline_labels参照)、右余白の確保は不要になった。その分グラフ
    # 全体を横に広げ、ラベルを置く余地を実際の描画領域の中に作る
    # (2026-09にユーザー指示)。
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(19, fig_height), dpi=260, sharex=True,
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
        ax1.plot(x, normalized, color=party_colors[party], linewidth=2.0, zorder=3)
        ax1.fill_between(x, [baseline] * n_months, normalized, color=party_colors[party], alpha=0.15, zorder=2)
        ax1.axhline(baseline, color="#cccccc", linewidth=0.8, zorder=1)
        # x座標をデータ単位(-3)で指定すると、sharex先の全パネルのxlimに
        # この余白分の空白が漏れ出る(2026-09にユーザー指摘: 重みシェア
        # パネルの左側に不要な空白ができていた)。x軸座標系だけAxes分率に
        # したtransformを使い、データ範囲を広げずにプロット枠のすぐ外側へ
        # ラベルを置く。
        ax1.text(
            -0.01, baseline, party, fontsize=15, ha="right", va="bottom", color=party_colors[party],
            transform=ax1.get_yaxis_transform(), clip_on=False,
        )

    for idx, _ in events:
        ax1.axvline(idx, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    for idx, label in events:
        ax1.text(idx, n_party * band + 0.3, label, fontsize=11, rotation=90, ha="left", va="bottom", color="#333333")

    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax1.axvspan(provisional_idx - 0.5, n_months - 1, color="#f2c14e", alpha=0.22, zorder=0)
        ax1.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.0, linestyle=":", zorder=4)
        ax1.text(
            provisional_idx - 0.5, n_party * band + 2.62, " 速報値 ", fontsize=14,
            ha="left", va="top", color="#5c4400", weight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff3cd", edgecolor="#b8860b", linewidth=1.3),
            zorder=6,
        )

    ax1.set_yticks([])
    ax1.set_ylim(-0.3, n_party * band + 2.7)
    ax1.set_xlim(0, n_months - 1 + n_months * 0.03)
    for spine in ["top", "right", "left"]:
        ax1.spines[spine].set_visible(False)

    level_data = {t: [h["level_weight_share"].get(t, 0.0) * 100 for h in history] for t in LEVEL_ORDER}
    ax2.stackplot(x, [level_data[t] for t in LEVEL_ORDER], colors=LEVEL_COLORS, alpha=0.85, zorder=2)
    if provisional_idx is not None and provisional_idx <= n_months - 1:
        ax2.axvline(provisional_idx - 0.5, color="#b8860b", linewidth=1.0, linestyle=":", zorder=4)
    for idx, _ in events:
        ax2.axvline(idx, color="#555555", linewidth=0.8, linestyle="--", zorder=5)

    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("重みシェア (%)", fontsize=15)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.spines[["top", "right"]].set_visible(False)

    if show_focus_panel:
        _render_focus_panel(
            ax4, focus_series, "地方重視率 (%)", n_months, provisional_idx, events, party_colors,
        )
    if show_executive_panel:
        _render_focus_panel(
            ax5, executive_series, "首長重視率 (%)", n_months, provisional_idx, events, party_colors,
        )

    for (start, end, name), color in zip(pm_periods, PM_COLORS):
        lo, hi = max(start, 0), min(end, n_months - 1)
        ax3.axvspan(lo, hi, color=color, alpha=0.55, zorder=1)
        ax3.text((lo + hi) / 2, 0.5, name, ha="center", va="center", fontsize=14, color="white", weight="bold", zorder=3)
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
    ax3.set_xticklabels(xlabels, fontsize=13, rotation=30, ha="right")

    # 左右の余白は政党名ラベル・凡例の実際の幅(インチ換算)で決まり、図の横幅を
    # 広げても比例して増やす必要はない。図を横に広げた分、比率としては
    # 余白を狭めて凡例が浮いて見えないようにする(2026-09にユーザー指摘)。
    # 凡例を含め、全ラベルがAxes内部(実データ+右側の小さな余白)に収まる
    # ようになったため、外側の余白はもう不要(2026-09にユーザー指摘の解消)。
    fig.tight_layout(rect=[0.055, 0.02, 0.99, 1])

    # 文字列全体の横幅が帯の範囲内に収まる区間を、tight_layoutで各Axesの
    # 最終位置が確定した後の実測サイズで探す必要があるため、ここで呼ぶ
    # (_draw_stack_labels参照)。
    _draw_stack_labels(ax2, LEVEL_ORDER, LEVEL_DISPLAY_NAMES, level_data)

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
    """政党ごとに「首長重視率」=首長C_p/(首長C_p+地方議会C_p)の平均を棒グラフで
    示す独立の図(monthly_index.build_month_snapshotsのexecutive_only=True・
    only_level={都道府県議会,市区町村議会}の2系列を組み合わせる、2026-09に
    ユーザー提案)。首長選は件数が少なく月によって0%/100%に振れやすいため、
    月次の時系列ではなく平均を主役にした——月次の時系列は図1の方に補助パネル
    として載せる(2026-09にユーザー指示、generate_main_chartのexecutive_history
    引数を参照)。

    平均は速報値期間(_provisional_start()以降、首長の推薦・支持データが
    未収録の期間)を除いた確定値期間だけで取る。速報値期間を含めて平均すると、
    無所属首長への推薦・支持の振り替えが未反映のまま平均に混ざり、政党によって
    は分類が完全に逆転することが判明したため(2026-09にユーザー指摘・検証、
    日本維新の会が確定値期間36.5%〈議会型〉・速報値期間91.7%〈首長型〉、
    国民民主党が確定値75.3%〈首長型〉・速報値38.9%〈議会型〉)。
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
    provisional_start = _provisional_start()
    if provisional_start is not None:
        shared_months = [ym for ym in shared_months if ym < provisional_start]

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
    ax.set_xlabel(f"首長重視率(確定値期間{len(shared_months)}か月平均, %)", fontsize=12)
    for y, (_, v) in zip(ys_pos, avg_rows):
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path
