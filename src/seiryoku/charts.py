"""図1(党勢指数チャート)を生成する。

2026-09のセッションで作り込んだmatplotlibスクリプト(政党別リッジライン・
選挙種別の重みシェア・首相在任期間の3段構成、太字フォント、8:2.2:0.9の
高さ比、タイトル/説明文は画像に含めずキャプションはnote本文側に書く)を、
data/cp_history.json・data/national_election_months.jsonから再生成できる
形で正式なコードに昇格したもの(月一更新の自動化)。
"""
from __future__ import annotations

import json
import math
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
# 党勢指数v2の動きの内訳の色。v2では市区町村議会が大半を占めるので、v1の灰色だと背景に見える
TREND_LEVEL_COLORS = {"衆院選": "#c0392b", "参院選": "#e67e22", "都道府県議会": "#2980b9", "市区町村議会": "#27ae60"}

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


PROVISIONAL_START_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "provisional_start_state.json"


def check_provisional_start_update() -> tuple[tuple[int, int] | None, bool]:
    """_provisional_start()の値を前回実行時の記録と比較する。

    jichisokenが新しい年版を公開するのは年1回(2024年版は2025年2月、2025年版は2026年1月)だけなので
    (design_document.tex \\S2.2参照)、境界はほとんどの月で変化しない。
    変化した月だけ毎月実行の出力で目立たせられるよう、月一更新の実行
    (cli.py)から呼び出す(2026-09にユーザー指示: 「どこかに書いておく」
    より、実際に毎月実行するコードにメッセージを残す方が忘れない)。
    戻り値は(現在の値, 前回の記録から変化したか)。
    """
    current = _provisional_start()
    previous = None
    if PROVISIONAL_START_STATE_PATH.exists():
        d = json.loads(PROVISIONAL_START_STATE_PATH.read_text(encoding="utf-8"))
        previous = (d["year"], d["month"])
    if current is not None and current != previous:
        PROVISIONAL_START_STATE_PATH.parent.mkdir(exist_ok=True)
        PROVISIONAL_START_STATE_PATH.write_text(
            json.dumps({"year": current[0], "month": current[1]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return current, (current != previous)


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
    市区町村長選挙の推薦・支持の振り替えが未反映のまま平均に混ざり、政党によって
    は分類が逆転するため(推薦・支持を名簿とWikipediaの和集合で数える現行の基準では、
    公明党が確定値期間63.8%〈首長型〉・速報値期間44.6%〈議会型〉)。
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


PARTY_MATRIX_OUT_PATH = Path(__file__).resolve().parents[2] / "notes" / "images" / "party_matrix.png"

# 地方重視率(全期間平均)・首長重視率(確定値期間88か月平均)の分類結果
# (design_document.tex「政党ごとの国政/地方への偏り」・note記事「おまけ2」参照)。
# 両指標とも閾値50%で単純に二分できるほど自明な計算ではなく、首長重視率は
# 速報値期間を除く必要があるなど個別の判断を要するため、値ではなく分類結果
# そのものをここに持つ。
PARTY_MATRIX = {
    ("国政型", "議会型"): ["自由民主党", "立憲民主党"],
    ("国政型", "首長型"): ["国民民主党", "日本維新の会"],
    ("地方型", "議会型"): ["日本共産党"],
    ("地方型", "首長型"): ["公明党"],
}


def generate_party_matrix_chart(out_path: Path = PARTY_MATRIX_OUT_PATH) -> Path:
    """「政党の型分類」表を画像化する。

    note.comの編集画面はテーブル(表組み)を作成できないため、他の数式
    (formula_cp.png)と同様に画像として埋め込む必要がある(2026-09に
    ユーザー指摘)。PDF版(design_document.tex)ではなくnote記事だけの都合
    だが、formula_cp.pngもnote・PDFの両方で同じ画像を使っている前例に
    合わせ、dai0kai_draft.texからもこの画像を参照する。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    party_colors = dict(zip(PARTY_ORDER, plt.cm.tab10(np.linspace(0, 1, max(10, len(PARTY_ORDER))))))
    rows = ["国政型", "地方型"]
    cols = ["議会型", "首長型"]

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=260)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.invert_yaxis()

    for i in range(4):
        ax.plot([i, i], [0, 3], color="#333333", linewidth=1.2, zorder=1)
        ax.plot([0, 3], [i, i], color="#333333", linewidth=1.2, zorder=1)

    for j, col in enumerate(cols):
        ax.text(1.5 + j, 0.5, col, ha="center", va="center", fontsize=16, weight="bold")
    for i, row in enumerate(rows):
        ax.text(0.5, 1.5 + i, row, ha="center", va="center", fontsize=16, weight="bold")

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            parties = PARTY_MATRIX[(row, col)]
            cx, cy = 1.5 + j, 1.5 + i
            n = len(parties)
            fontsize = 13 if n == 1 else 11
            step = min(0.5, 0.9 / n)
            for k, party in enumerate(parties):
                ax.text(
                    cx, cy + (k - (n - 1) / 2) * step, party, ha="center", va="center",
                    fontsize=fontsize, color=party_colors[party], weight="bold",
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


TREND_OUT_PATH = Path(__file__).resolve().parents[2] / "notes" / "images" / "trend_v2.png"
TREND_BY_PARTY_OUT_PATH = Path(__file__).resolve().parents[2] / "notes" / "images" / "trend_v2_by_party.png"
# 党ごとの図の並び。党勢の大きい順に読みやすく並べた編集上の選択
TREND_PARTY_ORDER = ["自由民主党", "立憲民主党", "国民民主党", "参政党", "公明党", "日本維新の会", "日本共産党", "いのちの党"]


def _setup_fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _party_colors(plt, parties: list[str]) -> dict:
    """図1と同じ政党に同じ色を当てる。"""
    order = list(PARTY_ORDER) + [p for p in parties if p not in PARTY_ORDER]
    return dict(zip(order, plt.cm.tab10(np.linspace(0, 1, max(10, len(order))))))


def _spread_labels(ys: list[float], gap: float) -> list[float]:
    """右端のラベルの縦位置を、元の順序を保ったまま最小間隔gapまで押し広げる。"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + gap)
    return out


def generate_trend_chart(result: dict, out_path: Path = TREND_OUT_PATH) -> Path:
    """党勢指数v2の図。1段目は全政党の党勢の推移、2段目は党勢の推定値の動き(|Δz|の合計)の
    選挙の種類ごとの内訳、最下段は首相の在任期間。resultはstate_space_glmの出力。
    1段目はその月までのデータだけによる値(x_f)で描く。平滑化の値は大きな国政選挙の跳びを
    前の月に割り振るので、選挙の前から党勢が動いていたように見えてしまう。"""
    plt = _setup_fonts()
    parties = [p for p in PARTY_ORDER if p in result["parties"]] + \
        [p for p in result["parties"] if p not in PARTY_ORDER]
    colors = _party_colors(plt, parties)
    months = sorted({r["tau"] for p in parties for r in result["parties"][p]["series"]})
    y0, m0 = int(months[0][:4]), int(months[0][5:7])
    n_months = len(months)
    pos = {t: i for i, t in enumerate(months)}

    def idx(y: int, m: int) -> int:
        return (y - y0) * 12 + (m - m0)

    events = []
    for (y, m), label in national_elections.compute_election_events():
        if 0 <= idx(y, m) < n_months:
            events.append((idx(y, m), f"{y % 100}/{m} {label}"))
    for y, m in _unified_local_election_months(y0, int(months[-1][:4])):
        if 0 <= idx(y, m) < n_months:
            events.append((idx(y, m), f"{y % 100}/{m} 統一地方選"))
    events.sort()
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(19, 14), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [8, 2.6, 0.9], "hspace": 0.06},
    )
    ends, ymax_by_party = {}, {}
    for party in parties:
        # 最初の衆院選までは衆院の尺度に合わせる手がかりがなく、その時点の値は事前分布と
        # 参院選からの換算のずれに左右されるので描かない
        shu = [n["vote_date"] for n in result["parties"][party].get("national", []) if n["level"] == "衆院選"]
        s = [r for r in result["parties"][party]["series"] if not shu or r["tau"] >= min(shu)]
        if not s:
            continue
        xs = [pos[r["tau"]] for r in s]
        ys = [r.get("x_f", r["x"]) * 100 for r in s]
        ax1.plot(xs, ys, color=colors[party], linewidth=2.4, zorder=3)
        ends[party] = ys[-1]
        ymax_by_party[party] = max(ys)
    drawn = [p for p in parties if p in ends]
    ymax = max(ymax_by_party[p] for p in drawn)
    # 選挙のラベルを線に重ねないよう、上に余白を取ってそこに置く
    ax1.set_ylim(0, ymax * 1.3)
    label_y = _spread_labels([ends[p] for p in drawn], gap=ymax * 0.045)
    for party, ly in zip(drawn, label_y):
        ax1.plot([n_months - 1, n_months - 1 + n_months * 0.012], [ends[party], ly],
                 color=colors[party], linewidth=1.0, zorder=3)
        ax1.text(n_months - 1 + n_months * 0.015, ly, party, fontsize=14, color=colors[party],
                 va="center", ha="left", weight="bold")
    for i, label in events:
        ax1.axvline(i, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
        ax1.text(i, ymax * 1.29, label, fontsize=11, rotation=90, ha="left", va="top", color="#333333")
    ax1.set_ylabel("党勢(衆院の議席率の尺度, %)", fontsize=15)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xlim(0, n_months - 1 + n_months * 0.13)
    ax1.spines[["top", "right"]].set_visible(False)

    totals = np.zeros(n_months)
    level_data = {lv: [0.0] * n_months for lv in LEVEL_ORDER}
    for party in parties:
        for r in result["parties"][party].get("movement", []):
            for lv in LEVEL_ORDER:
                level_data[lv][pos[r["tau"]]] += r.get(lv, 0.0)
    for lv in LEVEL_ORDER:
        totals += np.array(level_data[lv])
    levels = [lv for lv in LEVEL_ORDER if any(level_data[lv])]
    level_colors = [TREND_LEVEL_COLORS.get(lv, c) for lv, c in zip(LEVEL_ORDER, LEVEL_COLORS) if lv in levels]
    level_data = {lv: [v / t * 100 if t > 0 else 0.0 for v, t in zip(level_data[lv], totals)] for lv in levels}
    # 選挙のない月は空白のまま見せるため、隣の月と斜めにつながない段で描く
    ax2.stackplot(range(n_months), [level_data[lv] for lv in levels], colors=level_colors, alpha=0.85, zorder=2,
                  step="mid", labels=[LEVEL_DISPLAY_NAMES.get(lv, lv) for lv in levels])
    # 帯の大半が細い棒なので、帯の中に名前を置くと隣の棒を指して見える。右の余白に凡例として置く
    handles, labels = ax2.get_legend_handles_labels()
    # 凡例の左端を1段目の政党名の書き出しにそろえる
    ax2.legend(handles[::-1], labels[::-1], loc="center left", fontsize=13, frameon=False, handlelength=1.4,
               borderpad=0, borderaxespad=0,
               bbox_to_anchor=(n_months - 1 + n_months * 0.015, 50), bbox_transform=ax2.transData)
    for i, _ in events:
        ax2.axvline(i, color="#555555", linewidth=0.8, linestyle="--", zorder=5)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("動きの内訳 (%)", fontsize=15)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.spines[["top", "right"]].set_visible(False)

    for k, (y, m, name) in enumerate(PM_PERIODS):
        lo = max(idx(y, m), 0)
        hi = idx(*PM_PERIODS[k + 1][:2]) if k + 1 < len(PM_PERIODS) else n_months - 1
        hi = min(hi, n_months - 1)
        if hi <= 0:
            continue
        ax3.axvspan(lo, hi, color=PM_COLORS[k], alpha=0.55, zorder=1)
        ax3.text((lo + hi) / 2, 0.5, name, ha="center", va="center", fontsize=14, color="white", weight="bold")
    for i, _ in events:
        ax3.axvline(i, color="#333333", linewidth=0.8, linestyle="--", zorder=4)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax3.spines[spine].set_visible(False)
    ticks = list(range(0, n_months, 12))
    if n_months - 1 - ticks[-1] >= 10:
        ticks.append(n_months - 1)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([months[i][:7].replace("-", "/") for i in ticks], fontsize=13, rotation=30, ha="right")

    fig.tight_layout(rect=[0.0, 0.02, 0.99, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_trend_by_party_chart(result: dict, validation: dict | None = None,
                                  out_path: Path = TREND_BY_PARTY_OUT_PATH) -> Path:
    """党勢指数v2の政党別の図(横一列)。帯は±2標準偏差、点は国政選挙の結果(参院選は衆院の
    尺度に換算)、×は国政選挙を1回ずつ除いて補間した値。validationは補間の検証の出力。"""
    from datetime import date as _date
    import matplotlib.dates as mdates

    plt = _setup_fonts()
    parties = [p for p in TREND_PARTY_ORDER if p in result["parties"]]
    fig, axes = plt.subplots(1, len(parties), figsize=(3.75 * len(parties), 5.5), sharex=True, sharey=True)

    def logit(v: float) -> float:
        v = min(max(v, 1e-4), 1 - 1e-4)
        return math.log(v / (1 - v))

    for ax, party in zip(np.atleast_1d(axes), parties):
        r = result["parties"][party]
        a_san, load = r["alpha"].get("参院選"), r["params"]["san_load"]

        def to_x(v: float, level: str) -> float:
            if level != "参院選" or a_san is None:
                return v
            return 1 / (1 + math.exp(-(logit(v) - a_san) / load))

        ds = [_date.fromisoformat(s["tau"]) for s in r["series"]]
        ax.fill_between(ds, [s["x_lo"] for s in r["series"]], [s["x_hi"] for s in r["series"]],
                        color="C0", alpha=0.2, lw=0)
        ax.plot(ds, [s["x"] for s in r["series"]], color="C0", lw=1.5)
        for n in r.get("national", []):
            shu = n["level"] != "参院選"
            ax.plot(_date.fromisoformat(n["vote_date"]), to_x(n["y"], n["level"]), "o" if shu else "s", ms=6,
                    mfc="k" if shu else "w", mec="k", zorder=5)
        for h in (validation or {}).get("parties", {}).get(party, []):
            d = _date.fromisoformat(h["vote_date"])
            yp, lo, hi = (to_x(h[k], h["level"]) for k in ("pred", "lo", "hi"))
            ax.errorbar(d, yp, yerr=[[yp - lo], [hi - yp]], fmt="x", color="C3", ms=7, capsize=3, zorder=4)
        ax.set_title(party, fontsize=13)
        ax.set_ylim(0, 0.85)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [plt.Line2D([], [], color="C0", lw=1.5), plt.Line2D([], [], ls="", marker="o", color="k"),
               plt.Line2D([], [], ls="", marker="s", mfc="w", mec="k")]
    labels = ["推定した党勢(帯は±2標準偏差)", "衆院選の議席率", "参院選(衆院の尺度に換算)"]
    if validation:
        handles.append(plt.Line2D([], [], ls="", marker="x", color="C3"))
        labels.append("その選挙を除いて補間した値(±2標準偏差)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path


TREND_SUPPORT_OUT_PATH = Path(__file__).resolve().parents[2] / "notes" / "images" / "trend_v2_vs_support.png"
# 位置を決め打ちするラベル: {政党: (基準, 横のずれ(pt), 縦のずれ(pt), 横の寄せ, 縦の寄せ, 矢印)}。基準は
# "point"(●)、"top"(誤差棒の上端)、"bar"(●と上端の間の誤差棒の上)。点が密集する党の見た目のための
# 編集上の選択で、それ以外の党は自動で置く
_SUPPORT_LABEL_FIXED = {
    "立憲民主党": ("point", -7, 34, "right", "center", False),
    "国民民主党": ("top", -3, 20, "left", "center", True),
    "日本維新の会": ("point", 7, 30, "left", "center", False),
    "いのちの党": ("point", 6, 16, "left", "center", False),
}


def _place_point_labels(ax, items: list[tuple[str, float, float, str]], obstacles_px: list, segments_data: list,
                        boxes_px: list | None = None, diagonal: bool = True,
                        prefer: dict[str, list[int]] | None = None) -> None:
    """散布図の点にラベルを付ける。点のまわりの候補位置を近い順に試し、他のラベル・点・縦の線分・
    対角線(y=x)・軸の外に重ならない最初の位置に置く。点から離れた位置になったら矢印で結ぶ。
    itemsは(ラベル, x, y, 色)、obstacles_pxは避ける点の表示座標、segments_dataは避ける縦の線分
    (x, y0, y1)、boxes_pxは避ける矩形(凡例など)の表示座標(x0, y0, x1, y1)。preferは{ラベル: 先に試す向き
    (度)}で、その向きで置ける位置がなければ全方向を試す。描画位置が確定した後(tight_layoutの後)に呼ぶこと。"""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    frame = ax.get_window_extent(renderer)
    placed = list(boxes_px or [])
    segs = []
    for x, y0, y1 in segments_data:
        (sx, sy0), (_, sy1) = ax.transData.transform([(x, y0), (x, y1)])
        segs.append((sx - 3, min(sy0, sy1), sx + 3, max(sy0, sy1)))
    marks = [(px - 7, py - 7, px + 7, py + 7) for px, py in obstacles_px]

    def hits(b):
        box = (b.x0, b.y0, b.x1, b.y1)
        if box[0] < frame.x0 or box[2] > frame.x1 or box[1] < frame.y0 or box[3] > frame.y1:
            return True
        for o in placed + segs + marks:
            if box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3]:
                return True
        if diagonal:
            inv = ax.transData.inverted()
            signs = {np.sign(math.log10(cy) - math.log10(cx)) for cx, cy in
                     inv.transform([(box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])])}
            if len(signs) > 1:
                return True
        return False

    angles = [0, 180, 30, -30, 150, -150, 60, -60, 120, -120, 90, -90]
    for label, x, y, color in items:
        px, py = ax.transData.transform((x, y))
        chosen = None
        tries = [(r, a) for r in (9, 22, 38, 58, 82, 110) for a in (prefer or {}).get(label, [])] + \
            [(r, a) for r in (9, 22, 38, 58, 82, 110, 145) for a in angles]
        for r, a in tries:
            dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            tx = ax.text(px + dx * pt, py + dy * pt, label, fontsize=12, weight="bold", ha=ha, va="center",
                         transform=None)
            b = tx.get_window_extent(renderer)
            tx.remove()
            if not hits(b):
                chosen = (dx, dy, ha, b, r)
                break
        if chosen is None:
            chosen = (9, 0, "left", None, 9)
        dx, dy, ha, b, r = chosen
        arrow = dict(arrowstyle="->", color=color, lw=0.9, shrinkA=1, shrinkB=5) if r > 12 else None
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=color,
                    weight="bold", ha=ha, va="center", arrowprops=arrow)
        if b is not None:
            placed.append((b.x0 - 2, b.y0 - 2, b.x1 + 2, b.y1 + 2))


def generate_trend_support_scatter(result: dict, support: dict, out_path: Path = TREND_SUPPORT_OUT_PATH) -> Path:
    """政党支持率(無党派を除いて再計算)と、党勢指数v2の最新月の値(●、縦棒は±2標準偏差)・
    直近の衆院選の議席率(○)の散布図。○から●への矢印は、衆院選の後の党勢の動き。
    supportはsupport_historyの1件({"updated", "S_p"})。点線は両者が等しいところ。"""
    import matplotlib.ticker as mt

    plt = _setup_fonts()
    S = {p: v for p, v in support["S_p"].items() if p != "無党派"}
    total = sum(S.values())
    colors = _party_colors(plt, list(result["parties"]))
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    ax.plot([0.1, 100], [0.1, 100], color="#888", ls="--", lw=1.2, zorder=1)
    items, fixed, pts, segs, shu_date = [], [], [], [], None
    for party, r in result["parties"].items():
        if party not in S:
            continue
        s = r["series"][-1]
        x = S[party] / total * 100
        y, lo, hi = (s.get(k, s[k.replace("_f", "")]) * 100 for k in ("x_f", "x_f_lo", "x_f_hi"))
        c = colors[party]
        # 衆院選からの矢印を誤差棒の上で見せるため、誤差棒は薄く描く
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=1.0, capsize=3,
                    alpha=0.6, zorder=2)
        ax.plot(x, y, "o", ms=6.5, color=c, zorder=5)
        segs.append((x, lo, hi))
        pts.append((x, y))
        shu = [n for n in r.get("national", []) if n["level"] == "衆院選"]
        if shu:
            last = max(shu, key=lambda n: n["vote_date"])
            shu_date = last["vote_date"]
            ys = max(last["y"] * 100, 0.11)
            ax.plot(x, ys, "o", ms=6.5, mfc="white", mec=c, mew=1.5, zorder=4)
            ax.annotate("", (x, y), xytext=(x, ys), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, mutation_scale=12, shrinkA=4, shrinkB=4))
            segs.append((x, min(ys, y), max(ys, y)))
            pts.append((x, ys))
        if party in _SUPPORT_LABEL_FIXED:
            anchor, dx, dy, ha, va, arrow = _SUPPORT_LABEL_FIXED[party]
            ay = {"top": hi, "bar": math.sqrt(y * hi)}.get(anchor, y)
            fixed.append((party, x, ay, c, dx, dy, ha, va, arrow))
        else:
            items.append((party, x, y, c))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 100)
    ax.set_ylim(0.1, 100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"政党支持率(無党派を除いて再計算, %, {support['updated']}更新)", fontsize=13)
    ax.set_ylabel("党勢指数v2・衆院の議席率(%)", fontsize=13)
    ax.grid(alpha=0.3, which="major")
    handles = [plt.Line2D([], [], ls="", marker="o", color="#444", ms=6.5),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#444", ms=6.5, mew=1.5),
               plt.Line2D([], [], color="#888", ls="--", lw=1.2)]
    labels = [f"党勢指数v2({result['tau']}、縦棒は±2標準偏差)", f"衆院選の議席率({shu_date})", "支持率と等しい線"]
    ax.legend(handles, labels, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg = ax.get_legend().get_window_extent(renderer)
    boxes = [(leg.x0, leg.y0, leg.x1, leg.y1)]
    for party, x, y, c, dx, dy, ha, va, arrow in fixed:
        # 矢印はラベルの中央でなく点に近い側の端から出し、隣の党の誤差棒を横切らないようにする
        relpos = {"left": (0, 0.5), "right": (1, 0.5)}.get(ha, (0.5, 0.5))
        props = dict(arrowstyle="->", color=c, lw=0.9, shrinkA=1, shrinkB=1, relpos=relpos) if arrow else None
        a = ax.annotate(party, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=c,
                        weight="bold", ha=ha, va=va, arrowprops=props)
        b = a.get_window_extent(renderer)
        boxes.append((b.x0 - 2, b.y0 - 2, b.x1 + 2, b.y1 + 2))
    _place_point_labels(ax, sorted(items, key=lambda it: -it[2]), [tuple(ax.transData.transform(p)) for p in pts],
                        segs, boxes_px=boxes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path
