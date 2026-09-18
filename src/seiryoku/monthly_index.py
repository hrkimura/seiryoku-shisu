"""党勢指数$C_p(t)$を月次のフロー指標として再構成する。

2026-09にユーザー指示で、ストック指標(現時点で誰が何を保有しているかを
全自治体+国会について毎月積み上げる方式)からフロー指標(その月に実際に
選挙があった自治体・議会・国会だけを対象にする方式)へ設計を変更した。

理由: ストック方式では自治体の重み$\\sqrt{P_j}$が人口という定数だけで決まるため、
選挙結果が変わってもどの枠が動くかが変わるだけで、国会と地方全体の相対的な
重み自体はほぼ一切動かない(2026-09にユーザー指摘)。「インパクト」を見たい
という本プロジェクトの動機(\\S1)に照らすと、これは月ごとの変化を捉える指標
として適さない。そこで、その月に実際に選挙があった自治体・国会だけを
$\\sqrt{P_j}$で加重する方式にした——ある月にどれだけの人口を代表する選挙が
動いたか、その勝者の政党構成で$C_p(t)$を決める。

ここで$e$はその月($T$)に投票日があった選挙イベント(知事選・市区町村長選・
都道府県議会選・市区町村議会選・衆院選・参院選)、$P_e$はその選挙が代表する
人口(自治体の場合はその自治体の人口、国会の場合は全国人口)、$n_e$はその
選挙で決まる議席・ポストの数(知事・市区町村長選は常に1、議会選挙・衆院選・
参院選は実際の議席数)。選挙が無かった月は$C_p(T)$自体が定義できない
(欠測とする)。

2026-09にユーザー指摘・再設計: 当初はイベントごとに$\\sqrt{n_eP_e}$を計算して
単純合計していたが、平方根は凹関数のため、同じ人口を47都道府県のような
少数の大きな単位に集約するより、1700超の市区町村のような多数の小さな単位に
分割した方が$\\sum_e\\sqrt{P_e}$の合計が大きくなる($\\sqrt{a}+\\sqrt{b}>\\sqrt{a+b}$、
$a,b>0$)。実際、市区町村議会だけで平均52.9\\%を占め、ほぼ毎月どこかで選挙が
ある都道府県知事は平均3.0\\%にしかならないことが判明した——これは市区町村議会の
政治的重要性ではなく、単に行政区画を何個に分けているかというアーティファクト
だった(ペンローズの平方根則は本来「単位の数が固定された1つの代表機関の中」で
正当化される理論であり、単位の数も粒度も異なる複数階層をまたいだ単純合計に
転用したのが原因)。

そこで、階層$\\ell\\in L=\\{$衆院選, 参院選, 都道府県知事, 都道府県議会,
市区町村長, 市区町村議会$\\}$をそれぞれ1つの代表機関とみなし、階層をまたぐ
合成のときだけ$\\sqrt{}$を1回かける形に改めた(導出はnotes/new\\_index\\_formula.tex
参照)。

$$N_\\ell(T) = \\sum_{e\\in E_\\ell(T)} n_eP_e, \\qquad
  W_\\ell(T) = \\sqrt{N_\\ell(T)}$$

階層内の政党別シェア$\\phi_p^\\ell(T)$は、当初$n_eP_e$そのもので加重平均して
いたが、議員定数$n_e$は人口にほぼ比例せず(実測で$n\\propto P^{0.27\\sim0.41}$
程度)、$n_eP_e$を使うと大自治体の影響が人口の1.3〜1.4乗ぶん過大になって
いたことが判明した(2026-09にユーザー指摘)。「その階層の有権者を1人
ランダムに選び、その人の代表者($n_e$人のうち1人をさらにランダムに選ぶ)が
どの政党か」という設定で考えると、この確率はイベント$e$の中で既に
$\\phi_p(e)$として定義済みであり($n_e$はそこで使い切っている)、集計時に
人口へさらに$n_e$を掛け直す理由が無い。そこで$P_e$のみの加重平均に修正した。

$$\\Pi_\\ell(T) = \\sum_{e\\in E_\\ell(T)} P_e, \\qquad
  \\phi_p^\\ell(T) = \\frac{\\displaystyle\\sum_{e\\in E_\\ell(T)} P_e\\,\\phi_p(e)}{\\Pi_\\ell(T)}$$

$$C_p(T) = \\frac{\\displaystyle\\sum_{\\ell\\in L} W_\\ell(T)\\,\\phi_p^\\ell(T)}
                  {\\displaystyle\\sum_{\\ell\\in L} W_\\ell(T)}$$

ここで$E_\\ell(T)$は階層$\\ell$で投票日が$T$に含まれる選挙イベントの集合。
該当イベントが無い階層は和から除外する。$N_\\ell(T)$を先に合計してから
$W_\\ell(T)$の平方根を1回だけ取ることで、同じ$N_\\ell(T)$をいくつの自治体に
分割していても$\\phi_p^\\ell(T)$・$W_\\ell(T)$ともに変わらなくなる
(design_document.tex \\S2.7参照)。

首長選挙は常に$n_e=1$なので$\\sqrt{n_eP_e}=\\sqrt{P_e}$のままだが、国会の
選挙だけは実際の議席数(衆院選なら465、参院選の半数改選なら実際の改選数)を
掛ける(2026-09にユーザー指示で$\\sqrt{P_e}$のみの旧式から変更、design_document.tex
\\S2.2参照)。これを「1機関=1票」と数えるか「議員1人ずつが有権者を代表する」と
数えるかで重みが大きく変わるため、後者の解釈を採用した。衆院選と参院選は
別々の選挙イベントであり、$e$としても月としても常に別々に扱う——衆参の
議席数を合算してから1回だけ$\\sqrt{n_{国}P_{国}}$を掛けるようなことはしない
(2026-09にユーザー指摘、index.pyの\\texttt{compute_combined_index}にあった
同種の合算を分離した際の議論と同じ)。

首長側は既存のexecutive_term_chains(turnover.build_term_chain_cache())の
チェーン全体を舐め、投票日がその月に一致するエントリだけを拾う。国会側は
衆参それぞれの選挙結果ファイルから同様に拾う(衆院選は解散総選挙、参院選は
3年ごとの半数改選)。

議会側(都道府県議会・市区町村議会)はmunicipal_registry.load_gikai_term_chain_cache()
のチェーンを同様に舐める。首長と異なり、無所属候補の推薦・支持政党を記録した
情報源(jichisoken)が個人単位では存在しないため、jichisoken補正は試みず、
届出政党そのままの議席占有率($\\phi_p(e)$、index._phi)を国会の選挙と同じ
やり方で計算する。以前はこの「補正手段が無い」ことを理由に議会全体を$C_p(T)$の
対象から外していたが、2026年9月に公開グラフを政党間の水準比較から「各政党
自身の推移」の表示に切り替えたことで、この制約は障害でなくなった——ラベリングの
非対称性(自民党系候補は無所属で出馬しがちで共産党はほぼ必ず党名を明示する)は
政党間の水準比較を歪めるが、政党間で比較しないなら歪みは実害にならない
(design_document.tex \\S2.1参照)。

参院側は当初diet_history.fetch_sangiin_history()(参議院の「会派別所属議員数の
変遷」、選挙直後の参議院\\textbf{全体}のスナップショット)を使い、常会・臨時会・
特別会など召集ごとの構成変化を選挙イベントと誤カウントしないよう、半数改選の
周期に合致する年の8月以降だけを拾うフィルタ(旧_real_sangiin_election_snapshots)
を実装していた(2026-09に発見・修正——2018〜2026年の期間だけで222件中25件が
誤って選挙イベント扱いされていた)。しかしこのフィルタを施しても、抽出される
スナップショットは「その選挙で改選された議席」ではなく「改選されなかった残り
半分も含む参議院全体」の構成であり、$n_e$・$\\phi_p(e)$の定義(その選挙イベントで
決まったこと)と食い違うことが後に判明した(2026-09に発見、design_document.tex
\\S2.2参照)。そこで総務省が参院選ごとに公表する「党派別男女別新前元別当選人数」
ファイルから、その回に実際に改選された議席の当選者数だけを取得する
diet_history.fetch_sangiin_election_history()に切り替えた。このファイルは
選挙のあった回にしか存在しないため、上記のフィルタ自体が不要になった。

個別に追跡する政党の集合は、政党助成法上の「政党要件」(5議席以上、または
1議席以上かつ直近国政選挙で得票率2%以上、index.national_parties_meeting_requirement
参照)を満たす政党とする。月ごとに「その時点で要件を満たしていたか」を遡って
判定し直すのはせず、直近の国政選挙時点の判定を全期間に固定で適用する
(2026-09にユーザー指示、Wikipedia「日本の政党一覧」で現在の要件充足政党を
確認したのがきっかけ、_current_national_parties参照)。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

from .fetch import diet_history, population
from .index import (
    _effective_party_shares,
    _latest_sangiin_district_pct,
    _latest_shugiin_district_pct,
    _phi,
    national_parties_meeting_requirement,
)

MIN_YEAR = 2018

_YM_RE = re.compile(r"(\d{4})-(\d{2})")


def _ym(date_str: str | None) -> tuple[int, int] | None:
    if not date_str:
        return None
    m = _YM_RE.match(date_str)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    y, m = start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def _parsed_chain(chain: list[dict]) -> list[tuple[tuple[int, int], dict]]:
    parsed = [(_ym(c["vote_date"]), c) for c in chain]
    return [(k, v) for k, v in parsed if k is not None]


def _terms_in_month(parsed_chain: list[tuple[tuple[int, int], dict]], ym: tuple[int, int]) -> list[dict]:
    """投票日がちょうどymの月に一致する任期(通常0件か1件)を返す。"""
    return [term for k, term in parsed_chain if k == ym]


def _current_national_parties() -> set[str]:
    """政党要件(5議席以上、または1議席以上かつ直近国政選挙で得票率2%以上)を
    満たす政党の集合を、直近の衆参構成・得票率から1回だけ求める。

    月ごとに「その時点で要件を満たしていたか」を遡って判定し直すのはせず、
    全期間にこの1つの集合を適用する(2026-09にユーザー指示、「直近の国政選挙で
    要件を満たす政党のみでいい」という簡略化)。民主党のように既に解党した政党の
    過去の議席は、この集合に無い名前として無所属・諸派側に一括される。
    """
    from .fetch import diet

    diet_seats = diet.fetch_diet_seats()
    return national_parties_meeting_requirement(
        diet_seats,
        shugiin_district_pct=_latest_shugiin_district_pct(),
        sangiin_district_pct=_latest_sangiin_district_pct(),
    )


# 重みシェアの内訳(図1中段)で使う選挙種別ラベル。build_month_snapshotsが
# 各エントリに付与し、_snapshots_from_windowsが窓内で合算する。
LEVEL_TYPES = ["衆院選", "参院選", "都道府県知事", "都道府県議会", "市区町村長", "市区町村議会"]


@dataclass
class MonthSnapshot:
    year: int
    month: int
    C_p: dict[str, float]
    n_events: int
    level_weight_share: dict[str, float]


def build_month_snapshots(
    min_year: int = MIN_YEAR, window_months: int = 12, local_only: bool = False
) -> list[MonthSnapshot]:
    """min_year以降、月ごとにその月を含めてwindow_monthsか月分遡って実際にあった
    選挙を集計してC_p(t)を再構成する(モジュールdocstring参照)。既定の12か月は、
    どの時点で切っても暦月の構成が同じになり、統一地方選のような季節的な偏りを
    機械的にキャンセルできるための選択(2026-09にユーザー指摘・指示)。

    window_months=1にすると単月集計(季節変動を含む生の値)に戻る。窓を長くする
    ほど標本は増えて振れは小さくなるが、直近の変化への追従は遅くなるトレードオフがある。

    local_only=Trueにすると衆参の国政選挙イベントを一切含めない「地方選挙限定」の
    系列を再構成する。「地方選は国政選挙の前哨戦か」を検証するために2026-09に
    追加(precursor.py参照)。以前はdiet_history.fetch_sangiin_election_history等を
    空リストに差し替えるモンキーパッチで代用していたが、そのやり方は呼び出し側の
    コードを書き換える必要がありテストしにくいため、正式なパラメータに昇格した。
    """
    from . import municipal_registry, turnover
    from .fetch import jichisoken

    if local_only:
        sangiin_entries: list[tuple[tuple[int, int] | None, dict[str, int]]] = []
        shugiin_entries: list[tuple[tuple[int, int] | None, dict[str, int]]] = []
    else:
        sangiin_entries = [(_ym(s.date), s.seats) for s in diet_history.fetch_sangiin_election_history()]
        shugiin_entries = [(_ym(s.date), s.seats) for s in diet_history.fetch_shugiin_history()]
    national_pop = population.national_population()
    national_parties = _current_national_parties()

    turnover._ensure_governor_ids_loaded()
    gov_id_to_name = {jid: name for name, jid in turnover._GOVERNOR_JICHITAI_IDS.items()}
    chains = turnover.load_term_chain_cache()["chains"]

    pref_pop = population.prefecture_population_by_name()
    muni_pop = population.municipal_population_by_name()
    gov_by_year = jichisoken.governor_endorsements_by_year()
    muni_by_year = jichisoken.municipal_head_endorsements_by_year()

    local_entries = []
    for jid_str, chain in chains.items():
        jid = int(jid_str)
        parsed = _parsed_chain(chain)
        if jid in gov_id_to_name:
            name, size, by_year, level_type = (
                gov_id_to_name[jid], pref_pop.get(gov_id_to_name[jid]), gov_by_year, "都道府県知事",
            )
        else:
            name = municipal_registry.jurisdiction_name(jid)
            size, by_year = (muni_pop.get(name), muni_by_year) if name else (None, None)
            level_type = "市区町村長"
        if name and size is not None:
            local_entries.append((name, parsed, size, by_year, level_type))

    # 都道府県議会・市区町村議会(gikai)は、首長と違い候補者個人の推薦・支持政党を
    # 記録した情報源が無いため、jichisoken補正は試みず届出政党のみで$\\phi_p(e)$を
    # 計算する——国会の選挙と同じ扱い(2026年9月にユーザー指摘、design_document.tex
    # \\S2.1参照。以前はこの「補正手段が無い」ことを理由に議会全体を対象外にしていたが、
    # 政党間の水準比較をしない設計に転換したことで、この制約は障害でなくなった)。
    gikai_local_entries = []
    for jid_str, chain in municipal_registry.load_gikai_term_chain_cache()["chains"].items():
        jid = int(jid_str)
        parsed = _parsed_chain(chain)
        if jid in gov_id_to_name:
            size = pref_pop.get(gov_id_to_name[jid])
            level_type = "都道府県議会"
        else:
            name = municipal_registry.jurisdiction_name(jid)
            size = muni_pop.get(name) if name else None
            level_type = "市区町村議会"
        if size is not None:
            gikai_local_entries.append((parsed, size, level_type))

    today = date.today()
    months = _month_range((min_year, 1), (today.year, today.month))

    # 月ごとの生イベント(level_type, P_e, n_e, shares)をまず作っておき、
    # window_months分だけ遡って合算する(スライディングウィンドウ、
    # 2026-09にユーザー指示で追加)。P_eとn_eは別々に持っておき、階層内では
    # P_eだけの加重平均(n_eは掛けない)、階層をまたぐ重みだけn_eP_eの
    # 合計に平方根を取る(_level_aggregate参照、2026-09にユーザー指摘で
    # 再設計、notes/new_index_formula.tex参照)。当初はP_eとn_eの積を
    # 階層内の加重平均にも使っていたが、議員定数n_eは人口にほぼ比例せず
    # (実測でn∝P^0.27〜0.41程度)、n_eP_eをそのまま使うと大自治体の影響が
    # 意図せず人口の1.3〜1.4乗ぶん過大になっていたため、2026-09に分離した。
    entries_by_month: dict[tuple[int, int], list[tuple[str, float, float, dict[str, float]]]] = {}
    for ym in months:
        entries: list[tuple[str, float, float, dict[str, float]]] = []

        for name, parsed, size, by_year, level_type in local_entries:
            for term in _terms_in_month(parsed, ym):
                endorsement = jichisoken.endorsement_for_vote_year(by_year, name, term["vote_date"])
                endorsing = endorsement.endorsing_parties if endorsement else []
                shares = _effective_party_shares(term["party"], endorsing, national_parties)
                entries.append((level_type, size, 1.0, shares))  # 首長選挙は常にn_e=1

        for parsed, size, level_type in gikai_local_entries:
            for term in _terms_in_month(parsed, ym):
                seats = term["seats"]
                n_e = sum(seats.values())
                if n_e == 0:
                    continue
                entries.append((level_type, size, n_e, _phi(seats, national_parties)))

        for date_key, seats in sangiin_entries:
            if date_key == ym:
                n_e = sum(seats.values())
                entries.append(("参院選", national_pop, n_e, _phi(seats, national_parties)))
        for date_key, seats in shugiin_entries:
            if date_key == ym:
                n_e = sum(seats.values())
                entries.append(("衆院選", national_pop, n_e, _phi(seats, national_parties)))

        entries_by_month[ym] = entries

    return _snapshots_from_windows(entries_by_month, months, window_months)


def _months_back(ym: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """ym自身を含め、ymからn-1か月遡った月までのリストを新しい順で返す。"""
    result = []
    y, m = ym
    for _ in range(n):
        result.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return result


def _level_aggregate(
    window_entries: list[tuple[str, float, float, dict[str, float]]],
) -> tuple[dict[str, float], dict[str, float]]:
    """階層をまたぐ重みと階層内の政党別シェアを別々の量から求める、$C_p(T)$の
    定義(2026-09に再設計、notes/new_index_formula.tex参照)。

    階層をまたぐ重み$W_\\ell(T)=\\sqrt{N_\\ell(T)}$は$N_\\ell(T)=\\sum_e n_eP_e$
    (議席数×人口)から求める——複数の代表を1つの意思決定に束ねるという
    ペンローズ則の類推(design_document.tex \\S2.2)にならい、議席数を明示的に
    重みへ組み込む。

    一方、階層内の政党別シェア$\\phi_p^\\ell(T)$は$P_e$(人口)だけで加重平均し、
    $n_e$は掛けない。「その階層の有権者を1人ランダムに選び、その人の代表者
    ($n_e$人のうち1人をさらにランダムに選ぶ)がどの政党か」という設定を考えると、
    この確率はイベント$e$の中で既に$\\phi_p(e)$として定義済みであり、$n_e$は
    そこで使い切っている——集計時に人口へさらに$n_e$を掛け直す理由が無い。
    実際、議員定数$n_e$は人口にほぼ比例せず(実測で$n\\propto P^{0.27\\sim0.41}$
    程度)、$n_eP_e$をそのまま階層内の加重に使うと人口の1.3〜1.4乗ぶん
    大自治体の影響が過大になっていた(2026-09にユーザー指摘で発覚、修正)。

    戻り値は(C_p, level_weight_share)のタプルで、level_weight_shareは
    $W_\\ell(T)$を正規化したもの(図1中段の内訳表示用)。該当イベントが
    1件も無ければ両方とも空dictを返す。"""
    raw_totals: dict[str, float] = {}  # N_ell(T) = sum n_e*P_e、W_ellの計算専用
    pop_totals: dict[str, float] = {}  # 階層内のphi_p^ellの分母、P_eのみの合計
    party_totals: dict[str, dict[str, float]] = {}
    for level_type, P_e, n_e, shares in window_entries:
        raw_totals[level_type] = raw_totals.get(level_type, 0.0) + n_e * P_e
        pop_totals[level_type] = pop_totals.get(level_type, 0.0) + P_e
        level_party_totals = party_totals.setdefault(level_type, {})
        for party, share in shares.items():
            level_party_totals[party] = level_party_totals.get(party, 0.0) + P_e * share

    level_weights: dict[str, float] = {}
    phi_by_level: dict[str, dict[str, float]] = {}
    for level_type, total in raw_totals.items():
        if total <= 0:
            continue
        level_weights[level_type] = math.sqrt(total)
        pop_total = pop_totals[level_type]
        phi_by_level[level_type] = {p: v / pop_total for p, v in party_totals[level_type].items()}

    total_weight = sum(level_weights.values())
    if not total_weight:
        return {}, {}

    C_p: dict[str, float] = {}
    for level_type, w in level_weights.items():
        for party, phi in phi_by_level[level_type].items():
            C_p[party] = C_p.get(party, 0.0) + w * phi
    C_p = {p: v / total_weight for p, v in C_p.items()}

    level_weight_share = {t: level_weights.get(t, 0.0) / total_weight for t in LEVEL_TYPES}
    level_weight_share = {t: v for t, v in level_weight_share.items() if v > 0}
    return C_p, level_weight_share


def _snapshots_from_windows(
    entries_by_month: dict[tuple[int, int], list[tuple[str, float, float, dict[str, float]]]],
    months: list[tuple[int, int]],
    window_months: int,
) -> list[MonthSnapshot]:
    results: list[MonthSnapshot] = []
    for ym in months:
        if ym not in entries_by_month:
            continue
        window_entries: list[tuple[str, float, float, dict[str, float]]] = []
        for w in _months_back(ym, window_months):
            window_entries.extend(entries_by_month.get(w, []))
        if not window_entries:
            continue
        C_p, level_weight_share = _level_aggregate(window_entries)
        results.append(
            MonthSnapshot(
                year=ym[0],
                month=ym[1],
                C_p=C_p,
                n_events=len(window_entries),
                level_weight_share=level_weight_share,
            )
        )
    return results
