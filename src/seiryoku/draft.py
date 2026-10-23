"""月次更新のnote記事下書きを、事実データ(数値)を元にAnthropic APIで生成する。

数値の確定・整形はこのモジュール(Python側)で行い、Anthropic APIには
「与えられた数値をもとに文章を書く」役割だけを持たせる——数値の計算・比較を
LLMに委ねると誤りうるため(design_document.tex \\S7.2のTimesFM-3採用ルール自体も
プールした検定結果に基づく数値であり、記事の文章がそれと矛盾しないように、
事実は必ずこちらで確定させてから渡す)。
"""
from __future__ import annotations

import json
from pathlib import Path

from . import municipal_registry, national_elections, precursor, turnover
from . import forecast as forecast_mod

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"

# 選挙結果キャッシュの更新クロールが未完了のときに案内するコマンド。曖昧な
# 「更新クロールを完走させて」だけでも通じるはずだが(refresh_cli.pyのdocstring
# 参照)、確実性のため実行環境(共有venvのフルパス)まで明記する(2026-09に
# ユーザー指示)。
REFRESH_CLI_COMMAND = "~/Desktop/program/色々/bin/python3 -m seiryoku.refresh_cli"

# 前回、実際に記事(下書き)を書いた対象月の末尾を記録する。月一更新を時計のように
# 正確に回す気はなく、実行間隔が空くこともある前提のため、「今月」固定ではなく
# 「前回の更新月の次の月から、データが取得できる直近の月まで」を対象期間として
# 扱う(2026-09にユーザー指示)。
LAST_REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "last_report.json"


def load_last_report() -> tuple[int, int] | None:
    if LAST_REPORT_PATH.exists():
        d = json.loads(LAST_REPORT_PATH.read_text(encoding="utf-8"))
        return (d["year"], d["month"])
    return None


def save_last_report(year: int, month: int) -> None:
    LAST_REPORT_PATH.parent.mkdir(exist_ok=True)
    LAST_REPORT_PATH.write_text(
        json.dumps({"year": year, "month": month}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


SYSTEM_PROMPT = """あなたは「党勢指数」というnote連載の月次更新記事の下書きを書く。
読者は既に第0回の記事(指数の設計・TimesFM-3採用ルール・前哨戦検証の説明)を
読んでいる前提でよく、指数の定義自体を説明し直す必要はない。

厳守するスタイル:
- ダッシュ(——)を使わない。区切りたければ句点で文を分けるか括弧書きにする。
- 太字・箇条書きは実際に列挙可能な内容にのみ使い、機械的に増やさない。
- 「実測」「疑う」「照合」「入口」「土台」「事故」(比喩として)「別物」「既定」
  「定番」「効く」「瞬間」「静かに」「壊れる」「黙って」のような語彙を避け、
  普通の言い回しに置き換える。
- 「まとめ」という見出しを安易に付けない。
- 数値の変化が大きくても原因を断定せず、「要確認」「詳細は不明」のように留保する。
  因果関係を勝手に説明しない。
- 与えられた事実(数値)以外を創作しない。事実に無い数値・出来事を書かない。
- 常体(である調)で書く。
- 党勢指数の表には各党の前月からの変化だけを載せ、水準(現在の値)は載せない。水準を並べると
  政党どうしを比べる表に見えるため。並びは変化の大きい順にする。
- 図と表には毎回キャプションを付ける(図・表の直後の行に「*図1: ...*」「*表1: ...*」の形で)。
  noteの画像にはキャプション欄があり、note用の変換(note_export.py)でそこに貼る形に直す。

毎月必ず入れる節:
- 選挙講評: その月にあった主な選挙(事実データの「窓に入った選挙」の上位など)の結果を短くまとめる。
- 指数への影響の考察: 事実データの前月差の分解をもとに、窓に入った選挙と窓から外れた選挙
  (12か月前の選挙)のどちらが、どの政党の値をどれだけ動かしたかを説明する。分解は計算上の
  寄与であり、それ以上の因果(選挙の勝因など)は書かない。
- 政党支持率: 事実データの政党支持率(三春充希氏の集計)の全項目を、更新日と出典リンクを
  添えて表にする。党勢指数とは分母も意味も違うので水準を比べない。連動分析は、蓄積が
  足りない間は「まだ検定できる長さではない」とだけ書く。
"""


INDEPENDENT = "無所属"


def exclude_independent(c_p: dict[str, float]) -> dict[str, float]:
    total = sum(v for p, v in c_p.items() if p != INDEPENDENT)
    if total == 0:
        return {}
    return {p: v / total for p, v in c_p.items() if p != INDEPENDENT}


def build_month_facts(
    history: list[dict],
    election_months: list[tuple[int, int]] | None = None,
    precursor_rows: list[dict] | None = None,
    since: tuple[int, int] | None = None,
    attribution: dict | None = None,
    support: dict | None = None,
    trend: dict | None = None,
) -> dict:
    """cp_history.load_history()の結果(年月順である必要はない)から、
    下書きに必要な事実だけを取り出して整形する。

    対象期間は「前回更新した月の次の月」から「データが取得できる直近の月」まで
    (sinceを省略した場合はload_last_report()の値を使う)。前回更新が無ければ
    (初回、またはsinceが履歴の範囲外なら)直近1か月だけを対象にする。複数か月分を
    まとめて報告する場合、C_pの差分は期間の合計(期間開始の直前月から直近月まで)
    で見る——月ごとに分けても、間の月の値は結局「まだ報告していない中間状態」に
    過ぎず読者には意味がないため。"""
    history = sorted(history, key=lambda h: (h["year"], h["month"]))
    latest = history[-1]
    year, month = latest["year"], latest["month"]

    if since is None:
        since = load_last_report()

    if since is None:
        period = [latest]
    else:
        period = [h for h in history if (h["year"], h["month"]) > since]
        if not period:
            period = [latest]

    period_start_idx = history.index(period[0])
    baseline = history[period_start_idx - 1] if period_start_idx > 0 else None

    # 記事では図1の1段目と同じく、無所属を除いて正規化した割合で各党の増減を見る。
    # 無所属は直近の首長選挙の推薦・支持の反映が遅れるうえ、統一的な政策集団でもないため。
    latest_cp = exclude_independent(latest["C_p"])
    diffs: dict[str, float] = {}
    if baseline is not None:
        baseline_cp = exclude_independent(baseline["C_p"])
        parties = set(latest_cp) | set(baseline_cp)
        diffs = {p: latest_cp.get(p, 0.0) - baseline_cp.get(p, 0.0) for p in parties}

    if election_months is None:
        election_months = national_elections.load_election_months()
    period_yms = {(h["year"], h["month"]) for h in period}
    events_in_period = sorted(
        {label for ym, label in national_elections.compute_election_events() if ym in period_yms}
    )

    n_events = sum(h["n_events"] for h in period)

    forecast_history = forecast_mod.load_forecast_history()
    forecast_check: dict[str, dict[str, dict[str, float]]] = {}
    for i, h in enumerate(history):
        if (h["year"], h["month"]) not in period_yms or i == 0:
            continue
        predicted = forecast_history.get(f"{h['year']:04d}-{h['month']:02d}")
        if predicted is None:
            continue
        prev_h = history[i - 1]
        forecast_check[f"{h['year']:04d}-{h['month']:02d}"] = {
            party: {
                "predicted": predicted.get(party, 0.0),
                "actual": h["C_p"].get(party, 0.0) - prev_h["C_p"].get(party, 0.0),
            }
            for party in predicted
        }

    next_year, next_month = forecast_mod._next_month(year, month)
    will_forecast_next = national_elections.is_election_linked_month(next_year, next_month, election_months)

    if precursor_rows is None:
        precursor_rows = precursor.load_precursor_table()
    precursor_summary = {
        party: s for party, s in precursor.summarize_by_party(precursor_rows).items() if s["total"] > 0
    }

    return {
        "year": year,
        "month": month,
        "period_start": {"year": period[0]["year"], "month": period[0]["month"]},
        "C_p": latest_cp,
        "diffs": diffs,
        "independent_share": latest["C_p"].get(INDEPENDENT, 0.0),
        "attribution": attribution,
        "support": support,
        "trend": trend,
        "n_events": n_events,
        "level_weight_share": latest["level_weight_share"],
        "events_in_period": events_in_period,
        "forecast_check": forecast_check or None,
        "next_month": {"year": next_year, "month": next_month, "will_forecast": will_forecast_next},
        "precursor_summary": precursor_summary,
        "cache_refresh_incomplete": _cache_refresh_incomplete(),
    }


def _cache_refresh_incomplete() -> bool:
    """選挙結果キャッシュの更新クロール(refresh_cli.py)が完了しないまま
    中断された進捗ファイルが残っているとTrue。この場合、今月の数値に本来
    反映されるべき新しい選挙結果が漏れている可能性がある(2026-09にユーザー
    指摘: クロールが月をまたいで完了しないまま記事生成日を迎えると、沖縄県
    知事選のような見落としが静かに起こりうる)。"""
    return turnover.REFRESH_PROGRESS_PATH.exists() or municipal_registry.GIKAI_REFRESH_PROGRESS_PATH.exists()


def _fmt_effect(effect: dict[str, float], threshold: float = 5e-5) -> str:
    items = sorted(((p, v) for p, v in effect.items() if abs(v) >= threshold), key=lambda kv: -abs(kv[1]))
    return "、".join(f"{p} {100 * v:+.3f}" for p, v in items) or "ほぼ変化なし"


def _fmt_shares(shares: dict[str, float]) -> str:
    items = sorted(shares.items(), key=lambda kv: -kv[1])
    return "、".join(f"{p} {v:.2f}" for p, v in items)


def _format_attribution(a: dict) -> list[str]:
    """前月差の分解(attribution.decompose)を記事の「指数への影響の考察」用に整形する。"""
    ex = a["exiting_month"]
    lines = [
        f"{a['year']}年{a['month']}月の前月差の分解(無所属を除いて正規化した割合、単位はポイント):",
        f"  - 全体: {_fmt_effect(a['total'])}",
        f"  - 窓に入った選挙({a['year']}年{a['month']}月、{a['n_entered']}件)の効果: {_fmt_effect(a['entered'])}",
        f"  - 窓から外れた選挙({ex['year']}年{ex['month']}月、{a['n_exited']}件)の効果: {_fmt_effect(a['exited'])}",
        "  - 窓に入った選挙の効果の階層別内訳:",
    ]
    for level, effect in a["entered_by_level"].items():
        lines.append(f"    - {level}: {_fmt_effect(effect)}")
    lines.append("  - 窓に入った選挙のうち影響の大きいもの(その1件を抜いた場合との差):")
    for r in a["top_entered"]:
        lines.append(f"    - {r['vote_date']} {r['name']}({r['level']}) 構成[{_fmt_shares(r['shares'])}] 効果: {_fmt_effect(r['effect'])}")
    lines.append("  - 窓から外れた選挙のうち影響の大きいもの(その1件だけが外れた場合の変化):")
    for r in a["top_exited"]:
        lines.append(f"    - {r['vote_date']} {r['name']}({r['level']}) 構成[{_fmt_shares(r['shares'])}] 効果: {_fmt_effect(r['effect'])}")
    return lines


def _format_support(s: dict) -> list[str]:
    """政党支持率(三春充希氏の集計)の記録値と、党勢指数との連動分析の状態を整形する。

    記事には毎月、最新の値の全項目を出典付きで表にして載せる。"""
    lines = []
    if s.get("fetch_error"):
        lines.append(f"警告: 今回の実行では政党支持率を取得できなかった({s['fetch_error']})。下の値は前回までに記録したもの。")
    latest = s.get("latest")
    if latest:
        lines.append(f"政党支持率(三春充希氏による各社世論調査の平均、{latest['source']})の{latest['updated']}更新時点の値(%、記事の掲載順):")
        for party, v in latest["S_p"].items():
            lines.append(f"  - {party}: {v}")
    lines.append(f"政党支持率の蓄積: {s['n_months']}か月分(党勢指数との連動分析の状態は以下)")
    for party, r in s.get("by_party", {}).items():
        line = f"  - {party}: 重なる月 {r['n_months']}か月、{r['status']}"
        corr = r.get("level_corr") if r.get("primary") == "level" else r.get("diff_corr")
        if r.get("primary") and corr:
            line += f"、{r['primary']}相関 r={corr['r']:+.2f}(n={corr['n']})"
        if "lag_peak" in r:
            line += f"、ラグ相関のピーク k={r['lag_peak']}"
        if r.get("reason"):
            line += f"({r['reason']})"
        lines.append(line)
    return lines


def _format_support_diff(d: dict | None, horizon: int) -> list[str]:
    """党勢指数v3と政党支持率のhorizonか月の差(support_analysis.trend_support_diffs)を整形する。"""
    if not d or not d["points"]:
        return [f"党勢指数v3と政党支持率の{horizon}か月差の散布図: {horizon}か月離れた月末の支持率がまだそろわないため、今回はなし。"]
    _METHOD = {"exact": "更新日が月末", "interpolated": "前後の更新から補間", "nearest": "最も近い更新"}
    lines = [f"党勢指数v3と政党支持率(無党派を除いて再計算)の{horizon}か月差(重ならない{len(d['periods'])}期間・"
             f"{d['n']}点、どちらもポイント):"]
    for p in d["points"]:
        lines.append(f"  - {p['party']} {p['from']}→{p['to']}: v3 {p['dx']:+.2f}、支持率 {p['ds']:+.2f}")
    if d["pearson"]:
        c = d["pearson"]
        ci = f"、95%区間 {c['ci95'][0]:+.2f}〜{c['ci95'][1]:+.2f}" if c["ci95"] else ""
        lines.append(f"  相関 r={c['r']:+.2f}(n={c['n']}{ci}、同じ月の党どうしは独立でないので区間は目安)")
    if d["spearman"]:
        lines.append(f"  順位相関 ρ={d['spearman']['rho']:+.2f}")
    lines.append("  支持率を月末にそろえた方法:")
    for tau, a in d["alignment"].items():
        lines.append(f"    - {tau}: {_METHOD[a['method']]}({'・'.join(a['dates'])}、月末から最も近い更新まで{a['offset_days']}日)")
    return lines


def _format_trend(t: dict) -> list[str]:
    """党勢指数v3(state_space_glm.month_facts)を整形する。"""
    if t.get("error"):
        return [f"警告: 党勢指数v3を計算できなかった({t['error']})。"]
    lines = []
    if t.get("by_election_error"):
        lines.append(f"警告: 議会補選の取り込みに失敗し、前回までの補選で計算した({t['by_election_error']})。")
    if t.get("refit_reason"):
        lines.append(f"党勢指数v3のパラメータを今回推定し直した({t['refit_reason']})。")
    lines.append(f"党勢指数v3({t['tau']}時点、衆院の議席率の尺度、その月までの選挙による値。"
                 "括弧内は±2標準偏差の幅、変化は前月末からのポイント差、変化の大きい順):")
    rows = sorted(t["parties"].items(), key=lambda kv: -abs(kv[1]["change"] or 0.0))
    for party, r in rows:
        ch = f"{r['change'] * 100:+.1f}" if r["change"] is not None else "—"
        lines.append(f"  - {party}: {r['x'] * 100:.1f}%({r['x_lo'] * 100:.1f}〜{r['x_hi'] * 100:.1f}%)、変化 {ch}")
    for h, d in (t.get("support_diff") or {}).items():
        lines.extend(_format_support_diff(d, h))
    if t["impacts"]:
        lines.append("その月の選挙のうち党勢を大きく動かしたもの(選挙前の見込みと実際の議席の割合、"
                     "党勢の動きのポイント換算、動きの大きい順):")
        for c in t["impacts"]:
            lines.append(f"  - {c['party']} {c['vote_date']} {c['name']}({c['level']}、{c['n']:.0f}議席): "
                         f"見込み {c['expected'] * 100:.1f}% / 実際 {c['actual'] * 100:.1f}%、党勢 {c['dx'] * 100:+.2f}")
    return lines


def _format_facts_for_prompt(facts: dict) -> str:
    ps = facts["period_start"]
    if (ps["year"], ps["month"]) == (facts["year"], facts["month"]):
        lines = [f"対象月: {facts['year']}年{facts['month']}月(前回更新のすぐ翌月)"]
    else:
        lines = [
            f"対象期間: {ps['year']}年{ps['month']}月〜{facts['year']}年{facts['month']}月"
            "(前回の更新から間が空いたため、複数か月分をまとめて報告する)"
        ]
    if facts["cache_refresh_incomplete"]:
        lines.append(
            "警告: 選挙結果キャッシュの更新クロールが完了しないまま中断されている。"
            "今月の数値に、本来反映されるべき新しい選挙結果が漏れている可能性がある。"
            "公開前に手動で確認するか、更新クロール(refresh_cli.py)を再実行してから出し直すこと。"
        )
    lines.append(f"この期間にあった選挙イベント数の合計(直近12か月窓内の合計ではなく単純合計): {facts['n_events']}")
    if facts["events_in_period"]:
        lines.append(f"この期間に投票日があった国政選挙: {', '.join(facts['events_in_period'])}")

    if facts["diffs"]:
        lines.append(
            f"政党ごとのC_p(無所属を除いて正規化した割合。期間開始直前から{facts['year']}年{facts['month']}月までの差分、絶対値が大きい順):"
        )
        for party, d in sorted(facts["diffs"].items(), key=lambda kv: -abs(kv[1])):
            current = facts["C_p"].get(party, 0.0)
            lines.append(f"  - {party}: {d:+.4f}(現在の値 {current:.4f})")
    else:
        lines.append("比較対象の月のデータが無いため、差分を計算していない(今回が初回の可能性)。")
    if "independent_share" in facts:
        lines.append(f"参考: 無所属を含めた全体に占める無所属の割合 {facts['independent_share']:.4f}")
    if facts.get("attribution"):
        lines.extend(_format_attribution(facts["attribution"]))
    if facts.get("support"):
        lines.extend(_format_support(facts["support"]))
    if facts.get("trend"):
        lines.extend(_format_trend(facts["trend"]))

    if facts["level_weight_share"]:
        lines.append(f"直近({facts['year']}年{facts['month']}月)の重みシェア内訳(どの種類の選挙がC_pを動かしたか):")
        for level, share in sorted(facts["level_weight_share"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {level}: {share:.2f}")

    if facts["forecast_check"]:
        lines.append("この期間中にTimesFM-3で事前に出していた予測との答え合わせ(月ごと):")
        for ym, per_party in sorted(facts["forecast_check"].items()):
            lines.append(f"  {ym}:")
            for party, c in per_party.items():
                lines.append(f"    - {party}: 予測 {c['predicted']:+.4f} / 実際 {c['actual']:+.4f}")
    else:
        lines.append("この期間分の事前予測は無かった(対象月が予測対象月ではなかった)。")

    nm = facts["next_month"]
    if nm["will_forecast"]:
        lines.append(f"{nm['year']}年{nm['month']}月は国政選挙が絡む月にあたるため、次回更新でTimesFM-3による予測を行う予定である。")
    else:
        lines.append(f"{nm['year']}年{nm['month']}月は国政選挙が絡まない月にあたるため、次回更新では予測を出さない予定である。")

    if facts["precursor_summary"]:
        lines.append("「地方選は国政選挙の前哨戦か」の検証(地方限定C_pの増減方向と実際の議席シェア増減方向の一致数、これまでに確定した選挙ペアの累計):")
        for party, s in facts["precursor_summary"].items():
            lines.append(f"  - {party}: {s['match']}/{s['total']}")

    return "\n".join(lines)


def generate_monthly_draft(facts: dict, client=None) -> str:
    """factsを事実として渡し、Anthropic APIに月次更新記事の下書き本文を書かせる。

    テストではclientに偽オブジェクトを注入できるようにし、anthropicパッケージが
    未インストールの環境でも(実際にAPIを呼ばない限り)importエラーにならない
    ようにする(forecast.pyのtimesfmと同じ方針)。"""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    warning_instruction = ""
    if facts["cache_refresh_incomplete"]:
        warning_instruction = (
            "なお、事実データ中に「選挙結果キャッシュの更新クロールが完了しないまま"
            "中断されている」という警告がある場合、記事の見出しより前(本文の一番最初)に"
            "「公開前に必ず確認: 」で始まる警告文を1行入れ、以下のコマンドを完走させてから"
            f"出し直すよう促すこと(コマンドはそのまま引用すること): `{REFRESH_CLI_COMMAND}`\n\n"
        )

    user_prompt = (
        "以下は今回の更新分の党勢指数の事実データである(更新間隔は一定ではなく、"
        "複数か月分をまとめて報告することもある)。これをもとに、note連載の"
        "更新記事の下書き(Markdown形式)を書いてほしい。見出しは"
        f"「# 党勢指数 {facts['year']}年{facts['month']}月更新」から始めること。\n\n"
        + warning_instruction
        + _format_facts_for_prompt(facts)
    )

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def save_draft(text: str, year: int, month: int, out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"draft_{year:04d}-{month:02d}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_facts(facts: dict, out_dir: Path = OUTPUT_DIR) -> Path:
    """ANTHROPIC_API_KEY未設定の場合のフォールバック。API課金無しで自動化できる
    範囲(数値の整形)まではここで済ませ、実際の文章化は人間が対話環境(Claude Code等、
    従量課金APIとは別のサブスクリプションで無料)で行う運用にする(2026-09に
    ユーザー指示「無料の範囲でなるべく自動化する」)。"""
    out_dir.mkdir(exist_ok=True)
    year, month = facts["year"], facts["month"]
    path = out_dir / f"draft_facts_{year:04d}-{month:02d}.txt"
    header = (
        f"# {year}年{month}月の党勢指数、下書き用の事実データ\n"
        "# このファイルをClaude Code等の対話環境に渡して記事の文章を書いてもらう想定。\n"
        "# スタイルの指示はdraft.SYSTEM_PROMPTを参照。\n\n"
    )
    if facts["cache_refresh_incomplete"]:
        # ファイルの一番最初、通常の見出しより前に置く。人間がこのファイルを
        # そのままこの会話に貼っても、真っ先にこの指示が目に入るようにするため
        # (2026-09にユーザー指示「下書き先頭部分に再度指示してくれ」)。
        header = (
            "!!! 公開前に必ず確認: 選挙結果キャッシュの更新クロールが完了しないまま"
            "中断されている。今月の数値に反映漏れがある可能性があるため、以下を実行して"
            "完走させてからcli.pyを再実行し、この下書きを出し直すこと。\n"
            f"    {REFRESH_CLI_COMMAND}\n"
            "!!!\n\n"
        ) + header
    path.write_text(header + _format_facts_for_prompt(facts), encoding="utf-8")
    return path
