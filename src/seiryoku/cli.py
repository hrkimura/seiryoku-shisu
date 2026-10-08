"""月次実行のエントリポイント。データを取得して指数を計算し、JSONで保存する。"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from . import (
    attribution, coverage_history, cp_history, municipal_registry, national_elections, precursor, registry,
    support_analysis, support_history, turnover,
)
from . import draft as draft_mod
from . import forecast as forecast_mod
from .fetch import diet, miharu
from .index import compute_combined_index
from .monthly_index import build_month_snapshots

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def monthly_image_dir(history: list[dict], base: Path = OUTPUT_DIR) -> Path:
    """月次実行の図の保存先。データの最新月ごとに分ける。

    notes/images/には公開済みの記事が参照する図が置かれているので、月次実行では
    そこに書き込まない。記事に使う図は、ここから回ごとの名前で手動でコピーする。
    """
    last = max(history, key=lambda h: (h["year"], h["month"]))
    return base / "images" / f"{last['year']:04d}-{last['month']:02d}"


def _snapshot_to_dict(s) -> dict:
    return {
        "year": s.year, "month": s.month, "C_p": s.C_p,
        "n_events": s.n_events, "level_weight_share": s.level_weight_share,
    }


def _update_monthly_pipeline() -> dict:
    """$C_p(T)$の月次履歴・図・前哨戦検証表・翌月予測を再構成して保存する
    (月一更新の自動化、design_document.tex \\S2参照)。既存のrun()が計算する
    その日の断面(C_p_coverage等)とは別建てで、note記事に使う月次系列を担当する。
    """
    from .monthly_index import _snapshots_from_windows, build_month_entries

    month_entries = build_month_entries()
    history = cp_history.save_history(_snapshots_from_windows(month_entries.entries, month_entries.months, 12))

    election_months = national_elections.compute_election_months()
    national_elections.save_election_months(election_months)

    from . import charts

    image_dir = monthly_image_dir(history)

    local_history = [_snapshot_to_dict(s) for s in build_month_snapshots(local_only=True)]
    precursor_rows = precursor.update_precursor_table(local_history)

    national_history = [_snapshot_to_dict(s) for s in build_month_snapshots(national_only=True)]
    charts.generate_national_local_scatter(
        national_history, local_history, out_path=image_dir / "national_vs_local.png",
    )

    executive_history = [_snapshot_to_dict(s) for s in build_month_snapshots(executive_only=True)]
    local_assembly_history = [
        _snapshot_to_dict(s)
        for s in build_month_snapshots(only_level=frozenset({"都道府県議会", "市区町村議会"}))
    ]
    charts.generate_executive_focus_chart(
        executive_history, local_assembly_history, out_path=image_dir / "executive_focus.png",
    )

    charts.generate_main_chart(
        history, national_history=national_history, local_history=local_history,
        executive_history=executive_history, local_assembly_history=local_assembly_history,
        out_path=image_dir / "cp_and_level_share.png",
    )

    # TimesFM-3個別予測はバックテスト再検証の結果、機械的な符号ルール(下記)を
    # 上回る予測力を確認できなかったため、いったん見送っている
    # (design_document.tex \S7.3・note記事「予測とサプライズについて」参照。
    # forecast_next_month自体のコードは残してあり、データが積み上がれば再検討する)。
    last = sorted(history, key=lambda h: (h["year"], h["month"]))[-1]
    forecast_result = forecast_mod.direction_hint(last["year"], last["month"], election_months)
    if forecast_result is not None:
        next_ym = forecast_mod._next_month(last["year"], last["month"])
        forecast_mod.record_forecast(*next_ym, forecast_result)

    last_ym = (last["year"], last["month"])
    attribution_result = attribution.decompose(month_entries, last_ym)

    return {
        "history": history,
        "image_dir": str(image_dir),
        "attribution": attribution_result,
        "election_months": election_months,
        "precursor_summary": precursor.summarize_by_party(precursor_rows),
        "forecast_next_month": forecast_result,
    }


def _update_support_analysis(history: list[dict]) -> dict:
    """政党支持率$S_p(t)$を1件取得・追記し、$C_p(t)$との連動を分析し直す。

    取得元は現在値しか公開していないので、取得に失敗した月はその月の値が
    欠測になるだけで、後から取り戻せない。失敗しても月次実行全体は止めない。
    """
    fetch_error = None
    appended = False
    snapshot = None
    try:
        snapshot = miharu.fetch_support()
        appended = support_history.record_snapshot(snapshot)
    except Exception as e:
        fetch_error = f"{type(e).__name__}: {e}"

    from . import charts
    monthly_support = support_history.monthly_series(support_history.load_history())
    results = support_analysis.analyze_all(history, monthly_support, charts._provisional_start())
    support_analysis.save_results(results)
    return {
        "fetch_error": fetch_error,
        "appended": appended,
        "updated": snapshot["updated"] if snapshot else None,
        "unknown_labels": snapshot["unknown_labels"] if snapshot else [],
        "n_support_months": len(monthly_support),
        "by_party": results,
    }


def run() -> dict:
    diet_seats = diet.fetch_diet_seats()

    gov_registry = registry.refresh_from_seijiyama(registry.load_or_init_registry())
    registry.save_registry(gov_registry)

    # municipal_registry_coverageは運用監視用に残す(市区町村長データのクロール進捗)。
    muni_state = municipal_registry.load_checkpoint()
    muni_coverage = municipal_registry.coverage_report(muni_state)

    # 実行のたびにカバレッジを1件記録し、note記事のカバレッジ推移グラフに使う
    # 月次の実データを積み上げる(2026-09にユーザー指示)。
    coverage_history.record_snapshot(
        date.today().isoformat(),
        gov_covered=sum(registry.coverage_report(gov_registry).values()),
        gov_total=len(gov_registry),
        muni_head_covered=muni_coverage["head_covered"],
        muni_gikai_covered=muni_coverage["gikai_covered"],
        muni_total=muni_coverage["total_jichitai"],
    )

    # C_p(t): 国会+既知の知事・市区町村長を人口の平方根で一律に加重した合成指数
    # (2026-09にユーザー指示でI_p(t)・E_p(t)の別建てから統合、index.py参照)。
    C_p = compute_combined_index()
    # 党派転換指数は「その時点の断面」を見る指標であり、月をまたいだトレンドとして
    # 素朴に線を追う設計にはなっていない(直近30日等の短いウィンドウは標本が薄く
    # 振れが大きい)。そこで$C_p(T)$と同じ12か月累積の断面を直近1点だけ、前月分と
    # 合わせて取得する(2026-09にユーザー指示、W_p^turnoverは首長のみ実装——議会側は
    # 前回選挙結果の新規取得が必要なため未着手、[[seiryoku_shisu_design]]参照)。
    turnover_latest, turnover_prev = turnover.latest_turnover_snapshot()

    monthly = _update_monthly_pipeline()
    support = _update_support_analysis(monthly["history"])

    return {
        "date": date.today().isoformat(),
        "C_p": C_p["share"],
        "C_p_coverage": {
            "known_jurisdictions": C_p["known_jurisdictions"],
            "total_jurisdictions": C_p["total_jurisdictions"],
        },
        "W_p_turnover": {
            "year": turnover_latest.year,
            "month": turnover_latest.month,
            "n_events": turnover_latest.n_events,
            "W_p": turnover_latest.W_p,
            "L_p": turnover_latest.L_p,
            "net_share": turnover_latest.net_share,
            "prev_net_share": turnover_prev.net_share if turnover_prev else {},
        },
        "governor_registry_coverage": registry.coverage_report(gov_registry),
        "municipal_registry_coverage": muni_coverage,
        "monthly_index": {
            "n_history_months": len(monthly["history"]),
            "image_dir": monthly["image_dir"],
            "attribution": monthly["attribution"],
            "precursor_summary": monthly["precursor_summary"],
            "forecast_next_month": monthly["forecast_next_month"],
        },
        "support_analysis": support,
        "raw": {
            "diet_seats": diet_seats,
        },
    }


def _print_support_summary(sa: dict) -> None:
    print("\n政党支持率 S_p(t)(三春充希氏の集計)との連動分析:")
    if sa["fetch_error"]:
        print(f"  [取得失敗] 今月の支持率は記録できませんでした: {sa['fetch_error']}")
    elif sa["appended"]:
        print(f"  {sa['updated']}更新の値を追記しました")
    else:
        print(f"  {sa['updated']}更新の値は記録済みのため追記なし")
    if sa["unknown_labels"]:
        print(f"  [要確認] 略称表にない政党名: {', '.join(sa['unknown_labels'])}"
              "(fetch/miharu.pyのABBREVIATIONSに追加してください)")
    print(f"  支持率の蓄積: {sa['n_support_months']}か月分")
    for party, r in sa["by_party"].items():
        line = f"  {party}: {r['n_months']}か月, {r['status']}"
        corr = r.get("level_corr") if r.get("primary") == "level" else r.get("diff_corr")
        if r.get("primary") and corr:
            line += f", {r['primary']}相関 r={corr['r']:+.2f} (n={corr['n']})"
        if "lag_peak" in r:
            line += f", ラグ相関のピーク k={r['lag_peak']}"
        print(line)


def main() -> None:
    result = run()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{result['date']}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out_path}")

    print("\n政党インパクト指数 C_p:")
    for party, v in sorted(result["C_p"].items(), key=lambda kv: -kv[1]):
        print(f"  {party}: {v:.4f}")

    print("\n知事の議席台帳カバー率:", result["governor_registry_coverage"])

    mi = result["monthly_index"]
    print(f"\n党勢指数 C_p(T)の月次履歴: {mi['n_history_months']}か月分を保存")
    print(f"図の保存先: {mi['image_dir']}")
    if mi["forecast_next_month"] is not None:
        print("翌月の方向の目安(国政/地方の型からの機械的ルール):")
        for party, v in sorted(mi["forecast_next_month"].items(), key=lambda kv: -kv[1]):
            print(f"  {party}: {'上昇' if v > 0 else '下降'}")
    else:
        print("翌月は国政選挙が絡む月ではないため方向の目安なし")

    _print_support_summary(result["support_analysis"])

    # jichisoken(全国首長名簿)の年版更新は年1回(収録期間末の4月から9〜10か月後の1〜2月ごろ)しか起きないため
    # (design_document.tex \S2.2参照)、変化した月だけ目立たせる。
    from . import charts
    provisional_start, provisional_start_changed = charts.check_provisional_start_update()
    if provisional_start_changed:
        y, m = provisional_start
        print(f"\n[更新あり] 確定値/速報値の境界が {y}年{m}月 に変わりました。首長重視率の分類を再確認してください。")

    history = cp_history.load_history()
    facts = draft_mod.build_month_facts(history, attribution=mi["attribution"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        draft_text = draft_mod.generate_monthly_draft(facts)
        draft_path = draft_mod.save_draft(draft_text, facts["year"], facts["month"])
        print(f"\n記事下書きを保存しました: {draft_path}")
    else:
        # API課金を避けるため、文章化はせず数値の整形だけ自動で済ませる
        # (2026-09にユーザー指示「無料の範囲でなるべく自動化する」)。
        facts_path = draft_mod.save_facts(facts)
        print(f"\nANTHROPIC_API_KEY未設定のため記事下書きの生成はスキップし、"
              f"事実データのみ保存しました: {facts_path}")

    # 次回実行時の対象期間の起点として、今回報告した直近月を記録する(月一更新を
    # 時計のように正確に回す気はなく、間隔が空いても前回からの続きを報告する
    # 方針、2026-09にユーザー指示)。
    draft_mod.save_last_report(facts["year"], facts["month"])


if __name__ == "__main__":
    main()
