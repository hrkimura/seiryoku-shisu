"""月次実行のエントリポイント。データを取得して指数を計算し、JSONで保存する。"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from . import coverage_history, cp_history, municipal_registry, national_elections, precursor, registry, turnover
from . import draft as draft_mod
from . import forecast as forecast_mod
from .fetch import diet
from .index import compute_combined_index
from .monthly_index import build_month_snapshots

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


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
    history = cp_history.save_history(build_month_snapshots())

    election_months = national_elections.compute_election_months()
    national_elections.save_election_months(election_months)

    from . import charts

    local_history = [_snapshot_to_dict(s) for s in build_month_snapshots(local_only=True)]
    precursor_rows = precursor.update_precursor_table(local_history)

    national_history = [_snapshot_to_dict(s) for s in build_month_snapshots(national_only=True)]
    charts.generate_national_local_scatter(national_history, local_history)

    executive_history = [_snapshot_to_dict(s) for s in build_month_snapshots(executive_only=True)]
    local_assembly_history = [
        _snapshot_to_dict(s)
        for s in build_month_snapshots(only_level=frozenset({"都道府県議会", "市区町村議会"}))
    ]
    charts.generate_executive_focus_chart(executive_history, local_assembly_history)

    charts.generate_main_chart(
        history, national_history=national_history, local_history=local_history,
        executive_history=executive_history, local_assembly_history=local_assembly_history,
    )

    # timesfmは重い任意依存であり、本番のrequirements.txtには含まれるが未導入の
    # 実行環境でパイプライン全体を止めたくないため、無ければ予測をスキップする
    # (forecast.py参照、「ゼロ予測ではなく予測なし」の方針と同じ考え方)。
    forecast_result = None
    try:
        forecast_result = forecast_mod.forecast_next_month(history, election_months)
    except ImportError as exc:
        print(f"予測をスキップ(timesfm未導入): {exc}")

    if forecast_result is not None:
        last = sorted(history, key=lambda h: (h["year"], h["month"]))[-1]
        next_ym = forecast_mod._next_month(last["year"], last["month"])
        forecast_mod.record_forecast(*next_ym, forecast_result)

    return {
        "history": history,
        "election_months": election_months,
        "precursor_summary": precursor.summarize_by_party(precursor_rows),
        "forecast_next_month": forecast_result,
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
            "precursor_summary": monthly["precursor_summary"],
            "forecast_next_month": monthly["forecast_next_month"],
        },
        "raw": {
            "diet_seats": diet_seats,
        },
    }


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
    if mi["forecast_next_month"] is not None:
        print("翌月のTimesFM-3予測:")
        for party, v in sorted(mi["forecast_next_month"].items(), key=lambda kv: -kv[1]):
            print(f"  {party}: {v:+.4f}")
    else:
        print("翌月は予測対象月ではない、またはtimesfm未導入のため予測なし")

    history = cp_history.load_history()
    facts = draft_mod.build_month_facts(history)
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
