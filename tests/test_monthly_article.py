import json
import subprocess

from seiryoku import monthly_article as ma


def _trend(national=(), impacts=(), one_month_beyond=False):
    ch = [{"months": 1, "beyond_error": one_month_beyond}, {"months": 12, "beyond_error": True}]
    return {"parties": {"A": {"changes": ch}}, "national": list(national), "impacts": list(impacts)}


def test_article_reasons_need_a_national_election_a_beyond_error_result_or_a_long_gap():
    prev = {"2026-10": 2}
    # 12か月の変化が誤差を超えていても、それだけでは記事にしない
    assert ma.article_reasons(_trend(), 2026, 11, prev) == []
    assert ma.article_reasons(_trend(national=[{"level": "衆院選"}]), 2026, 11, prev) == ["国政選挙があった"]
    assert ma.article_reasons(_trend(one_month_beyond=True), 2026, 11, prev) == ["1か月の変化が誤差を超えた党がある"]
    assert ma.article_reasons(_trend(impacts=[{"level": "市区町村議会"}]), 2026, 11, prev) == [
        "見込みから誤差を超えて外れた地方選挙がある"]
    # 国政選挙の外れはimpactsにも入るが、地方選挙の理由にはしない
    assert ma.article_reasons(_trend(national=[{"level": "衆院補選"}], impacts=[{"level": "衆院補選"}]),
                              2026, 11, prev) == ["国政選挙があった"]
    assert ma.article_reasons(_trend(), 2027, 4, prev) == ["前回の記事から6か月たった"]
    assert ma.article_reasons(_trend(), 2027, 3, prev) == []
    assert ma.article_reasons({"error": "x"}, 2026, 11, prev) == []
    assert ma.article_reasons(None, 2026, 11, {}) == ["最初の記事"]


def test_decide_numbers_articles_consecutively_and_keeps_the_number_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "ROOT", tmp_path)
    monkeypatch.setattr(ma, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "data").mkdir()
    ma.articles_path().write_text(json.dumps({"2026-10": 2}))
    d = ma.decide(_trend(), 2026, 11, ["[参考] 支持率の取得が1回失敗"])
    assert d == {"article": False, "n": None, "reasons": [], "needs_claude": False}
    assert ma.decide(_trend(), 2026, 11, ["[要確認] 略称表にない政党名: X"])["needs_claude"]
    d = ma.decide(_trend(national=[{"level": "衆院選"}]), 2027, 1, [])
    assert d["article"] and d["n"] == 3
    assert ma.article_number(2027, 1) == 3
    # やり直しで理由が変わっても、振った番号は変えない
    assert ma.decide(_trend(), 2027, 1, [])["n"] == 3
    assert ma.load_decision(2027, 1)["n"] == 3


def test_copy_images_uses_longest_diff_horizon(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    for name in ["trend_v2.png", "trend_v2_vs_support.png", "trend_v2_support_diff_1m.png",
                 "trend_v2_support_diff_3m.png", "trend_v2_by_party.png"]:
        (src / name).write_text(name)
    copied = ma.copy_images(src, 4, dest)
    assert sorted(p.name for p in copied) == [
        "dai4kai_trend_v2.png", "dai4kai_trend_v2_support_diff.png", "dai4kai_trend_v2_vs_support.png"]
    assert (dest / "dai4kai_trend_v2_support_diff.png").read_text() == "trend_v2_support_diff_3m.png"


def test_copy_images_skips_missing_figures(tmp_path):
    (tmp_path / "trend_v2.png").write_text("x")
    assert [p.name for p in ma.copy_images(tmp_path, 2, tmp_path / "d")] == ["dai2kai_trend_v2.png"]


def test_collect_warnings_gathers_facts_and_run_warnings():
    facts = "# 見出し\n警告: 党勢指数v3を計算できなかった(X)。\n本文\n!!! 公開前に必ず確認: クロール\n"
    w = ma.collect_warnings(facts, ["新党"], "国政選挙", True, False, 2)
    assert w[0].startswith("警告: 党勢指数v3")
    assert w[1].startswith("!!!")
    assert any("新党" in x for x in w)
    assert any("補間の検証も回し直した" in x for x in w)
    assert any("2回失敗" in x for x in w)
    assert ma.collect_warnings("本文\n", [], None, False, False, 0) == []


def test_write_status_lists_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "OUTPUT_DIR", tmp_path)
    path = ma.write_status(2026, 10, 2, [], [tmp_path / "a.png"], ["国政選挙があった"])
    text = path.read_text()
    assert path.name == "status_2026-10.txt"
    assert "第2回" in text and "なし" in text and "a.png" in text and "理由: 国政選挙があった" in text
    assert "今月は記事にしない" in ma.write_status(2026, 11, None, [], []).read_text()


def test_latest_facts_and_prompt(tmp_path, monkeypatch):
    for ym in ["2026-08", "2026-10", "2026-09"]:
        (tmp_path / f"draft_facts_{ym}.txt").write_text("")
    (tmp_path / "draft_facts_2026-10.txt.bak").write_text("")
    assert ma.latest_facts(tmp_path) == (2026, 10)
    assert ma.latest_facts(tmp_path / "none") is None
    monkeypatch.setattr(ma, "ROOT", tmp_path)
    monkeypatch.setattr(ma, "OUTPUT_DIR", tmp_path)
    (tmp_path / "data").mkdir()
    ma.decide(_trend(national=[{"level": "衆院選"}]), 2026, 10, [])
    assert ma.build_prompt(2026, 10, "facts_{{YM}} dai{{N}}kai 第{{N}}回 {{REASONS}}") == \
        "facts_2026-10 dai0kai 第0回 国政選挙があった、最初の記事"


def test_prompt_templates_have_no_leftover_placeholders():
    prompt = ma.build_prompt(2026, 10)
    assert "{{" not in prompt
    assert "output/draft_facts_2026-10.txt" in prompt and "notes/dai2kai_draft.md" in prompt
    maintenance = ma.build_prompt(2026, 10, maintenance=True)
    assert "{{" not in maintenance and "output/status_2026-10.txt" in maintenance


def _setup_main(tmp_path, monkeypatch, result):
    out, notes = tmp_path / "output", tmp_path / "notes"
    out.mkdir()
    notes.mkdir()
    (out / "draft_facts_2026-10.txt").write_text("")
    monkeypatch.setattr(ma, "OUTPUT_DIR", out)
    monkeypatch.setattr(ma, "NOTES_DIR", notes)
    monkeypatch.setattr(ma, "ROOT", tmp_path)
    monkeypatch.setattr(ma, "build_prompt", lambda y, m, maintenance=False: "対処" if maintenance else "prompt")
    (out / "article_decision_2026-10.json").write_text(json.dumps(
        {"article": True, "n": 2, "reasons": ["国政選挙があった"], "needs_claude": True}))
    monkeypatch.setattr(ma, "notify", lambda *a: None)
    calls = []

    def fake_claude(prompt):
        calls.append(prompt)
        (notes / "dai2kai_draft.md").write_text("途中")
        return result
    monkeypatch.setattr(ma, "run_claude", fake_claude)
    return out, notes, calls


def test_main_writes_log_and_skips_next_time(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "報告", ""))
    assert ma.main([]) == 0
    assert (out / "draft_log_2026-10.txt").read_text() == "報告"
    assert ma.main([]) == 0
    assert len(calls) == 1
    assert ma.main(["--force"]) == 0
    assert len(calls) == 2


def test_main_failure_removes_partial_draft(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 1, "", "network"))
    assert ma.main([]) == 1
    assert not (notes / "dai2kai_draft.md").exists()
    assert not (out / "draft_log_2026-10.txt").exists()


def test_main_keeps_existing_draft(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "", ""))
    (notes / "dai2kai_draft.md").write_text("手で書いた")
    assert ma.main([]) == 0
    assert calls == []
    assert (notes / "dai2kai_draft.md").read_text() == "手で書いた"


def test_national_by_lines_reports_added_pending_and_errors():
    r = {"vote_date": "2027-04-25", "house": "衆議院", "district": "甲県第1区", "winner": "A", "party": "B"}
    lines = ma.national_by_lines({"added": [r], "pending": ["2027年 10月の補欠選挙"], "error": "X"})
    assert "衆議院甲県第1区" in lines[0] and "10月の補欠選挙" in lines[1] and "X" in lines[2]
    assert ma.national_by_lines(None) == [] and ma.national_by_lines({"added": [], "pending": []}) == []
    assert len(ma.collect_warnings("", [], None, False, False, 0, {"added": [r], "pending": []})) == 1


def test_main_gives_up_after_max_failures(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 1, "", "network"))
    assert ma.main([]) == 1
    assert ma.main([]) == 1
    assert ma.main([]) == ma.EXIT_GIVE_UP
    assert ma.main([]) == 0
    assert len(calls) == 3
    monkeypatch.setattr(ma, "run_claude", lambda p: subprocess.CompletedProcess([], 0, "報告", ""))
    assert ma.main(["--force"]) == 0
    assert not (out / "draft_failures_2026-10").exists()


def test_main_success_clears_failure_count(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "報告", ""))
    (out / "draft_failures_2026-10").write_text("2\n")
    assert ma.main([]) == 0
    assert not (out / "draft_failures_2026-10").exists()


def _no_article(out, needs_claude):
    (out / "article_decision_2026-10.json").write_text(json.dumps(
        {"article": False, "n": None, "reasons": [], "needs_claude": needs_claude}))


def test_main_without_article_and_warnings_does_not_start_claude(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "報告", ""))
    _no_article(out, False)
    assert ma.main([]) == 0
    assert calls == []


def test_main_without_article_asks_claude_only_to_handle_warnings(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "対処の報告", ""))
    _no_article(out, True)
    assert ma.main([]) == 0
    assert calls == ["対処"]
    assert (out / "draft_log_2026-10.txt").read_text() == "対処の報告"
    assert ma.main([]) == 0
    assert len(calls) == 1


def test_main_needs_the_decision_of_the_monthly_run(tmp_path, monkeypatch):
    out, notes, calls = _setup_main(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, "", ""))
    (out / "article_decision_2026-10.json").unlink()
    assert ma.main([]) == 1
    assert calls == []
