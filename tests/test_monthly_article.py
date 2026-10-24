import subprocess

from seiryoku import monthly_article as ma


def test_article_number_counts_months_from_dai0kai():
    assert ma.article_number(2026, 8) == 0
    assert ma.article_number(2026, 10) == 2
    assert ma.article_number(2027, 1) == 5


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
    path = ma.write_status(2026, 10, 2, [], [tmp_path / "a.png"])
    text = path.read_text()
    assert path.name == "status_2026-10.txt"
    assert "第2回" in text and "なし" in text and "a.png" in text


def test_latest_facts_and_prompt(tmp_path):
    for ym in ["2026-08", "2026-10", "2026-09"]:
        (tmp_path / f"draft_facts_{ym}.txt").write_text("")
    (tmp_path / "draft_facts_2026-10.txt.bak").write_text("")
    assert ma.latest_facts(tmp_path) == (2026, 10)
    assert ma.latest_facts(tmp_path / "none") is None
    assert ma.build_prompt(2026, 10, "facts_{{YM}} dai{{N}}kai 第{{N}}回") == "facts_2026-10 dai2kai 第2回"


def test_prompt_template_has_placeholders():
    prompt = ma.build_prompt(2026, 10)
    assert "{{" not in prompt
    assert "output/draft_facts_2026-10.txt" in prompt and "notes/dai2kai_draft.md" in prompt


def _setup_main(tmp_path, monkeypatch, result):
    out, notes = tmp_path / "output", tmp_path / "notes"
    out.mkdir()
    notes.mkdir()
    (out / "draft_facts_2026-10.txt").write_text("")
    monkeypatch.setattr(ma, "OUTPUT_DIR", out)
    monkeypatch.setattr(ma, "NOTES_DIR", notes)
    monkeypatch.setattr(ma, "ROOT", tmp_path)
    monkeypatch.setattr(ma, "build_prompt", lambda y, m: "prompt")
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
