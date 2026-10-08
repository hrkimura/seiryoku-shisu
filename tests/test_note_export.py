from seiryoku import note_export


def test_math_to_text_handles_delta_and_mathrm():
    assert note_export.math_to_text(r"\mathrm{Corr}(\Delta S_p(t), \Delta C_p(t+k))") == "Corr(ΔS_p(t), ΔC_p(t+k))"
    assert note_export.math_to_text("C_p(T)") == "C_p(T)"


def test_convert_rewrites_math_images_captions_and_joins_paragraph_lines(tmp_path):
    md = (
        "# 題\n\n"
        "指数$${C_p(T)}$$は\n続く行。\n\n"
        "![図](images/fig.png)\n*図1: 説明*\n\n"
        "- 項目1\n- 項目2\n\n"
        "> 引用\n"
    )
    assert note_export.convert(md, tmp_path, "x") == (
        "# 題\n\n"
        "指数C_p(T)は続く行。\n\n"
        "[ここに画像: fig.png]\n図1: 説明\n\n"
        "- 項目1\n- 項目2\n\n"
        "> 引用\n"
    )


def test_convert_turns_tables_into_images(tmp_path):
    md = "前文\n\n| 政党 | 値 |\n|---|---|\n| A | 1 |\n| B | 2 |\n\n後文\n"
    out = note_export.convert(md, tmp_path, "dai1kai")
    assert "[ここに画像: dai1kai_table1.png]" in out
    assert "|" not in out
    assert (tmp_path / "dai1kai_table1.png").stat().st_size > 0


def test_parse_table_drops_separator_row():
    header, rows = note_export._parse_table(["| a | b |", "|---|:--:|", "| 1 | 2 |"])
    assert header == ["a", "b"] and rows == [["1", "2"]]
