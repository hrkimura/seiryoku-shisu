from seiryoku.fetch.diet_history import (
    _norm,
    _parse_sangiin_result_sheet,
    _parse_shugiin_result_sheet,
    _strip_by_election_suffix,
)


def _sangiin_block(parties: list[str], grand_totals: list) -> list[tuple]:
    """参院選の当選人数シート1ブロックぶんの行を組み立てる。

    実データでは「比例代表・選挙区・合計」の3区分がそれぞれ「新・現・元・計」の
    行を持ち、各区分の「計」行にも(政党の絞り込みに使う「計」列だけでなく)
    「男」列に実際の数値が入る。この「男」列の数値が政党名の見出しと誤認されない
    ことを確認するため、途中の区分の計行にもダミーの数値を入れる(2026-09に
    発見した回帰のためのテストデータ)。最後の区分(合計)の計行だけが実際の
    grand_totalsを持つ。
    """
    header: list = [None, None]
    for name in parties:
        header += [name, None, None]  # 実データでは政党名セルは1回だけ(結合セル)
    danjokei = [None, None] + ["男", "女", "計"] * len(parties)

    def blank_row(label0, label1):
        return tuple([label0, label1] + [None] * (len(header) - 2))

    def total_row(label0, label1, values):
        row = [label0, label1]
        for v in values:
            row += [3, 1, v]  # 男・女はダミー値、政党見出しと誤認されないことを確認する
        return tuple(row)

    zeros = [0] * len(parties)
    rows = [
        tuple(header),
        tuple(danjokei),
        blank_row("比", "新"), blank_row("例", "現"), blank_row("代", "元"),
        total_row(None, "計", zeros),
        blank_row("選", "新"), blank_row("挙", "現"), blank_row("区", "元"),
        total_row(None, "計", zeros),
        blank_row(None, "新"), blank_row("合", "現"), blank_row("計", "元"),
        total_row(None, "計", grand_totals),
    ]
    return rows


def _block(parties: list[str], totals: list[int]) -> list[tuple]:
    """1ブロックぶんの行を組み立てる(タイトル行・空行は呼び出し側で付ける)。
    小選挙区・比例代表の内訳は問わないので、合計の「計」行にだけ実際の値を入れる。
    """
    header = ["", ""] + [name for name in parties for _ in range(3)]
    kubun = ["区  分"] + [""] * (len(header) - 1)
    danjokei = ["", ""] + ["男", "女", "計"] * len(parties)
    filler = lambda label0, label1: [label0, label1] + [""] * (len(header) - 2)
    total_row = ["", "計"]
    for t in totals:
        total_row += ["", "", t]
    rows = [
        tuple(header),
        tuple(kubun),
        tuple(danjokei),
        tuple(filler("小", "新")),
        tuple(filler("選", "前")),
        tuple(filler("挙", "元")),
        tuple(filler("区", "計")),
        tuple(filler("比", "新")),
        tuple(filler("例", "前")),
        tuple(filler("代", "元")),
        tuple(filler("表", "計")),
        tuple(filler("", "新")),
        tuple(filler("合", "前")),
        tuple(filler("計", "元")),
        tuple(total_row),
    ]
    return rows


def test_norm_strips_fullwidth_and_ascii_whitespace():
    assert _norm("小　　計") == "小計"
    assert _norm("  無所属 ") == "無所属"
    assert _norm(None) == ""
    assert _norm(42) == ""


def test_parse_single_block_sheet():
    title = [("3. 開票結果", "", "", "", "", "（１）届出政党等別男女別新前元別当選人数(小選挙区、比例代表)")]
    blank = [("",)]
    rows = title + blank + _block(["自由民主党", "民主党"], [119, 308])
    assert _parse_shugiin_result_sheet(rows) == {"自由民主党": 119, "民主党": 308}


def test_parse_sheet_with_multiple_stacked_blocks():
    """1シートに政党名の見出し行が複数回繰り返されるケース(実データで確認済み)。
    ブロックをまたいで最後の「計」行を拾うと、後のブロックの値が前のブロックの
    政党に紐付いてしまうバグが過去にあったための回帰テスト。
    """
    title = [("3. 開票結果", "", "", "", "", "（１）届出政党等別男女別新前元別当選人数(小選挙区、比例代表)")]
    blank = [("",)]
    block1 = _block(["自由民主党", "公明党"], [281, 29])
    block2 = _block(["日本共産党", "社会民主党"], [12, 2])
    rows = title + blank + block1 + blank + block2
    assert _parse_shugiin_result_sheet(rows) == {
        "自由民主党": 281,
        "公明党": 29,
        "日本共産党": 12,
        "社会民主党": 2,
    }


def test_parse_sheet_counts_self_filed_party_members_as_others():
    """本人・推薦届出の表の「小計」「合計」は再集計列なので除外し、「無所属」は1カテゴリとして残す。
    「政党等所属」は政党の届出でなく本人届出で立った政党所属の当選者で、党派別の表に入らない
    (2024年の日本保守党の小選挙区の1議席)ので、諸派として数える。"""
    title = [("3. 開票結果", "", "", "", "", "（１）届出政党等別男女別新前元別当選人数(小選挙区、比例代表)")]
    subtitle = [("", "", "本人･推薦届出")]
    blank = [("", "", "", "", "", "", "", "", "", "", "", "　　     合　計")]
    header = ["区  分", "", "        政党等所属", "", "", "        無所属", "", "", "        小   計", "", "", ""]
    danjokei = ["", ""] + ["男", "女", "計"] * 3 + [""]
    filler = lambda label0, label1: [label0, label1] + [""] * (len(header) - 2)
    total_row = ["", "計", "", "", 1, "", "", 12, "", "", 13, "", "", 465]
    rows = (
        title
        + subtitle
        + blank
        + [tuple(header), tuple(danjokei)]
        + [tuple(filler(a, b)) for a, b in [("小", "新"), ("選", "前"), ("挙", "元"), ("区", "計")]]
        + [tuple(filler(a, b)) for a, b in [("比", "新"), ("例", "前"), ("代", "元"), ("表", "計")]]
        + [tuple(filler(a, b)) for a, b in [("", "新"), ("合", "前"), ("計", "元")]]
        + [tuple(total_row)]
    )
    seats = _parse_shugiin_result_sheet(rows)
    assert seats == {"諸派": 1, "無所属": 12}


def test_parse_sheet_raises_when_no_header_found():
    import pytest

    with pytest.raises(ValueError):
        _parse_shugiin_result_sheet([("", "", ""), ("", "", "")])


def test_strip_by_election_suffix_drops_bundled_by_election_count():
    # 「（　）書は、通常選挙と合併して行われた補欠選挙の当選人の数で外書」
    assert _strip_by_election_suffix("21(1)") == 21
    assert _strip_by_election_suffix("14") == 14
    assert _strip_by_election_suffix(9) == 9
    assert _strip_by_election_suffix("") == 0
    assert _strip_by_election_suffix(None) == 0


def test_parse_sangiin_sheet_uses_last_total_row_as_grand_total():
    """参院選のシートは選挙区・比例代表それぞれにも「計」行があるため、
    最後(合計区分)の「計」行だけをその回の改選当選者数として使う。"""
    title = [("3. 開票結果", "", "", "", "", "（１）党派別男女別新現元別当選人数（比例代表、選挙区）")]
    rows = title + _sangiin_block(["自由民主党", "公明党"], [63, 13])
    assert _parse_sangiin_result_sheet(rows) == {"自由民主党": 63, "公明党": 13}


def test_parse_sangiin_sheet_strips_by_election_suffix_from_grand_total():
    title = [("3. 開票結果",)]
    rows = title + _sangiin_block(["自由民主党", "公明党"], ["21(1)", 8])
    assert _parse_sangiin_result_sheet(rows) == {"自由民主党": 21, "公明党": 8}


def test_parse_sangiin_sheet_does_not_mistake_numeric_value_cells_for_headers():
    """途中の区分の「計」行にある「男」列の数値(バグ発見時は「23」のような
    政党名見出しに見える文字列)が新しいブロックの見出しと誤認されないことの
    回帰テスト(2026-09に発見)。"""
    title = [("3. 開票結果",)]
    rows = title + _sangiin_block(["自由民主党", "公明党"], [23, 4])
    assert _parse_sangiin_result_sheet(rows) == {"自由民主党": 23, "公明党": 4}


def test_parse_sangiin_sheet_with_multiple_stacked_blocks():
    title = [("3. 開票結果",)]
    rows = (
        title
        + _sangiin_block(["自由民主党", "公明党"], [63, 13])
        + title
        + _sangiin_block(["日本共産党", "無所属"], [4, 5])
    )
    assert _parse_sangiin_result_sheet(rows) == {
        "自由民主党": 63,
        "公明党": 13,
        "日本共産党": 4,
        "無所属": 5,
    }


def test_parse_sangiin_sheet_joins_a_party_name_wrapped_onto_two_header_rows():
    """2019年のシートでは「労働の解放をめざす/労働者党」が2行に折り返され、続きの行が
    別の見出しとみなされて同じブロックの社民・れいわ・NHK党が抜けていた。"""
    title = [("3. 開票結果",)]
    block = _sangiin_block(["社会民主党", "れいわ新選組", "労働の解放をめざす"], [1, 2, 0])
    i = next(j for j, row in enumerate(block) if "社会民主党" in row)
    cont = tuple(["区　分", None] + [None] * 6 + ["労働者党", None, None])
    rows = title + _sangiin_block(["自由民主党"], [57]) + title + block[: i + 1] + [cont] + block[i + 1 :]
    assert _parse_sangiin_result_sheet(rows) == {
        "自由民主党": 57,
        "社会民主党": 1,
        "れいわ新選組": 2,
        "労働の解放をめざす労働者党": 0,
    }


def test_parse_shugiin_sheet_joins_a_party_name_wrapped_onto_two_header_rows():
    """2021年のシートでは「ＮＨＫと裁判してる党/弁護士法72条違反で」が2行に折り返され、
    同じブロックのれいわ・社民が抜けていた。"""
    header = ("", "", "ＮＨＫと裁判してる党", "", "", "れいわ新選組", "", "", "社会民主党", "", "")
    cont = ("", "", "弁護士法72条違反で", "", "", "", "", "", "", "", "")
    danjokei = ("", "") + ("男", "女", "計") * 3
    total = ("", "計", 0, 0, 0, 2, 1, 3, 1, 0, 1)
    seats = _parse_shugiin_result_sheet([header, cont, danjokei, total])
    assert seats == {"ＮＨＫと裁判してる党弁護士法72条違反で": 0, "れいわ新選組": 3, "社会民主党": 1}


def test_parse_sangiin_sheet_raises_when_no_header_found():
    import pytest

    with pytest.raises(ValueError):
        _parse_sangiin_result_sheet([("", "", ""), ("", "", "")])
