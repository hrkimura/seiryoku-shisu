"""記事の下書き(Markdown)を、note.comの編集画面に貼り付ける形に変換する。

noteの編集画面は数式記法・表組み・斜体を扱えないため、次のように置き換える。
- $${...}$$の数式はプレーンテキスト(C_p(T)、ΔS_p(t)など)にする
- 表は画像にして、本文には「[ここに画像: ファイル名]」と書く
- 画像の埋め込みも同じ「[ここに画像: ファイル名]」にする(noteでは手で挿入する)
- 図・表の直後の斜体1行(*図1: ...*)はキャプションとみなし、「[キャプション: ...]」として
  画像の直後に出す(noteでは画像のキャプション欄に貼る。図・表には毎回キャプションを付ける)
- 段落内の改行をつなげる
"""
from __future__ import annotations

import re
from pathlib import Path

_MATH_RE = re.compile(r"\$\$\{(.*?)\}\$\$")
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\((?:[^)]*/)?([^)/]+)\)$")
_CAPTION_RE = re.compile(r"^\*([^*].*[^*])\*$")
_TEX_SYMBOLS = {r"\Delta": "Δ", r"\ell": "ℓ", r"\phi": "φ", r"\beta": "β", r"\,": ""}


def math_to_text(expr: str) -> str:
    t = expr
    t = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
    for tex, plain in _TEX_SYMBOLS.items():
        t = t.replace(tex + " ", plain).replace(tex, plain)
    return t.replace("{", "").replace("}", "")


def _is_block_line(line: str) -> bool:
    return (
        line.startswith(("#", "- ", "* ", "> ", "|", "[ここに画像", "[キャプション", "!["))
        or re.match(r"^\d+\. ", line) is not None
    )


def _parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    cells = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    header, body = cells[0], [r for r in cells[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    return header, body


def render_table_image(header: list[str], rows: list[list[str]], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    fig, ax = plt.subplots(figsize=(2.6 * len(header), 0.42 * (len(rows) + 1)), dpi=220)
    ax.axis("off")
    # bbox=[0, 0, 1, 1]で表を軸いっぱいに広げないと、行数に関係なく上下に余白が残る。
    table = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def convert(markdown: str, table_image_dir: Path, table_prefix: str) -> str:
    """下書きの本文をnote用に変換する。表の画像はtable_image_dirに
    「{table_prefix}_table{番号}.png」として書き出す。"""
    out: list[str] = []
    lines = markdown.splitlines()
    i = 0
    n_tables = 0
    while i < len(lines):
        line = _MATH_RE.sub(lambda m: math_to_text(m.group(1)), lines[i])
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(_MATH_RE.sub(lambda m: math_to_text(m.group(1)), lines[i]))
                i += 1
            n_tables += 1
            name = f"{table_prefix}_table{n_tables}.png"
            render_table_image(*_parse_table(block), table_image_dir / name)
            out.append(f"[ここに画像: {name}]")
            continue
        image = _IMAGE_RE.match(line)
        caption = _CAPTION_RE.match(line)
        if image:
            out.append(f"[ここに画像: {image.group(1)}]")
        elif caption and out and out[-1].startswith("[ここに画像"):
            out.append(f"[キャプション: {caption.group(1)}]")
        elif caption:
            out.append(caption.group(1))
        elif out and line and out[-1] and not _is_block_line(line) and not _is_block_line(out[-1]):
            out[-1] += line
        else:
            out.append(line)
        i += 1
    return "\n".join(out) + "\n"


MAX_CHARS = 5000


_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def char_count(note_text: str) -> int:
    """note用の本文の字数。画像とキャプションの行、改行、リンクのURLは数えない(リンクは表示される文字だけ)。"""
    return sum(len(_LINK_RE.sub(r"\1", l)) for l in note_text.splitlines()
               if not l.startswith(("[ここに画像", "[キャプション")))


def export(draft_path: Path) -> Path:
    """notes/xxx_draft.md -> notes/xxx_draft_for_note.md(表の画像はnotes/images/へ)。"""
    stem = draft_path.stem
    note_path = draft_path.with_name(f"{stem}_for_note.md")
    prefix = stem.removesuffix("_draft")
    note_path.write_text(
        convert(draft_path.read_text(encoding="utf-8"), draft_path.parent / "images", prefix),
        encoding="utf-8",
    )
    return note_path


if __name__ == "__main__":
    import sys

    note_path = export(Path(sys.argv[1]))
    n = char_count(note_path.read_text(encoding="utf-8"))
    print(note_path)
    print(f"{n}字" + (f"(上限の{MAX_CHARS}字を超えている)" if n > MAX_CHARS else ""))
