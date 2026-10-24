"""月次記事の準備(図のコピー、警告のまとめと通知)と、Claude Codeによる下書きの生成。

月次実行(cli)の最後に prepare() が図を記事用の名前でコピーし、警告を
output/status_YYYY-MM.txt に書いてデスクトップに通知する。その後、別のunit
(seiryoku-shisu-draft.service)が main() で `claude -p` を起動し、
notes/monthly_draft_prompt.md の指示で下書きを書かせる。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
NOTES_DIR = ROOT / "notes"
PROMPT_PATH = NOTES_DIR / "monthly_draft_prompt.md"
CLAUDE = Path.home() / ".local" / "bin" / "claude"

# 第0回が2026年8月分。以後、月ごとに1回ずつ進む。
FIRST_ARTICLE = (2026, 8)

ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch",
    "Bash(PYTHONPATH=src python3 -m seiryoku.note_export:*)",
    "Bash(git status:*)", "Bash(git diff:*)",
]


def article_number(year: int, month: int) -> int:
    return (year - FIRST_ARTICLE[0]) * 12 + month - FIRST_ARTICLE[1]


def copy_images(image_dir: Path, n: int, dest: Path = NOTES_DIR / "images") -> list[Path]:
    """記事に載せる図を notes/images/ に回ごとの名前でコピーする。差の散布図は、
    出ている中でいちばん長い期間のものを使う。"""
    pairs = [("trend_v2.png", "trend_v2"), ("trend_v2_vs_support.png", "trend_v2_vs_support")]
    diffs = sorted(image_dir.glob("trend_v2_support_diff_*m.png"),
                   key=lambda p: int(re.search(r"_(\d+)m\.png$", p.name).group(1)))
    if diffs:
        pairs.append((diffs[-1].name, "trend_v2_support_diff"))
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_name, name in pairs:
        src = image_dir / src_name
        if src.exists():
            copied.append(Path(shutil.copy2(src, dest / f"dai{n}kai_{name}.png")))
    return copied


def support_fetch_failures(since: str = "1 month ago") -> int | None:
    """毎日の支持率の取得が失敗した回数。journalctlが使えなければNone。"""
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "seiryoku-shisu-support.service", "--since", since, "--no-pager"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.count("[取得失敗]")


def collect_warnings(facts_text: str, unknown_labels: list[str], refit_reason: str | None,
                     validated: bool, provisional_start_changed: bool, support_failures: int | None) -> list[str]:
    warnings = [l.strip() for l in facts_text.splitlines() if l.startswith(("警告:", "!!!"))]
    if unknown_labels:
        warnings.append(f"[要確認] 支持率の略称表にない政党名: {', '.join(unknown_labels)}"
                        "(src/seiryoku/fetch/miharu.py の ABBREVIATIONS に追加する)")
    if refit_reason:
        warnings.append(f"[要確認] パラメータを推定し直した({refit_reason})。"
                        + ("補間の検証も回し直した。" if validated else "補間の検証は失敗した。"))
    if provisional_start_changed:
        warnings.append("[要確認] 確定値/速報値の境界が変わった。首長重視率の分類を見直す。")
    if support_failures:
        warnings.append(f"[参考] 支持率の毎日の取得がこの1か月で{support_failures}回失敗した(翌日に取り直される)。")
    return warnings


def status_path(year: int, month: int) -> Path:
    return OUTPUT_DIR / f"status_{year:04d}-{month:02d}.txt"


def write_status(year: int, month: int, n: int, warnings: list[str], copied: list[Path]) -> Path:
    lines = [f"# {year}年{month}月分(第{n}回)の月次実行", ""]
    lines += ["## 警告", *(warnings or ["なし"]), ""]
    lines += ["## 記事用にコピーした図", *(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                                         for p in copied)]
    path = status_path(year, month)
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare(year: int, month: int, image_dir: Path, facts_text: str, unknown_labels: list[str],
            refit_reason: str | None, validated: bool, provisional_start_changed: bool) -> Path:
    n = article_number(year, month)
    copied = copy_images(image_dir, n)
    warnings = collect_warnings(facts_text, unknown_labels, refit_reason, validated,
                                provisional_start_changed, support_fetch_failures())
    path = write_status(year, month, n, warnings, copied)
    notify(f"{year}年{month}月分の月次実行が終わりました",
           f"警告{len(warnings)}件。続けて第{n}回の下書きを生成します" if warnings
           else f"警告なし。続けて第{n}回の下書きを生成します")
    return path


def notify(title: str, body: str) -> None:
    """デスクトップ通知。通知できない環境(ログアウト中など)では何もしない。MATEの通知は数秒で消えて
    履歴にも残らないので、クリックするまで消えないようにする。"""
    try:
        subprocess.run(["notify-send", "--app-name=党勢指数", "--urgency=critical", "--expire-time=0", title, body],
                       timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def latest_facts(output_dir: Path | None = None) -> tuple[int, int] | None:
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    months = [tuple(map(int, m.groups())) for p in output_dir.glob("draft_facts_*.txt")
              if (m := re.fullmatch(r"draft_facts_(\d{4})-(\d{2})\.txt", p.name))]
    return max(months) if months else None


def build_prompt(year: int, month: int, template: str | None = None) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8") if template is None else template
    return template.replace("{{YM}}", f"{year:04d}-{month:02d}").replace("{{N}}", str(article_number(year, month)))


def draft_log_path(year: int, month: int) -> Path:
    return OUTPUT_DIR / f"draft_log_{year:04d}-{month:02d}.txt"


def run_claude(prompt: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return subprocess.run(
        [str(CLAUDE), "-p", prompt, "--permission-mode", "acceptEdits", "--allowedTools", ",".join(ALLOWED_TOOLS)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="最新の事実データから、Claude Codeに月次記事の下書きを書かせる")
    ap.add_argument("--print-prompt", action="store_true", help="依頼文を表示するだけで、Claudeは起動しない")
    ap.add_argument("--force", action="store_true", help="その月の下書きを書き終えていても書き直させる")
    args = ap.parse_args(argv)

    ym = latest_facts()
    if ym is None:
        print("output/draft_facts_*.txt がない")
        return 1
    prompt = build_prompt(*ym)
    if args.print_prompt:
        print(prompt)
        return 0
    n = article_number(*ym)
    log = draft_log_path(*ym)
    draft = NOTES_DIR / f"dai{n}kai_draft.md"
    if (log.exists() or draft.exists()) and not args.force:
        print(f"第{n}回の下書きはすでにある({draft})")
        return 0
    proc = run_claude(prompt)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep="\n")
        # 起動前に下書きがないことを確かめているので、残っているのは途中で止まった下書きだけ。
        # 消しておかないと、取り直しが「すでにある」で止まる。
        draft.unlink(missing_ok=True)
        notify(f"第{n}回の下書きの生成に失敗", "seiryoku-shisu-draft.service がしばらくして取り直します")
        return 1
    log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    notify(f"第{n}回の下書きができました", f"notes/dai{n}kai_draft.md(報告は {log.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
