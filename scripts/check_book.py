#!/usr/bin/env python3
"""Validate the self-study book's local links and runnable examples.

This script deliberately uses only Python's standard library so it works in a
fresh checkout and in GitHub Actions without installing the robot stack.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


BOOK_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CHAPTER_RE = re.compile(r"^(\d{2})_[a-z0-9_]+\.md$")
EXAMPLES = (
    "bandit_incremental_mean.py",
    "gridworld_value_iteration.py",
    "tabular_q_learning.py",
    "ppo_clip_demo.py",
)
EXERCISE_HEADING_RE = re.compile(
    r"^#{2,3} .*?(?:Exercises|Check your understanding|Lab:|exercise|"
    r"What to reproduce|Microduck experiment|Capstone)",
    re.IGNORECASE | re.MULTILINE,
)


def local_target(raw_target: str) -> str | None:
    """Return the path portion of a local Markdown target, if it has one."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        # Markdown titles follow the URL after whitespace. This book does not
        # use spaces in local filenames, so splitting is unambiguous.
        target = target.split(maxsplit=1)[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return path or None


def check_local_links() -> list[str]:
    errors: list[str] = []
    for markdown in sorted(BOOK_ROOT.rglob("*.md")):
        text = markdown.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = local_target(match.group(1))
            if target is None:
                continue
            resolved = (markdown.parent / target).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{markdown.relative_to(BOOK_ROOT)}:{line}: "
                    f"missing local target {target!r}"
                )
    return errors


def check_chapter_index() -> list[str]:
    errors: list[str] = []
    chapters = sorted(
        path for path in BOOK_ROOT.glob("*.md") if CHAPTER_RE.match(path.name)
    )
    actual = [int(CHAPTER_RE.match(path.name).group(1)) for path in chapters]
    expected = list(range(1, 21))
    if actual != expected:
        errors.append(f"chapter sequence is {actual}; expected {expected}")

    index = (BOOK_ROOT / "README.md").read_text(encoding="utf-8")
    for chapter in chapters:
        occurrences = index.count(f"({chapter.name})")
        if occurrences != 1:
            errors.append(
                f"README.md links {chapter.name} {occurrences} times; expected once"
            )
    return errors


def check_markdown_conventions() -> tuple[list[str], int, int]:
    """Check GitHub math blocks and chapter-end folded solution placement."""

    errors: list[str] = []
    math_blocks = 0
    solution_folds = 0
    for markdown in sorted(BOOK_ROOT.glob("*.md")):
        text = markdown.read_text(encoding="utf-8")
        lines = text.splitlines()
        math_start: int | None = None

        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped == "$$":
                errors.append(
                    f"{markdown.name}:{number}: use a fenced ```math block, not $$"
                )
            if r"\operatorname" in line:
                errors.append(
                    f"{markdown.name}:{number}: use GitHub-safe \\mathrm notation"
                )
            if stripped == "```math":
                if math_start is not None:
                    errors.append(
                        f"{markdown.name}:{number}: nested math fence opened; "
                        f"previous start is line {math_start}"
                    )
                math_start = number
                math_blocks += 1
            elif stripped == "```" and math_start is not None:
                math_start = None

        if math_start is not None:
            errors.append(
                f"{markdown.name}:{math_start}: unclosed fenced math block"
            )

        opened = len(re.findall(r"^<details(?:\s[^>]*)?>$", text, re.MULTILINE))
        closed = len(re.findall(r"^</details>$", text, re.MULTILINE))
        if opened != closed:
            errors.append(
                f"{markdown.name}: details tags are unbalanced: {opened} open, "
                f"{closed} closed"
            )
        solution_folds += opened

        if re.search(r"^### Solution$", text, re.MULTILINE):
            errors.append(
                f"{markdown.name}: solutions must be folded at the chapter end, "
                "not inline after a problem"
            )

        exercise_matches = list(EXERCISE_HEADING_RE.finditer(text))
        if exercise_matches:
            last_prompt = exercise_matches[-1].start()
            folded = re.search(
                r"^## .*Folded .*",
                text[last_prompt:],
                re.IGNORECASE | re.MULTILINE,
            )
            if folded is None:
                errors.append(
                    f"{markdown.name}: exercise/lab prompts need a later "
                    "chapter-end folded solution or rubric"
                )

    return errors, math_blocks, solution_folds


def run_examples() -> list[str]:
    errors: list[str] = []
    for name in EXAMPLES:
        path = BOOK_ROOT / "examples" / name
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=BOOK_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"example {name} failed ({result.returncode}): {detail}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-examples", action="store_true", help="check documents only"
    )
    args = parser.parse_args()

    convention_errors, math_blocks, solution_folds = check_markdown_conventions()
    errors = check_local_links() + check_chapter_index() + convention_errors
    if not args.skip_examples:
        errors += run_examples()
    if errors:
        for error in errors:
            print(f"BOOK_CHECK_ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "BOOK_CHECK_OK: 20 chapters, local links, examples, "
        f"{math_blocks} math blocks, and {solution_folds} solution folds are valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
