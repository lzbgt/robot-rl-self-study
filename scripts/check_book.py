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

    errors = check_local_links() + check_chapter_index()
    if not args.skip_examples:
        errors += run_examples()
    if errors:
        for error in errors:
            print(f"BOOK_CHECK_ERROR: {error}", file=sys.stderr)
        return 1

    print("BOOK_CHECK_OK: 20 chapters, local links, and examples are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
