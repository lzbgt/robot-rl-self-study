#!/usr/bin/env python3
"""Build the complete book PDF from the GitHub-first Markdown sources.

The source format deliberately uses fenced ``math`` blocks because GitHub
renders them reliably. This builder converts those fences to Pandoc display
math in a generated file. It also expands the HTML ``details`` answer folds so
the PDF is complete when read offline. Source Markdown is never rewritten.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"
STEM = "robot-rl-self-study"
PDF = DIST / f"{STEM}.pdf"
CHECKSUMS = DIST / "SHA256SUMS"
TEX = BUILD / f"{STEM}.tex"
COMBINED = BUILD / f"{STEM}.md"
REPOSITORY = "https://github.com/lzbgt/robot-rl-self-study"
CHAPTERS = sorted(ROOT.glob("[0-9][0-9]_*.md"))
SOURCES = [ROOT / "pdf" / "frontmatter.md", *CHAPTERS, ROOT / "SOURCES.md"]
BUILD_ENV = os.environ.copy()
BUILD_ENV.update(
    {
        # TeX and xdvipdfmx honor this standard reproducible-build timestamp.
        # It matches the edition date in metadata.yaml (2026-08-31 UTC).
        "SOURCE_DATE_EPOCH": "1788134400",
        "FORCE_SOURCE_DATE": "1",
        "TZ": "UTC",
    }
)

SUMMARY_RE = re.compile(r"^<summary>(.*?)</summary>\s*$")
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
CHAPTER_RE = re.compile(r"^(\d{2})_[a-z0-9_]+\.md$")


def require_tools() -> None:
    missing = [name for name in ("pandoc", "xelatex") if shutil.which(name) is None]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"missing PDF build tool(s): {names}. See README.md#pdf-edition"
        )


def rewrite_link(match: re.Match[str], source: Path) -> str:
    """Make chapter links internal and other repository links web-accessible."""

    label, raw_target = match.groups()
    target = raw_target.strip()
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return match.group(0)

    path_text, marker, fragment = target.partition("#")
    basename = Path(path_text).name
    chapter = CHAPTER_RE.match(basename)
    if chapter:
        destination = f"#{fragment}" if marker and fragment else f"#chapter-{chapter.group(1)}"
    elif basename == "SOURCES.md":
        destination = f"#{fragment}" if marker and fragment else "#sources"
    elif basename == "README.md":
        destination = REPOSITORY + (f"#{fragment}" if marker and fragment else "")
    else:
        relative = (source.parent / path_text).resolve().relative_to(ROOT)
        destination = f"{REPOSITORY}/blob/main/{relative.as_posix()}"
        if marker and fragment:
            destination += f"#{fragment}"
    return f"[{label}]({destination})"


def preprocess(source: Path) -> str:
    """Translate GitHub presentation syntax into print presentation syntax."""

    lines = source.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_math = False

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "```math":
            if in_math:
                raise ValueError(f"{source.name}:{number}: nested math fence")
            output.append("$$")
            in_math = True
            continue
        if in_math and stripped == "```":
            output.append("$$")
            in_math = False
            continue
        if stripped in {"<details>", "</details>"}:
            continue
        summary = SUMMARY_RE.match(stripped)
        if summary:
            title = html.unescape(summary.group(1)).strip()
            output.extend((f"### {title}", ""))
            continue
        output.append(line)

    if in_math:
        raise ValueError(f"{source.name}: unclosed math fence")

    text = "\n".join(output).strip() + "\n"
    chapter = CHAPTER_RE.match(source.name)
    if chapter:
        text = re.sub(
            r"^(# .+)$",
            rf"\1 {{#chapter-{chapter.group(1)}}}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    elif source.name == "SOURCES.md":
        text = re.sub(
            r"^(# .+)$", r"\1 {#sources}", text, count=1, flags=re.MULTILINE
        )

    return LINK_RE.sub(lambda match: rewrite_link(match, source), text)


def run(command: list[str], *, log: Path | None = None, quiet: bool = False) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=BUILD_ENV,
    )
    if log is not None:
        log.write_text(result.stdout, encoding="utf-8")
    if result.stdout and (not quiet or result.returncode):
        print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    require_tools()
    if len(CHAPTERS) != 20:
        raise SystemExit(f"expected 20 chapters, found {len(CHAPTERS)}")

    BUILD.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    for suffix in ("aux", "log", "out", "pdf", "toc"):
        generated = BUILD / f"{STEM}.{suffix}"
        if generated.exists():
            generated.unlink()
    combined = "\n\n\\newpage\n\n".join(preprocess(path) for path in SOURCES)
    COMBINED.write_text(combined, encoding="utf-8")

    run(
        [
            "pandoc",
            str(COMBINED),
            "--from=markdown+tex_math_dollars+raw_tex+pipe_tables+fenced_code_blocks",
            "--to=latex",
            "--standalone",
            "--top-level-division=chapter",
            "--metadata-file=pdf/metadata.yaml",
            "--include-in-header=pdf/latex-header.tex",
            "--syntax-highlighting=idiomatic",
            "--output",
            str(TEX),
        ],
        log=BUILD / "pandoc.log",
    )

    # A clean build needs three passes: create the contents, account for its
    # pages, then confirm the shifted page references are stable.
    xelatex = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={BUILD}",
        str(TEX),
    ]
    for pass_number in range(1, 4):
        run(
            xelatex,
            log=BUILD / f"xelatex-pass-{pass_number}.txt",
            quiet=True,
        )

    built_pdf = BUILD / PDF.name
    if not built_pdf.exists():
        raise SystemExit(f"XeLaTeX did not create {built_pdf}")
    shutil.copy2(built_pdf, PDF)
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    CHECKSUMS.write_text(f"{digest}  {PDF.name}\n", encoding="utf-8")
    print(
        f"PDF_BUILD_OK: {PDF.relative_to(ROOT)} ({PDF.stat().st_size:,} bytes, "
        f"sha256 {digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
