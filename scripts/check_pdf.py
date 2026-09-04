#!/usr/bin/env python3
"""Validate the generated PDF's metadata, contents, fonts, and TeX layout log."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "dist" / "robot-rl-self-study.pdf"
LOG = ROOT / "build" / "robot-rl-self-study.log"
CHECKSUMS = ROOT / "dist" / "SHA256SUMS"
LATEX_HEADER = ROOT / "pdf" / "latex-header.tex"
COVER_ART = ROOT / "pdf" / "cover-art.png"
COVER_PROVENANCE = ROOT / "pdf" / "cover-art.provenance.md"
PDF_MARKDOWN = [
    ROOT / "pdf" / "frontmatter.md",
    *sorted(ROOT.glob("[0-9][0-9]_*.md")),
    ROOT / "SOURCES.md",
]
COLOR_RE = re.compile(
    r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}"
)


def relative_luminance(hex_color: str) -> float:
    """Return WCAG relative luminance for an sRGB hexadecimal color."""

    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio for two hexadecimal sRGB colors."""

    first = relative_luminance(foreground)
    second = relative_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def check_semantic_palette() -> tuple[list[str], float]:
    """Keep every text/background pairing above the normal-text threshold."""

    if not LATEX_HEADER.exists():
        return ["missing semantic style source: pdf/latex-header.tex"], 0.0
    palette = dict(COLOR_RE.findall(LATEX_HEADER.read_text(encoding="utf-8")))
    palette["PaperWhite"] = "FFFFFF"
    pairs = (
        ("BookInk", "PaperWhite"),
        ("ConceptNavy", "PaperWhite"),
        ("ProcessTeal", "PaperWhite"),
        ("EquationNavy", "PaperWhite"),
        ("LinkTeal", "PaperWhite"),
        ("AnswerAmber", "AnswerTint"),
        ("BookInk", "CodeTint"),
        ("ConceptNavy", "CodeTint"),
        ("ProcessTeal", "CodeTint"),
        ("AnswerAmber", "CodeTint"),
    )
    errors: list[str] = []
    ratios: list[float] = []
    for foreground, background in pairs:
        missing = [name for name in (foreground, background) if name not in palette]
        if missing:
            errors.append(
                "semantic palette is missing color(s): " + ", ".join(missing)
            )
            continue
        ratio = contrast_ratio(palette[foreground], palette[background])
        ratios.append(ratio)
        if ratio < 4.5:
            errors.append(
                f"{foreground} on {background} has {ratio:.2f}:1 contrast; "
                "normal-size text requires at least 4.5:1"
            )
    return errors, min(ratios, default=0.0)


def command(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join(args)} failed: {detail}")
    return result.stdout


def main() -> int:
    errors: list[str] = []
    palette_errors, minimum_contrast = check_semantic_palette()
    errors.extend(palette_errors)
    for tool in ("pdfinfo", "pdftotext", "pdffonts"):
        if shutil.which(tool) is None:
            errors.append(f"missing validation tool: {tool} (install Poppler)")
    if errors:
        return report(errors)
    if not PDF.exists():
        return report([f"missing generated PDF: {PDF.relative_to(ROOT)}"])
    if not LOG.exists():
        return report([f"missing XeLaTeX log: {LOG.relative_to(ROOT)}"])
    if not CHECKSUMS.exists():
        errors.append(f"missing checksum file: {CHECKSUMS.relative_to(ROOT)}")
    else:
        expected = CHECKSUMS.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(PDF.read_bytes()).hexdigest()
        if expected != f"{actual}  {PDF.name}":
            errors.append("dist/SHA256SUMS does not match the generated PDF")

    if not COVER_ART.exists():
        errors.append(f"missing cover artwork: {COVER_ART.relative_to(ROOT)}")
    if not COVER_PROVENANCE.exists():
        errors.append(
            f"missing cover provenance: {COVER_PROVENANCE.relative_to(ROOT)}"
        )
    elif COVER_ART.exists():
        provenance = COVER_PROVENANCE.read_text(encoding="utf-8")
        recorded_digest = re.search(
            r"^- SHA-256: `([0-9a-f]{64})`$", provenance, re.MULTILINE
        )
        actual_digest = hashlib.sha256(COVER_ART.read_bytes()).hexdigest()
        if recorded_digest is None:
            errors.append("cover provenance has no valid SHA-256 record")
        elif recorded_digest.group(1) != actual_digest:
            errors.append("cover artwork does not match its provenance SHA-256")

    info = command("pdfinfo", str(PDF))
    title = re.search(r"^Title:\s*(.+)$", info, re.MULTILINE)
    pages = re.search(r"^Pages:\s*(\d+)$", info, re.MULTILINE)
    if title is None or title.group(1).strip() != "Robot Reinforcement Learning":
        errors.append("PDF title metadata is missing or incorrect")
    page_count = int(pages.group(1)) if pages else 0
    if page_count < 100:
        errors.append(f"PDF has only {page_count} pages; the full book was not built")
    if PDF.stat().st_size < 500_000:
        errors.append("PDF is unexpectedly small; fonts or chapters may be missing")

    text_path = ROOT / "build" / "robot-rl-self-study.txt"
    command("pdftotext", "-layout", str(PDF), str(text_path))
    extracted = text_path.read_text(encoding="utf-8", errors="replace")
    cover_text = command(
        "pdftotext", "-f", "1", "-l", "1", "-layout", str(PDF), "-"
    )
    for phrase in (
        "Robot Reinforcement Learning",
        "From First Principles to Real Robots",
        "Bruce Lu",
    ):
        if phrase not in cover_text:
            errors.append(f"PDF cover is missing expected text: {phrase!r}")
    required_text = (
        "Reinforcement Learning Foundations",
        "Proximal Policy Optimization (PPO) from Equations to Code",
        "Microduck",
        "JumpRover",
        "FastTD3",
        "Problem 1: observation dimensions",
        "Primary Sources and Open-Source Study Index",
    )
    for phrase in required_text:
        if phrase not in extracted:
            errors.append(f"PDF text is missing expected section: {phrase!r}")
    answer_banners = len(
        re.findall(
            r"^\s*REFERENCE ANSWER · CHECK AFTER ATTEMPTING\s*$",
            extracted,
            re.MULTILINE,
        )
    )
    expected_answer_banners = sum(
        path.read_text(encoding="utf-8").count("<summary>")
        for path in PDF_MARKDOWN
    )
    if answer_banners != expected_answer_banners:
        errors.append(
            f"PDF contains {answer_banners} reference-answer banners; expected "
            f"{expected_answer_banners} from the Markdown sources"
        )

    fonts = command("pdffonts", str(PDF))
    font_rows = [line for line in fonts.splitlines()[2:] if line.strip()]
    if not font_rows:
        errors.append("PDF contains no detectable fonts")
    for row in font_rows:
        # Poppler aligns three yes/no columns after variable-width type and
        # encoding names: embedded, subset, Unicode map.
        if re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", row) is None:
            errors.append(f"PDF font is not embedded: {row.split()[0]}")

    log = LOG.read_text(encoding="utf-8", errors="replace")
    forbidden = {
        r"Overfull \\[hv]box": "layout overflow",
        r"Undefined control sequence": "undefined LaTeX command",
        r"Missing character:": "missing font glyph",
        r"Package fancyhdr Warning: \\headheight is too small": (
            "undersized running-header layout"
        ),
        r"LaTeX Error:": "LaTeX error",
        r"Emergency stop": "emergency TeX stop",
        r"Rerun to get cross-references right": "unresolved cross-reference pass",
    }
    for pattern, label in forbidden.items():
        matches = list(re.finditer(pattern, log))
        if matches:
            locations: list[str] = []
            for match in matches[:5]:
                line_number = log.count("\n", 0, match.start()) + 1
                line = log.splitlines()[line_number - 1].strip()
                locations.append(f"line {line_number}: {line}")
            errors.append(
                f"XeLaTeX log contains {len(matches)} {label} warning(s): "
                + "; ".join(locations)
            )

    if errors:
        return report(errors)
    print(
        f"PDF_CHECK_OK: {page_count} pages, release and cover checksums, cover "
        f"text, embedded fonts, expected sections, semantic text contrast >= "
        f"{minimum_contrast:.2f}:1, and no TeX overflow warnings"
    )
    return 0


def report(errors: list[str]) -> int:
    for error in errors:
        print(f"PDF_CHECK_ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
