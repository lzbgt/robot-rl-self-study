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
    required_text = (
        "Reinforcement Learning Foundations",
        "PPO from Equations to Code",
        "Microduck",
        "Problem 1: observation dimensions",
        "Primary Sources and Open-Source Study Index",
    )
    for phrase in required_text:
        if phrase not in extracted:
            errors.append(f"PDF text is missing expected section: {phrase!r}")

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
        r"LaTeX Error:": "LaTeX error",
        r"Emergency stop": "emergency TeX stop",
        r"Rerun to get cross-references right": "unresolved cross-reference pass",
    }
    for pattern, label in forbidden.items():
        matches = re.findall(pattern, log)
        if matches:
            errors.append(f"XeLaTeX log contains {len(matches)} {label} warning(s)")

    if errors:
        return report(errors)
    print(
        f"PDF_CHECK_OK: {page_count} pages, checksum, embedded fonts, expected "
        "sections, and no TeX overflow warnings"
    )
    return 0


def report(errors: list[str]) -> int:
    for error in errors:
        print(f"PDF_CHECK_ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
