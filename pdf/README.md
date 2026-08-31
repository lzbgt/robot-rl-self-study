# PDF Build Environment and Procedure

The PDF is a generated edition of the same Markdown that GitHub displays. The
entire build implementation and configuration live in this repository:

- `scripts/build_pdf.py` combines the front matter, 20 chapters, and source
  index; converts fenced `math` blocks for Pandoc; expands folded solutions;
  and runs three XeLaTeX passes;
- `metadata.yaml` defines the book metadata, A4 page geometry, and fonts;
- `latex-header.tex` defines code wrapping, URL breaking, table layout, glyph
  handling, and table-of-contents widths;
- `scripts/check_pdf.py` rejects incomplete books, missing sections,
  unembedded fonts, missing glyphs, unresolved references, and horizontal or
  vertical TeX overflow; and
- `../.github/workflows/book-check.yml` installs a clean Ubuntu environment,
  runs the complete build/check pipeline, and uploads the PDF as an artifact.

The generated release files are `dist/robot-rl-self-study.pdf` and its
`dist/SHA256SUMS` integrity record. Intermediates in `build/` are intentionally
ignored.

## Semantic visual system

The PDF palette is a learning interface, not decoration:

| Style | Semantic role | Redundant non-color cue |
| --- | --- | --- |
| concept navy | chapters and conceptual subheadings | large sans-serif bold type and a chapter rule |
| process teal | section navigation, links, page markers, table rules | heading level, placement, and rules |
| answer amber on pale amber | expanded reference answers | explicit `REFERENCE ANSWER` label and banner |
| blue-gray code panel | code, commands, data, and text diagrams | monospaced type, background, and frame |
| equation navy | display equations | centered mathematical layout and surrounding whitespace |

All text colors are deliberately dark on a white or very pale background. The
validator requires every foreground/background pair to meet the
[WCAG 2.2 normal-text contrast threshold of 4.5:1](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html);
the current minimum is 6.16:1. Meaning does not depend on hue, so the edition
remains usable in grayscale and for common color-vision differences. The
source of truth is
`pdf/latex-header.tex`; solution banners are emitted by
`scripts/build_pdf.py`. When changing either, run the overflow/font
validator and visually inspect a chapter opening, a code-heavy page, a wide
table, an equation page, and a reference-answer page.

## Required commands and fonts

The scripts require Python 3.10 or newer and these commands on `PATH`:

```text
pandoc
xelatex
pdfinfo
pdftotext
pdffonts
```

The last three are supplied by Poppler. XeLaTeX must be able to find these
font-family names exactly:

```text
Noto Serif
Noto Sans
Noto Sans Mono
Noto Sans Math
```

## Arch Linux — tested local environment

```bash
sudo pacman -S --needed \
    python make pandoc-cli \
    texlive-xetex texlive-latexextra texlive-fontsrecommended \
    noto-fonts poppler
```

If package metadata is stale, refresh it before installing:

```bash
sudo pacman -Syy
```

## Ubuntu and Debian-family systems — CI environment

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
    python3 make pandoc \
    texlive-xetex texlive-latex-extra texlive-fonts-recommended \
    lmodern fonts-noto-core poppler-utils
```

This is the setup executed by GitHub Actions on every push and pull request.

## macOS with Homebrew

Install Pandoc, Poppler, a headless MacTeX distribution, and the four fonts:

```bash
brew install pandoc poppler
brew install --cask \
    mactex-no-gui \
    font-noto-serif font-noto-sans font-noto-sans-mono font-noto-sans-math
```

MacTeX may require a new terminal before `xelatex` is visible. Alternatively,
run:

```bash
eval "$(/usr/libexec/path_helper)"
```

The package names above link to the official [Homebrew Pandoc](https://formulae.brew.sh/formula/pandoc),
[MacTeX](https://formulae.brew.sh/cask/mactex-no-gui),
[Poppler](https://formulae.brew.sh/formula/poppler), and
[Noto Sans Math](https://formulae.brew.sh/cask/font-noto-sans-math) records.

## Windows with MiKTeX

1. Install Python 3 and Pandoc, and enable their `PATH` options.
2. Install the [Basic MiKTeX distribution](https://miktex.org/howto/install-miktex).
   Enable automatic installation of missing LaTeX packages, or install the
   packages named in `latex-header.tex` through MiKTeX Console.
3. Install the four Noto font families above at the operating-system level.
4. Install Poppler and place `pdfinfo.exe`, `pdftotext.exe`, and
   `pdffonts.exe` on `PATH`.
5. Open a new PowerShell window and run the version checks below.

GNU Make is optional on Windows. The direct commands are:

```powershell
py -3 scripts/build_pdf.py
py -3 scripts/check_pdf.py
```

The builder depends on the `xelatex` interface, not on a TeX Live-specific
directory layout, so MiKTeX is a supported provider when all packages and fonts
resolve. The automated reference build uses TeX Live; a MiKTeX-built PDF should
pass the same content and overflow validator.

## Verify the environment before building

```bash
python3 --version
pandoc --version
xelatex --version
pdfinfo -v
```

On systems with Fontconfig, verify the chosen fonts too:

```bash
fc-match "Noto Serif"
fc-match "Noto Sans"
fc-match "Noto Sans Mono"
fc-match "Noto Sans Math"
```

Each command should report the requested family, not a fallback.

## Build and validate

From the repository root:

```bash
make pdf
make pdf-check
```

For the Markdown examples, source conventions, and PDF together:

```bash
make check
```

A successful run ends with both `BOOK_CHECK_OK` and `PDF_CHECK_OK`. The PDF
validator reads `build/robot-rl-self-study.log`; do not discard `build/` before
running it.

The builder fixes `SOURCE_DATE_EPOCH` to the edition date before invoking
Pandoc and XeLaTeX. Consequently, identical source and tool versions produce a
stable artifact instead of embedding the wall-clock build time.

## Troubleshooting

- **Font not found:** install all four Noto families, refresh the OS font cache,
  and open a new terminal. Do not silently substitute another font because it
  changes wrapping and can hide overflow.
- **Undefined LaTeX command:** the TeX distribution is incomplete. Install the
  package that provides the command rather than deleting layout protections.
- **Overfull box:** inspect the reported line in
  `build/robot-rl-self-study.tex`. Fix the source table/token or the general
  wrapping rule; do not suppress the warning in the validator.
- **PDF changes after a no-source-change rebuild:** confirm the same Pandoc,
  XeLaTeX, fonts, and Poppler versions are used. `SOURCE_DATE_EPOCH` removes
  clock variance, not rendering differences between tool versions.
- **GitHub math differs from PDF math:** run `python3 scripts/check_book.py`.
  Source display equations must use fenced `math` blocks; the builder performs
  the PDF-only conversion in `build/`.
