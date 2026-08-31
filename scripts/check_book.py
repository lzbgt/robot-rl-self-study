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
    "returns_and_occupancy.py",
    "gaussian_policy_math.py",
    "gae_walkthrough.py",
    "cem_mpc_point_mass.py",
    "pd_joint_and_kalman.py",
    "evaluation_statistics.py",
    "realtime_data_age.py",
    "behavior_cloning_shift.py",
    "camera_latency_and_stopping.py",
    "gridworld_value_iteration.py",
    "tabular_q_learning.py",
    "ppo_clip_demo.py",
)
EXERCISE_HEADING_RE = re.compile(
    r"^#{2,3} .*?(?:Exercises|Check your understanding|Lab:|exercise|"
    r"What to reproduce|Microduck experiment|Capstone)",
    re.IGNORECASE | re.MULTILINE,
)
ABBREVIATION_EXPANSIONS = {
    "AArch64": r"64-bit Arm architecture\s+\(AArch64\)",
    "ACT": r"Action Chunking with Transformers\s+\(ACT\)",
    "AI": r"artificial intelligence\s+\(AI\)",
    "API": r"application programming interfaces?\s+\(APIs?\)",
    "BAM": r"Better Actuator Models\s+\(BAM\)",
    "BC": r"behavior cloning\s+\(BC\)",
    "BLDC": r"brushless direct[- ]current(?: motor)?\s+\(BLDC\)",
    "CAD": r"computer-aided design\s+\(CAD\)",
    "CEM": r"cross-entropy method\s+\(CEM\)",
    "CoM": r"center of mass\s+\(CoM\)",
    "CPU": r"central processing units?\s+\(CPUs?\)",
    "CQL": r"Conservative Q-Learning\s+\(CQL\)",
    "CUDA": r"Compute Unified Device Architecture\s+\(CUDA\)",
    "CVAE": r"conditional variational autoencoder\s+\(CVAE\)",
    "D4RL": r"Datasets for Deep Data-Driven Reinforcement Learning\s+\(D4RL\)",
    "DAgger": r"Dataset Aggregation\s+\(DAgger\)",
    "DQN": r"Deep Q-Network\s+\(DQN\)",
    "DP": r"dynamic programming\s+\(DP\)",
    "DoF": r"degrees? of freedom\s+\(DoFs?\)",
    "DR": r"domain randomization\s+\(DR\)",
    "ELU": r"exponential linear unit\s+\(ELU\)",
    "FOC": r"field-oriented control\s+\(FOC\)",
    "FAST": r"Frequency-space Action Sequence Tokenization\s+\(FAST\)",
    "FPS": r"frames per second\s+\(FPS\)",
    "GAE": r"Generalized Advantage Estimation\s+\(GAE\)",
    "GPU": r"graphics processing units?\s+\(GPUs?\)",
    "GR00T": r"Generalist Robot 00 Technology\s+\(GR00T\)",
    "IK": r"inverse kinematics\s+\(IK\)",
    "IMU": r"inertial measurement unit\s+\(IMU\)",
    "IQL": r"Implicit Q-Learning\s+\(IQL\)",
    "ID": r"identifiers?\s+\(IDs?\)",
    "JSON": r"JavaScript Object Notation\s+\(JSON\)",
    "KL": r"Kullback[–—-]Leibler\s+\(KL\)",
    "LLM": r"large language model\s+\(LLM\)",
    "LM": r"language model\s+\(LM\)",
    "LiDAR": r"light detection and ranging\s+\(LiDAR\)",
    "LoRA": r"Low-Rank Adaptation\s+\(LoRA\)",
    "MC": r"Monte Carlo\s+\(MC\)",
    "MCU": r"microcontroller unit\s+\(MCU\)",
    "MDP": r"Markov Decision Process\s+\(MDP\)",
    "MJCF": (
        r"MuJoCo(?:'s)? (?:Extensible Markup Language \(XML\)|XML) "
        r"model format\s+\(MJCF\)"
    ),
    "MLP": r"multilayer perceptron\s+\(MLP\)",
    "MPC": r"Model Predictive Control\s+\(MPC\)",
    "NaN": r"not-a-number\s+\(NaN\)",
    "ONNX": r"Open Neural Network Exchange\s+\(ONNX\)",
    "PD": r"proportional[–—-]derivative\s+\(PD\)",
    "PID": r"proportional[–—-]integral[–—-]derivative\s+\(PID\)",
    "POMDP": r"Partially Observable Markov Decision Process\s+\(POMDP\)",
    "PPO": r"Proximal Policy Optimization\s+\(PPO\)",
    "PWM": r"pulse-width modulation\s+\(PWM\)",
    "RGB": r"red-green-blue\s+\(RGB\)",
    "RL": r"reinforcement learning\s+\(RL\)",
    "RMA": r"Rapid Motor Adaptation\s+\(RMA\)",
    "ReLU": r"rectified linear unit\s+\(ReLU\)",
    "RT-1": r"Robotics Transformer 1\s+\(RT-1\)",
    "RT-2": r"Robotics Transformer 2\s+\(RT-2\)",
    "RT-X": r"Robotics Transformer X\s+\(RT-X\)",
    "RSL-RL": (
        r"Robotic Systems Lab(?:'s)? reinforcement learning\s+\(RSL-RL\)"
    ),
    "SAC": r"Soft Actor-Critic\s+\(SAC\)",
    "SARSA": r"State[–—-]Action[–—-]Reward[–—-]State[–—-]Action\s+\(SARSA\)",
    "SHA-256": r"Secure Hash Algorithm 256-bit\s+\(SHA-256\)",
    "SLAM": r"simultaneous localization and mapping\s+\(SLAM\)",
    "TD": r"temporal[–—-]difference\s+\(TD\)",
    "TD-MPC2": (
        r"Temporal Difference Learning for Model Predictive Control, "
        r"second generation\s+\(TD-MPC2\)"
    ),
    "TD3": r"Twin Delayed Deep Deterministic Policy Gradient\s+\(TD3\)",
    "UI": r"user interface\s+\(UI\)",
    "VLA": r"vision-language-action\s+\(VLAs?\)",
    "VLM": r"vision-language model\s+\(VLM\)",
    "WCET": r"worst-case execution time\s+\(WCET\)",
    "XML": r"Extensible Markup Language\s+\(XML\)",
    "X11": r"X Window System version 11\s+\(X11\)",
    "YAML": r"YAML Ain't Markup Language\s+\(YAML\)",
}


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

        in_nonmath_fence = False
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "\t" in line:
                errors.append(
                    f"{markdown.name}:{number}: tab character can corrupt "
                    "inline LaTeX; use spaces and $...$ math"
                )
            if "$$" in line:
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
            elif stripped.startswith("```") and math_start is None:
                in_nonmath_fence = not in_nonmath_fence
            elif math_start is None and not in_nonmath_fence:
                if r"\(" in line or r"\)" in line or r"\[" in line or r"\]" in line:
                    errors.append(
                        f"{markdown.name}:{number}: use $...$ inline math or a "
                        "fenced ```math block for GitHub/PDF parity"
                    )
                dollars = len(re.findall(r"(?<!\\)\$", line))
                if dollars % 2:
                    errors.append(
                        f"{markdown.name}:{number}: inline $...$ math must open "
                        "and close on the same source line"
                    )

        if math_start is not None:
            errors.append(
                f"{markdown.name}:{math_start}: unclosed fenced math block"
            )
        if in_nonmath_fence:
            errors.append(f"{markdown.name}: unclosed fenced code block")

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


def check_chapter_depth() -> tuple[list[str], int, int, int]:
    """Enforce a multi-dimensional teaching rubric, not a line-count target."""

    errors: list[str] = []
    prose_counts: list[int] = []
    mechanism_counts: list[int] = []
    chapters = sorted(BOOK_ROOT.glob("[0-9][0-9]_*.md"))
    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        prose = prose_without_code(text)
        prose_words = len(
            re.findall(r"\b[A-Za-z][A-Za-z0-9’'-]*\b", prose)
        )
        math_blocks = len(re.findall(r"^```math$", text, re.MULTILINE))
        code_blocks = len(
            re.findall(r"^```(?:[A-Za-z0-9_+.-]+)$", text, re.MULTILINE)
        ) - math_blocks
        mechanisms = math_blocks + code_blocks
        headings = len(re.findall(r"^##+\s+", text, re.MULTILINE))
        links = len(LINK_RE.findall(text))
        folds = len(re.findall(r"^<details(?:\s[^>]*)?>$", text, re.MULTILINE))

        prose_counts.append(prose_words)
        mechanism_counts.append(mechanisms)
        if prose_words < 2500:
            errors.append(
                f"{chapter.name}: only {prose_words} prose words; deepen the "
                "problem background, derivation, interpretation, or evidence"
            )
        if mechanisms < 8:
            errors.append(
                f"{chapter.name}: only {mechanisms} equation/code mechanisms; "
                "map more theory to calculable or executable artifacts"
            )
        if headings < 12:
            errors.append(
                f"{chapter.name}: only {headings} teaching sections; add a "
                "coherent concept-to-experiment progression"
            )
        if links < 1:
            errors.append(
                f"{chapter.name}: has no source or cross-chapter link"
            )
        if folds < 1:
            errors.append(
                f"{chapter.name}: has no folded chapter-end answer/rubric"
            )

    return (
        errors,
        min(prose_counts, default=0),
        min(mechanism_counts, default=0),
        sum(prose_counts),
    )


def check_example_catalog() -> list[str]:
    """Keep runnable files, the checker manifest, and learner catalog aligned."""

    errors: list[str] = []
    expected = set(EXAMPLES)
    actual = {
        path.name
        for path in (BOOK_ROOT / "examples").glob("*.py")
        if not path.name.startswith("_")
    }
    for name in sorted(expected - actual):
        errors.append(f"example manifest names missing file: {name}")
    for name in sorted(actual - expected):
        errors.append(f"runnable example is not checked: examples/{name}")

    catalog = (BOOK_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    for name in sorted(expected):
        if catalog.count(f"`{name}`") < 1:
            errors.append(f"examples/README.md does not catalog {name}")
    return errors


def prose_without_code(text: str) -> str:
    """Preserve prose line numbers while removing code, math, and link targets."""

    output: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            output.append("")
        elif in_fence:
            output.append("")
        else:
            line = re.sub(r"`[^`\n]+`", "", line)
            line = re.sub(r"\]\([^)]+\)", "]", line)
            line = line.replace("**", "").replace("__", "")
            output.append(line)
    return "\n".join(output)


def check_first_use_expansions() -> list[str]:
    """Require important abbreviations to be expanded in every chapter."""

    errors: list[str] = []
    for chapter in sorted(BOOK_ROOT.glob("[0-9][0-9]_*.md")):
        prose = prose_without_code(chapter.read_text(encoding="utf-8"))
        search_text = prose.replace("\n", " ")
        for short, full_pattern in ABBREVIATION_EXPANSIONS.items():
            plural = "s?" if short not in {"CoM", "DAgger", "LoRA"} else ""
            use_re = re.compile(
                rf"(?<![A-Za-z0-9-]){re.escape(short)}{plural}(?![A-Za-z0-9-])"
            )
            first_use = use_re.search(search_text)
            if first_use is None:
                continue
            window = search_text[
                max(0, first_use.start() - 180) : first_use.end() + 30
            ]
            window = re.sub(r"\s+", " ", window)
            valid = re.search(full_pattern, window, re.IGNORECASE) is not None
            if not valid:
                line = prose.count("\n", 0, first_use.start()) + 1
                errors.append(
                    f"{chapter.name}:{line}: first prose use of {short} must "
                    f"spell out the term as full name ({short})"
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

    convention_errors, math_blocks, solution_folds = check_markdown_conventions()
    depth_errors, min_prose, min_mechanisms, total_prose = check_chapter_depth()
    errors = (
        check_local_links()
        + check_chapter_index()
        + convention_errors
        + depth_errors
        + check_first_use_expansions()
        + check_example_catalog()
    )
    if not args.skip_examples:
        errors += run_examples()
    if errors:
        for error in errors:
            print(f"BOOK_CHECK_ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "BOOK_CHECK_OK: 20 chapters, local links, examples, "
        f"{math_blocks} math blocks, and {solution_folds} solution folds are valid; "
        f"depth rubric: {total_prose:,} prose words total, >= {min_prose:,} per "
        f"chapter, >= {min_mechanisms} equation/code mechanisms per chapter"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
