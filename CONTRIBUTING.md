# Contributing

Thank you for helping make robot reinforcement learning easier to learn and
harder to misrepresent.

## What a strong contribution contains

A theory contribution should define new terminology inline, explain each new
equation in words, give a small numerical or code example, and connect the idea
to an experimental decision.

A research contribution should use the primary paper and official project
artifact. State the exact task, embodiment, information available at training
and deployment, data/compute setting, evaluation protocol, and limitation. Do
not use “state of the art” without its dated benchmark and comparison scope.

A code contribution should be small enough for a beginner to trace, identify
its dependencies, use a fixed seed where randomness matters, and contain an
assertion or reported quantity that makes failure visible.

A robot-safety contribution must preserve the separation among model output,
local planning, bounded control, realtime safety, and physical emergency stop.
Simulation performance alone is not hardware-release evidence.

## Pull-request checklist

- [ ] New jargon is explained at first use and, when broadly useful, added to
      Chapter 20.
- [ ] Equations define every symbol, unit, and tensor shape used.
- [ ] Claims link to primary sources or official repositories.
- [ ] Evidence boundaries say what a paper or experiment does *not* establish.
- [ ] Commands and code were run in a fresh or documented environment.
- [ ] Local links and examples pass with `python scripts/check_book.py`.
- [ ] No credential, private dataset, copyrighted paper copy, or robot secret
      is committed.

## Source and citation policy

Prefer peer-reviewed proceedings, publisher/OpenReview records, or the
authors' arXiv record. Link code only when it is official or explicitly label
its relationship to the authors. A newer preprint can be valuable, but mark it
as a preprint and avoid converting a paper's scoped result into a universal
ranking.

This book paraphrases papers and links to their records. Do not paste long
passages, figures, or tables without permission compatible with this
repository's Apache-2.0 license.
