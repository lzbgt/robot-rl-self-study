#!/usr/bin/env python3
"""Small, dependency-free statistics for reinforcement-learning evaluation.

The scores below represent independently trained seeds. Repeated episodes from
one checkpoint should not be placed in this list as if they were new training
seeds; use a clustered or hierarchical analysis for nested data.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from statistics import fmean, median


Statistic = Callable[[Sequence[float]], float]


def interquartile_mean(values: Sequence[float]) -> float:
    """Average the middle 50%, with fractional weights at trim boundaries."""

    if not values:
        raise ValueError("IQM requires at least one value")
    ordered = sorted(values)
    lower = 0.25 * len(ordered)
    upper = 0.75 * len(ordered)
    weighted_sum = 0.0

    # Observation i occupies one unit interval [i, i + 1] in the empirical
    # distribution. Keep only its overlap with the middle-half interval.
    for index, value in enumerate(ordered):
        overlap = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        weighted_sum += overlap * value

    return weighted_sum / (upper - lower)


def bootstrap_interval(
    values: Sequence[float],
    statistic: Statistic,
    *,
    repetitions: int = 10_000,
    seed: int = 7,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over independent units."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [rng.choice(values) for _ in values]
        estimates.append(statistic(sample))
    estimates.sort()

    tail = (1.0 - confidence) / 2.0
    low_index = max(0, math.floor(tail * repetitions))
    high_index = min(repetitions - 1, math.ceil((1.0 - tail) * repetitions) - 1)
    return estimates[low_index], estimates[high_index]


def paired_probability_of_improvement(
    baseline: Sequence[float], treatment: Sequence[float]
) -> float:
    """Fraction of paired seeds on which treatment strictly improves."""

    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired score lists must have equal nonzero length")
    improvements = sum(new > old for old, new in zip(baseline, treatment))
    return improvements / len(baseline)


def paired_mean_difference_interval(
    baseline: Sequence[float], treatment: Sequence[float]
) -> tuple[float, float]:
    """Bootstrap the mean of treatment-minus-baseline paired differences."""

    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired score lists must have equal nonzero length")
    differences = [new - old for old, new in zip(baseline, treatment)]
    return bootstrap_interval(differences, fmean)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for one binomial success probability."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
    )
    return center - half_width, center + half_width


def main() -> None:
    baseline = [0.42, 0.55, 0.61, 0.67, 0.71, 0.76, 0.80, 0.95]
    treatment = [0.50, 0.60, 0.58, 0.74, 0.78, 0.81, 0.88, 0.93]

    print("independent training seeds:", len(baseline))
    print("baseline mean:  ", round(fmean(baseline), 3))
    print("baseline median:", round(median(baseline), 3))
    print("baseline IQM:   ", round(interquartile_mean(baseline), 3))
    print(
        "baseline mean 95% bootstrap interval:",
        tuple(round(value, 3) for value in bootstrap_interval(baseline, fmean)),
    )
    print(
        "paired probability treatment > baseline:",
        round(paired_probability_of_improvement(baseline, treatment), 3),
    )
    print(
        "paired mean-difference 95% bootstrap interval:",
        tuple(
            round(value, 3)
            for value in paired_mean_difference_interval(baseline, treatment)
        ),
    )
    print(
        "18/20 success 95% Wilson interval:",
        tuple(round(value, 3) for value in wilson_interval(18, 20)),
    )

    assert abs(interquartile_mean([1.0, 2.0, 3.0, 4.0]) - 2.5) < 1e-12
    low, high = wilson_interval(18, 20)
    assert 0.69 < low < 0.71 and 0.96 < high < 0.98
    assert paired_probability_of_improvement(baseline, treatment) == 0.75


if __name__ == "__main__":
    main()
