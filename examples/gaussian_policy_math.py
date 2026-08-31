#!/usr/bin/env python3
"""Check the scalar Gaussian equations used by continuous-action policies."""

from __future__ import annotations

import math


def gaussian_log_probability(action: float, mean: float, std: float) -> float:
    """Log N(action; mean, std^2), written to mirror the book equation."""

    variance = std * std
    return -0.5 * (
        ((action - mean) ** 2) / variance
        + 2.0 * math.log(std)
        + math.log(2.0 * math.pi)
    )


def score_with_respect_to_mean(action: float, mean: float, std: float) -> float:
    """Derivative of log probability with respect to the Gaussian mean."""

    return (action - mean) / (std * std)


def main() -> None:
    action, mean, std = 0.3, 0.1, 0.2
    analytic = score_with_respect_to_mean(action, mean, std)

    epsilon = 1e-6
    numeric = (
        gaussian_log_probability(action, mean + epsilon, std)
        - gaussian_log_probability(action, mean - epsilon, std)
    ) / (2.0 * epsilon)

    assert math.isclose(analytic, numeric, rel_tol=1e-8, abs_tol=1e-8)
    print(f"log probability: {gaussian_log_probability(action, mean, std):.6f}")
    print(f"analytic score: {analytic:.6f}")
    print(f"finite-difference score: {numeric:.6f}")

    entropy = 0.5 * math.log(2.0 * math.pi * math.e * std * std)
    print(f"Gaussian entropy: {entropy:.6f}")


if __name__ == "__main__":
    main()
