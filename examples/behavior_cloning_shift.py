#!/usr/bin/env python3
"""Two toy failure mechanisms for one-step behavior cloning.

This is arithmetic, not a robot benchmark: squared loss averages action modes,
and a small uncompensated closed-loop drift accumulates trajectory cost.
"""

from __future__ import annotations

from statistics import fmean


def squared_loss(prediction: float, labels: list[float]) -> float:
    return fmean((prediction - label) ** 2 for label in labels)


def uncompensated_drift(
    error_per_step_m: float, horizon: int
) -> tuple[float, float]:
    position_error = 0.0
    cumulative_absolute_error = 0.0
    for _ in range(horizon):
        position_error += error_per_step_m
        cumulative_absolute_error += abs(position_error)
    return position_error, cumulative_absolute_error


def main() -> None:
    left_or_right = [-1.0, 1.0]
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    losses = {x: squared_loss(x, left_or_right) for x in candidates}
    best = min(losses, key=losses.get)

    final_error, trajectory_cost = uncompensated_drift(0.02, horizon=25)

    print("two valid demonstrated actions:", left_or_right)
    print("squared losses:", losses)
    print("minimum-loss deterministic action:", best)
    print("final drift after 25 steps (m):", round(final_error, 3))
    print("sum of absolute drift over trajectory (m-step):", round(trajectory_cost, 3))

    assert best == 0.0  # the mean was never a demonstrated action
    assert abs(final_error - 0.5) < 1e-12
    # 0.02 * (1 + ... + 25) = 0.02 * 325 = 6.5
    assert abs(trajectory_cost - 6.5) < 1e-12


if __name__ == "__main__":
    main()
