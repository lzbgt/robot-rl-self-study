#!/usr/bin/env python3
"""Model-predictive control with the cross-entropy method on a point mass.

This is deliberately a planner, not a learning algorithm. It makes the
model/plan/recede loop in Chapter 6 executable with only the standard library.
"""

from __future__ import annotations

import math
import random


DT = 0.1
HORIZON = 15
SAMPLES = 500
ELITES = 50
REFINEMENTS = 5
ACTION_LIMIT = 2.0


def dynamics(state: tuple[float, float], action: float) -> tuple[float, float]:
    """Integrate position/velocity for one constant-acceleration step."""

    position, velocity = state
    clipped_action = max(-ACTION_LIMIT, min(ACTION_LIMIT, action))
    next_position = position + DT * velocity + 0.5 * DT * DT * clipped_action
    next_velocity = velocity + DT * clipped_action
    return next_position, next_velocity


def sequence_cost(
    state: tuple[float, float], actions: list[float], target: float
) -> float:
    position, velocity = state
    cost = 0.0
    for action in actions:
        position, velocity = dynamics((position, velocity), action)
        cost += 0.01 * action * action
    return (position - target) ** 2 + 0.2 * velocity * velocity + cost


def plan_action(
    state: tuple[float, float], target: float, rng: random.Random
) -> float:
    """Fit a Gaussian action-sequence distribution to low-cost samples."""

    means = [0.0] * HORIZON
    standard_deviations = [ACTION_LIMIT] * HORIZON

    for _ in range(REFINEMENTS):
        population: list[tuple[float, list[float]]] = []
        for _ in range(SAMPLES):
            actions = [
                max(
                    -ACTION_LIMIT,
                    min(ACTION_LIMIT, rng.gauss(mean, std)),
                )
                for mean, std in zip(means, standard_deviations)
            ]
            population.append((sequence_cost(state, actions, target), actions))

        population.sort(key=lambda item: item[0])
        elite_actions = [actions for _, actions in population[:ELITES]]
        for step in range(HORIZON):
            values = [actions[step] for actions in elite_actions]
            means[step] = sum(values) / ELITES
            variance = sum((value - means[step]) ** 2 for value in values) / ELITES
            standard_deviations[step] = max(0.05, math.sqrt(variance))

    return means[0]


def main() -> None:
    rng = random.Random(4)
    state = (0.0, 0.0)
    target = 1.0

    # Receding horizon: discard the rest of each planned sequence and replan
    # after observing the next real state.
    for step in range(30):
        action = plan_action(state, target, rng)
        state = dynamics(state, action)
        print(
            f"step={step:02d} position={state[0]: .3f} "
            f"velocity={state[1]: .3f} action={action: .3f}"
        )

    assert abs(state[0] - target) < 0.08


if __name__ == "__main__":
    main()
