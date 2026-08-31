#!/usr/bin/env python3
"""Connect discounted-return equations to executable bookkeeping.

The example intentionally uses only Python's standard library. It shows the
backward recurrence used by real rollout buffers and estimates how a policy
changes the states represented in its own training data.
"""

from __future__ import annotations

import random
from collections import Counter


def discounted_returns(
    rewards: list[float], gamma: float, bootstrap_value: float = 0.0
) -> list[float]:
    """Return G_t for every t using G_t = r_t + gamma * G_{t+1}."""

    returns = [0.0] * len(rewards)
    future = bootstrap_value
    for step in range(len(rewards) - 1, -1, -1):
        future = rewards[step] + gamma * future
        returns[step] = future
    return returns


def sample_two_state_policy(
    cautious_probability: float, episodes: int, seed: int = 7
) -> Counter[str]:
    """Estimate state occupancy for a tiny balance/fall process."""

    rng = random.Random(seed)
    visits: Counter[str] = Counter()
    for _ in range(episodes):
        state = "balanced"
        for _ in range(20):
            visits[state] += 1
            cautious = rng.random() < cautious_probability
            stay_probability = 0.98 if cautious else 0.70
            if rng.random() > stay_probability:
                state = "fallen"
                visits[state] += 1
                break
    return visits


def main() -> None:
    rewards = [2.0, 1.0, 4.0]
    returns = discounted_returns(rewards, gamma=0.5)
    assert returns == [3.5, 3.0, 4.0]
    print("backward discounted returns:", returns)

    for cautious_probability in (0.1, 0.9):
        visits = sample_two_state_policy(cautious_probability, episodes=5_000)
        total = sum(visits.values())
        fallen_fraction = visits["fallen"] / total
        print(
            f"cautious_probability={cautious_probability:.1f} "
            f"fallen occupancy={fallen_fraction:.3f} visits={total}"
        )


if __name__ == "__main__":
    main()
