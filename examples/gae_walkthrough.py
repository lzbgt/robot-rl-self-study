#!/usr/bin/env python3
"""A transparent Generalized Advantage Estimation backward pass."""

from __future__ import annotations


def generalized_advantage_estimation(
    rewards: list[float],
    values: list[float],
    terminated: list[bool],
    bootstrap_value: float,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Compute advantages and value targets for one rollout fragment."""

    if not (len(rewards) == len(values) == len(terminated)):
        raise ValueError("rollout fields must have equal length")

    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = bootstrap_value

    for step in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if terminated[step] else 1.0
        delta = (
            rewards[step]
            + gamma * continuation * next_value
            - values[step]
        )
        next_advantage = (
            delta + gamma * lam * continuation * next_advantage
        )
        advantages[step] = next_advantage
        next_value = values[step]

    value_targets = [
        value + advantage for value, advantage in zip(values, advantages)
    ]
    return advantages, value_targets


def main() -> None:
    rewards = [1.0, 0.5, -0.2, 2.0]
    values = [0.4, 0.6, 0.7, 0.2]
    terminated = [False, False, False, True]
    advantages, targets = generalized_advantage_estimation(
        rewards,
        values,
        terminated,
        bootstrap_value=99.0,  # ignored because the last transition terminates
        gamma=0.99,
        lam=0.95,
    )

    print("step reward value advantage value_target")
    for step, (reward, value, advantage, target) in enumerate(
        zip(rewards, values, advantages, targets)
    ):
        print(
            f"{step:>4} {reward:>6.2f} {value:>5.2f} "
            f"{advantage:>9.5f} {target:>12.5f}"
        )

    assert abs(advantages[-1] - 1.8) < 1e-12
    assert all(abs((v + a) - t) < 1e-12 for v, a, t in zip(values, advantages, targets))


if __name__ == "__main__":
    main()
