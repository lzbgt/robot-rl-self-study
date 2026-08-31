"""A dependency-free epsilon-greedy multi-armed bandit.

Read Chapter 3 before changing this file. The learner never sees TRUE_MEANS;
it estimates each arm from sampled rewards.
"""

from __future__ import annotations

import random


TRUE_MEANS = (-0.2, 0.0, 0.3, 0.7)
REWARD_STD = 1.0
STEPS = 5_000
EPSILON = 0.1
SEED = 7


def argmax(values: list[float], rng: random.Random) -> int:
    """Choose randomly among exact ties so arm zero has no hidden privilege."""
    best = max(values)
    candidates = [index for index, value in enumerate(values) if value == best]
    return rng.choice(candidates)


def run_bandit(epsilon: float = EPSILON) -> tuple[list[int], list[float], float]:
    rng = random.Random(SEED)
    estimates = [0.0 for _ in TRUE_MEANS]
    counts = [0 for _ in TRUE_MEANS]
    total_reward = 0.0

    for _ in range(STEPS):
        if rng.random() < epsilon:
            action = rng.randrange(len(TRUE_MEANS))
        else:
            action = argmax(estimates, rng)

        reward = rng.gauss(TRUE_MEANS[action], REWARD_STD)
        counts[action] += 1

        # New mean = old mean + 1/N * prediction error.
        estimates[action] += (reward - estimates[action]) / counts[action]
        total_reward += reward

    return counts, estimates, total_reward / STEPS


def main() -> None:
    counts, estimates, average_reward = run_bandit()

    print(f"seed={SEED} steps={STEPS} epsilon={EPSILON}")
    print("arm  true_mean  selections  estimate")
    for arm, (truth, count, estimate) in enumerate(
        zip(TRUE_MEANS, counts, estimates, strict=True)
    ):
        print(f"{arm:>3}  {truth:>9.2f}  {count:>10}  {estimate:>8.3f}")
    print(f"average reward: {average_reward:.3f}")

    assert sum(counts) == STEPS
    assert counts[3] == max(counts), "the best arm should dominate this seeded run"


if __name__ == "__main__":
    main()

