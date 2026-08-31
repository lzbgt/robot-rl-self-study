"""Print the PPO clipped surrogate for simple probability-ratio cases."""

from __future__ import annotations


CLIP_EPSILON = 0.2


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def clipped_surrogate(ratio: float, advantage: float, epsilon: float) -> float:
    unclipped = ratio * advantage
    clipped = clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
    return min(unclipped, clipped)


def main() -> None:
    ratios = (0.5, 0.8, 1.0, 1.2, 1.5)
    advantages = (2.0, -2.0)

    print(f"clip epsilon={CLIP_EPSILON}")
    print("ratio  advantage  ratio*A  clipped surrogate")
    for advantage in advantages:
        for ratio in ratios:
            objective = clipped_surrogate(ratio, advantage, CLIP_EPSILON)
            print(
                f"{ratio:>5.2f}  {advantage:>9.2f}  "
                f"{ratio * advantage:>7.2f}  {objective:>17.2f}"
            )

    # With positive advantage, increasing probability above 1.2 earns no
    # further surrogate benefit. With negative advantage, reducing probability
    # below 0.8 earns no further surrogate benefit.
    assert clipped_surrogate(1.5, 2.0, 0.2) == 2.4
    assert clipped_surrogate(0.5, -2.0, 0.2) == -1.6
    assert clipped_surrogate(1.5, -2.0, 0.2) == -3.0


if __name__ == "__main__":
    main()
