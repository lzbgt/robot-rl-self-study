"""Off-policy Q-learning in a small cliff grid.

The agent starts at S. Stepping on C gives -100 and resets to S. Every other
non-goal step gives -1. The learned greedy policy should go around the cliff.
"""

from __future__ import annotations

import random


ROWS = 4
COLS = 6
START = (3, 0)
GOAL = (3, 5)
CLIFF = {(3, col) for col in range(1, 5)}
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ALPHA = 0.4
GAMMA = 0.95
EPISODES = 1_000
SEED = 11


def step(state: tuple[int, int], action: str) -> tuple[tuple[int, int], float, bool, bool]:
    delta_row, delta_col = ACTIONS[action]
    row = min(max(state[0] + delta_row, 0), ROWS - 1)
    col = min(max(state[1] + delta_col, 0), COLS - 1)
    next_state = (row, col)

    if next_state in CLIFF:
        return START, -100.0, False, True
    if next_state == GOAL:
        return GOAL, 0.0, True, False
    return next_state, -1.0, False, False


def choose_action(
    state: tuple[int, int],
    q: dict[tuple[tuple[int, int], str], float],
    epsilon: float,
    rng: random.Random,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(list(ACTIONS))
    values = {action: q[(state, action)] for action in ACTIONS}
    best = max(values.values())
    return rng.choice([action for action, value in values.items() if value == best])


def train() -> tuple[dict[tuple[tuple[int, int], str], float], list[float], int]:
    rng = random.Random(SEED)
    states = [(row, col) for row in range(ROWS) for col in range(COLS)]
    q = {(state, action): 0.0 for state in states for action in ACTIONS}
    returns: list[float] = []
    cliff_entries = 0

    for episode in range(EPISODES):
        state = START
        episode_return = 0.0
        epsilon = max(0.02, 0.3 * (1.0 - episode / EPISODES))

        for _ in range(200):
            action = choose_action(state, q, epsilon, rng)
            next_state, reward, done, hit_cliff = step(state, action)
            cliff_entries += int(hit_cliff)

            next_best = 0.0 if done else max(q[(next_state, a)] for a in ACTIONS)
            td_error = reward + GAMMA * next_best - q[(state, action)]
            q[(state, action)] += ALPHA * td_error

            episode_return += reward
            state = next_state
            if done:
                break
        returns.append(episode_return)

    return q, returns, cliff_entries


def greedy_action(
    state: tuple[int, int], q: dict[tuple[tuple[int, int], str], float]
) -> str:
    return max(ACTIONS, key=lambda action: q[(state, action)])


def main() -> None:
    q, returns, cliff_entries = train()
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    print(f"seed={SEED} episodes={EPISODES} cliff entries during training={cliff_entries}")
    print(f"mean return, first 100 episodes: {sum(returns[:100]) / 100:.2f}")
    print(f"mean return, last 100 episodes:  {sum(returns[-100:]) / 100:.2f}")
    print("greedy policy after training:")

    for row in range(ROWS):
        symbols = []
        for col in range(COLS):
            state = (row, col)
            if state == START:
                symbols.append("S")
            elif state == GOAL:
                symbols.append("G")
            elif state in CLIFF:
                symbols.append("C")
            else:
                symbols.append(arrow[greedy_action(state, q)])
        print(" ".join(symbols))

    assert sum(returns[-100:]) / 100 > sum(returns[:100]) / 100
    assert greedy_action(START, q) == "U"


if __name__ == "__main__":
    main()

