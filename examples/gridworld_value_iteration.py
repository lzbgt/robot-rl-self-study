"""Value iteration in a deterministic 4x4 grid with a known model."""

from __future__ import annotations


ROWS = 4
COLS = 4
GOAL = (3, 3)
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DISCOUNT = 0.95
STEP_REWARD = -1.0
TOLERANCE = 1e-10


def transition(state: tuple[int, int], action: str) -> tuple[tuple[int, int], float]:
    """Return the deterministic next state and immediate reward."""
    if state == GOAL:
        return GOAL, 0.0

    delta_row, delta_col = ACTIONS[action]
    row = min(max(state[0] + delta_row, 0), ROWS - 1)
    col = min(max(state[1] + delta_col, 0), COLS - 1)
    return (row, col), STEP_REWARD


def solve() -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], str], int]:
    states = [(row, col) for row in range(ROWS) for col in range(COLS)]
    values = {state: 0.0 for state in states}
    sweeps = 0

    while True:
        sweeps += 1
        new_values = values.copy()
        largest_change = 0.0

        for state in states:
            if state == GOAL:
                continue
            candidates = []
            for action in ACTIONS:
                next_state, reward = transition(state, action)
                candidates.append(reward + DISCOUNT * values[next_state])
            new_values[state] = max(candidates)
            largest_change = max(largest_change, abs(new_values[state] - values[state]))

        values = new_values
        if largest_change < TOLERANCE:
            break

    policy: dict[tuple[int, int], str] = {}
    for state in states:
        if state == GOAL:
            continue
        policy[state] = max(
            ACTIONS,
            key=lambda action: (
                transition(state, action)[1]
                + DISCOUNT * values[transition(state, action)[0]]
            ),
        )
    return values, policy, sweeps


def main() -> None:
    values, policy, sweeps = solve()
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    print(f"converged in {sweeps} full Bellman sweeps")
    print("optimal values:")
    for row in range(ROWS):
        print(" ".join(f"{values[(row, col)]:7.3f}" for col in range(COLS)))

    print("greedy policy:")
    for row in range(ROWS):
        symbols = []
        for col in range(COLS):
            state = (row, col)
            symbols.append("G" if state == GOAL else arrow[policy[state]])
        print(" ".join(symbols))

    assert values[GOAL] == 0.0
    assert policy[(3, 2)] == "R"


if __name__ == "__main__":
    main()

