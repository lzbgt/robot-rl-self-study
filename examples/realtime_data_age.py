#!/usr/bin/env python3
"""Demonstrate how queue policy changes feedback-data age.

A sensor captures at 200 Hz and packets arrive 2 ms later. A 50 Hz controller
needs only the freshest state. One controller release is skipped to represent a
temporary slowdown. Times are deterministic integer milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    capture_ms: int
    arrival_ms: int


def simulate(strategy: str) -> list[int]:
    if strategy not in {"fifo", "latest"}:
        raise ValueError("strategy must be 'fifo' or 'latest'")

    samples = [Sample(t, t + 2) for t in range(0, 141, 5)]
    controller_releases = [20, 40, 60, 100, 120, 140]  # 80 ms release missed
    unread: list[Sample] = []
    next_sample = 0
    ages: list[int] = []

    for release_ms in controller_releases:
        while (
            next_sample < len(samples)
            and samples[next_sample].arrival_ms <= release_ms
        ):
            unread.append(samples[next_sample])
            next_sample += 1

        if not unread:
            raise RuntimeError("no sensor sample available at controller release")

        if strategy == "fifo":
            selected = unread.pop(0)
        else:
            selected = unread[-1]
            unread.clear()  # overwritten stale samples are counted in real code

        ages.append(release_ms - selected.capture_ms)

    return ages


def main() -> None:
    fifo_ages = simulate("fifo")
    latest_ages = simulate("latest")

    print("controller releases (ms):", [20, 40, 60, 100, 120, 140])
    print("FIFO sample ages (ms):   ", fifo_ages)
    print("latest sample ages (ms): ", latest_ages)
    print("FIFO maximum age (ms):   ", max(fifo_ages))
    print("latest maximum age (ms): ", max(latest_ages))

    assert fifo_ages == [20, 35, 50, 85, 100, 115]
    assert latest_ages == [5, 5, 5, 5, 5, 5]
    assert max(fifo_ages) > 20 * max(latest_ages)


if __name__ == "__main__":
    main()
