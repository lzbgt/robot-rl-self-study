#!/usr/bin/env python3
"""Map one-joint dynamics, PD control, delay, and a Kalman update to code."""

from __future__ import annotations

import random
from collections import deque


def scalar_kalman_update(
    prior_mean: float,
    prior_variance: float,
    measurement: float,
    measurement_variance: float,
) -> tuple[float, float, float]:
    """Fuse one scalar prior and measurement; return posterior and gain."""

    gain = prior_variance / (prior_variance + measurement_variance)
    posterior_mean = prior_mean + gain * (measurement - prior_mean)
    posterior_variance = (1.0 - gain) * prior_variance
    return posterior_mean, posterior_variance, gain


def simulate_joint(delay_steps: int) -> tuple[float, float]:
    """Control I*q_ddot = torque - viscous_friction with delayed feedback."""

    inertia = 0.02
    viscous_friction = 0.05
    kp, kd = 8.0, 0.8
    torque_limit = 1.5
    dt = 0.002
    target = 0.5
    position = 0.0
    velocity = 0.0
    history = deque([(position, velocity)] * (delay_steps + 1), maxlen=delay_steps + 1)
    largest_position = position

    for _ in range(1_500):
        measured_position, measured_velocity = history[0]
        torque = kp * (target - measured_position) - kd * measured_velocity
        torque = max(-torque_limit, min(torque_limit, torque))

        acceleration = (torque - viscous_friction * velocity) / inertia
        velocity += dt * acceleration
        position += dt * velocity
        history.append((position, velocity))
        largest_position = max(largest_position, position)

    return position, largest_position


def main() -> None:
    rng = random.Random(3)
    true_angle = 0.4
    measurement = true_angle + rng.gauss(0.0, 0.1)
    posterior, posterior_variance, gain = scalar_kalman_update(
        prior_mean=0.0,
        prior_variance=0.25,
        measurement=measurement,
        measurement_variance=0.01,
    )
    print(
        f"measurement={measurement:.3f} kalman_gain={gain:.3f} "
        f"posterior={posterior:.3f} variance={posterior_variance:.4f}"
    )

    for delay_steps in (0, 5, 15):
        final_position, peak = simulate_joint(delay_steps)
        print(
            f"feedback_delay={delay_steps * 2:>2} ms "
            f"final={final_position:.4f} peak={peak:.4f}"
        )

    final_position, _ = simulate_joint(delay_steps=0)
    assert abs(final_position - 0.5) < 1e-3


if __name__ == "__main__":
    main()
