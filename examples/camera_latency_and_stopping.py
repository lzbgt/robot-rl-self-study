#!/usr/bin/env python3
"""Back-project a depth pixel and compute an uncertainty-aware speed limit.

The example uses only Python's standard library. It makes Chapter 16's
geometry and stopping equations executable; it is not a safety controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


def back_project(
    u_px: float,
    v_px: float,
    depth_m: float,
    camera: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Return camera-frame X, Y, Z for one ideal undistorted depth pixel."""

    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    x_m = depth_m * (u_px - camera.cx_px) / camera.fx_px
    y_m = depth_m * (v_px - camera.cy_px) / camera.fy_px
    return x_m, y_m, depth_m


def stopping_distance(
    speed_mps: float,
    reaction_s: float,
    brake_accel_mps2: float,
    margin_m: float,
) -> float:
    """Distance traveled during delay, constant braking, and margin."""

    if min(speed_mps, reaction_s, brake_accel_mps2, margin_m) < 0.0:
        raise ValueError("inputs must be nonnegative")
    if brake_accel_mps2 == 0.0:
        return float("inf")
    return (
        speed_mps * reaction_s
        + speed_mps**2 / (2.0 * brake_accel_mps2)
        + margin_m
    )


def maximum_speed(
    mean_clearance_m: float,
    clearance_std_m: float,
    standard_deviations: float,
    reaction_s: float,
    brake_accel_mps2: float,
    margin_m: float,
) -> float:
    """Solve the stopping inequality using conservative clearance."""

    conservative_m = mean_clearance_m - standard_deviations * clearance_std_m
    usable_m = conservative_m - margin_m
    if usable_m <= 0.0 or brake_accel_mps2 <= 0.0:
        return 0.0
    return max(
        0.0,
        -brake_accel_mps2 * reaction_s
        + sqrt(
            (brake_accel_mps2 * reaction_s) ** 2
            + 2.0 * brake_accel_mps2 * usable_m
        ),
    )


def main() -> None:
    camera = CameraIntrinsics(400.0, 400.0, 320.0, 240.0)
    point_m = back_project(360.0, 220.0, 1.2, camera)
    expected_m = (0.12, -0.06, 1.2)
    assert all(abs(a - b) < 1e-12 for a, b in zip(point_m, expected_m))

    map_age_s = 0.22
    requested_speed_mps = 0.5
    unobserved_travel_m = requested_speed_mps * map_age_s
    assert abs(unobserved_travel_m - 0.11) < 1e-12

    speed_limit_mps = maximum_speed(
        mean_clearance_m=0.80,
        clearance_std_m=0.10,
        standard_deviations=2.0,
        reaction_s=0.18,
        brake_accel_mps2=0.80,
        margin_m=0.12,
    )
    distance_m = stopping_distance(speed_limit_mps, 0.18, 0.80, 0.12)
    conservative_clearance_m = 0.80 - 2.0 * 0.10
    assert abs(distance_m - conservative_clearance_m) < 1e-12

    print(f"camera point [m]: {point_m}")
    print(f"travel while map ages [m]: {unobserved_travel_m:.3f}")
    print(f"uncertainty-aware speed limit [m/s]: {speed_limit_mps:.3f}")
    print(f"stopping envelope at limit [m]: {distance_m:.3f}")


if __name__ == "__main__":
    main()
