"""aha_manipulation.arm_plan: trajectory checks and sampling (no ROS)."""

import math

import pytest

from aha_manipulation.arm_plan import JointLimits, Point, plan_trajectory

NAMES = ["a", "b", "c"]
LIMITS = [JointLimits(n, -1.0, 1.0, 1.0, 2.0) for n in NAMES]
START = {"a": 0.1, "b": -0.2, "c": 0.0}


def plan(points, names=("a",), start=START, now=10.0, tolerance=0.05):
    return plan_trajectory(LIMITS, list(names), points, start, now, tolerance)


def pt(positions, t, velocities=(), accelerations=()):
    return Point(tuple(positions), tuple(velocities), tuple(accelerations), t)


def test_linear_from_the_measured_start_and_others_hold():
    p = plan([pt([0.6], 2.0)])
    assert p.times == [10.0, 12.0]
    assert p.sample(10.0) == START
    assert p.sample(11.0) == pytest.approx({"a": 0.35, "b": -0.2, "c": 0.0})
    assert p.sample(20.0) == p.final == {"a": 0.6, "b": -0.2, "c": 0.0}
    assert p.velocities is None


def test_point_at_time_zero_is_replaced_by_the_start():
    p = plan([pt([0.12], 0.0), pt([0.5], 1.0)])
    assert p.times == [10.0, 11.0]
    assert p.sample(10.0)["a"] == 0.1


def test_point_at_time_zero_far_from_the_start_is_rejected():
    with pytest.raises(ValueError, match="from the current position"):
        plan([pt([0.3], 0.0), pt([0.5], 1.0)])


def test_only_a_start_point_holds_the_start():
    p = plan([pt([0.1], 0.0)])
    assert p.end_time == 10.0 and p.final == START


def test_hermite_with_velocities_starts_and_ends_at_rest():
    # 0.1 -> 0.5 in 1 s with zero end velocities: a smooth S curve.
    p = plan([pt([0.1], 0.0, [0.0]), pt([0.5], 1.0, [0.0])])
    assert p.velocities is not None
    assert p.sample(10.5)["a"] == pytest.approx(0.3)
    assert p.sample(10.1)["a"] - 0.1 < 0.02  # linear would be 0.04


def test_samples_stay_inside_the_position_limits():
    # A Hermite segment toward the limit with a large end velocity bulges past it.
    limits = [JointLimits("a", -1.0, 1.0, 10.0, 100.0)]
    p = plan_trajectory(
        limits,
        ["a"],
        [pt([1.0], 0.5, [5.0]), pt([1.0], 1.0, [0.0])],
        {"a": 0.0},
        0.0,
        0.05,
    )
    samples = [p.sample(t / 100) for t in range(101)]
    assert max(s["a"] for s in samples) == 1.0


def test_positions_outside_the_limits_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        plan([pt([1.2], 5.0)])


def test_too_fast_segment_is_rejected():
    with pytest.raises(ValueError, match="rad/s"):
        plan([pt([0.9], 0.5)])  # 0.8 rad in 0.5 s


def test_too_fast_start_from_rest_is_rejected():
    # From rest, d rad in T s needs 2 d / T^2 (velocity change over half the segment).
    plan([pt([0.9], 1.0)])  # 0.8 rad/s, 1.6 rad/s^2
    with pytest.raises(ValueError, match="rad/s\\^2"):
        plan([pt([0.5], 0.4)])  # 1.0 rad/s, 5 rad/s^2


def test_given_velocity_and_acceleration_over_the_limits_are_rejected():
    with pytest.raises(ValueError, match="velocity"):
        plan([pt([0.5], 1.0, [1.5])])
    with pytest.raises(ValueError, match="acceleration"):
        plan([pt([0.5], 1.0, [0.5], [3.0])])


def test_limits_allow_roundoff():
    # Exactly at the velocity limit plus a relative 1e-6.
    plan([pt([0.1 + 0.5 * (1 + 1e-6)], 0.5, [1.0 + 1e-6])])


@pytest.mark.parametrize(
    "names,points,match",
    [
        ([], [pt([], 1.0)], "no joint names"),
        (["a", "a"], [pt([0.1, 0.1], 1.0)], "duplicate"),
        (["x"], [pt([0.1], 1.0)], "unknown joints"),
        (["a"], [], "no points"),
        (["a"], [pt([0.1, 0.2], 1.0)], "positions for"),
        (["a"], [pt([0.1], 1.0, [0.0, 0.0])], "velocities for"),
        (["a"], [pt([math.nan], 1.0)], "not finite"),
        (["a"], [pt([0.2], 1.0), pt([0.3], 1.0)], "does not increase"),
        (["a"], [pt([0.2], -1.0)], "does not increase"),
    ],
)
def test_malformed_trajectories_are_rejected(names, points, match):
    with pytest.raises(ValueError, match=match):
        plan(points, names=names)
