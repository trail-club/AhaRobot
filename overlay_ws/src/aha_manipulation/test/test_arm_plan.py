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
    assert p.sample(10.0)["a"] == pytest.approx(0.1)
    assert p.sample(11.0)["a"] == 0.5


def test_start_gap_fades_out_over_the_trajectory():
    # MoveIt-like: 0.1 s first segment, the arm 0.01 rad off point 0. Moving
    # the first knot alone would need about 6 * 0.01 / 0.1^2 = 6 rad/s^2 more.
    start = {"a": 0.11, "b": -0.2, "c": 0.0}
    # Trapezoid at 0.4 rad/s^2 from 0.1: 0.002 rad and 0.04 rad/s at 0.1 s.
    points = [
        pt([0.1], 0.0, [0.0]),
        pt([0.102], 0.1, [0.04]),
        pt([0.15], 0.5, [0.2]),
        pt([0.2], 1.0, [0.0]),
    ]
    p = plan(points, start=start)
    assert p.sample(10.0)["a"] == pytest.approx(0.11)
    assert p.sample(11.0)["a"] == pytest.approx(0.2)
    # Half way the gap is half gone.
    shifted = plan(points, start={"a": 0.1, "b": -0.2, "c": 0.0})
    assert p.sample(10.5)["a"] - shifted.sample(10.5)["a"] == pytest.approx(0.005)


def test_point_at_time_zero_far_from_the_start_is_rejected():
    with pytest.raises(ValueError, match="from the current position"):
        plan([pt([0.3], 0.0), pt([0.5], 1.0)])


def test_only_a_start_point_holds_the_start():
    p = plan([pt([0.1], 0.0)])
    assert p.end_time == 10.0 and p.final == START


def test_hermite_with_velocities_starts_and_ends_at_rest():
    # 0.1 -> 0.4 in 1 s with zero end velocities: a smooth S curve.
    p = plan([pt([0.1], 0.0, [0.0]), pt([0.4], 1.0, [0.0])])
    assert p.velocities is not None
    assert p.sample(10.5)["a"] == pytest.approx(0.25)
    assert p.sample(10.1)["a"] - 0.1 < 0.015  # linear would be 0.03


def test_velocity_is_the_derivative_of_the_samples():
    linear = plan([pt([0.6], 2.0)])
    hermite = plan([pt([0.5], 1.0, [0.4]), pt([0.6], 2.0, [0.0])])
    for p in (linear, hermite):
        for t in (10.3, 10.99, 11.5):
            h = 1e-6
            numeric = (p.sample(t + h)["a"] - p.sample(t - h)["a"]) / (2 * h)
            assert p.velocity(t)["a"] == pytest.approx(numeric, abs=1e-6)
            assert p.velocity(t)["b"] == 0.0
        assert p.velocity(9.0)["a"] == p.velocity(12.0)["a"] == 0.0


def test_hermite_peak_velocity_between_the_points_is_rejected():
    # Rest to rest, d rad in T s peaks at 1.5 d / T in the middle: 0.6 rad in
    # 1 s averages 0.6 rad/s but reaches 0.9 rad/s; 0.7 rad in 0.9 s reaches
    # 1.17 rad/s (6 d / T^2 = 5.2 rad/s^2 too, the velocity is checked first).
    limits = [JointLimits("a", -1.0, 1.0, 1.0, 100.0)]
    start = {"a": 0.0}
    plan_trajectory(limits, ["a"], [pt([0.6], 1.0, [0.0])], start, 0.0, 0.05)
    with pytest.raises(ValueError, match="reaches 1.167 rad/s between"):
        plan_trajectory(limits, ["a"], [pt([0.7], 0.9, [0.0])], start, 0.0, 0.05)


def test_hermite_peak_acceleration_is_rejected():
    # The review's case: 0.8 rad in 1 s from rest to rest averages 0.8 rad/s
    # (linear needs only 1.6 rad/s^2) but the cubic starts at 6 d / T^2.
    plan([pt([0.9], 1.0)], start={"a": 0.1, "b": 0.0, "c": 0.0})
    with pytest.raises(ValueError, match="rad/s between"):
        plan([pt([0.9], 1.0, [0.0])], start={"a": 0.1, "b": 0.0, "c": 0.0})
    # Within the velocity limit, only the acceleration: 0.3 rad in 0.6 s
    # reaches 0.75 rad/s and 5 rad/s^2.
    with pytest.raises(ValueError, match="5.000 rad/s\\^2 between"):
        plan([pt([0.4], 0.6, [0.0])])


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
    # Exactly at the limits plus a relative 1e-6.
    over = 1 + 1e-6
    limits = [JointLimits("a", -2.0, 2.0, 1.0, 2.0)]
    # Linear from rest: 1 rad/s mean, 2 rad/s^2 at the knots (2 d / T^2).
    plan_trajectory(limits, ["a"], [pt([1.0 * over], 1.0)], {"a": 0.0}, 0.0, 0.05)
    # Hermite rest to rest: 1.5 d / T = 1 rad/s and 6 d / T^2 = 2 rad/s^2.
    plan_trajectory(
        limits, ["a"], [pt([4 / 3 * over], 2.0, [0.0])], {"a": 0.0}, 0.0, 0.05
    )


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
