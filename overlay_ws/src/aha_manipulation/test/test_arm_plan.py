"""Unit tests for the pure arm planner (no ROS)."""

import pytest

from aha_manipulation.arm_plan import ArmJoint, plan_arm

NAMES = ["j0", "j1", "j2"]


def _joints():
    return [ArmJoint("j0", -1.5, 1.5), ArmJoint("j1", -3.0, 3.0), ArmJoint("j2", -1.0, 1.0)]


def test_single_point_stretches_to_speed_cap():
    plan = plan_arm(_joints(), NAMES, [("j0", 1.0, 0.0)], now=0.0, max_speed=2.0)
    assert plan.segments[0].send_at == pytest.approx(0.5)
    assert plan.end_time == pytest.approx(0.5)
    assert plan.segments[0].targets["j0"] == pytest.approx(1.0)


def test_declared_time_dominates_when_slow_enough():
    plan = plan_arm(_joints(), NAMES, [("j0", 1.0, 5.0)], now=0.0, max_speed=2.0)
    assert plan.segments[0].send_at == pytest.approx(5.0)


def test_speed_cap_dominates_when_time_too_short():
    plan = plan_arm(_joints(), NAMES, [("j0", 1.0, 0.1)], now=0.0, max_speed=2.0)
    assert plan.segments[0].send_at == pytest.approx(0.5)


def test_slowest_joint_travel_stretches():
    plan = plan_arm(_joints(), NAMES, [("j0", 0.5, 1.0), ("j1", 3.0, 1.0)], now=0.0, max_speed=1.0)
    assert plan.segments[0].send_at == pytest.approx(3.0)


def test_unlisted_joints_hold_start_pose():
    plan = plan_arm(
        _joints(), NAMES, [("j0", 0.5, 1.0)], now=0.0, max_speed=1.0, start_pose={"j1": 0.25}
    )
    assert plan.segments[0].targets["j0"] == pytest.approx(0.5)
    assert plan.segments[0].targets["j1"] == pytest.approx(0.25)
    assert plan.segments[0].targets["j2"] == pytest.approx(0.0)


def test_static_keyframe_is_merged_away():
    plan = plan_arm(_joints(), NAMES, [("j0", 0.2, 0.0), ("j0", 0.2, 3.0)], now=0.0, max_speed=1.0)
    assert len(plan.segments) == 1
    assert plan.segments[0].send_at == pytest.approx(0.2)
    assert plan.position_at(3.0)["j0"] == pytest.approx(0.2)


def test_same_time_points_merge_into_one_keyframe():
    plan = plan_arm(_joints(), NAMES, [("j0", 0.3, 0.0), ("j1", 0.4, 0.0)], now=0.0, max_speed=1.0)
    assert len(plan.segments) == 1
    assert plan.segments[0].targets["j1"] == pytest.approx(0.4)
    assert plan.segments[0].targets["j2"] == pytest.approx(0.0)


def test_clamping_is_recorded():
    plan = plan_arm(_joints(), NAMES, [("j2", 2.0, 1.0)], now=0.0, max_speed=1.0)
    assert plan.segments[0].targets["j2"] == pytest.approx(1.0)
    assert plan.clamped == [("j2", 2.0, 1.0)]


def test_position_at_interpolates_and_holds():
    plan = plan_arm(_joints(), NAMES, [("j0", 1.0, 1.0)], now=0.0, max_speed=10.0)
    assert plan.position_at(-1.0)["j0"] == pytest.approx(0.0)
    assert plan.position_at(0.0)["j0"] == pytest.approx(0.0)
    assert plan.position_at(0.5)["j0"] == pytest.approx(0.5)
    assert plan.position_at(1.0)["j0"] == pytest.approx(1.0)
    assert plan.position_at(5.0)["j0"] == pytest.approx(1.0)


def test_dense_sampling_never_exceeds_max_speed():
    points = [
        ("j0", 0.8, 0.2),
        ("j1", -2.5, 0.4),
        ("j2", 0.9, 0.6),
        ("j0", -1.2, 0.8),
    ]
    plan = plan_arm(_joints(), NAMES, points, now=100.0, max_speed=0.5)
    limits = {"j0": (-1.5, 1.5), "j1": (-3.0, 3.0), "j2": (-1.0, 1.0)}
    times = [100.0] + [segment.send_at for segment in plan.segments]
    assert times == sorted(times)
    assert all(a < b for a, b in zip(times, times[1:]))
    t = 100.0
    prev = plan.position_at(t)
    steps = 0
    while t < plan.end_time:
        t += 0.01
        current = plan.position_at(t)
        for name in NAMES:
            low, high = limits[name]
            assert low - 1e-9 <= current[name] <= high + 1e-9
            assert abs(current[name] - prev[name]) / 0.01 <= 0.5 + 1e-6
        prev = current
        steps += 1
    assert steps > 20


def test_non_monotonic_times_normalize():
    plan = plan_arm(_joints(), NAMES, [("j0", 0.5, 2.0), ("j0", -0.5, 0.5)], now=0.0, max_speed=1.0)
    assert plan.end_time == pytest.approx(2.0)
    assert plan.segments[0].send_at == pytest.approx(0.5)
    assert plan.segments[0].targets["j0"] == pytest.approx(-0.5)
    assert plan.segments[-1].targets["j0"] == pytest.approx(0.5)


def test_start_pose_is_clamped():
    plan = plan_arm(
        _joints(), NAMES, [("j0", 0.5, 1.0)], now=0.0, max_speed=1.0, start_pose={"j0": 9.9}
    )
    assert plan.start_pose["j0"] == pytest.approx(1.5)


def test_empty_trajectory_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [])


def test_unknown_joint_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [("j9", 0.0, 0.0)])


def test_duplicate_command_names_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), ["j0", "j0"], [("j0", 0.0, 0.0)])


def test_bad_limits_rejected():
    with pytest.raises(ValueError):
        plan_arm([ArmJoint("j0", 1.0, -1.0)], ["j0"], [("j0", 0.0, 0.0)])


def test_non_finite_value_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [("j0", float("nan"), 0.0)])


def test_negative_time_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [("j0", 0.1, -0.5)])


def test_zero_max_speed_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [("j0", 0.1, 0.0)], max_speed=0.0)


def test_bad_start_pose_name_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(), NAMES, [("j0", 0.1, 0.0)], start_pose={"j9": 1.0})
