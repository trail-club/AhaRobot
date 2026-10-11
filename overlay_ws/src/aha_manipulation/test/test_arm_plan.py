"""Unit tests for the pure arm planner (no ROS)."""

import pytest

from aha_manipulation.arm_map import ArmJoint
from aha_manipulation.arm_plan import plan_arm


def _joints(n=5, lim=1.57):
    return [ArmJoint(f"j{i}", -lim, lim) for i in range(n)]


def test_first_point_is_issued_at_the_speed_minimum():
    plan = plan_arm(_joints(1, 1.57), ["j0"], [("j0", 1.0, 0.0)], now=0.0, max_speed=100.0)
    assert plan.segments[0].targets == {"j0": 1.0}
    assert plan.segments[0].send_at == pytest.approx(0.01, abs=1e-6)  # 1 rad / 100 rad/s


def test_speed_limited_stretch():
    # 1 rad of travel at 0.5 rad/s takes 2 s; the declared 1 s is shorter, so stretch wins
    plan = plan_arm(_joints(1, 1.57), ["j0"], [("j0", 0.0, 0.0), ("j0", 1.0, 1.0)], now=0.0, max_speed=0.5)
    assert plan.segments[0].targets == {"j0": 0.0}
    assert plan.segments[1].targets == {"j0": 1.0}
    assert plan.segments[1].send_at == pytest.approx(2.0, abs=1e-6)


def test_declared_time_dominates():
    # 0.5 rad at 2.0 rad/s is 0.25 s; the declared 5 s is slower, so declared wins
    plan = plan_arm(_joints(1, 1.57), ["j0"], [("j0", 0.0, 0.0), ("j0", 0.5, 5.0)], now=0.0, max_speed=2.0)
    assert plan.segments[1].send_at == pytest.approx(5.0, abs=1e-6)


def test_clamped_to_limits():
    plan = plan_arm(_joints(1, 1.0), ["j0"], [("j0", 0.0, 0.0), ("j0", 5.0, 5.0)], now=0.0, max_speed=0.5)
    assert plan.segments[-1].targets == {"j0": 1.0}


def test_static_points_hold_at_declared_time():
    plan = plan_arm(
        _joints(1, 1.57), ["j0"], [("j0", 0.2, 0.0), ("j0", 0.2, 3.0), ("j0", 0.3, 6.0)], now=0.0, max_speed=1.0
    )
    assert [s.targets for s in plan.segments] == [{"j0": 0.2}, {"j0": 0.2}, {"j0": 0.3}]
    assert [s.send_at for s in plan.segments] == pytest.approx([0.0, 3.0, 6.0])


def test_unlisted_joint_holds_its_start():
    plan = plan_arm(
        _joints(2, 1.57), ["j0", "j1"], [("j0", 1.0, 0.0), ("j0", 0.0, 2.0)], now=0.0, max_speed=1.0
    )
    assert all(s.targets.get("j1", 0.0) == 0.0 for s in plan.segments)
    assert [s.targets["j0"] for s in plan.segments] == [1.0, 0.0]


def test_empty_points():
    plan = plan_arm(_joints(2), ["j0", "j1"], [], now=0.0)
    assert plan.segments == []
    assert plan.end_time == 0.0


def test_end_time_is_final_send_time():
    plan = plan_arm(_joints(1, 1.57), ["j0"], [("j0", 0.0, 0.0), ("j0", 1.0, 1.0)], now=0.0, max_speed=0.5)
    assert plan.end_time == pytest.approx(plan.segments[-1].send_at)


def test_unknown_joint_rejected():
    with pytest.raises(ValueError):
        plan_arm(_joints(1), ["j0"], [("j9", 0.0, 0.0)])


def test_bad_limit_rejected():
    with pytest.raises(ValueError):
        plan_arm([ArmJoint("j0", 1.0, -1.0)], ["j0"], [("j0", 0.0, 0.0)])
