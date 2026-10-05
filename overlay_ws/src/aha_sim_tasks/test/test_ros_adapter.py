"""Controller transport regressions without launching ROS processes."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from builtin_interfaces.msg import Time  # noqa: E402
from aha_sim_tasks.api import Action, CONTROLLER_JOINTS, GRIPPER_JOINTS, JOINT_LIMITS  # noqa: E402
from aha_sim_tasks.ros_environment import RosEnvironment  # noqa: E402


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def environment():
    return SimpleNamespace(
        positions={name: 0.0 for name in JOINT_LIMITS},
        last_targets={},
        last_trajectory_times={},
        base=Publisher(),
        trajectories={name: Publisher() for name in CONTROLLER_JOINTS},
        grippers={name: Publisher() for name in GRIPPER_JOINTS},
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))
        ),
    )


def test_partial_lift_command_clamps_numerical_drift_in_held_joint():
    node = environment()
    node.positions["joint_l1"] = -1e-14
    RosEnvironment.apply(node, Action(joint_positions={"joint_r1": 0.18}))
    trajectory = node.trajectories["lift_controller"].messages[-1]
    assert trajectory.joint_names == ["joint_l1", "joint_r1"]
    assert list(trajectory.points[0].positions) == [0.0, 0.18]
    assert trajectory.points[0].time_from_start.sec == 2


def test_omitted_joint_preserves_previous_command():
    node = environment()
    RosEnvironment.apply(
        node, Action(joint_positions={"joint_l1": 0.1, "joint_r1": 0.2})
    )
    node.positions["joint_l1"] = 0.05
    RosEnvironment.apply(node, Action(joint_positions={"joint_r1": 0.3}))
    assert list(
        node.trajectories["lift_controller"].messages[-1].points[0].positions
    ) == [
        0.1,
        0.3,
    ]


def test_invalid_action_publishes_nothing():
    node = environment()
    with pytest.raises(ValueError):
        RosEnvironment.apply(node, Action(joint_positions={"joint_r1": float("nan")}))
    assert node.base.messages == []
    assert all(not publisher.messages for publisher in node.trajectories.values())


def test_unreached_target_is_reissued_after_trajectory_duration():
    node = environment()
    action = Action(joint_positions={"joint_r1": 0.18})
    RosEnvironment.apply(node, action)
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 1
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=14))
    )
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 2
    node.positions["joint_r1"] = 0.18
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=18))
    )
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 2
