"""Controller transport regressions without launching ROS processes."""

from pathlib import Path
import json
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


def scoring_environment():
    received = []
    return SimpleNamespace(
        world=None,
        previous_object=None,
        previous_robot=None,
        mark=received.append,
        received=received,
    )


def sample(t, contacts, x):
    from std_msgs.msg import String

    return String(
        data=json.dumps(
            {
                "schema_version": 1,
                "frame_id": "world",
                "stamp": {"sec": int(t), "nanosec": round((t - int(t)) * 1e9)},
                "contact_window_start": {
                    "sec": int(t - 0.05),
                    "nanosec": round(((t - 0.05) % 1) * 1e9),
                },
                "robot_pose": [x, 0, 0],
                "object_position": [x, 0, 0.3],
                "finger_contacts": contacts,
            }
        )
    )


def test_scoring_state_keeps_contacts_and_poses_from_same_sample():
    node = scoring_environment()
    RosEnvironment.on_state(node, sample(1, 2, 0.5))
    RosEnvironment.on_state(node, sample(1.05, 0, 0.55))
    assert node.world.sim_time == pytest.approx(1.05)
    assert node.world.finger_contacts == 0
    assert node.world.object_position == (0.55, 0, 0.3)
    assert node.world.object_speed == pytest.approx(1)
    assert node.world.robot_speed == pytest.approx(1)
    RosEnvironment.on_state(node, sample(1.02, 2, 100))
    RosEnvironment.on_state(node, sample(1.05, 1, 100))
    assert node.world.finger_contacts == 0
    assert len(node.received) == 2
    RosEnvironment.on_state(node, sample(1.1, 1, 0.6))
    assert node.world.finger_contacts == 1
    assert node.world.object_speed == pytest.approx(1)


@pytest.mark.parametrize(
    "change",
    [
        {"finger_contacts": 3},
        {"finger_contacts": True},
        {"object_position": [0, 0, float("nan")]},
        {"robot_pose": [0, 0]},
        {"schema_version": 2},
        {"frame_id": "odom"},
    ],
)
def test_bad_scoring_sample_does_not_change_last_state(change):
    from std_msgs.msg import String

    node = scoring_environment()
    RosEnvironment.on_state(node, sample(1, 2, 0.5))
    previous = node.world
    data = json.loads(sample(1.05, 0, 0.55).data)
    data.update(change)
    with pytest.raises(ValueError):
        RosEnvironment.on_state(node, String(data=json.dumps(data)))
    assert node.world is previous
    assert node.previous_object == (1, (0.5, 0, 0.3))
