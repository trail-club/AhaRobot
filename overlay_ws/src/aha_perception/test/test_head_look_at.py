"""head_look_at.py: map targets are aimed at again as the base turns.

The node runs in-process; transforms go straight into its buffer and the
published trajectories are captured instead of sent.
"""

import importlib.util
import math
from pathlib import Path

from geometry_msgs.msg import PointStamped, TransformStamped
import pytest
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from aha_perception.head_config import look_angles

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "head_look_at.py"
PIVOT_Z = 1.429


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("head_look_at", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def node(module):
    rclpy.init()
    head = module.HeadLookAt()
    head.sent = []
    head.pub = type("Capture", (), {"publish": lambda self, m: head.sent.append(m)})()
    set_tf(head, "base_link", "link_head_tilt", 0.05, 0.0, PIVOT_Z, 0.0)
    set_tf(head, "map", "base_link", 0.0, 0.0, 0.0, 0.0)
    yield head
    head.destroy_node()
    rclpy.shutdown()


def set_tf(head, parent, child, x, y, z, yaw):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y = x, y
    t.transform.translation.z = z
    t.transform.rotation.z, t.transform.rotation.w = (
        math.sin(yaw / 2),
        math.cos(yaw / 2),
    )
    head.tf_buffer.set_transform(t, "test")


def target(frame, x, y, z):
    msg = PointStamped()
    msg.header.frame_id = frame
    msg.point.x, msg.point.y, msg.point.z = x, y, z
    return msg


def pan_tilt(trajectory):
    return tuple(trajectory.points[-1].positions)


def test_look_angles_signs():
    pan, tilt = look_angles(1.0, 1.0, -1.0)
    assert pan == pytest.approx(-math.pi / 4)  # left is negative pan
    assert tilt > 0.0  # below is positive tilt
    assert look_angles(0.0, 0.0, 0.0) is None


def test_map_target_is_aimed_at_again_as_the_base_turns(node):
    node.on_target(target("map", 2.05, 0.0, PIVOT_Z))
    assert pan_tilt(node.sent[-1]) == pytest.approx((0.0, 0.0), abs=1e-6)
    node.track()
    assert len(node.sent) == 1  # unchanged: nothing sent within the deadband
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.05)  # base turned left
    node.track()
    pan, _ = pan_tilt(node.sent[-1])
    assert pan == pytest.approx(0.05, abs=0.01)  # head turns right to keep the point
    step = node.sent[-1].points[-1].time_from_start
    # A small step lasts one tracking period, shorter than a new target's move.
    assert step.sec + step.nanosec * 1e-9 == pytest.approx(node.track_period)
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.5)
    node.track()
    assert pan_tilt(node.sent[-1])[0] == pytest.approx(0.5, abs=0.02)


def test_a_followed_target_moves_in_short_steps(node):
    node.on_target(target("map", 2.05, 0.0, PIVOT_Z))
    first = node.sent[-1].points[-1].time_from_start
    assert first.sec + first.nanosec * 1e-9 >= node.config.min_duration
    node.current = (0.0, 0.0)
    node.on_target(target("map", 2.05, -0.1, PIVOT_Z))  # moved 0.1 m right
    step = node.sent[-1].points[-1].time_from_start
    assert step.sec + step.nanosec * 1e-9 == pytest.approx(node.track_period)


def test_a_small_re_aim_during_a_long_move_keeps_its_pace(node):
    node.current = (1.5, 0.0)  # far right, as after a turn
    node.on_target(target("map", 2.05, 0.0, PIVOT_Z))
    node.moving = True
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.05)  # base still settling
    node.current = (1.2, 0.0)  # on the way
    node.track()
    step = node.sent[-1].points[-1].time_from_start
    # 1.15 rad at 1 rad/s, not one tracking period.
    assert step.sec + step.nanosec * 1e-9 == pytest.approx(1.15, abs=0.02)


def test_a_head_stopped_off_target_is_sent_again(node):
    node.current = (0.0, 0.0)
    node.on_target(target("map", 2.05, 0.0, PIVOT_Z))
    node.sent_at -= 10.0
    node.current, node.moving = (-1.57, 0.0), False  # overshot to the limit
    node.track()
    assert len(node.sent) == 2
    assert pan_tilt(node.sent[-1]) == pytest.approx((0.0, 0.0), abs=1e-6)
    node.current = (0.0, 0.0)
    node.track()
    assert len(node.sent) == 2


def test_a_target_crossing_behind_keeps_the_head_on_its_side(node):
    node.on_target(target("map", -2.0, -0.3, PIVOT_Z))  # behind, to the right
    assert pan_tilt(node.sent[-1])[0] == pytest.approx(node.config.pan_max)
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.3)  # now just behind-left
    node.track()
    assert pan_tilt(node.sent[-1])[0] == pytest.approx(node.config.pan_max)
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 1.5)  # clearly on the left
    node.track()
    assert pan_tilt(node.sent[-1])[0] == pytest.approx(node.config.pan_min)


def test_base_frame_target_is_not_tracked(node):
    node.on_target(target("base_link", 2.0, 0.5, PIVOT_Z))
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.5)
    node.track()
    assert len(node.sent) == 1


def test_another_nodes_head_command_stops_tracking(node):
    node.on_target(target("map", 2.05, 0.0, PIVOT_Z))
    node.on_command(node.sent[-1])  # our own command comes back: still tracking
    assert node.target is not None
    other = JointTrajectory(
        joint_names=["joint_head_pan", "joint_head_tilt"],
        points=[JointTrajectoryPoint(positions=[0.3, 0.1])],
    )
    node.on_command(other)
    assert node.target is None
    set_tf(node, "map", "base_link", 0.0, 0.0, 0.0, 0.5)
    node.track()
    assert len(node.sent) == 1
