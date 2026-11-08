"""gripper_handover.py keeps only states stamped after forget_state (the arm
may be moved by hand while the script waits for Enter). No spin."""

import importlib.util
import sys
from pathlib import Path

import pytest
import rclpy
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gripper_handover.py"
ARM = ["joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6"]


@pytest.fixture(scope="module")
def handover():
    spec = importlib.util.spec_from_file_location("gripper_handover", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node(handover):
    rclpy.init()
    node = handover.GripperNode("right")
    yield node
    node.destroy_node()
    rclpy.shutdown()


def state(stamp_ns, value):
    msg = JointState(name=ARM + ["joint_r7r"], position=[value] * 5 + [0.02])
    msg.header.stamp = Time(
        sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
    )
    return msg


def test_states_stamped_before_forget_are_ignored(node):
    now = node.get_clock().now().nanoseconds
    node._on_state(state(now - 10_000_000, 0.1))
    assert node.arm["joint_r2"] == pytest.approx(0.1)
    node.forget_state()
    assert node.arm == {} and node.position is None
    # Queued while waiting for Enter: older than the forget.
    node._on_state(state(now - 5_000_000, 0.2))
    assert node.arm == {} and node.position is None
    node._on_state(state(node.get_clock().now().nanoseconds + 1_000_000, 0.3))
    assert node.arm["joint_r2"] == pytest.approx(0.3)
    assert node.position == pytest.approx(0.02)


def test_unstamped_states_are_kept(node):
    node.forget_state()
    node._on_state(state(0, 0.4))
    assert node.arm["joint_r2"] == pytest.approx(0.4)
