#!/usr/bin/env python3
"""Real arm_node gripper: hand an object to the robot, or take it back.

  ros2 run aha_test_manipulation gripper_handover.py open
  ros2 run aha_test_manipulation gripper_handover.py close --width 0.03
  ros2 run aha_test_manipulation gripper_handover.py close              # no width: contact detection
  ros2 run aha_test_manipulation gripper_handover.py handover --arm left --open 0.03

Commands astra_controller's arm_node directly (JointCommand on
/<arm>/arm/gripper_joint_command, joint_<r|l>7r, position_cmd per side) instead
of going through place_server.py. arm_node takes each command as a target at
full speed, so a single big step (observed: 0.03 -> 0.0 m onto an object)
browned out the arm board; every ramp here starts from the measured finger
position (``/<arm>/arm/joint_states``) and steps toward the target at --rate
Hz over --ramp s. The ramp sequence, close target and contact detection are
aha_test_manipulation.gripper_ramp (ROS free, unit tested).

``close`` without --width ramps slowly toward fully closed but stops
advancing once the state stops following the command (the fingers are
pushing on the object) and holds --squeeze short of that contact point, so
it never squeezes all the way to 0. ``handover`` opens, waits for Enter
(hand the object in, or take it out), then closes the same way.
"""

import argparse
import sys
import time
from collections import deque

import rclpy
from astra_controller_interfaces.msg import JointCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from aha_test_manipulation import gripper_ramp as ramp

INITIAL_STATE_TIMEOUT = 3.0  # wall s, waiting for the first finger reading
INITIAL_STATE_POLL = 0.05  # wall s


class GripperNode(Node):
    """One arm's finger command/state topics (astra_controller arm_node)."""

    def __init__(self, arm: str):
        super().__init__("gripper_handover")
        self.joint = ramp.finger_joint(arm)
        self.position = None  # latest measured finger position (m, per side)
        self.command_pub = self.create_publisher(
            JointCommand, f"/{arm}/arm/gripper_joint_command", qos_profile_sensor_data
        )
        self.create_subscription(
            JointState,
            f"/{arm}/arm/joint_states",
            self._on_state,
            qos_profile_sensor_data,
        )

    def _on_state(self, msg: JointState):
        if self.joint in msg.name:
            self.position = msg.position[msg.name.index(self.joint)]

    def send(self, value: float):
        msg = JointCommand(name=[self.joint], position_cmd=[float(value)])
        msg.header.stamp = self.get_clock().now().to_msg()
        self.command_pub.publish(msg)

    def check_subscribed(self, timeout: float = INITIAL_STATE_TIMEOUT):
        """Wait up to ``timeout`` s for the arm_node's subscription: commands
        published before discovery finishes are lost."""
        deadline = time.monotonic() + timeout
        while (
            self.command_pub.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=INITIAL_STATE_POLL)
        if self.command_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"nothing subscribes {self.command_pub.topic_name}: is arm_node running?"
            )


def wait_for_state(node: GripperNode, timeout: float = INITIAL_STATE_TIMEOUT):
    """The first measured finger position, or None after ``timeout`` s."""
    deadline = time.monotonic() + timeout
    while node.position is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=INITIAL_STATE_POLL)
    return node.position


def drive(node: GripperNode, sequence, rate: float):
    """Publish ``sequence``'s commands at ``rate`` Hz, feeding back the
    latest measured position each tick; warns once if the state looks frozen
    while commanding (aha_test_manipulation.gripper_ramp.is_stale)."""
    period = 1.0 / rate
    history = deque()
    warned = False
    while not sequence.done:
        command = sequence.next_command(node.position)
        node.send(command)
        # spin_once returns as soon as any one callback is ready, so with a
        # state topic faster than --rate it would return almost instantly and
        # the loop would run far faster than --rate; spin in short slices
        # until the tick's deadline instead, to actually hold the rate (the
        # ramp timing and the staleness window below both depend on it).
        deadline = time.monotonic() + period
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(node, timeout_sec=remaining)
        now = time.monotonic()
        history.append((now, node.position))
        while history and now - history[0][0] > ramp.STALE_WINDOW + 0.5:
            history.popleft()
        if not warned and ramp.is_stale(list(history)):
            node.get_logger().warn(
                f"{node.joint} state looks frozen (identical for over "
                f"{ramp.STALE_WINDOW:.0f} s) while commanding; it may not be live"
            )
            warned = True
    return node.position


def run_open(node: GripperNode, args) -> float:
    start = wait_for_state(node)
    if start is None:
        raise RuntimeError(
            f"no state on {node.get_name()}'s joint_states within "
            f"{INITIAL_STATE_TIMEOUT:.0f} s; is arm_node running and publishing?"
        )
    node.check_subscribed()
    sequence = ramp.SimpleRamp(start, args.open, args.ramp, args.rate)
    return drive(node, sequence, args.rate)


def run_close(node: GripperNode, args) -> tuple[float, float]:
    """(final measured position, estimated object width)."""
    start = wait_for_state(node)
    if start is None:
        raise RuntimeError(
            f"no state on {node.get_name()}'s joint_states within "
            f"{INITIAL_STATE_TIMEOUT:.0f} s; is arm_node running and publishing?"
        )
    node.check_subscribed()
    if args.width is not None:
        # The target already accounts for the squeeze, so the final measured
        # position (once the ramp reaches it) inverts back to --width.
        target = ramp.close_target(args.width, args.squeeze)
        sequence = ramp.SimpleRamp(start, target, args.ramp, args.rate)
        final = drive(node, sequence, args.rate)
        return final, ramp.estimated_width(final, args.squeeze)
    sequence = ramp.ContactRamp(start, 0.0, args.ramp, args.rate, squeeze=args.squeeze)
    final = drive(node, sequence, args.rate)
    # Contact detection holds the command squeeze below the contact point,
    # but the state stays pinned at the object (it cannot close further), so
    # the contact position itself (not the held command) is the half-width;
    # without contact the ramp closed to 0 (empty gripper) and final is that.
    half_width = (
        sequence.contact_position if sequence.contact_position is not None else final
    )
    return final, 2 * half_width


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("action", choices=["open", "close", "handover"])
    parser.add_argument("--arm", choices=["right", "left"], default="right")
    parser.add_argument(
        "--open", type=float, default=0.03, help="opening per side, m (default 0.03)"
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="object width across the fingers, m; omit for contact detection",
    )
    parser.add_argument(
        "--squeeze",
        type=float,
        default=0.003,
        help="per side beyond the contact point, m (default 0.003)",
    )
    parser.add_argument(
        "--ramp", type=float, default=1.0, help="ramp duration, s (default 1.0)"
    )
    parser.add_argument(
        "--rate", type=float, default=20.0, help="command rate, Hz (default 20)"
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    rclpy.init()
    node = GripperNode(args.arm)
    try:
        if args.action == "open":
            final = run_open(node, args)
            print(f"final opening: {final:.4f} m per side")
        elif args.action == "close":
            final, width = run_close(node, args)
            print(
                f"final opening: {final:.4f} m per side; estimated object width: {width:.4f} m"
            )
        else:  # handover
            opened = run_open(node, args)
            print(f"opened to {opened:.4f} m per side")
            input("hand the object in (or take it out), then press Enter to close... ")
            final, width = run_close(node, args)
            print(
                f"final opening: {final:.4f} m per side; estimated object width: {width:.4f} m"
            )
        return 0
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
