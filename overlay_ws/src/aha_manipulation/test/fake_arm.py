#!/usr/bin/env python3
"""Minimal arm stand-in for aha_manipulation tests.

Mirrors the upstream ``astra_controller`` ``arm_node`` command interface: it
accepts a ``JointCommand`` on ``command_topic`` and re-emits the commanded
positions as a ``JointState`` on ``state_topic``. Unlike the real arm (which
ramps on its own firmware) it reaches each goal instantaneously, so tests can
assert the paced bridge drives it to the intended pose.

Topics are relative, so running the node in a namespace (or leaving it in the
root) with matching ``command_topic`` / ``state_topic`` params is all the test
needs.
"""

import argparse
import threading

import rclpy
from astra_controller_interfaces.msg import JointCommand
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState


class FakeArm(Node):
    def __init__(self, *, name="fake_arm", context=None, params_file=None) -> None:
        overrides = (
            [Parameter("parameters_file", Parameter.Type.STRING, params_file)]
            if params_file
            else []
        )
        super().__init__(name, namespace="", context=context, parameter_overrides=overrides)
        self.declare_parameter("joint_names", ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5"])
        self.declare_parameter("command_topic", "arm/joint_command")
        self.declare_parameter("state_topic", "arm/joint_states")
        self.declare_parameter("publish_rate", 50.0)

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._position = {name: 0.0 for name in self._joint_names}
        self._lock = threading.Lock()

        self._pub_state = self.create_publisher(
            JointState, self.get_parameter("state_topic").value, 10
        )
        self.create_subscription(
            JointCommand, self.get_parameter("command_topic").value, self._on_command, 10
        )
        self._timer = self.create_timer(
            1.0 / float(self.get_parameter("publish_rate").value), self._publish_state
        )

    def _on_command(self, msg: JointCommand) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position_cmd):
                if name in self._position:
                    self._position[name] = float(position)

    def _publish_state(self) -> None:
        with self._lock:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(self._joint_names)
            msg.position = [self._position[name] for name in self._joint_names]
        self._pub_state.publish(msg)


def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default=None)
    parsed, _ = parser.parse_known_args(args)
    rclpy.init(args=args)
    node = FakeArm(name="fake_arm", context=rclpy.get_default_context(), params_file=parsed.params_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
