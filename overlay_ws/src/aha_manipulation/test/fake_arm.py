#!/usr/bin/env python3
"""Stand-in for the upstream astra_controller arm_node in the bridge tests.

Like the arm_node it takes a JointCommand on `command_topic` only with
exactly `joint_names` (others are counted on `fake/rejected`, where the
arm_node would raise) and publishes the joint feedback on `state_topic`. The
joints start at `initial` and move toward the goal at `max_rate` rad/s.

Test controls (std_msgs/Bool): `fake/stall` stops the motion, `fake/mute`
stops the feedback.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from astra_controller_interfaces.msg import JointCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt32


class FakeArm(Node):
    def __init__(self):
        super().__init__("fake_arm")
        self.names = list(self.declare_parameter("joint_names", [""]).value)
        initial = list(self.declare_parameter("initial", [0.0] * len(self.names)).value)
        self.max_rate = self.declare_parameter("max_rate", 2.0).value
        rate = self.declare_parameter("publish_rate", 50.0).value
        self.dt = 1.0 / rate
        self.position = dict(zip(self.names, initial))
        self.goal = dict(self.position)
        self.stall = False
        self.mute = False
        self.rejected = 0

        self.state_pub = self.create_publisher(
            JointState, self.declare_parameter("state_topic", "/joint_states").value, 10
        )
        self.rejected_pub = self.create_publisher(UInt32, "fake/rejected", 10)
        # Same QoS as the arm_node's subscription.
        self.create_subscription(
            JointCommand,
            self.declare_parameter("command_topic", "joint_command").value,
            self._on_command,
            qos_profile_sensor_data,
        )
        self.create_subscription(Bool, "fake/stall", self._on_stall, 10)
        self.create_subscription(Bool, "fake/mute", self._on_mute, 10)
        self.create_timer(self.dt, self._step)

    def _on_command(self, msg):
        if list(msg.name) != self.names:
            self.rejected += 1
            return
        self.goal = dict(zip(self.names, msg.position_cmd))

    def _on_stall(self, msg):
        self.stall = msg.data

    def _on_mute(self, msg):
        self.mute = msg.data

    def _step(self):
        if not self.stall:
            step = self.max_rate * self.dt
            for name in self.names:
                diff = self.goal[name] - self.position[name]
                self.position[name] += max(-step, min(step, diff))
        self.rejected_pub.publish(UInt32(data=self.rejected))
        if self.mute:
            return
        msg = JointState(name=list(self.names))
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [self.position[n] for n in self.names]
        self.state_pub.publish(msg)
        # The arm_node publishes the gripper on the same topic (arms.launch.py).
        self.state_pub.publish(
            JointState(name=["joint_l7l", "joint_l7r"], position=[0.0, 0.0])
        )


def main():
    rclpy.init()
    node = FakeArm()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
