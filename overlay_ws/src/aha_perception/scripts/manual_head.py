#!/usr/bin/env python3
"""Stand-in for the head servos when only the camera is connected.

Publishes /head_controller/joint_states from the head trajectories sent by the
control panel, head_look_at.py or teleop_head.py, so the TF of a camera that is
tilted by hand (or fixed on a stand) can be matched to the real camera by
moving the sliders. The head moves to the last point of each trajectory at the
configured max_speed; positions are clamped to the head.yaml limits.

  in   /head_controller/joint_trajectory (trajectory_msgs/JointTrajectory)
  out  /head_controller/joint_states (sensor_msgs/JointState, 30 Hz)

Parameters: head.* (config/head.yaml), initial_pan / initial_tilt (rad).
"""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from aha_perception.head_config import (
    PAN_JOINT,
    TILT_JOINT,
    TRAJECTORY_TOPIC,
    declare_head_config,
)
from aha_perception.head_follow import CommandedHead

STATE_TOPIC = "/head_controller/joint_states"
RATE_HZ = 30.0


class ManualHead(Node):
    def __init__(self):
        super().__init__("aha_manual_head")
        head = declare_head_config(self)
        pan = float(self.declare_parameter("initial_pan", 0.0).value)
        tilt = float(self.declare_parameter("initial_tilt", 0.0).value)
        self.follower = CommandedHead(head, pan, tilt)
        self.last = time.monotonic()
        self.pub = self.create_publisher(JointState, STATE_TOPIC, 10)
        self.create_subscription(
            JointTrajectory, TRAJECTORY_TOPIC, self.on_trajectory, 10
        )
        self.create_timer(1.0 / RATE_HZ, self.on_timer)
        pan, tilt = self.follower.position
        self.get_logger().info(
            f"manual head: pan {pan:+.3f} tilt {tilt:+.3f} rad; "
            f"follows {TRAJECTORY_TOPIC}"
        )

    def on_trajectory(self, msg):
        if self.follower.command(msg):
            pan, tilt = self.follower.target
            self.get_logger().warn(
                f"target clamped to pan {pan:+.3f} tilt {tilt:+.3f} rad"
            )

    def on_timer(self):
        now = time.monotonic()
        position = self.follower.step(now - self.last)
        self.last = now
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [PAN_JOINT, TILT_JOINT]
        msg.position = position
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ManualHead()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
