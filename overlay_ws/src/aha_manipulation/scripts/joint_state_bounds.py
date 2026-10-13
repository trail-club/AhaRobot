#!/usr/bin/env python3
"""Relay joint states to MoveIt with physics roundoff at the limits removed.

Gazebo reports a joint resting on a limit slightly outside it (the lift at
its lower limit reads -1e-14 m), and MoveIt's CheckStartStateBounds then
rejects every plan from that state. Positions within TOLERANCE of a URDF
limit are moved onto the limit; larger excursions are relayed unchanged so
that real limit violations stay visible.

Input `joint_states` (remapped by manip.launch.py), output OUTPUT_TOPIC.
Limits come from the robot_description parameter.
"""

import sys
import xml.etree.ElementTree as ET

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

OUTPUT_TOPIC = "/aha/manipulation/joint_states"
TOLERANCE = 1e-6


def position_limits(urdf):
    """{joint: (lower, upper)} for revolute and prismatic joints of a URDF string."""
    limits = {}
    for joint in ET.fromstring(urdf).findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") in ("revolute", "prismatic") and limit is not None:
            limits[joint.get("name")] = (
                float(limit.get("lower", 0.0)),
                float(limit.get("upper", 0.0)),
            )
    return limits


def clamp_roundoff(value, lower, upper, tolerance=TOLERANCE):
    if lower - tolerance <= value < lower:
        return lower
    if upper < value <= upper + tolerance:
        return upper
    return value


def clamp_positions(msg, limits):
    """Clamp msg.position in place; a message without positions passes unchanged."""
    if len(msg.position) == len(msg.name):
        msg.position = [
            clamp_roundoff(value, *limits[name]) if name in limits else value
            for name, value in zip(msg.name, msg.position)
        ]
    return msg


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("aha_joint_state_bounds")
    limits = position_limits(node.declare_parameter("robot_description", "").value)
    publisher = node.create_publisher(JointState, OUTPUT_TOPIC, 10)

    node.create_subscription(
        JointState,
        "joint_states",
        lambda msg: publisher.publish(clamp_positions(msg, limits)),
        qos_profile_sensor_data,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
