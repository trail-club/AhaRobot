#!/usr/bin/env python3
"""Relay joint states to MoveIt with small excursions past the limits removed.

MoveIt's CheckStartStateBounds rejects every plan from a state outside the
URDF limits by any amount. Gazebo reports a joint resting on a limit about
1e-14 outside it, and its physics limits lie 0.005 outside the URDF, so a
joint can also stop, or pass while moving, up to 0.005 outside (measured:
joint_r4 5e-3 during a move to its limit).

Positions within TOLERANCE outside a URDF limit are moved onto the limit.
TOLERANCE stays below trajectory_execution.allowed_start_tolerance
(config/controllers.yaml), so a plan from the clamped state still starts
within the tolerance of the actual position. Larger excursions are relayed
unchanged so that real limit violations stay visible.

Input `joint_states` (remapped by manip.launch.py), output OUTPUT_TOPIC.
Limits come from the robot_description parameter.
"""

import sys
import xml.etree.ElementTree as ET

import rclpy
from sensor_msgs.msg import JointState

OUTPUT_TOPIC = "/aha/manipulation/joint_states"
TOLERANCE = 0.005  # m for prismatic, rad for revolute joints


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

    # Reliable like the publishers (joint_state_broadcaster, the evaluation),
    # so no state is dropped under load.
    node.create_subscription(
        JointState,
        "joint_states",
        lambda msg: publisher.publish(clamp_positions(msg, limits)),
        10,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
