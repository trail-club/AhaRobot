#!/usr/bin/env python3
"""Relay joint states to MoveIt with small excursions past the limits removed.

MoveIt's CheckStartStateBounds rejects every plan from a state outside the
URDF limits by any amount, while a joint resting on a limit is reported
slightly outside it (Gazebo: about 1e-14).

Positions at most the limit_tolerance parameter outside a URDF limit are
moved onto the limit; larger excursions are relayed unchanged so that real
limit violations stay visible. The default covers roundoff only. A simulator
whose physics limits lie outside the URDF passes its margin (sim.launch.py).

MoveIt plans from, and checks the execution start against, the relayed state,
so it never sees the difference. The trajectory controllers close it, at most
limit_tolerance, as the execution starts.

Input `joint_states` (remapped by manip.launch.py), output OUTPUT_TOPIC.
Limits come from the robot_description parameter.
"""

import sys
import xml.etree.ElementTree as ET

import rclpy
from sensor_msgs.msg import JointState

OUTPUT_TOPIC = "/aha/manipulation/joint_states"
ROUNDOFF = 1e-6  # m for prismatic, rad for revolute joints


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


def clamp_near_limit(value, lower, upper, tolerance):
    """Value moved onto a limit when at most tolerance outside it."""
    if lower - tolerance <= value < lower:
        return lower
    if upper < value <= upper + tolerance:
        return upper
    return value


def clamp_positions(msg, limits, tolerance):
    """Clamp msg.position in place; a message without positions passes unchanged."""
    if len(msg.position) == len(msg.name):
        msg.position = [
            clamp_near_limit(value, *limits[name], tolerance)
            if name in limits
            else value
            for name, value in zip(msg.name, msg.position)
        ]
    return msg


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("aha_joint_state_bounds")
    limits = position_limits(node.declare_parameter("robot_description", "").value)
    tolerance = node.declare_parameter("limit_tolerance", ROUNDOFF).value
    publisher = node.create_publisher(JointState, OUTPUT_TOPIC, 10)

    # Reliable like the publishers (joint_state_broadcaster, the evaluation),
    # so no state is dropped under load.
    node.create_subscription(
        JointState,
        "joint_states",
        lambda msg: publisher.publish(clamp_positions(msg, limits, tolerance)),
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
