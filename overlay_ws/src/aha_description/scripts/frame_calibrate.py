#!/usr/bin/env python3
"""Calibrate the arm mount yaw of the frame model (aha_description/frame.py).

The arm joints read URDF angles (zero pose and signs calibrated on the arms,
docs/context/motor.md), but how each arm stands on the frame is not known:
the yaw of link_<l|r>1 in the frame config. With the arms' torque off, for
each arm:

1. forward: put the upper arm straight forward (parallel to base_link x)
   and the forearm in line with it. The shoulder angle gives the yaw that
   points link_<l|r>2 forward; the elbow angle should then read the URDF
   straight pose (joint_<l|r>3 = -its origin yaw).
2. outward: turn the upper arm straight outward (away from the robot). The
   shoulder angle must change by +90 deg (left) or -90 deg (right) about z;
   another change means a wrong sign or a slipping joint.

Needs /joint_states from the arms (aha_bringup arms.launch.py; torque off
with `ros2 topic pub --once /<side>/arm/torque_enable std_msgs/msg/UInt8
"{data: 0}"`) and /robot_description (robot_state_publisher). With --write
the yaw values in the frame config are replaced in place.

  ros2 run aha_description frame_calibrate.py --config <frame.yaml> [--write]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ARMS = {"left": "l", "right": "r"}
OUTWARD = {"left": math.pi / 2, "right": -math.pi / 2}  # shoulder change, rad
STRAIGHT_TOLERANCE = math.radians(5.0)
OUTWARD_TOLERANCE = math.radians(15.0)
SAMPLE_S = 1.0


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def origin_yaw(urdf: str, joint: str) -> float:
    """z rotation of a joint origin whose roll and pitch are zero."""
    element = ET.fromstring(urdf).find(f"joint[@name='{joint}']/origin")
    if element is None:
        raise ValueError(f"URDF has no origin for {joint}")
    roll, pitch, yaw = (float(v) for v in element.get("rpy", "0 0 0").split())
    if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
        raise ValueError(f"{joint} origin is not a pure z rotation")
    return yaw


def mount_yaw(shoulder_offset: float, shoulder: float) -> float:
    """link_<x>1 yaw that points link_<x>2 along base_link x at this angle."""
    return wrap(-(shoulder_offset + shoulder))


def straight_error(elbow_offset: float, elbow: float) -> float:
    """Forearm direction relative to the upper arm (0 = in line)."""
    return wrap(elbow_offset + elbow)


def outward_error(side: str, forward: float, outward: float) -> float:
    """Shoulder change from forward to outward minus the expected one."""
    return wrap((outward - forward) - OUTWARD[side])


def replace_yaw(text: str, side: str, yaw: float) -> str:
    """Config text with the ``yaw`` of ``side``'s arm entry set to ``yaw``."""
    pattern = re.compile(rf"^(\s*{side}:\s*\{{.*?yaw:\s*)(-?[0-9.eE+-]+)(.*)$", re.M)
    text, count = pattern.subn(lambda m: f"{m.group(1)}{yaw:.4f}{m.group(3)}", text)
    if count != 1:
        raise ValueError(f"no single '{side}: {{... yaw: ...}}' line in the config")
    return text


class JointReader:
    def __init__(self):
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("aha_frame_calibrate")
        self.samples = {}
        self.urdf = None
        self.node.create_subscription(JointState, "/joint_states", self.on_joints, 50)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(
            String, "/robot_description", self.on_description, latched
        )

    def on_joints(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.samples.setdefault(name, []).append(position)

    def on_description(self, msg):
        self.urdf = msg.data

    def spin(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_description(self, timeout: float = 10.0) -> str:
        deadline = time.monotonic() + timeout
        while self.urdf is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.urdf is None:
            raise RuntimeError("no /robot_description (robot_state_publisher)")
        return self.urdf

    def mean(self, joints) -> list[float]:
        """Mean angles over SAMPLE_S of fresh messages."""
        self.samples.clear()
        self.spin(SAMPLE_S)
        missing = [j for j in joints if not self.samples.get(j)]
        if missing:
            raise RuntimeError(f"no /joint_states for {missing}")
        # Circular mean: safe next to +-pi.
        return [
            math.atan2(
                sum(math.sin(a) for a in self.samples[j]),
                sum(math.cos(a) for a in self.samples[j]),
            )
            for j in joints
        ]

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def calibrate(reader, urdf, side) -> float:
    letter = ARMS[side]
    shoulder, elbow = f"joint_{letter}2", f"joint_{letter}3"
    shoulder_offset = origin_yaw(urdf, shoulder)
    elbow_offset = origin_yaw(urdf, elbow)
    input(
        f"\n[{side}] Put the upper arm straight FORWARD (parallel to the robot's "
        "x axis) and the forearm in line with it, then press Enter."
    )
    forward, bend = reader.mean([shoulder, elbow])
    yaw = mount_yaw(shoulder_offset, forward)
    straight = straight_error(elbow_offset, bend)
    print(
        f"[{side}] {shoulder} {math.degrees(forward):+.1f} deg, {elbow} "
        f"{math.degrees(bend):+.1f} deg -> mount yaw {math.degrees(yaw):+.1f} deg"
    )
    if abs(straight) > STRAIGHT_TOLERANCE:
        print(
            f"[{side}] warning: the forearm reads {math.degrees(straight):+.1f} deg "
            "off the upper arm line (elbow zero or sign, or the pose)"
        )
    input(
        f"[{side}] Turn the upper arm straight OUTWARD (away from the robot), "
        "then press Enter."
    )
    (outward,) = reader.mean([shoulder])
    error = outward_error(side, forward, outward)
    print(
        f"[{side}] {shoulder} {math.degrees(outward):+.1f} deg: changed "
        f"{math.degrees(wrap(outward - forward)):+.1f} deg "
        f"(expected {math.degrees(OUTWARD[side]):+.0f})"
    )
    if abs(error) > OUTWARD_TOLERANCE:
        print(
            f"[{side}] warning: {math.degrees(error):+.1f} deg from the expected "
            "change (shoulder sign, or the pose)"
        )
    return yaw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frame YAML")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--write", action="store_true", help="update the yaw values")
    args = parser.parse_args(argv)

    reader = JointReader()
    try:
        urdf = reader.wait_description()
        results = {side: calibrate(reader, urdf, side) for side in args.arms}
    finally:
        reader.close()
    print()
    for side, yaw in results.items():
        print(f"{side}: yaw {yaw:.4f}  # {math.degrees(yaw):+.1f} deg")
    if args.write:
        text = args.config.read_text()
        for side, yaw in results.items():
            text = replace_yaw(text, side, yaw)
        args.config.write_text(text)
        print(f"written to {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
