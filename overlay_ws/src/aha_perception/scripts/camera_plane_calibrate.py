#!/usr/bin/env python3
"""Correct the head camera mount of the frame model from a reference plane.

Runs next to plane detection on the real robot (aha_bringup real.launch.py).
Over --samples detections it takes the largest horizontal plane whose height
is within --tolerance of --reference (m in base_footprint: the table top the
frame stands on, or 0 for the floor), averages its fit and computes the
camera tilt and height correction (aha_perception/plane_calibration.py).
The new camera_joint (link_head_tilt -> camera_link) is printed and, with
--write, set as ``camera`` in the frame config (aha_description/frame.py).
Point the head at a large free area of the reference plane; the camera's
horizontal position and yaw are not observable this way. The correction holds
for the head pose it was measured in (the error may come from the pan / tilt
axes): calibrate in the working pose and check other poses afterwards.

  ros2 run aha_perception camera_plane_calibrate.py --reference 0.38 \
    --config <frame.yaml> [--write]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from aha_perception.plane_calibration import (
    PlaneFit,
    correction,
    corrected_mount,
    matrix_rpy,
)

TARGET_FRAME = "base_footprint"
CAMERA_FRAME = "camera_color_optical_frame"
MOUNT_PARENT = "link_head_tilt"
MOUNT_CHILD = "camera_link"
MAX_TILT_DEG = 15.0
TIMEOUT_S = 60.0


def pick_plane(msg, reference: float, tolerance: float):
    """PlaneFit of the largest near-horizontal plane at the reference height."""
    best = None
    for plane in msg.planes:
        if not all(math.isfinite(v) for v in (plane.offset, plane.height, plane.area)):
            continue
        fit = PlaneFit.of(
            (plane.normal.x, plane.normal.y, plane.normal.z), -plane.offset
        )
        tilt = math.degrees(math.acos(min(1.0, abs(float(fit.normal[2])))))
        if tilt > MAX_TILT_DEG or abs(plane.height - reference) > tolerance:
            continue
        if best is None or plane.area > best[0]:
            best = (plane.area, fit)
    return None if best is None else best[1]


def mean_fit(fits) -> PlaneFit:
    normal = np.mean([f.normal for f in fits], axis=0)
    offset = float(np.mean([f.offset for f in fits]))
    return PlaneFit.of(normal, offset)


def set_camera(text: str, xyz, rpy) -> str:
    """Frame config text with ``camera: {xyz, rpy}`` set (replaced or added
    before ``arms:``)."""
    line = (
        "camera: {xyz: [" + ", ".join(f"{v:.4f}" for v in xyz) + "], rpy: ["
        + ", ".join(f"{v:.4f}" for v in rpy) + "]}"
    )
    pattern = re.compile(r"^camera:.*$", re.M)
    if pattern.search(text):
        return pattern.sub(line, text, count=1)
    arms = re.compile(r"^arms:", re.M).search(text)
    if arms is None:
        raise ValueError("no 'arms:' in the frame config")
    return text[: arms.start()] + line + "\n\n" + text[arms.start() :]


def matrix_of(transform) -> np.ndarray:
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = (t.x, t.y, t.z)
    return m


def collect(args):
    """(mean plane, mean camera position, base->parent, parent->camera)."""
    import rclpy
    from aha_msgs.msg import PlaneArray
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("aha_camera_plane_calibrate")
    buffer = Buffer()
    TransformListener(buffer, node)
    samples = []

    def lookup(target, source, stamp):
        for when in (Time.from_msg(stamp), Time()):
            try:
                return matrix_of(buffer.lookup_transform(target, source, when).transform)
            except TransformException:
                continue
        return None

    def on_planes(msg):
        if msg.header.frame_id != TARGET_FRAME:
            node.get_logger().error(
                f"planes are in {msg.header.frame_id!r}, not {TARGET_FRAME}: "
                "run plane detection with target_frame:=base_footprint",
                throttle_duration_sec=5.0,
            )
            return
        fit = pick_plane(msg, args.reference, args.tolerance)
        if fit is None:
            node.get_logger().warn(
                f"no horizontal plane within {args.tolerance} m of {args.reference} m",
                throttle_duration_sec=5.0,
            )
            return
        camera = lookup(TARGET_FRAME, CAMERA_FRAME, msg.header.stamp)
        parent = lookup(TARGET_FRAME, MOUNT_PARENT, msg.header.stamp)
        mount = lookup(MOUNT_PARENT, MOUNT_CHILD, msg.header.stamp)
        if camera is None or parent is None or mount is None:
            node.get_logger().warn("TF not available yet", throttle_duration_sec=5.0)
            return
        samples.append((fit, camera[:3, 3], parent, mount))
        print(f"sample {len(samples)}/{args.samples}", flush=True)

    node.create_subscription(
        PlaneArray, "/aha/perception/planes", on_planes, qos_profile_sensor_data
    )
    deadline = time.monotonic() + TIMEOUT_S
    try:
        while len(samples) < args.samples and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if len(samples) < args.samples:
        raise RuntimeError(f"only {len(samples)} samples in {TIMEOUT_S:.0f} s")
    plane = mean_fit([s[0] for s in samples])
    camera = np.mean([s[1] for s in samples], axis=0)
    # The head should stay still: use the last transforms.
    return plane, camera, samples[-1][2], samples[-1][3]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=float, required=True, help="plane height, m")
    parser.add_argument("--tolerance", type=float, default=0.08, help="m")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--config", type=Path, help="frame YAML (for --write)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write and args.config is None:
        parser.error("--write needs --config")

    plane, camera, base_to_parent, parent_to_camera = collect(args)
    fix = correction(plane, camera, args.reference)
    mount = corrected_mount(base_to_parent, parent_to_camera, fix)
    rpy = matrix_rpy(mount)
    print(
        f"plane tilt {fix.tilt_deg:.2f} deg, camera {fix.distance:.3f} m above it\n"
        f"correction: roll {math.degrees(fix.roll):+.2f} deg, pitch "
        f"{math.degrees(fix.pitch):+.2f} deg, height {fix.dz * 1000:+.1f} mm\n"
        f"camera_joint: xyz {np.round(mount[:3, 3], 4).tolist()} "
        f"rpy {np.round(rpy, 4).tolist()}"
    )
    if args.write:
        args.config.write_text(set_camera(args.config.read_text(), mount[:3, 3], rpy))
        print(f"written to {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
