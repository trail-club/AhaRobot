#!/usr/bin/env python3
"""Correct the head camera's horizontal position and yaw from gripper clicks.

Runs on the real robot (aha_bringup real.launch.py) after
camera_plane_calibrate.py has set the camera tilt and height. For each pose
an arm (torque off) is held still with its gripper tip in the point cloud and
the tip is clicked in RViz with Publish Point (--topic). The click, seen
through the modelled camera, is paired with the same point from the arm
model's TF at the click time: the origin of the nearest of --links plus
--offset in that link, moved --radius toward the camera (the click lands on
the surface facing it). A pair is rejected when no link or more than one is
within --max-error of the click, when the gripper moved in the last 0.5 s,
or when the head moved since the first pair. Enter 'u' to drop the last
pair and 'q' to finish.

The fit (aha_perception/camera_arm_calibration.py) gives the camera's
horizontal shift and yaw in base_footprint with per-pair residuals; the new
camera_joint (link_head_tilt -> camera_link) is printed and, with --write,
set as ``camera`` in the frame config, keeping the tilt and height. The
correction holds for the head pose it was measured in.

The RViz of real.launch.py remaps Publish Point to the head look-at, so a
click would turn the head: start real.launch.py with rviz:=false and RViz on
its own (aha_bringup README, Calibration).

  ros2 run aha_perception camera_arm_calibrate.py --config <frame.yaml> [--write]
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aha_perception.camera_arm_calibration import (
    corrected_mount,
    facing_point,
    horizontal_correction,
    nearest,
)
from aha_perception.plane_calibration import (
    matrix_rpy,
    set_camera,
    transform_matrix,
)

TARGET_FRAME = "base_footprint"
MOUNT_PARENT = "link_head_tilt"
MOUNT_CHILD = "camera_link"
STILL_WINDOW_S = 0.5
STILL_TOLERANCE = 0.005  # m the model point may move within STILL_WINDOW_S
HEAD_TOLERANCE = (0.003, math.radians(0.5))  # camera position m, rotation rad
TF_WAIT_S = 1.0
TOPIC_WAIT_S = 2.0


@dataclass(frozen=True)
class Pair:
    link: str
    seen: np.ndarray  # (3,) in TARGET_FRAME
    model: np.ndarray  # (3,) in TARGET_FRAME
    base_to_parent: np.ndarray
    parent_to_camera: np.ndarray

    @property
    def camera(self) -> np.ndarray:
        return self.base_to_parent @ self.parent_to_camera


def head_moved(reference: np.ndarray, camera: np.ndarray) -> bool:
    """Whether ``camera`` (4x4) is off ``reference`` by more than HEAD_TOLERANCE."""
    shift = float(np.linalg.norm(camera[:3, 3] - reference[:3, 3]))
    relative = reference[:3, :3].T @ camera[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0)))
    return shift > HEAD_TOLERANCE[0] or angle > HEAD_TOLERANCE[1]


class Clicks:
    """Clicked points and TF lookups."""

    def __init__(self, topic: str):
        import rclpy
        from geometry_msgs.msg import PointStamped
        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformException, TransformListener

        self.rclpy, self.Duration, self.Time = rclpy, Duration, Time
        self.TransformException = TransformException
        rclpy.init()
        self.node = rclpy.create_node("aha_camera_arm_calibrate")
        self.buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.listener = TransformListener(self.buffer, self.node)
        self.topic = topic
        self.click = None
        self.node.create_subscription(PointStamped, topic, self.on_click, 10)

    def on_click(self, msg):
        self.click = msg

    def spin(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def check_topic(self):
        self.spin(TOPIC_WAIT_S)
        if self.node.count_publishers(self.topic) == 0:
            print(
                f"warning: nothing publishes {self.topic}: start RViz without the "
                "look-at remap (real.launch.py rviz:=false, then rviz2 on its own)",
                flush=True,
            )

    def lookup(self, target: str, source: str, when, wait: float = 0.0):
        """4x4 target <- source at ``when``, waiting up to ``wait`` s for it."""
        deadline = time.monotonic() + wait
        while True:
            try:
                return transform_matrix(
                    self.buffer.lookup_transform(target, source, when).transform
                )
            except self.TransformException:
                if time.monotonic() >= deadline:
                    return None
                self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def next_event(self):
        """('click', PointStamped) or ('line', text) of the next user input."""
        self.click = None
        while True:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.click is not None:
                return "click", self.click
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                line = sys.stdin.readline()
                return "line", line.strip().lower() if line else "q"

    def pair(self, msg, args) -> Pair:
        """Pair of a click; ValueError with the reason when it is rejected."""
        stamp = self.Time.from_msg(msg.header.stamp)
        stamped = stamp.nanoseconds > 0
        if not stamped:
            stamp = self.Time()
        latest = self.Time()
        fixed = self.lookup(TARGET_FRAME, msg.header.frame_id, stamp, TF_WAIT_S)
        if fixed is None:
            fixed = self.lookup(TARGET_FRAME, msg.header.frame_id, latest)
        parent = self.lookup(TARGET_FRAME, MOUNT_PARENT, stamp, TF_WAIT_S)
        mount = self.lookup(MOUNT_PARENT, MOUNT_CHILD, latest)
        if fixed is None or parent is None or mount is None:
            raise ValueError(f"no TF {TARGET_FRAME} -> {msg.header.frame_id} / camera")
        p = msg.point
        seen = (fixed @ np.array([p.x, p.y, p.z, 1.0]))[:3]
        camera = (parent @ mount)[:3, 3]
        offset = np.array([*args.offset, 1.0])
        # Without a stamp (latest TF) the motion before the click is unknown.
        before_stamp = (
            stamp - self.Duration(seconds=STILL_WINDOW_S) if stamped else None
        )
        candidates, moved = {}, {}
        for link in args.links:
            now = self.lookup(TARGET_FRAME, link, stamp, TF_WAIT_S)
            before = (
                self.lookup(TARGET_FRAME, link, before_stamp) if before_stamp else None
            )
            if now is None:
                continue
            point = (now @ offset)[:3]
            candidates[link] = facing_point(point, camera, args.radius)
            if before is not None:
                moved[link] = float(np.linalg.norm(point - (before @ offset)[:3]))
        if not candidates:
            raise ValueError(f"no TF for any of {args.links}")
        link = nearest(seen, candidates, args.max_error)
        if moved.get(link, 0.0) > STILL_TOLERANCE:
            raise ValueError(
                f"{link} moved {moved[link] * 1000:.0f} mm in the last "
                f"{STILL_WINDOW_S} s: hold the arm still while clicking"
            )
        return Pair(link, seen, candidates[link], parent, mount)

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def fit(pairs):
    reference = pairs[0].camera
    return horizontal_correction(
        [p.seen for p in pairs], [p.model for p in pairs], reference[:3, 3]
    )


def summary(fix) -> str:
    return (
        f"camera shift x {fix.dx * 1000:+.1f} mm, y {fix.dy * 1000:+.1f} mm, "
        f"yaw {math.degrees(fix.dyaw):+.2f} deg; RMS residual {fix.rms * 1000:.1f} mm"
    )


def collect(clicks: Clicks, args) -> list[Pair]:
    pairs: list[Pair] = []
    print(
        "Hide the RobotModel display in RViz (a click on the model mesh gives the\n"
        "model point, not the camera's). Use 6 or more poses spread over the arms'\n"
        "reach, gripper pointing down.",
        flush=True,
    )
    while True:
        print(
            f"\npose {len(pairs) + 1}: hold the gripper tip still in view and click "
            "it in RViz (Publish Point); 'u' + Enter: undo, 'q' + Enter: finish",
            flush=True,
        )
        kind, value = clicks.next_event()
        if kind == "line":
            if value == "q":
                return pairs
            if value == "u" and pairs:
                pairs.pop()
                print(f"dropped pair {len(pairs) + 1}")
            continue
        try:
            pair = clicks.pair(value, args)
            if pairs and head_moved(pairs[0].camera, pair.camera):
                raise ValueError(
                    "the head moved since the first pair: keep it still "
                    "('q' and start again if it has to move)"
                )
        except ValueError as e:
            print(f"rejected: {e}")
            continue
        pairs.append(pair)
        d = (pair.model - pair.seen) * 1000
        print(
            f"pair {len(pairs)} ({pair.link}): seen "
            f"{np.round(pair.seen, 3).tolist()}, model - seen "
            f"[{d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}] mm"
        )
        if len(pairs) >= 2:
            try:
                print("so far: " + summary(fit(pairs)))
            except ValueError as e:
                print(f"so far: {e}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="frame YAML (for --write)")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--topic", default="/clicked_point")
    parser.add_argument(
        "--links", nargs="+", default=["link_ree", "link_lee"], help="gripper links"
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="clicked point in the link frame, m",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="m from the point to the surface the camera sees",
    )
    parser.add_argument("--max-error", type=float, default=0.10, help="m")
    args = parser.parse_args(argv)
    if args.write and args.config is None:
        parser.error("--write needs --config")
    if args.radius < 0.0 or args.max_error <= 0.0:
        parser.error("--radius must be >= 0 and --max-error > 0")

    clicks = Clicks(args.topic)
    try:
        clicks.check_topic()
        pairs = collect(clicks, args)
    finally:
        clicks.close()
    if len(pairs) < 2:
        print(f"no result: {len(pairs)} pair(s), at least 2 are needed")
        return 1
    try:
        fix = fit(pairs)
    except ValueError as e:
        print(f"no result: {e}")
        return 1
    mount = corrected_mount(pairs[0].base_to_parent, pairs[0].parent_to_camera, fix)
    rpy = matrix_rpy(mount)
    print("\npair  link        residual x, y [mm]   |r| [mm]  model - seen z [mm]")
    for i, (pair, r, dz) in enumerate(zip(pairs, fix.residuals, fix.dz), 1):
        print(
            f"{i:4d}  {pair.link:10s}  {r[0] * 1000:+7.1f} {r[1] * 1000:+7.1f}"
            f"   {np.linalg.norm(r) * 1000:8.1f}  {dz * 1000:+8.1f}"
        )
    print(
        f"{summary(fix)}; mean model - seen z {np.mean(fix.dz) * 1000:+.1f} mm "
        "(not applied)\n"
        f"camera_joint: xyz {np.round(mount[:3, 3], 4).tolist()} "
        f"rpy {np.round(rpy, 4).tolist()}"
    )
    if args.write:
        args.config.write_text(set_camera(args.config.read_text(), mount[:3, 3], rpy))
        print(f"written to {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
