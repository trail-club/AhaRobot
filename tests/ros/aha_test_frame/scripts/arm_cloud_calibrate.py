#!/usr/bin/env python3
"""Calibrate an arm's mount yaw and joint offsets from the head camera's cloud.

Runs on the real robot (aha_test_frame real.launch.py, camera on) after the
camera tilt and height are calibrated (camera_plane_calibrate.py). The arm
(torque off) is held still in a pose inside the camera's view; Enter takes a
snapshot: the latest point cloud (--cloud) in base_footprint via TF at its
stamp, and the arm's joints from /<side>/arm/joint_states averaged over
STILL_WINDOW_S. A snapshot is rejected when the joints moved, when the cloud
is older than the joints by MAX_CLOUD_AGE_S, when the pose repeats an earlier
snapshot, or when the head moved since the first. Joints reading exactly the
same for FROZEN_S only warn: an arm at rest with the torque on reads constant,
and so do the frozen readings of an arm board that was reset (check the model
against the arm in RViz).
'u' drops the last snapshot, 'q' fits
(aha_test_frame/arm_cloud_calibration.py): the --free parameters
(shoulder: mount yaw, elbow / wrist: joint_<x>3 / joint_<x>4 offsets,
camera: horizontal position and yaw) that put the cloud points on the arm
model. It prints per-snapshot RMS distances and inlier counts, the estimates
with their observability checks, and the new config values; --write updates
``yaw`` / ``joint_offsets`` of the arm and, with camera free, ``camera`` in
the frame config (refused when a check fails, unless --force).

The model is /robot_description (what TF and the cloud use); it must match
the frame config (--config) the deltas are added to. --save keeps the
snapshots in an .npz after each one, --load fits saved snapshots offline.
--snapshots N --interval S takes N snapshots S seconds apart without input.

  ros2 run aha_test_frame arm_cloud_calibrate.py --config <frame.yaml> \
    --arm right [--free shoulder,elbow[,wrist][,camera]] [--write]
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from aha_test_frame.frame import (
    ARMS,
    replace_joint_offsets,
    replace_shoulder_xyz,
    replace_yaw,
)
from aha_test_frame.arm_cloud_calibration import (
    DEFAULT_FREE,
    GROUPS,
    ArmModel,
    Snapshot,
    arm_joints,
    camera_moved,
    fit,
    free_parameters,
    prepare,
    unit,
)
from aha_test_frame.camera_arm_calibration import (
    HorizontalCorrection,
    corrected_mount,
)
from aha_test_frame.plane_calibration import matrix_rpy, set_camera, transform_matrix

TARGET_FRAME = "base_footprint"
MOUNT_PARENT = "link_head_tilt"
MOUNT_CHILD = "camera_link"
STILL_WINDOW_S = 0.5
STILL_TOLERANCE = 0.005  # rad a joint may move within STILL_WINDOW_S
FROZEN_S = 1.0
MAX_CLOUD_AGE_S = 0.5
HEAD_TOLERANCE = (0.003, math.radians(0.5))  # camera position m, rotation rad
MIN_POSE_CHANGE = 0.1  # rad, some joint must differ from every other snapshot
MODEL_TOLERANCE = 1e-4  # m or rad between /robot_description and the config
TF_WAIT_S = 1.0
GUIDE = """\
Put the {side} arm (torque off) in 3 or more clearly different poses over
the table, inside the camera's view: vary the shoulder and the elbow, keep
the forearm and the gripper visible, hide nothing in front of the arm, and
hold it still for each snapshot. The head must not move."""


def circular_mean(values) -> float:
    values = np.asarray(values, float)
    return math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean()))


def joint_problem(times, values) -> str | None:
    """Why the joint readings (``values`` N x J at ``times``, s) do not make
    a snapshot, or None."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    recent = times >= times[-1] - STILL_WINDOW_S if len(times) else times
    if recent.sum() < 3:
        return "no fresh joint states"
    swing = np.ptp(values[recent], axis=0).max()
    if swing > STILL_TOLERANCE:
        return f"the joints moved {swing:.3f} rad in {STILL_WINDOW_S} s: hold still"
    if times[-1] - times[0] >= FROZEN_S and np.ptp(values, axis=0).max() == 0.0:
        return (
            f"the joints read exactly the same for {times[-1] - times[0]:.1f} s: "
            "frozen readings after an arm board reset? (fine for an arm at rest)"
        )
    return None


def origin_values(urdf: str, joints) -> dict:
    root = ET.fromstring(urdf)
    out = {}
    for name in joints:
        origin = root.find(f"joint[@name='{name}']/origin")
        if origin is None:
            raise ValueError(f"URDF has no origin for {name}")
        out[name] = [
            float(v)
            for key in ("xyz", "rpy")
            for v in (origin.get(key) or "0 0 0").split()
        ]
    return out


def model_mismatch(running: str, config: str, side: str) -> list[str]:
    """Joints whose origins differ between the two URDFs."""
    joints = [*arm_joints(side).values(), "camera_joint"]
    a, b = origin_values(running, joints), origin_values(config, joints)
    return [
        name
        for name in joints
        if np.max(np.abs(np.subtract(a[name], b[name]))) > MODEL_TOLERANCE
    ]


def updated_config(text: str, side: str, free, params: dict) -> str:
    """Frame config text with the fitted deltas added to the arm's ``yaw``,
    ``joint_offsets`` and ``shoulder_xyz`` height."""
    arm = yaml.safe_load(text)["arms"][side]
    if "shoulder" in free:
        text = replace_yaw(text, side, float(arm["yaw"]) + params["shoulder"])
    if "height" in free:
        x, y, z = (float(v) for v in arm["shoulder_xyz"])
        text = replace_shoulder_xyz(text, side, (x, y, z + params["height"]))
    names = arm_joints(side)
    offsets = dict(arm.get("joint_offsets") or {})
    changed = False
    for key in ("elbow", "wrist"):
        if key in free:
            joint = names[key]
            offsets[joint] = float(offsets.get(joint, 0.0)) + params[key]
            changed = True
    if changed:
        text = replace_joint_offsets(text, side, offsets)
    return text


# --- ROS capture ------------------------------------------------------------


class Capture:
    def __init__(self, args):
        import rclpy
        from rclpy.duration import Duration
        from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import JointState, PointCloud2
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformException, TransformListener

        self.rclpy, self.Time, self.Duration = rclpy, Time, Duration
        self.TransformException = TransformException
        rclpy.init()
        self.node = rclpy.create_node("aha_arm_cloud_calibrate")
        self.buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.listener = TransformListener(self.buffer, self.node)
        self.args = args
        letter = ARMS[args.arm]
        self.arm = [f"joint_{letter}{i}" for i in range(2, 7)]
        self.fingers = [f"joint_{letter}7l", f"joint_{letter}7r"]
        self.history = deque()  # (receive s, stamp ns, values)
        self.gripper = {}
        self.cloud = None
        self.urdf = None
        self.node.create_subscription(
            JointState, f"/{args.arm}/arm/joint_states", self.on_joints, 50
        )
        self.node.create_subscription(
            PointCloud2, args.cloud, self.on_cloud, qos_profile_sensor_data
        )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(
            String, "/robot_description", self.on_description, latched
        )

    def on_joints(self, msg):
        positions = dict(zip(msg.name, msg.position))
        for name in self.fingers:
            if name in positions:
                self.gripper[name] = positions[name]
        if not all(name in positions for name in self.arm):
            return
        now = time.monotonic()
        stamp = self.Time.from_msg(msg.header.stamp).nanoseconds
        self.history.append((now, stamp, [positions[n] for n in self.arm]))
        while self.history and self.history[0][0] < now - 2.0 * FROZEN_S:
            self.history.popleft()

    def on_cloud(self, msg):
        self.cloud = msg

    def on_description(self, msg):
        self.urdf = msg.data

    def spin(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait_description(self, timeout: float = 10.0) -> str:
        deadline = time.monotonic() + timeout
        while self.urdf is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.urdf is None:
            raise RuntimeError("no /robot_description (robot_state_publisher)")
        return self.urdf

    def lookup(self, target: str, source: str, when, wait: float = 0.0):
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

    def snapshot(self) -> dict:
        """One snapshot; ValueError with the reason when it is rejected."""
        from sensor_msgs_py import point_cloud2

        self.spin(STILL_WINDOW_S + 0.1)
        if self.history and self.history[-1][0] - self.history[0][0] < FROZEN_S:
            # Long enough to tell frozen readings.
            self.spin(FROZEN_S - (self.history[-1][0] - self.history[0][0]) + 0.1)
        if not self.history:
            raise ValueError(f"no /{self.args.arm}/arm/joint_states")
        times = [h[0] for h in self.history]
        values = [h[2] for h in self.history]
        problem = joint_problem(times, values)
        if problem and "frozen" in problem:
            print(f"warning: {problem}")
        elif problem:
            raise ValueError(problem)
        recent = [v for t, _, v in self.history if t >= times[-1] - STILL_WINDOW_S]
        joints = {
            name: circular_mean([v[i] for v in recent])
            for i, name in enumerate(self.arm)
        }
        joints.update(self.gripper)
        cloud = self.cloud
        if cloud is None:
            raise ValueError(f"no point cloud on {self.args.cloud}")
        stamp = self.Time.from_msg(cloud.header.stamp)
        age = (self.history[-1][1] - stamp.nanoseconds) * 1e-9
        if stamp.nanoseconds and self.history[-1][1] and age > MAX_CLOUD_AGE_S:
            raise ValueError(
                f"the cloud is {age:.2f} s older than the joints: is the camera running?"
            )
        frame = cloud.header.frame_id
        fixed = self.lookup(TARGET_FRAME, frame, stamp, TF_WAIT_S)
        if fixed is None:
            fixed = self.lookup(TARGET_FRAME, frame, self.Time())
            if fixed is None:
                raise ValueError(f"no TF {TARGET_FRAME} -> {frame}")
            print(f"warning: no TF at the cloud stamp, using the latest for {frame}")
        parent = self.lookup(TARGET_FRAME, MOUNT_PARENT, stamp, TF_WAIT_S)
        if parent is None:
            parent = self.lookup(TARGET_FRAME, MOUNT_PARENT, self.Time())
        mount = self.lookup(MOUNT_PARENT, MOUNT_CHILD, self.Time())
        if parent is None or mount is None:
            raise ValueError(f"no TF {TARGET_FRAME} -> {MOUNT_PARENT} -> {MOUNT_CHILD}")
        xyz = point_cloud2.read_points_numpy(
            cloud, field_names=("x", "y", "z"), skip_nans=True
        ).astype(float)
        points = xyz @ fixed[:3, :3].T + fixed[:3, 3]
        return {
            "points": points,
            "joints": joints,
            "camera": fixed,
            "base_to_parent": parent,
            "parent_to_camera": mount,
        }

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


# --- snapshots on disk ------------------------------------------------------


def save(path: Path, urdf: str, side: str, snaps):
    data = {"urdf": np.array(urdf), "side": np.array(side), "count": len(snaps)}
    for i, snap in enumerate(snaps):
        data[f"points_{i}"] = snap["points"].astype(np.float32)
        data[f"joint_names_{i}"] = np.array(list(snap["joints"]))
        data[f"joint_values_{i}"] = np.array(list(snap["joints"].values()))
        for key in ("camera", "base_to_parent", "parent_to_camera"):
            data[f"{key}_{i}"] = snap[key]
    np.savez_compressed(path, **data)


def load(path: Path):
    data = np.load(path)
    snaps = []
    for i in range(int(data["count"])):
        names = [str(n) for n in data[f"joint_names_{i}"]]
        snaps.append(
            {
                "points": data[f"points_{i}"].astype(float),
                "joints": dict(zip(names, data[f"joint_values_{i}"].tolist())),
                "camera": data[f"camera_{i}"],
                "base_to_parent": data[f"base_to_parent_{i}"],
                "parent_to_camera": data[f"parent_to_camera_{i}"],
            }
        )
    return str(data["urdf"]), str(data["side"]), snaps


# --- main -------------------------------------------------------------------


def check(model, snap, args, snaps) -> str:
    """One line on a snapshot; ValueError when it cannot be used."""
    if snaps and camera_moved(snaps[0]["camera"], snap["camera"], HEAD_TOLERANCE):
        raise ValueError("the head moved since the first snapshot: keep it still")
    for i, other in enumerate(snaps, 1):
        change = pose_change(other["joints"], snap["joints"], args.arm)
        if change < MIN_POSE_CHANGE:
            raise ValueError(
                f"the pose is within {change:.2f} rad of snapshot {i}: move the arm"
            )
    prepared = prepare(model, to_snapshot(snap, args))
    if len(prepared.points) < 100:
        raise ValueError(
            f"only {len(prepared.points)} cloud points near the arm model: "
            "is the arm in view?"
        )
    support = "none" if prepared.support is None else f"{prepared.support:.3f} m"
    return f"{len(prepared.points)} points near the arm, support {support}"


def pose_change(a: dict, b: dict, side: str) -> float:
    """Largest change of the arm joints 2..6 between two readings, rad."""
    letter = ARMS[side]
    names = [f"joint_{letter}{i}" for i in range(2, 7)]
    return max(abs(a.get(n, 0.0) - b.get(n, 0.0)) for n in names)


def to_snapshot(snap: dict, args) -> Snapshot:
    return Snapshot(
        snap["points"], snap["joints"], snap["camera"][:3, 3], support=args.support
    )


def collect(capture: Capture, model, args, snaps: list, urdf: str) -> list:
    print(GUIDE.format(side=args.arm), flush=True)
    automatic = args.snapshots > 0
    while True:
        if automatic:
            if len(snaps) >= args.snapshots:
                return snaps
            capture.spin(args.interval)
            command = ""
        else:
            print(
                f"\nsnapshot {len(snaps) + 1}: Enter = take, 'u' = undo, 'q' = fit",
                flush=True,
            )
            command = read_line(capture)
        if command == "q":
            return snaps
        if command == "u":
            if snaps:
                snaps.pop()
                print(f"dropped snapshot {len(snaps) + 1}")
            continue
        try:
            snap = capture.snapshot()
            line = check(model, snap, args, snaps)
        except ValueError as e:
            print(f"rejected: {e}", flush=True)
            continue
        snaps.append(snap)
        print(f"snapshot {len(snaps)}: {line}", flush=True)
        if args.save:
            save(args.save, urdf, args.arm, snaps)


def read_line(capture: Capture) -> str:
    while True:
        capture.rclpy.spin_once(capture.node, timeout_sec=0.05)
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            line = sys.stdin.readline()
            return line.strip().lower() if line else "q"


def report(result, args) -> None:
    print("\nsnapshot  points  inliers before/after  RMS before/after [mm]  support")
    for i, s in enumerate(result.snapshots, 1):
        support = "-" if s.support is None else f"{s.support:.3f}"
        print(
            f"{i:8d}  {s.points:6d}  {s.inliers_before:7d} / {s.inliers_after:<7d}"
            f"  {s.rms_before * 1000:8.2f} / {s.rms_after * 1000:<8.2f}  {support}"
        )
    print("\nparameter   delta        noise 1-sigma  leave-one-out")
    for name in result.free:
        spread = result.leave_one_out
        loo = "-" if spread is None else f"{spread[name]:.4f}"
        value = result.params[name]
        shown = f"{value:+.4f} {unit(name)}"
        if unit(name) == "rad":
            shown += f" ({math.degrees(value):+.2f} deg)"
        print(f"{name:10s}  {shown:26s} {result.std[name]:.4f}  {loo}")
    if len(result.free) > 1:
        print("correlation:")
        for name, row in zip(result.free, result.correlation):
            print(f"  {name:10s} " + " ".join(f"{v:+.3f}" for v in row))
    print(
        f"condition {result.condition:.3g}; {result.iterations} iterations, "
        f"{result.seconds:.1f} s"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frame YAML")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--free",
        default=",".join(DEFAULT_FREE),
        help=f"comma-separated, of {', '.join(GROUPS)}",
    )
    parser.add_argument("--write", action="store_true", help="update the config")
    parser.add_argument("--force", action="store_true", help="write despite warnings")
    parser.add_argument("--cloud", default="/camera/depth/points")
    parser.add_argument("--support", type=float, help="support surface height, m")
    parser.add_argument(
        "--exclude-fingers",
        action="store_true",
        help="leave the finger points out of the fit (only the hand may be in view)",
    )
    parser.add_argument("--save", type=Path, help="keep the snapshots in this .npz")
    parser.add_argument("--load", type=Path, help="fit the snapshots of this .npz")
    parser.add_argument(
        "--snapshots", type=int, default=0, help="take this many without input"
    )
    parser.add_argument("--interval", type=float, default=3.0, help="s between them")
    args = parser.parse_args(argv)
    try:
        free = free_parameters([g for g in args.free.split(",") if g])
    except ValueError as e:
        parser.error(str(e))
    if "camera" in args.free.split(",") and "shoulder" in args.free.split(","):
        print(
            "warning: with one arm the camera correction and the shoulder offset "
            "move the arm alike; free the camera only with the shoulder fixed"
        )

    from aha_test_frame.frame import frame_description

    config_urdf = frame_description(args.config)
    if args.load:
        urdf, side, snaps = load(args.load)
        if side != args.arm:
            parser.error(f"{args.load} has snapshots of the {side} arm")
        model = ArmModel.from_urdf(urdf, args.arm)
    else:
        capture = Capture(args)
        try:
            urdf = capture.wait_description()
            model = ArmModel.from_urdf(urdf, args.arm)
            snaps = collect(capture, model, args, [], urdf)
        finally:
            capture.close()
    mismatch = model_mismatch(urdf, config_urdf, args.arm)
    if mismatch:
        print(
            f"warning: the snapshots' model (/robot_description) differs from "
            f"{args.config} at {', '.join(mismatch)}: restart the launch with "
            "this config and take them again before --write"
        )
    prepared = [prepare(model, to_snapshot(s, args)) for s in snaps]
    if not prepared:
        print("no snapshots")
        return 1
    try:
        result = fit(model, prepared, free, exclude_fingers=args.exclude_fingers)
    except ValueError as e:
        print(f"no result: {e}")
        return 1
    report(result, args)

    text = args.config.read_text()
    new = updated_config(text, args.arm, free, result.params)
    arm = yaml.safe_load(new)["arms"][args.arm]
    print(f"\n{args.arm}: yaw {float(arm['yaw']):.4f}", end="")
    if arm.get("joint_offsets"):
        print(f", joint_offsets {arm['joint_offsets']}", end="")
    print()
    if "camera_yaw" in free:
        p = result.params
        fix = HorizontalCorrection(
            dx=p["camera_x"],
            dy=p["camera_y"],
            dyaw=p["camera_yaw"],
            pivot=result.pivot,
            residuals=np.zeros((0, 2)),
            dz=np.zeros(0),
        )
        mount = corrected_mount(
            snaps[0]["base_to_parent"], snaps[0]["parent_to_camera"], fix
        )
        rpy = matrix_rpy(mount)
        new = set_camera(new, mount[:3, 3], rpy)
        print(
            f"camera_joint: xyz {np.round(mount[:3, 3], 4).tolist()} "
            f"rpy {np.round(rpy, 4).tolist()}"
        )
    if args.write:
        if mismatch:
            print(
                f"not written: the snapshots' model differs from {args.config} at "
                f"{', '.join(mismatch)}"
            )
            return 1
        if not result.reliable and not args.force:
            print("not written: see the warnings (--force to write anyway)")
            return 1
        args.config.write_text(new)
        print(f"written to {args.config}; restart the launch to load it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
