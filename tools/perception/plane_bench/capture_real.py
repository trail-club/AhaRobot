#!/usr/bin/env python3
"""Capture real head-camera depth frames for offline plane detection.

Subscribes the registered depth, color, CameraInfo and /joint_states, looks up
TF target_frame <- depth frame at each image stamp (latest TF as fallback) and
writes a scene directory with real_frames.save_scene. Run it in the ROS
container with aha_perception sourced; see README.md.
"""

import argparse
import datetime
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener

from aha_perception.head_config import PAN_JOINT, TILT_JOINT
from aha_perception.plane_detection_io import depth_from_buffer, transform_to_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from real_frames import save_scene, scene_dir_name  # noqa: E402

DEPTH_TOPIC = "/camera/depth_registered/image_rect"
COLOR_TOPIC = "/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
JOINT_STATES_TOPIC = "/joint_states"

TF_TIMEOUT = Duration(seconds=0.2)
# Wait this long for /joint_states before capturing without head joints.
JOINT_WAIT_S = 2.0
JPEG_QUALITY = 90
WARN_PERIOD_S = 5.0


def color_to_rgb(msg):
    """rgb8 / bgr8 Image to an (H, W, 3) uint8 RGB array."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported color encoding {msg.encoding!r}")
    rows = np.frombuffer(msg.data, dtype=np.uint8, count=msg.height * msg.step)
    image = rows.reshape(msg.height, msg.step)[:, : msg.width * 3]
    image = image.reshape(msg.height, msg.width, 3)
    return np.ascontiguousarray(image[..., ::-1] if msg.encoding == "bgr8" else image)


def encode_jpeg(rgb):
    """JPEG bytes with cv2 or PIL; None when neither is available."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        ok, buffer = cv2.imencode(
            ".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buffer.tobytes()
    try:
        import io

        from PIL import Image as PILImage
    except ImportError:
        return None
    out = io.BytesIO()
    PILImage.fromarray(rgb).save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Capture(Node):
    def __init__(self, target_frame):
        super().__init__("aha_plane_bench_capture")
        self.target_frame = target_frame
        self.tf_buffer = Buffer()
        # A separate node spun in its own thread fills the buffer while the main
        # loop blocks in lookup_transform(timeout=...).
        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.info = None
        self.depth = None
        self.colors = deque(maxlen=10)
        self.joints = {}
        qos = qos_profile_sensor_data
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, qos)
        self.create_subscription(Image, COLOR_TOPIC, self.on_color, qos)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.on_info, qos)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.on_joints, qos)

    def close(self):
        # Stop the listener's executor thread before rclpy shuts down, or it
        # raises InvalidHandle on the destroyed subscriptions.
        self.tf_listener.executor.shutdown()
        self.tf_listener.dedicated_listener_thread.join()
        self.destroy_node()

    def on_depth(self, msg):
        self.depth = msg

    def on_color(self, msg):
        self.colors.append(msg)

    def on_info(self, msg):
        self.info = msg

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def lookup(self, frame, stamp):
        """(4x4 camera_to_base, used the latest TF) or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, frame, Time.from_msg(stamp), timeout=TF_TIMEOUT
            )
            latest = False
        except TransformException as e:
            try:
                tf = self.tf_buffer.lookup_transform(self.target_frame, frame, Time())
            except TransformException:
                self.get_logger().warn(
                    f"TF {self.target_frame} <- {frame} unavailable: {e}",
                    throttle_duration_sec=WARN_PERIOD_S,
                )
                return None
            self.get_logger().warn(
                f"TF at the image stamp unavailable, using the latest: {e}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            latest = True
        t, q = tf.transform.translation, tf.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w)), latest

    def nearest_color(self, stamp):
        target = stamp_seconds(stamp)
        colors = list(self.colors)
        if not colors:
            return None
        return min(colors, key=lambda m: abs(stamp_seconds(m.header.stamp) - target))

    def grab(self, msg):
        info = self.info
        if info is None:
            self.get_logger().warn(
                "waiting for camera_info", throttle_duration_sec=WARN_PERIOD_S
            )
            return None
        if (info.width, info.height) != (msg.width, msg.height):
            self.get_logger().warn(
                f"depth {msg.width}x{msg.height} does not match camera_info "
                f"{info.width}x{info.height}, skipped",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        result = self.lookup(msg.header.frame_id, msg.header.stamp)
        if result is None:
            return None
        camera_to_base, latest_tf = result
        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        joints = [self.joints.get(name, np.nan) for name in (PAN_JOINT, TILT_JOINT)]
        return {
            "msg": msg,
            "depth": depth,
            "camera_to_base": camera_to_base,
            "latest_tf": latest_tf,
            "stamp": stamp_seconds(msg.header.stamp),
            "head_joints": joints,
            "info": info,
        }

    def collect(self, count, interval, timeout):
        frames = []
        started = time.monotonic()
        deadline = started + timeout
        next_grab = 0.0
        last_stamp = None
        while len(frames) < count and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            msg = self.depth
            if msg is None or time.monotonic() < next_grab:
                continue
            waiting = time.monotonic() < started + JOINT_WAIT_S
            if waiting and (not self.joints or not self.colors):
                continue
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            if frames and (msg.width, msg.height) != frames[0]["depth"].shape[::-1]:
                continue
            frame = self.grab(msg)
            if frame is None:
                continue
            if not frames:
                frame["color"] = self.nearest_color(msg.header.stamp)
            frames.append(frame)
            next_grab = time.monotonic() + interval
            self.get_logger().info(f"frame {len(frames)}/{count}")
        return frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="scene name in the dir name")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds")
    parser.add_argument("--out-dir", type=Path, default=HERE / "real_frames")
    parser.add_argument("--target-frame", default="base_footprint")
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--base-height",
        type=float,
        default=None,
        help="height of the target_frame origin above the floor (m); "
        "unset when unknown",
    )
    parser.add_argument(
        "--filter-label", default="", help="depth filtering, e.g. filtered / raw"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="give up after seconds"
    )
    args = parser.parse_args(argv)
    if args.frames < 1 or args.interval < 0.0 or args.timeout <= 0.0:
        parser.error("--frames >= 1, --interval >= 0 and --timeout > 0 required")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = datetime.datetime.now().astimezone()
    scene_dir = args.out_dir / scene_dir_name(args.scene, started)
    if scene_dir.exists():
        print(f"error: {scene_dir} already exists", file=sys.stderr)
        return 1

    rclpy.init()
    node = None
    frames, ros_time = [], 0.0
    try:
        node = Capture(args.target_frame)
        frames = node.collect(args.frames, args.interval, args.timeout)
        ros_time = node.get_clock().now().nanoseconds * 1e-9
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        rclpy.try_shutdown()

    if not frames:
        print("error: no frame captured", file=sys.stderr)
        return 1
    if len(frames) < args.frames:
        print(
            f"warning: only {len(frames)}/{args.frames} frames within the timeout",
            file=sys.stderr,
        )

    first = frames[0]
    color_jpeg = None
    if first["color"] is None:
        print("warning: no color image received, color.jpg skipped", file=sys.stderr)
    else:
        color_jpeg = encode_jpeg(color_to_rgb(first["color"]))
        if color_jpeg is None:
            print("warning: neither cv2 nor PIL, color.jpg skipped", file=sys.stderr)

    info = first["info"]
    meta = {
        "scene": args.scene,
        "note": args.note,
        "filter_label": args.filter_label,
        "target_frame": args.target_frame,
        "depth_frame": first["msg"].header.frame_id,
        "depth_encoding": first["msg"].encoding,
        "base_height": args.base_height,
        "ros_time": ros_time,
        "captured_at": started.isoformat(timespec="seconds"),
        "interval": args.interval,
        "topics": {
            "depth": DEPTH_TOPIC,
            "color": COLOR_TOPIC,
            "camera_info": CAMERA_INFO_TOPIC,
            "joint_states": JOINT_STATES_TOPIC,
        },
        "distortion_model": info.distortion_model,
        "d": [float(v) for v in info.d],
        "latest_tf_frames": [i for i, f in enumerate(frames) if f["latest_tf"]],
    }
    save_scene(
        scene_dir,
        depth=np.stack([f["depth"] for f in frames]),
        camera_to_base=np.stack([f["camera_to_base"] for f in frames]),
        stamps=np.array([f["stamp"] for f in frames]),
        k=np.array(info.k, dtype=np.float64),
        meta=meta,
        head_joints=np.array([f["head_joints"] for f in frames], dtype=np.float64),
        head_joint_names=(PAN_JOINT, TILT_JOINT),
        color_jpeg=color_jpeg,
    )
    size = sum(p.stat().st_size for p in scene_dir.iterdir()) / 1e6
    print(f"wrote {len(frames)} frames to {scene_dir} ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
