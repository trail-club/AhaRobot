#!/usr/bin/env python3
"""Accumulate the head camera cloud over time and head poses in a voxel map.

Subscribes the depth image registered to color (32FC1 m, or 16UC1 mm), the
color image (rgb8 / bgr8; the one nearest to the depth stamp within
color_tolerance, waiting up to COLOR_WAIT_S for one at or after it; gray
without one: the sim bridge does not always give both the same stamp) and the
CameraInfo. Each depth frame (every stride-th row and column, the latest
only, at most max_rate per second on the wall clock, in a worker thread)
updates the aha_perception.cloud_map.VoxelMap in fixed_frame:

1. Camera pose: TF fixed_frame <- camera at the image stamp, waiting up to
   TF_TIMEOUT; the frame is skipped without it (no fallback to the latest
   TF: a stale head or arm pose would smear the map). If fixed_frame has
   never had that TF, the map is kept in base_frame with a warning and reset
   when fixed_frame appears.
2. Base check: the map is cleared when base_frame has moved more than
   reset_translation / reset_rotation_deg in fixed_frame since the first
   frame after the last reset (odometry drift would smear the map), and
   voxels not observed for max_age seconds of image time are forgotten
   (0: never).
3. Still check: the camera pose at stamp - settle_time must be within
   still_translation / still_rotation_deg of the pose at the stamp (TF comes
   from /joint_states and the wheel odometry, so this catches a moving head,
   lift or base). Frames while it moves, or without TF that far back, stop
   here.
4. View test: stored voxels the camera now sees through are removed:
   aha_perception.surfaces.CameraView.states ABSENT (measured depth farther
   than the voxel by max(clear_tolerance, 3 sigma) of the D435i noise model),
   against the smallest valid depth of each 3x3 neighborhood (no range cut,
   so a far background still clears). Voxels out of view, nearer than
   min_range, occluded or without a depth return are kept.
5. Insert: pixels with depth in [min_range, max_range], voxelized, cropped
   to crop_min / crop_max in base_frame, minus the robot (self_filter: the
   aha_perception.self_filter link boxes of /robot_description grown by
   self_padding, at their TF at the image stamp; insertion waits for
   /robot_description and for every link's TF). Only new points are
   filtered: a stored voxel the arm moves into stays until the view test or
   max_age removes it. Objects held in the gripper are not filtered.

Output, at most publish_rate per second, stamped with the last frame:
  /aha/perception/merged_points  sensor_msgs/PointCloud2 (x, y, z, rgb) in
      the map frame: the voxel centers seen in at least min_hits frames
Service:
  /aha/perception/merged_points/clear  std_srvs/Trigger: empties the map

The map and the TF buffer are cleared when a /tf stamp goes back (sim restart
or reset); a frame being processed then, or during a clear, is dropped.
Statistics are logged every SUMMARY_PERIOD_S.
"""

import sys
import threading
import time
from collections import Counter, OrderedDict

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException

from aha_perception.cloud_map import (
    VoxelMap,
    in_box,
    key_centers,
    min_filter_depth,
    pose_change,
    transform_points,
    voxelize,
)
from aha_perception.plane_detection_io import (
    POINT_DTYPE,
    depth_from_buffer,
    depth_to_cloud,
    stamp_to_seconds,
    transform_to_matrix,
    xyzrgb_records,
)
from aha_perception.planes import DepthNoiseModel
from aha_perception.self_filter import link_boxes, points_in_boxes
from aha_perception.surfaces import ABSENT, CameraView, TrackerOptions
from aha_perception.tf_jump import (
    TF_RESET_MIN_INTERVAL_S,
    JumpDetectingTransformListener,
)

NODE_NAME = "aha_perception_cloud_accumulator"
CLOUD_TOPIC = "/aha/perception/merged_points"
CLEAR_SERVICE = "/aha/perception/merged_points/clear"

TF_TIMEOUT = Duration(seconds=0.2)
GRAY = (128, 128, 128)
COLOR_BUFFER = 30  # color images kept to match the depth stamps
COLOR_WAIT_S = 0.2  # wait for the color image of a depth frame (wall clock)
WARN_PERIOD_S = 5.0
SUMMARY_PERIOD_S = 10.0
WORKER_JOIN_S = 3.0
EXECUTOR_THREADS = 2  # the clear service beside the subscriptions and TF

CLOUD_FIELDS = [
    PointField(
        name=name,
        offset=POINT_DTYPE.fields[name][1],
        datatype=PointField.FLOAT32,
        count=1,
    )
    for name in POINT_DTYPE.names
]


def make_cloud(header, xyz, rgb):
    records = xyzrgb_records(xyz, rgb)
    return PointCloud2(
        header=header,
        height=1,
        width=len(records),
        fields=CLOUD_FIELDS,
        is_bigendian=False,
        point_step=POINT_DTYPE.itemsize,
        row_step=POINT_DTYPE.itemsize * len(records),
        data=records.tobytes(),
        is_dense=True,
    )


def color_from_msg(msg, stride):
    """(H, W, 3) uint8 RGB of every stride-th row and column, or None."""
    if msg.encoding not in ("rgb8", "bgr8"):
        return None
    rows = np.frombuffer(msg.data, dtype=np.uint8, count=msg.height * msg.step)
    image = rows.reshape(msg.height, msg.step)[:, : msg.width * 3]
    image = image.reshape(msg.height, msg.width, 3)[::stride, ::stride]
    return image[..., ::-1] if msg.encoding == "bgr8" else image


def stamp_key(stamp):
    return stamp.sec, stamp.nanosec


def nearest_color(colors, stamp, tolerance):
    """The message of colors (stamp key -> Image) nearest to stamp, or None."""
    target = stamp_to_seconds(stamp)
    best, best_dt = None, tolerance
    for (sec, nanosec), msg in colors.items():
        dt = abs(sec + nanosec * 1e-9 - target)
        if dt <= best_dt:
            best, best_dt = msg, dt
    return best


def matrix_of(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class CloudAccumulator(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        p = self.declare_parameter
        depth_topic = p("depth_topic", "/camera/depth_registered/image_rect").value
        color_topic = p("color_topic", "/camera/color/image_raw").value
        info_topic = p("camera_info_topic", "/camera/color/camera_info").value
        self.color_tolerance = float(p("color_tolerance", 0.1).value)
        self.fixed_frame = p("fixed_frame", "odom").value
        self.base_frame = p("base_frame", "base_footprint").value
        max_rate = float(p("max_rate", 5.0).value)
        self.period = 1.0 / max_rate if max_rate > 0.0 else 0.0
        publish_rate = float(p("publish_rate", 2.0).value)
        self.publish_period = 1.0 / publish_rate if publish_rate > 0.0 else 0.0
        self.stride = int(p("stride", 2).value)
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        voxel_size = float(p("voxel_size", 0.01).value)
        self.min_range = float(p("min_range", 0.3).value)
        self.max_range = float(p("max_range", 2.0).value)
        self.crop_min = np.array(p("crop_min", [-0.5, -1.5, -0.05]).value, float)
        self.crop_max = np.array(p("crop_max", [2.0, 1.5, 2.2]).value, float)
        if self.crop_min.shape != (3,) or self.crop_max.shape != (3,):
            raise ValueError("crop_min and crop_max need 3 values")
        self.self_filter = bool(p("self_filter", True).value)
        self.self_padding = float(p("self_padding", 0.02).value)
        self.settle_time = float(p("settle_time", 0.3).value)
        self.still_translation = float(p("still_translation", 0.005).value)
        self.still_rotation = np.radians(float(p("still_rotation_deg", 0.5).value))
        self.reset_translation = float(p("reset_translation", 0.05).value)
        self.reset_rotation = np.radians(float(p("reset_rotation_deg", 3.0).value))
        self.max_age = float(p("max_age", 60.0).value)
        self.min_hits = int(p("min_hits", 2).value)
        clear_tolerance = float(p("clear_tolerance", 0.03).value)
        self.noise_subpixel = float(p("noise_subpixel", 0.12).value)
        self.view_options = TrackerOptions(occlusion_tol=clear_tolerance)

        self.map = VoxelMap(voxel_size)
        self.map_lock = threading.Lock()
        self.map_frame = None
        self.anchor = None  # base_frame pose in the map frame after the reset
        self.last_stamp = None
        self.last_publish = -float("inf")
        # Incremented by the clear service and on_tf_jump: a frame being
        # processed before is dropped.
        self.epoch = 0
        self.boxes = None
        self.stats = Counter()

        self.tf_buffer = Buffer()
        self.tf_listener = JumpDetectingTransformListener(
            self.tf_buffer, self, self.on_tf_jump
        )
        self.last_tf_reset = -float("inf")
        self.camera_info = None
        self.colors_cond = threading.Condition()
        self.colors = OrderedDict()
        self.color_seen = False

        self.cond = threading.Condition()
        self.pending = None
        self.stopping = False

        self.create_subscription(
            CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, depth_topic, self.on_depth, qos_profile_sensor_data
        )
        if color_topic:
            self.create_subscription(
                Image, color_topic, self.on_color, qos_profile_sensor_data
            )
        if self.self_filter:
            self.create_subscription(
                String,
                "/robot_description",
                self.on_robot_description,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            CLOUD_TOPIC,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.create_service(
            Trigger,
            CLEAR_SERVICE,
            self.on_clear,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_timer(SUMMARY_PERIOD_S, self.log_summary)

        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        rate = f"{max_rate:g} Hz" if max_rate > 0.0 else "unlimited rate"
        self.get_logger().info(
            f"{depth_topic} -> {CLOUD_TOPIC} in {self.fixed_frame}, {rate}, "
            f"stride {self.stride}, voxel {voxel_size:g} m, range "
            f"{self.min_range:g}-{self.max_range:g} m, self filter "
            f"{'on' if self.self_filter else 'off'}, max age {self.max_age:g} s, "
            f"min hits {self.min_hits}"
        )

    # Callbacks

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_color(self, msg):
        with self.colors_cond:
            self.colors[stamp_key(msg.header.stamp)] = msg
            self.color_seen = True
            while len(self.colors) > COLOR_BUFFER:
                self.colors.popitem(last=False)
            self.colors_cond.notify_all()

    def on_depth(self, msg):
        with self.cond:
            self.pending = msg
            self.cond.notify()

    def on_robot_description(self, msg):
        boxes, warnings = link_boxes(msg.data, self.self_padding)
        for warning in warnings:
            self.get_logger().warn(f"self filter: {warning}")
        self.boxes = boxes
        self.get_logger().info(f"self filter: {len(boxes)} link boxes")

    def on_clear(self, request, response):
        with self.map_lock:
            count = len(self.map)
            self.reset_map()
            self.epoch += 1
        self.publish(force=True)
        response.success = True
        response.message = f"cleared {count} voxels"
        return response

    def on_tf_jump(self):
        if time.monotonic() - self.last_tf_reset < TF_RESET_MIN_INTERVAL_S:
            return False
        self.tf_buffer.clear()
        self.tf_listener.forget()
        self.last_tf_reset = time.monotonic()
        with self.cond:
            self.pending = None
        with self.colors_cond:
            self.colors.clear()
        with self.map_lock:
            self.reset_map()
            self.epoch += 1
        self.get_logger().info(
            "/tf time went back (sim restart or reset): cleared the TF buffer "
            "and the map"
        )
        return True

    def reset_map(self):
        """Empty the map (map_lock held); the next frame sets the anchor."""
        self.map.clear()
        self.anchor = None

    # Worker

    def stop(self):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        self.worker.join(timeout=WORKER_JOIN_S)

    def run_worker(self):
        next_start = 0.0
        while True:
            with self.cond:
                while not self.stopping:
                    wait = next_start - time.monotonic()
                    if self.pending is not None and wait <= 0.0:
                        break
                    self.cond.wait(None if self.pending is None else wait)
                if self.stopping:
                    return
                msg, self.pending = self.pending, None
            next_start = time.monotonic() + self.period
            try:
                self.process(msg)
            except Exception as e:  # keep the worker alive
                if self.stopping or not rclpy.ok():
                    return
                self.get_logger().error(
                    f"cloud accumulation failed: {e!r}",
                    throttle_duration_sec=WARN_PERIOD_S,
                )
            self.publish()

    def lookup(self, target, source, stamp, timeout=Duration()):
        """4x4 source -> target at stamp, or None."""
        if target == source:
            return np.eye(4)
        try:
            tf = self.tf_buffer.lookup_transform(target, source, stamp, timeout=timeout)
        except TransformException:
            return None
        return matrix_of(tf)

    def camera_pose(self, camera_frame, stamp):
        """(map frame, camera -> map) at stamp, or None to skip the frame."""
        when = Time.from_msg(stamp)
        pose = self.lookup(self.fixed_frame, camera_frame, when, TF_TIMEOUT)
        if pose is not None:
            return self.fixed_frame, pose
        if self.map_frame == self.fixed_frame:
            self.stats["no tf"] += 1
            self.get_logger().warn(
                f"TF {self.fixed_frame} <- {camera_frame} at the image stamp "
                "unavailable, frame skipped",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        pose = self.lookup(self.base_frame, camera_frame, when)
        if pose is None:
            self.stats["no tf"] += 1
            self.get_logger().warn(
                f"TF {self.base_frame} <- {camera_frame} at the image stamp "
                "unavailable, frame skipped",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        self.get_logger().warn(
            f"TF {self.fixed_frame} unavailable, accumulating in "
            f"{self.base_frame} until it is (the base must not move)",
            once=True,
        )
        return self.base_frame, pose

    def is_still(self, frame, camera_frame, stamp, pose):
        if stamp_to_seconds(stamp) < self.settle_time:
            return False
        before = Time.from_msg(stamp) - Duration(seconds=self.settle_time)
        earlier = self.lookup(frame, camera_frame, before)
        if earlier is None:
            return False
        translation, rotation = pose_change(earlier, pose)
        return translation <= self.still_translation and rotation <= self.still_rotation

    def color_for(self, stamp, shape):
        """Color image nearest to the depth stamp within color_tolerance, else gray.

        Waits up to COLOR_WAIT_S for a color image at or after the stamp.
        """
        key = stamp_key(stamp)
        with self.colors_cond:
            if self.color_seen:
                self.colors_cond.wait_for(
                    lambda: self.stopping or any(k >= key for k in self.colors),
                    timeout=COLOR_WAIT_S,
                )
            msg = nearest_color(self.colors, stamp, self.color_tolerance)
        color = color_from_msg(msg, self.stride) if msg is not None else None
        if color is None or color.shape[:2] != shape:
            self.stats["no color"] += 1
            return np.broadcast_to(np.array(GRAY, np.uint8), shape + (3,))
        return color

    def robot_mask(self, frame, centers, stamp):
        """(N,) bool of the points on the robot, or None if it cannot be known."""
        if not self.self_filter:
            return np.zeros(len(centers), dtype=bool)
        boxes = self.boxes
        if boxes is None:
            self.get_logger().warn(
                "self filter: waiting for /robot_description, not inserting",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        when = Time.from_msg(stamp)
        poses = []
        for box in boxes:
            pose = self.lookup(frame, box.link, when)
            if pose is None:
                self.get_logger().warn(
                    f"self filter: TF {frame} <- {box.link} at the image stamp "
                    "unavailable, not inserting",
                    throttle_duration_sec=WARN_PERIOD_S,
                )
                return None
            poses.append(pose)
        return points_in_boxes(centers, boxes, poses)

    def update_base(self, epoch, frame, base_to_map, stamp):
        """Map frame, base motion reset and forgetting, for every located frame.

        Runs before the still check, so the map is cleared while the base
        drives. False if a reset since ``epoch`` drops the frame.
        """
        seconds = stamp_to_seconds(stamp)
        with self.map_lock:
            if self.epoch != epoch:
                return False
            if frame != self.map_frame:
                if self.map_frame is not None:
                    self.get_logger().info(
                        f"accumulating in {frame} (was {self.map_frame}), map reset"
                    )
                self.reset_map()
                self.map_frame = frame
            if self.anchor is None:
                self.anchor = base_to_map
            else:
                translation, rotation = pose_change(self.anchor, base_to_map)
                if (
                    translation > self.reset_translation
                    or rotation > self.reset_rotation
                ):
                    self.get_logger().info(
                        f"{self.base_frame} moved {translation:.2f} m / "
                        f"{np.degrees(rotation):.1f} deg, map reset"
                    )
                    self.reset_map()
                    self.anchor = base_to_map
            self.map.forget_older(seconds, self.max_age)
            self.last_stamp = stamp
        return True

    def process(self, msg):
        info = self.camera_info
        if info is None:
            self.get_logger().warn(
                "waiting for CameraInfo", throttle_duration_sec=WARN_PERIOD_S
            )
            return
        with self.map_lock:
            epoch = self.epoch
        stamp = msg.header.stamp
        camera_frame = msg.header.frame_id
        located = self.camera_pose(camera_frame, stamp)
        if located is None:
            return
        frame, camera_to_map = located
        base_to_map = self.lookup(frame, self.base_frame, Time.from_msg(stamp))
        if base_to_map is None:
            self.stats["no tf"] += 1
            return
        if not self.update_base(epoch, frame, base_to_map, stamp):
            self.stats["reset"] += 1
            return
        if not self.is_still(frame, camera_frame, stamp, camera_to_map):
            self.stats["moving"] += 1
            return

        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        cloud = depth_to_cloud(depth, fx, fy, cx, cy, self.stride)
        z = cloud[..., 2]
        color = self.color_for(stamp, z.shape)
        view = CameraView(
            camera_to_tracking=camera_to_map,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=info.width,
            height=info.height,
            min_depth=self.min_range,
            max_depth=float("inf"),
            depth=min_filter_depth(z),
            stride=self.stride,
        )
        noise = DepthNoiseModel(focal=fx, subpixel=self.noise_subpixel)

        valid = np.isfinite(z) & (z >= self.min_range) & (z <= self.max_range)
        points = transform_points(camera_to_map, cloud[valid])
        keys, rgb = voxelize(points, color[valid], self.map.voxel_size)
        centers = key_centers(keys, self.map.voxel_size)
        inside = in_box(
            transform_points(np.linalg.inv(base_to_map), centers),
            self.crop_min,
            self.crop_max,
        )
        keys, rgb, centers = keys[inside], rgb[inside], centers[inside]
        on_robot = self.robot_mask(frame, centers, stamp)

        seconds = stamp_to_seconds(stamp)
        with self.map_lock:
            if self.epoch != epoch or frame != self.map_frame:
                self.stats["reset"] += 1
                return
            if len(self.map):
                states = view.states(self.map.centers(), noise, self.view_options)
                self.map.keep(states != ABSENT)
            if on_robot is not None:
                self.map.insert(keys[~on_robot], rgb[~on_robot], seconds)
                self.stats["inserted"] += 1
            else:
                self.stats["not inserted"] += 1

    def publish(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < self.publish_period:
            return
        with self.map_lock:
            if self.map_frame is None or self.last_stamp is None:
                return
            xyz, rgb = self.map.output(self.min_hits)
            header = Header(stamp=self.last_stamp, frame_id=self.map_frame)
        self.last_publish = now
        self.cloud_pub.publish(make_cloud(header, xyz, rgb))

    def log_summary(self):
        with self.map_lock:
            voxels = len(self.map)
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.stats.items()))
        self.get_logger().info(
            f"{voxels} voxels in {self.map_frame}; frames: {counts or 'none'}"
        )
        self.stats.clear()


def main():
    rclpy.init()
    try:
        node = CloudAccumulator()
    except ValueError as e:
        rclpy.logging.get_logger(NODE_NAME).fatal(str(e))
        rclpy.try_shutdown()
        return 1
    executor = MultiThreadedExecutor(num_threads=EXECUTOR_THREADS)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
