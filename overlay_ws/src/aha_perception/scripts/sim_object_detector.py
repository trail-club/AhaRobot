#!/usr/bin/env python3
"""Simulation only: object detections from Gazebo ground-truth poses.

Publishes aha_msgs/DetectedObjectArray on output_topic for the Gazebo models
whose names match an entry of the object list (objects_config,
aha_perception.object_list) and that the head camera sees: the object center
or top face center projects into the depth image and the depth there is not
closer than that point (aha_perception.sim_object_detection.object_visible,
depth_tolerance). Model poses come from Gazebo's /world/<world>/pose/info
(gz-transport, GZ_PARTITION of the environment) relative to the robot model
(robot_model, its origin is base_footprint), so the detections are relative
to the robot like those of a camera detector. They are published in
output_frame (default odom) through the TF at the depth stamp (latest TF if
unavailable; base_frame if output_frame has no TF).

class_name is the label, track_id a number per model fixed for the node's
life, size the list's size, the pose the true pose plus Gaussian noise
(pose_noise, m, per axis and detection) with that variance in the
covariance. miss_rate / mislabel_rate drop / relabel (to another label of
the list) an object for as long as seed is unchanged (drawn per seed and
model name). The parameters can be changed at run time. Runs at most at rate
(Hz, wall clock) on the latest depth image.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from aha_msgs.msg import DetectedObject, DetectedObjectArray
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from aha_perception.object_list import load_object_list
from aha_perception.plane_detection_io import (
    depth_from_buffer,
    quaternion_from_matrix,
    transform_to_matrix,
)
from aha_perception.sim_object_detection import corruption, object_visible

NODE_NAME = "aha_perception_sim_object_detector"
WARN_PERIOD_S = 5.0
WORLD_TIMEOUT_S = 60.0
POSE_MAX_AGE_S = 2.0  # wall clock
DEFAULT_CONFIG = "config/objects/robocup_objects.yaml"


def tf_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class GazeboPoses:
    """Latest top-level model poses of a world from gz-transport.

    The subscription is throttled to ``rate`` messages per second and a
    message is only parsed when asked for (pose/info is large and frequent)."""

    def __init__(self, world: str, timeout: float, rate: float):
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node as GzNode
        from gz.transport13 import SubscribeOptions

        self.node = GzNode()
        self.world = world or self._find_world(timeout)
        self.lock = threading.Lock()
        self.msg = None
        self.stamp = 0.0
        options = SubscribeOptions()
        options.msgs_per_sec = max(int(rate), 1)
        topic = f"/world/{self.world}/pose/info"
        if not self.node.subscribe(Pose_V, topic, self._on_poses, options):
            raise RuntimeError(f"cannot subscribe to {topic}")

    def _find_world(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for topic in self.node.topic_list():
                parts = topic.split("/")
                if len(parts) == 4 and parts[1] == "world" and parts[3] == "clock":
                    return parts[2]
            time.sleep(0.5)
        raise RuntimeError("no Gazebo world found (check GZ_PARTITION)")

    def _on_poses(self, msg):
        with self.lock:
            self.msg = msg
            self.stamp = time.monotonic()

    def latest(self) -> tuple[dict, float]:
        """(world pose 4x4 by name, age in s of the message)."""
        with self.lock:
            msg, stamp = self.msg, self.stamp
        poses = {}
        for pose in msg.pose if msg is not None else ():
            p, q = pose.position, pose.orientation
            poses[pose.name] = transform_to_matrix(
                (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
            )
        return poses, time.monotonic() - stamp


class SimObjectDetector(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        share = Path(get_package_share_directory("aha_perception"))
        config = self.declare_parameter("objects_config", "").value or str(
            share / DEFAULT_CONFIG
        )
        self.objects = load_object_list(config)
        world = self.declare_parameter("world", "").value
        self.robot_model = self.declare_parameter("robot_model", "aha_robot").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.output_frame = self.declare_parameter("output_frame", "odom").value
        depth_topic = self.declare_parameter(
            "depth_topic", "/camera/depth_registered/image_rect"
        ).value
        info_topic = self.declare_parameter(
            "camera_info_topic", "/camera/color/camera_info"
        ).value
        output_topic = self.declare_parameter(
            "output_topic", "/aha/perception/objects"
        ).value
        rate = float(self.declare_parameter("rate", 2.0).value)
        self.declare_parameter("depth_tolerance", 0.03)
        self.declare_parameter("image_margin", 4.0)  # px
        self.declare_parameter("pose_noise", 0.01)
        self.declare_parameter("miss_rate", 0.0)
        self.declare_parameter("mislabel_rate", 0.0)
        self.declare_parameter("seed", 0)

        self.gz = GazeboPoses(world, WORLD_TIMEOUT_S, 2.0 * rate)
        self.track_ids = {}
        self.noise_seed = None
        self.rng = None
        self.camera_info = None
        self.depth_msg = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, depth_topic, self.on_depth, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(DetectedObjectArray, output_topic, 10)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self.on_timer)
        self.get_logger().info(
            f"world {self.gz.world}: {len(self.objects.objects)} labels from {config} "
            f"-> {output_topic} ({self.output_frame})"
        )

    def param(self, name):
        return self.get_parameter(name).value

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_depth(self, msg):
        self.depth_msg = msg

    def lookup(self, target, source, stamp):
        for when in (Time.from_msg(stamp), Time()):
            try:
                return tf_matrix(self.tf_buffer.lookup_transform(target, source, when))
            except TransformException:
                continue
        return None

    def on_timer(self):
        msg, info = self.depth_msg, self.camera_info
        if msg is None or info is None or info.k[0] <= 0.0:
            self.get_logger().warn(
                "waiting for depth and camera_info", throttle_duration_sec=WARN_PERIOD_S
            )
            return
        poses, age = self.gz.latest()
        world_base = poses.get(self.robot_model)
        if world_base is None or age > POSE_MAX_AGE_S:
            self.get_logger().warn(
                f"no recent Gazebo pose of {self.robot_model}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        stamp = msg.header.stamp
        base_cam = self.lookup(self.base_frame, msg.header.frame_id, stamp)
        if base_cam is None:
            self.get_logger().warn(
                f"no TF {self.base_frame} <- {msg.header.frame_id}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        frame = self.output_frame
        out_base = self.lookup(frame, self.base_frame, stamp)
        if out_base is None:
            self.get_logger().warn(
                f"no TF {frame} <- {self.base_frame}, publishing in {self.base_frame}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            frame, out_base = self.base_frame, np.eye(4)
        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        seed = int(self.param("seed"))
        if seed != self.noise_seed:
            self.noise_seed, self.rng = seed, np.random.default_rng(seed)
        sigma = max(float(self.param("pose_noise")), 0.0)
        tolerance = float(self.param("depth_tolerance"))
        margin = float(self.param("image_margin"))
        miss_rate = float(self.param("miss_rate"))
        mislabel_rate = float(self.param("mislabel_rate"))
        cam_base = np.linalg.inv(base_cam)
        base_world = np.linalg.inv(world_base)

        out = DetectedObjectArray()
        out.header.stamp = stamp
        out.header.frame_id = frame
        for name in sorted(poses):
            entry = self.objects.for_model(name)
            if entry is None:
                continue
            base_object = base_world @ poses[name]
            if not object_visible(
                depth, K, cam_base @ base_object, entry.size, tolerance, margin
            ):
                continue
            missed, label = corruption(
                seed, name, entry.label, self.objects.labels, miss_rate, mislabel_rate
            )
            if missed:
                continue
            pose = out_base @ base_object
            if sigma > 0.0:
                pose[:3, 3] += self.rng.normal(0.0, sigma, 3)
            det = DetectedObject()
            det.header = out.header
            det.class_name = label
            det.track_id = self.track_ids.setdefault(name, len(self.track_ids) + 1)
            det.confidence = 1.0
            p = det.pose.pose.position
            p.x, p.y, p.z = (float(v) for v in pose[:3, 3])
            o = det.pose.pose.orientation
            o.x, o.y, o.z, o.w = quaternion_from_matrix(pose[:3, :3])
            for k in range(3):
                det.pose.covariance[7 * k] = sigma * sigma
            det.size.x, det.size.y, det.size.z = entry.size
            out.objects.append(det)
        self.pub.publish(out)


def main():
    rclpy.init()
    try:
        node = SimObjectDetector()
    except (RuntimeError, ValueError, OSError) as e:
        rclpy.logging.get_logger(NODE_NAME).fatal(str(e))
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
