#!/usr/bin/env python3
"""Detect the objects of the object list in the head RGB-D camera.

Each processed color image goes to aha_perception.object_worker on the DGX
Spark: SAM 3.1 finds candidate masks for generic phrases and the Qwen3.8 VLM
names each candidate with a label of the object list (objects_config) or
none. The node measures each named mask in the aligned depth, transforms it
to `target_frame` at the image time, tracks the objects and publishes the
ones seen in the image as aha_msgs/DetectedObjectArray, the same topic and
meaning as sim_object_detector.py. One image is in flight at a time, so the
rate is the DGX round trip (seconds), capped by `max_rate`.
"""

import json
from pathlib import Path
import shlex
import threading
import time

from aha_msgs.msg import DetectedObject, DetectedObjectArray
from ament_index_python.packages import get_package_share_directory
import message_filters
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.dgx import (
    DgxWorker,
    RequestError,
    WorkerOptions,
    object_spec,
)
from aha_perception.object_detection import (
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_PROMPTS,
    ObjectDetection,
    ObjectTracker,
    annotate,
    measure_object,
    merge_detections,
    near_any,
    size_fits,
    vlm_timeout,
)
from aha_perception.object_list import load_object_list
from aha_perception.people import decode_mask, encode_image, rotation_matrix

OBJECTS_TOPIC = "/aha/perception/objects"
MARKERS_TOPIC = "/aha/perception/objects/markers"
IMAGE_TOPIC = "/aha/perception/objects/image"
DEFAULT_CONFIG = "config/objects/robocup_objects.yaml"
# Gripper links (fingers and wrists) that the head camera sees.
ROBOT_FRAMES = ("link_l6", "link_l7l", "link_l7r", "link_r6", "link_r7l", "link_r7r")


def image_array(msg):
    """Image message as a numpy array: rgb8 / bgr8 -> HxWx3 RGB, depth -> metres."""
    if msg.encoding in ("rgb8", "bgr8"):
        data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        rgb = data[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(rgb[..., ::-1] if msg.encoding == "bgr8" else rgb)
    if msg.encoding == "32FC1":
        data = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)
        return data[:, : msg.width]
    if msg.encoding in ("16UC1", "mono16"):
        data = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)
        depth = data[:, : msg.width].astype(np.float32) / 1000.0
        depth[depth == 0] = np.nan
        return depth
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ObjectPerception(Node):
    def __init__(self):
        super().__init__("object_perception")
        p = self.declare_parameter
        defaults = WorkerOptions()
        self.color_topic = p("color_topic", "/camera/color/image_raw").value
        self.depth_topic = p("depth_topic", "/camera/depth_registered/image_rect").value
        self.info_topic = p("info_topic", "/camera/color/camera_info").value
        # A fixed frame with z up (map / odom) keeps tracks still while the
        # robot moves; empty: the camera frame, without the hidden depth.
        self.target_frame = p("target_frame", "map").value
        config = p("objects_config", "").value or str(
            Path(get_package_share_directory("aha_perception")) / DEFAULT_CONFIG
        )
        self.objects = load_object_list(config)
        self.labels = [
            {"label": o.label, "category": o.category, "size": list(o.size)}
            for o in self.objects.objects
        ]
        self.prompts = list(p("prompts", list(DEFAULT_PROMPTS)).value)
        self.threshold = p("threshold", DEFAULT_CANDIDATE_THRESHOLD).value
        self.max_candidates = p("max_candidates", DEFAULT_MAX_CANDIDATES).value
        self.max_size = p("max_size", 0.6).value
        self.border_px = p("border_px", 2).value
        # The robot's grippers show at the bottom of the image; nothing within
        # robot_margin of these links is an object on a surface.
        self.robot_frames = list(p("robot_frames", list(ROBOT_FRAMES)).value)
        self.robot_margin = p("robot_margin", 0.12).value
        self.merge_distance = p("merge_distance", 0.05).value
        self.max_rate = p("max_rate", 1.0).value
        self.request_timeout = p("request_timeout", 120.0).value
        # The worker gives up on the VLM first: a slow VLM fails the image
        # instead of answering after the timeout while the next images queue.
        self.vlm_timeout = vlm_timeout(self.request_timeout, self.max_candidates)
        self.startup_timeout = p("startup_timeout", 1800.0).value
        self.tf_timeout = p("tf_timeout", 0.5).value
        self.options = WorkerOptions(
            host=p("ssh_host", defaults.host).value,
            port=p("ssh_port", defaults.port).value,
            user=p("ssh_user", defaults.user).value,
            identity_file=p("identity_file", defaults.identity_file).value,
            image=p("image", defaults.image).value,
            checkpoint=p("checkpoint", defaults.checkpoint).value,
            vlm_url=p("vlm_url", defaults.vlm_url).value,
            vlm_model=p("vlm_model", defaults.vlm_model).value,
        )
        # Non-empty: save each processed image with its candidates there.
        self.debug_dir = p("debug_dir", "").value
        # Tests and relays: a local argv that speaks the worker protocol.
        self.worker_command = shlex.split(p("worker_command", "").value)
        self.tracker = ObjectTracker(
            gate_m=p("gate", 0.08).value,
            forget_after=p("forget_after", 120.0).value,
        )

        self.frame_ready = threading.Condition()
        self.frame = None
        self.stopping = threading.Event()
        self.error = None
        self.worker = None
        self.last_stamp = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.objects_pub = self.create_publisher(DetectedObjectArray, OBJECTS_TOPIC, 10)
        self.marker_pub = self.create_publisher(MarkerArray, MARKERS_TOPIC, 10)
        self.image_pub = self.create_publisher(Image, IMAGE_TOPIC, 2)
        # Best effort loses most fragmented 640x480 images from the reliable
        # simulation bridge and rosbridge; a best-effort camera needs false.
        reliable = p("reliable", True).value
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE
            if reliable
            else ReliabilityPolicy.BEST_EFFORT,
        )
        subscribers = [
            message_filters.Subscriber(self, Image, topic, qos_profile=qos)
            for topic in (self.color_topic, self.depth_topic)
        ]
        subscribers.append(
            message_filters.Subscriber(
                self, CameraInfo, self.info_topic, qos_profile=qos
            )
        )
        # The camera publishes color, aligned depth and info with one stamp.
        self.sync = message_filters.TimeSynchronizer(subscribers, 10)
        self.sync.registerCallback(self.on_frame)
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    # -- Input ---------------------------------------------------------------

    def on_frame(self, color, depth, info):
        with self.frame_ready:
            self.frame = (color, depth, info)
            self.frame_ready.notify()

    def next_frame(self):
        with self.frame_ready:
            while not self.stopping.is_set():
                if self.frame is not None:
                    frame, self.frame = self.frame, None
                    return frame
                self.frame_ready.wait(0.5)
        return None

    # -- Processing ------------------------------------------------------------

    def process(self):
        try:
            self.worker = DgxWorker(
                self.options,
                spec=object_spec(self.options, self.vlm_timeout),
                log=lambda text: self.get_logger().info(text),
                command=self.worker_command or None,
            )
            info = self.worker.start(self.startup_timeout)
            if not info.get("vlm"):
                raise RuntimeError("the DGX object worker has no VLM")
            self.get_logger().info(
                f"DGX object worker ready ({len(self.labels)} labels, "
                f"{len(self.prompts)} phrases); publishing {OBJECTS_TOPIC} in frame "
                f"{self.target_frame or 'camera'}"
            )
            next_time = 0.0
            while not self.stopping.is_set():
                frame = self.next_frame()
                if frame is None:
                    break
                wait = next_time - time.monotonic()
                if wait > 0:
                    self.stopping.wait(wait)
                    with self.frame_ready:  # a newer frame may have arrived
                        frame = self.frame or frame
                        self.frame = None
                next_time = time.monotonic() + 1.0 / self.max_rate
                try:
                    self.process_frame(*frame)
                except (
                    TimeoutError,
                    ValueError,
                    TransformException,
                    RequestError,
                ) as exc:
                    self.get_logger().warning(
                        f"Skipped an image: {exc}", throttle_duration_sec=5.0
                    )
        except Exception as exc:
            if not self.stopping.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                self.get_logger().error(f"Object perception stopped: {self.error}")

    def process_frame(self, color, depth_msg, info):
        stamp = color.header.stamp
        seconds = stamp_seconds(stamp)
        if self.last_stamp is not None and seconds < self.last_stamp - 1.0:
            # The simulation restarted; old tracks belong to another episode.
            self.tracker.clear()
        self.last_stamp = seconds
        rgb, depth = image_array(color), image_array(depth_msg)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth is not aligned to the color image")
        # The camera and gripper poses at the image time, before the robot moves
        # on and before the DGX round trip outlasts the TF buffer.
        rotation, translation = self.camera_pose(color.header.frame_id, stamp)
        robot = self.robot_points(self.target_frame or color.header.frame_id, stamp)
        started = time.monotonic()
        response = self.worker.request(
            {
                "type": "detect",
                "image": encode_image(rgb),
                "prompts": self.prompts,
                "labels": self.labels,
                "threshold": float(self.threshold),
                "max_candidates": int(self.max_candidates),
                "border_px": int(self.border_px),
            },
            self.request_timeout,
        )
        round_trip = time.monotonic() - started
        if tuple(response["shape"]) != rgb.shape[:2]:
            raise ValueError("the DGX worker answered for another image size")
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        detections, debug = [], []
        for found in response["objects"]:
            mask = decode_mask(found, rgb.shape[:2])
            label = found.get("label")
            entry = self.objects.get(label) if label else None
            measured = None
            if entry is not None:
                measured = measure_object(
                    mask,
                    depth,
                    fx,
                    fy,
                    cx,
                    cy,
                    rotation=rotation if self.target_frame else None,
                    translation=translation,
                    footprint=min(entry.size[:2]),
                )
            reason = ""
            if measured is None:
                reason = "" if entry is None else ", no depth"
            elif max(measured.size) > self.max_size:
                reason = ", too large"  # e.g. a box mask over a shelf
            elif not size_fits(measured.size, entry.size):
                reason = f", too large for {label}"
            elif near_any(measured.centre, robot, self.robot_margin):
                reason = ", the robot"
            if reason:
                measured = None
            if measured is not None:
                detections.append(
                    ObjectDetection(
                        label,
                        measured.centre,
                        measured.size,
                        float(found["score"]),
                        [int(v) for v in found["bbox"]],
                    )
                )
            text = f"{label or 'none'} ({found['phrase']} {found['score']:.2f}{reason})"
            debug.append((mask, text, measured is not None))
        detections = merge_detections(detections, self.merge_distance)
        seen = self.tracker.update(detections, seconds)
        frame = self.target_frame or color.header.frame_id
        self.publish(stamp, frame, seen)
        if self.image_pub.get_subscription_count() or self.debug_dir:
            annotated = annotate(rgb, debug)
            self.publish_image(color.header, annotated)
            if self.debug_dir:
                self.save_debug(seconds, annotated, response, seen)
        timing = response.get("seconds", {})
        self.get_logger().info(
            f"{len(seen)} object(s) of {response.get('candidates', 0)} candidate(s) "
            f"in {round_trip:.1f} s (SAM 3.1 {timing.get('sam', 0.0):.1f} s, "
            f"VLM {timing.get('vlm', 0.0):.1f} s): "
            + ", ".join(f"{t.id} {t.label}" for t in seen)
        )

    def robot_points(self, frame, stamp):
        """Positions of robot_frames in `frame` at the image time; frames without
        TF are left out. Waits at most tf_timeout in all for the TF, and warns
        when none of them has one: then the grippers are not excluded."""
        points = []
        deadline = time.monotonic() + self.tf_timeout
        for link in self.robot_frames:
            try:
                t = self.tf_buffer.lookup_transform(
                    frame,
                    link,
                    Time.from_msg(stamp),
                    timeout=Duration(seconds=max(0.0, deadline - time.monotonic())),
                ).transform.translation
            except TransformException:
                continue
            points.append(np.array([t.x, t.y, t.z]))
        if self.robot_frames and not points:
            self.get_logger().warning(
                f"No TF {frame} <- {', '.join(self.robot_frames)} at the image "
                "time; the robot's grippers are not excluded from the objects",
                throttle_duration_sec=10.0,
            )
        return points

    def camera_pose(self, camera_frame, stamp):
        if not self.target_frame or self.target_frame == camera_frame:
            return np.eye(3), np.zeros(3)
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            camera_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=self.tf_timeout),
        ).transform
        q, t = transform.rotation, transform.translation
        return rotation_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    # -- Output ----------------------------------------------------------------

    def publish(self, stamp, frame, seen):
        message = DetectedObjectArray()
        message.header.stamp = stamp
        message.header.frame_id = frame
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for track in seen:
            detected = DetectedObject()
            detected.header = message.header
            detected.class_name = track.label
            detected.track_id = track.id
            detected.confidence = float(track.score * track.agreement)
            position = detected.pose.pose.position
            position.x, position.y, position.z = (float(v) for v in track.centre)
            # A camera detector has no orientation: the box is axis-aligned.
            detected.pose.pose.orientation.w = 1.0
            detected.size.x, detected.size.y, detected.size.z = (
                float(v) for v in track.size
            )
            message.objects.append(detected)
            markers.markers.extend(self.track_markers(track))
        for marker in markers.markers:
            marker.header.frame_id = frame
            marker.header.stamp = stamp
        self.objects_pub.publish(message)
        self.marker_pub.publish(markers)

    @staticmethod
    def track_markers(track):
        box = Marker(ns="objects", id=track.id, type=Marker.CUBE)
        box.pose.position.x, box.pose.position.y, box.pose.position.z = (
            float(v) for v in track.centre
        )
        box.pose.orientation.w = 1.0
        box.scale.x, box.scale.y, box.scale.z = (float(v) for v in track.size)
        box.color.r, box.color.g, box.color.b, box.color.a = 0.1, 0.6, 1.0, 0.6
        label = Marker(ns="labels", id=track.id, type=Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y = (
            box.pose.position.x,
            (box.pose.position.y),
        )
        label.pose.position.z = float(track.centre[2] + track.size[2] / 2 + 0.05)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.06
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"{track.id} {track.label}"
        return [box, label]

    def publish_image(self, header, rgb):
        image = Image()
        image.header = header
        image.height, image.width = rgb.shape[:2]
        image.encoding = "rgb8"
        image.step = image.width * 3
        image.data = np.ascontiguousarray(rgb).tobytes()
        self.image_pub.publish(image)

    def save_debug(self, seconds, annotated, response, seen):
        folder = Path(self.debug_dir)
        name = f"{seconds:.2f}".replace(".", "_")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            PilImage.fromarray(annotated).save(folder / f"{name}.jpg", quality=90)
            record = {
                "stamp": seconds,
                "seconds": response.get("seconds", {}),
                "answers": response.get("answers", []),
                "candidates": [
                    {k: found[k] for k in ("label", "phrase", "score", "bbox")}
                    for found in response["objects"]
                ],
                "objects": [
                    {
                        "id": t.id,
                        "label": t.label,
                        "centre": [round(float(v), 4) for v in t.centre],
                        "size": [round(float(v), 4) for v in t.size],
                        "votes": t.votes,
                    }
                    for t in seen
                ],
            }
            (folder / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
        except OSError as exc:
            self.get_logger().warning(f"Cannot save debug images: {exc}")

    def stop(self):
        self.stopping.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        self.thread.join(timeout=5)
        if self.worker is not None and not self.worker.close():
            self.get_logger().error(
                "DGX cleanup was incomplete; see the messages above"
            )


def main():
    rclpy.init()
    try:
        node = ObjectPerception()
    except (RuntimeError, ValueError, OSError) as exc:
        rclpy.logging.get_logger("object_perception").fatal(str(exc))
        rclpy.try_shutdown()
        return 1
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    code = 0
    try:
        while rclpy.ok() and node.error is None:
            executor.spin_once(timeout_sec=0.2)
        code = 1 if node.error else 0
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
