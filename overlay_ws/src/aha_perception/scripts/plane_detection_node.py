#!/usr/bin/env python3
"""Detect planes (floor, table, shelf, wall) in the head depth image.

Subscribes the depth image registered to color (32FC1 m, or 16UC1 mm) and its
CameraInfo, back-projects it to an organized cloud, transforms it to
target_frame (origin on the floor, z up; base_footprint) with the TF at the
image stamp and runs aha_perception.planes.detect_planes (params merge and
DepthNoiseModel subpixel from the ROS parameters). Only horizontal
planes and walls are published (aha_perception.planes.select_output_planes);
oblique planes and small vertical planes are dropped.

The noise model is DepthNoiseModel(focal=fx of CameraInfo,
subpixel=noise_subpixel) with the default baseline; the detector params are
the defaults except merge (warned and ignored for a method without it). The
node defaults (merge on, subpixel 0.12) were tuned on calibrated D435i noise
and differ from the library defaults. If the TF at the image stamp is not
available within TF_TIMEOUT, the latest TF is used with a warning. The method,
merge and subpixel are logged at startup, and each detection as
"3 planes (floor 1, table 1, wall 1) of 14 detected".

stride N passes every N-th row and column to the detector (pixel coordinates
stay those of the full image). The detector's own RANSAC decimation follows
(params_for_input_stride: params.stride / N, at least 1) and so do the place
options (PlaceOptions.for_input_stride). Pixel-count thresholds such as
min_inliers are not scaled: scaling them by 1 / N^2 fragmented the planes in
the synthetic benchmark.

Only the latest depth frame is kept. Detection runs in a worker thread at most
max_rate times per second (wall clock), so frames arriving while it is busy
are dropped and the subscriptions are never blocked.

Outputs, stamped with the depth image stamp, in target_frame:
  /aha/perception/planes          aha_msgs/PlaneArray: the output planes
  /aha/perception/planes/cloud    sensor_msgs/PointCloud2 (x, y, z, rgb):
                                  inlier points, one color per plane
  /aha/perception/planes/markers  visualization_msgs/MarkerArray: DELETEALL,
                                  then per plane the convex hull (ns "hull")
                                  and a text (ns "label") colored by label:
                                  height above the floor ("table 0.72 m")
                                  for horizontal planes, area ("wall 2.3 m2")
                                  for vertical ones

The latest detection is kept for the service
/aha/perception/get_place_candidates (aha_msgs/GetPlaceCandidates), which
runs aha_perception.place.find_place_candidates on it and publishes the
candidates on /aha/perception/place/markers (MarkerArray: DELETEALL, then
per candidate a disc of the clearance radius (ns "clearance"), the center
(ns "center") and the rank (ns "rank")). Detections older than
max_detection_age seconds (wall clock) are not used.

The TF buffer is cleared when a /tf stamp goes back (sim restart or reset).
"""

import dataclasses
import sys
import threading
import time
from array import array
from collections import Counter

import numpy as np
import rclpy
from aha_msgs.msg import PlaceCandidate as PlaceCandidateMsg
from aha_msgs.msg import Plane as PlaneMsg
from aha_msgs.msg import PlaneArray
from aha_msgs.srv import GetPlaceCandidates
from geometry_msgs.msg import Point, Point32, Polygon, PoseStamped, Quaternion, Vector3
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.place import PlaceOptions, base_cloud, find_place_candidates
from aha_perception.plane_detection_io import (
    POINT_DTYPE,
    depth_from_buffer,
    depth_to_cloud,
    label_color,
    plane_palette,
    plane_points,
    plane_text,
    quaternion_from_z_axis,
    transform_point,
    transform_to_matrix,
    xyzrgb_records,
)
from aha_perception.planes import (
    DETECTORS,
    DepthNoiseModel,
    DetectionOptions,
    detect_planes,
    params_for_input_stride,
    select_output_planes,
)
from aha_perception.tf_jump import (
    TF_RESET_MIN_INTERVAL_S,
    JumpDetectingTransformListener,
)

NODE_NAME = "aha_perception_plane_detection"
PLANES_TOPIC = "/aha/perception/planes"
CLOUD_TOPIC = "/aha/perception/planes/cloud"
MARKER_TOPIC = "/aha/perception/planes/markers"
PLACE_SERVICE = "/aha/perception/get_place_candidates"
PLACE_MARKER_TOPIC = "/aha/perception/place/markers"

TF_TIMEOUT = Duration(seconds=0.2)
HULL_WIDTH = 0.015
TEXT_HEIGHT = 0.08
TEXT_OFFSET = 0.1
PLACE_DISC_HEIGHT = 0.004
PLACE_CENTER_SIZE = 0.03
PLACE_TEXT_HEIGHT = 0.05
PLACE_TEXT_OFFSET = 0.06
WARN_PERIOD_S = 5.0
SUMMARY_PERIOD_S = 10.0
WORKER_JOIN_S = 3.0

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
        data=array("B", records.tobytes()),
        is_dense=True,
    )


def to_point(p):
    return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


def make_markers(header, planes):
    markers = [Marker(action=Marker.DELETEALL)]
    up = np.array([0.0, 0.0, 1.0])
    for i, plane in enumerate(planes):
        r, g, b, a = label_color(plane.label)
        color = ColorRGBA(r=r, g=g, b=b, a=a)

        hull_points = plane.hull_points()
        if len(hull_points) >= 2:
            hull = Marker(header=header, ns="hull", id=i, type=Marker.LINE_STRIP)
            hull.action = Marker.ADD
            hull.pose.orientation.w = 1.0
            hull.scale.x = HULL_WIDTH
            hull.color = color
            hull.points = [to_point(p) for p in hull_points]
            hull.points.append(hull.points[0])
            markers.append(hull)

        text = Marker(header=header, ns="label", id=i, type=Marker.TEXT_VIEW_FACING)
        text.action = Marker.ADD
        text.pose.position = to_point(plane.centroid + TEXT_OFFSET * up)
        text.pose.orientation.w = 1.0
        text.scale.z = TEXT_HEIGHT
        text.color = color
        text.text = plane_text(plane, up)
        markers.append(text)
    return MarkerArray(markers=markers)


def to_vector(v):
    return Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def make_plane_msg(header, plane_id, plane):
    hull = Polygon(
        points=[
            Point32(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            for p in plane.hull_points()
        ]
    )
    return PlaneMsg(
        header=header,
        id=plane_id,
        label=plane.label.value,
        normal=to_vector(plane.normal),
        offset=float(plane.offset),
        centroid=to_point(plane.centroid),
        area=float(plane.area),
        hull=hull,
        height=float(plane.centroid[2]),
    )


def make_plane_array(header, planes, ids=None):
    ids = range(len(planes)) if ids is None else ids
    return PlaneArray(
        header=header,
        planes=[make_plane_msg(header, i, planes[i]) for i in ids],
    )


def make_candidate_msg(header, candidate):
    pose = PoseStamped(header=header)
    pose.pose.position = to_point(candidate.position)
    x, y, z, w = quaternion_from_z_axis(candidate.normal)
    pose.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)
    return PlaceCandidateMsg(
        pose=pose,
        clearance=float(candidate.clearance),
        plane_id=candidate.plane_index,
        surface_label=candidate.label,
        surface_height=float(candidate.surface_height),
        score=float(candidate.score),
    )


def make_place_markers(header, candidates):
    markers = [Marker(action=Marker.DELETEALL)]
    for rank, candidate in enumerate(candidates):
        x, y, z, w = quaternion_from_z_axis(candidate.normal)
        orientation = Quaternion(x=x, y=y, z=z, w=w)
        center = np.asarray(candidate.position)
        normal = np.asarray(candidate.normal)

        disc = Marker(header=header, ns="clearance", id=rank, type=Marker.CYLINDER)
        disc.action = Marker.ADD
        disc.pose.position = to_point(center + 0.5 * PLACE_DISC_HEIGHT * normal)
        disc.pose.orientation = orientation
        disc.scale.x = disc.scale.y = 2.0 * float(candidate.clearance)
        disc.scale.z = PLACE_DISC_HEIGHT
        disc.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.25)
        markers.append(disc)

        dot = Marker(header=header, ns="center", id=rank, type=Marker.SPHERE)
        dot.action = Marker.ADD
        dot.pose.position = to_point(center)
        dot.pose.orientation.w = 1.0
        dot.scale.x = dot.scale.y = dot.scale.z = PLACE_CENTER_SIZE
        if rank == 0:
            dot.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=1.0)
        else:
            dot.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.9)
        markers.append(dot)

        text = Marker(header=header, ns="rank", id=rank, type=Marker.TEXT_VIEW_FACING)
        text.action = Marker.ADD
        text.pose.position = to_point(center + PLACE_TEXT_OFFSET * normal)
        text.pose.orientation.w = 1.0
        text.scale.z = PLACE_TEXT_HEIGHT
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = str(rank + 1)
        markers.append(text)
    return MarkerArray(markers=markers)


@dataclasses.dataclass
class Detection:
    """Latest detection kept for the place service."""

    stamp: object
    planes: list
    cloud: np.ndarray  # organized camera-frame cloud the planes index into
    camera_to_base: np.ndarray
    noise: DepthNoiseModel
    wall_time: float  # time.monotonic() when the detection finished


class PlaneDetection(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.method = self.declare_parameter("method", "ransac_gravity_refined").value
        if self.method not in DETECTORS:
            raise ValueError(
                f"unknown method {self.method!r}, choose from {sorted(DETECTORS)}"
            )
        self.target_frame = self.declare_parameter(
            "target_frame", "base_footprint"
        ).value
        depth_topic = self.declare_parameter(
            "depth_topic", "/camera/depth_registered/image_rect"
        ).value
        info_topic = self.declare_parameter(
            "camera_info_topic", "/camera/color/camera_info"
        ).value
        max_rate = float(self.declare_parameter("max_rate", 5.0).value)
        self.period = 1.0 / max_rate if max_rate > 0.0 else 0.0
        self.detect_vertical = bool(
            self.declare_parameter("detect_vertical", True).value
        )
        self.min_depth = float(self.declare_parameter("min_depth", 0.3).value)
        self.max_depth = float(self.declare_parameter("max_depth", 3.0).value)
        self.max_points = int(
            self.declare_parameter("max_points_per_plane", 5000).value
        )
        # Every stride-th row and column; the detector's working stride and the
        # place options follow (params_for_input_stride, for_input_stride).
        self.stride = int(self.declare_parameter("stride", 2).value)
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        self.place_options = PlaceOptions().for_input_stride(self.stride)
        # Tuned on calibrated D435i noise, see
        # tools/perception/plane_bench/results/tuning.md.
        merge = bool(self.declare_parameter("merge", True).value)
        self.noise_subpixel = float(
            self.declare_parameter("noise_subpixel", 0.12).value
        )
        if not self.noise_subpixel > 0.0:
            raise ValueError(f"noise_subpixel must be > 0, got {self.noise_subpixel}")
        self.max_detection_age = float(
            self.declare_parameter("max_detection_age", 3.0).value
        )
        params_type = DETECTORS[self.method].params_type
        if "merge" in {f.name for f in dataclasses.fields(params_type)}:
            self.params = params_type(merge=merge)
            merge_text = "on" if merge else "off"
        else:
            self.params = params_type()
            merge_text = "unsupported"
            if merge:
                self.get_logger().warn(
                    f"{self.method} has no merge param, merge ignored"
                )
        self.params = params_for_input_stride(self.params, self.stride)

        self.tf_buffer = Buffer()
        self.tf_listener = JumpDetectingTransformListener(
            self.tf_buffer, self, self.on_tf_jump
        )
        self.last_tf_reset = -float("inf")
        self.camera_info = None

        self.cond = threading.Condition()
        self.pending = None
        self.stopping = False
        self.latest_lock = threading.Lock()
        self.latest = None

        self.create_subscription(
            CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, depth_topic, self.on_depth, qos_profile_sensor_data
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
        reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, reliable)
        self.plane_pub = self.create_publisher(PlaneArray, PLANES_TOPIC, reliable)
        self.place_marker_pub = self.create_publisher(
            MarkerArray, PLACE_MARKER_TOPIC, reliable
        )
        self.create_service(
            GetPlaceCandidates, PLACE_SERVICE, self.on_get_place_candidates
        )
        # Clear markers left in RViz by a previous run.
        self.clear_markers()

        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        rate = f"{max_rate:g} Hz" if max_rate > 0.0 else "unlimited rate"
        self.get_logger().info(
            f"{self.method} on {depth_topic} -> {self.target_frame}, {rate}, "
            f"walls {'on' if self.detect_vertical else 'off'}, stride {self.stride}, "
            f"merge {merge_text}, noise subpixel {self.noise_subpixel:g}"
        )

    def clear_markers(self):
        clear = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        self.marker_pub.publish(clear)
        self.place_marker_pub.publish(clear)

    def on_tf_jump(self):
        if time.monotonic() - self.last_tf_reset < TF_RESET_MIN_INTERVAL_S:
            return False
        self.tf_buffer.clear()
        self.tf_listener.forget()
        self.last_tf_reset = time.monotonic()
        with self.cond:
            self.pending = None
        with self.latest_lock:
            self.latest = None
        self.clear_markers()
        self.get_logger().info(
            "/tf time went back (sim restart or reset): cleared the TF buffer"
        )
        return True

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_depth(self, msg):
        with self.cond:
            self.pending = msg
            self.cond.notify()

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
                    f"plane detection failed: {e!r}",
                    throttle_duration_sec=WARN_PERIOD_S,
                )

    def lookup_camera_to_base(self, frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame, frame, Time.from_msg(stamp), timeout=TF_TIMEOUT
            )
        except TransformException as e:
            stamped_error = e
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, frame, Time()
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.target_frame} <- {frame} unavailable: {e}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        self.get_logger().warn(
            f"TF at the image stamp unavailable, using the latest: {stamped_error}",
            throttle_duration_sec=WARN_PERIOD_S,
        )
        return transform

    def process(self, msg):
        info = self.camera_info
        if info is None:
            self.get_logger().warn(
                "waiting for camera_info", throttle_duration_sec=WARN_PERIOD_S
            )
            return
        if (info.width, info.height) != (msg.width, msg.height):
            self.get_logger().warn(
                f"depth {msg.width}x{msg.height} does not match camera_info "
                f"{info.width}x{info.height}, skipped",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        fx, cx, fy, cy = info.k[0], info.k[2], info.k[4], info.k[5]
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn(
                "camera_info has no intrinsics", throttle_duration_sec=WARN_PERIOD_S
            )
            return
        tf = self.lookup_camera_to_base(msg.header.frame_id, msg.header.stamp)
        if tf is None:
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        camera_to_base = transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

        start = time.perf_counter()
        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        cloud = depth_to_cloud(depth, fx, fy, cx, cy, self.stride)
        options = DetectionOptions(
            camera_to_base=camera_to_base,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            detect_vertical=self.detect_vertical,
            # The noise model scales with the focal length of this camera.
            noise=DepthNoiseModel(focal=fx, subpixel=self.noise_subpixel),
        )
        detected_planes = detect_planes(cloud, self.method, options, self.params)
        planes = select_output_planes(detected_planes, options)
        detected = time.perf_counter()

        colors = [plane_palette(i) for i in range(len(planes))]
        xyz, rgb = plane_points(planes, cloud, camera_to_base, colors, self.max_points)
        header = Header(stamp=msg.header.stamp, frame_id=self.target_frame)
        if self.stopping:
            return
        with self.latest_lock:
            self.latest = Detection(
                stamp=msg.header.stamp,
                planes=planes,
                cloud=cloud,
                camera_to_base=camera_to_base,
                noise=options.noise,
                wall_time=time.monotonic(),
            )
        self.plane_pub.publish(make_plane_array(header, planes))
        self.cloud_pub.publish(make_cloud(header, xyz, rgb))
        self.marker_pub.publish(make_markers(header, planes))

        total_ms = (time.perf_counter() - start) * 1e3
        detect_ms = (detected - start) * 1e3
        counts = Counter(plane.label.value for plane in planes)
        summary = ", ".join(f"{label} {n}" for label, n in sorted(counts.items()))
        text = (
            f"{len(planes)} planes ({summary or 'none'}) of {len(detected_planes)} "
            f"detected in {total_ms:.0f} ms "
            f"(detect {detect_ms:.0f} ms)"
        )
        self.get_logger().debug(text)
        self.get_logger().info(text, throttle_duration_sec=SUMMARY_PERIOD_S)

    def near_in_target(self, near):
        """``near`` (PointStamped) in target_frame with the latest TF, or None."""
        p = (near.point.x, near.point.y, near.point.z)
        if near.header.frame_id == self.target_frame:
            return np.asarray(p, dtype=np.float64)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, near.header.frame_id, Time()
            )
        except TransformException as e:
            self.get_logger().warn(f"near point not transformed: {e}")
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        matrix = transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        return transform_point(matrix, p)

    def on_get_place_candidates(self, request, response):
        def fail(message):
            response.success = False
            response.message = message
            self.get_logger().warn(f"place candidates: {message}")
            return response

        with self.latest_lock:
            detection = self.latest
        if detection is None:
            return fail("no plane detection yet")
        age = time.monotonic() - detection.wall_time
        if self.max_detection_age > 0.0 and age > self.max_detection_age:
            return fail(
                f"latest detection is {age:.1f} s old "
                f"(max_detection_age {self.max_detection_age:g} s)"
            )
        size = (request.object_size.x, request.object_size.y, request.object_size.z)
        near = None
        if request.near.header.frame_id:
            near = self.near_in_target(request.near)
            if near is None:
                return fail(
                    f"no TF {self.target_frame} <- {request.near.header.frame_id}"
                )

        start = time.perf_counter()
        points, depth = base_cloud(detection.cloud, detection.camera_to_base)
        try:
            result = find_place_candidates(
                detection.planes,
                points,
                depth,
                size,
                detection.noise,
                self.place_options,
                labels=list(request.surface_labels) or None,
                near=near,
                max_candidates=int(request.max_candidates) or None,
            )
        except ValueError as e:
            return fail(str(e))
        elapsed_ms = (time.perf_counter() - start) * 1e3

        header = Header(stamp=detection.stamp, frame_id=self.target_frame)
        response.success = True
        response.stamp = detection.stamp
        response.candidates = [make_candidate_msg(header, c) for c in result.candidates]
        response.planes = make_plane_array(header, detection.planes, result.supports)
        response.message = (
            f"{len(result.candidates)} candidates on {len(result.supports)} support "
            f"planes, detection {age:.1f} s old"
        )
        self.place_marker_pub.publish(make_place_markers(header, result.candidates))
        self.get_logger().info(
            f"place {size[0]:g} x {size[1]:g} x {size[2]:g} m: {response.message}, "
            f"{elapsed_ms:.0f} ms"
        )
        return response


def main():
    rclpy.init()
    try:
        node = PlaneDetection()
    except ValueError as e:
        rclpy.logging.get_logger(NODE_NAME).fatal(str(e))
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
