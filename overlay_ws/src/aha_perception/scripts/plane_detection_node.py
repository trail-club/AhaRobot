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
                                  inlier points, one color per surface id
                                  (gray before the plane is tracked)
  /aha/perception/planes/markers  visualization_msgs/MarkerArray: DELETEALL,
                                  then per plane the convex hull (ns "hull")
                                  and a text (ns "label") colored by label:
                                  the surface id once tracked, then height
                                  above the floor ("S7 table 0.72 m") for
                                  horizontal planes, area ("wall 2.3 m2")
                                  for vertical ones
  /aha/perception/surfaces        aha_msgs/SurfaceArray in tracking_frame:
                                  every tracked surface after this detection

Surfaces: aha_perception.surfaces.SurfaceTracker is updated with the output
planes of every detection, the TF target_frame -> tracking_frame at the
image stamp (the latest TF if that is not available) and the camera view
(intrinsics, depth range, the decimated depth for the occlusion tests). If
tracking_frame has no TF yet, surfaces are tracked in target_frame (they
move with the robot) with a warning, and the tracker is reset when the TF
appears. Once tracking in tracking_frame, a detection without that TF does
not update the surfaces. The lifecycle parameters surface_confirm_hits,
surface_max_misses and surface_max_unseen are TrackerOptions confirm_hits,
max_misses and max_unseen_s.

The latest detection is kept for the service
/aha/perception/get_place_candidates (aha_msgs/GetPlaceCandidates), which
runs aha_perception.place.find_place_candidates and find_place_obstacles on
it and publishes the result on /aha/perception/place/markers (MarkerArray:
DELETEALL, then per candidate a disc of the clearance radius (ns
"clearance") and the center (ns "center"), per obstacle a box (ns
"obstacle") and per request exclude region a box (ns "exclude"); no rank:
a caller such as the place server tries the candidates in its own order).
Detections older than max_detection_age seconds (wall clock) are not used.
The service has its own callback group (spin with a MultiThreadedExecutor,
as main does), so TF and the image subscriptions keep running while it
computes. With surface_ids only the planes of the latest detection
assigned to those surfaces are supports (STATUS_NO_SURFACE if none is);
status is STATUS_NO_SURFACE without supports, STATUS_NO_SPACE without
candidates and STATUS_ERROR (success false) on errors. The request's near
and approach_from are transformed to target_frame with the TF at the
detection stamp (else the latest), like the candidates, which are in
target_frame at that stamp. The request's clearance is passed to
find_place_candidates, and its approach_from with yaw_offsets as the
Orientation: the candidates are then oriented, the pose's yaw the object
yaw checked.

Furniture (parameter furniture_config, a YAML of aha_perception.furniture;
empty = none, nothing below happens): after every detection the tracked
surfaces are associated with the furniture levels (associate_surfaces with
furniture_margin and furniture_height_tolerance) through the TF tracking
frame -> config frame at the image stamp (else the latest); without that TF
there is no association (throttled warning). The result fills
Surface.furniture / furniture_level, clips the hulls of the planes of those
surfaces to the furniture footprint grown by furniture_clip_margin
(furniture.clip_plane: depth edge noise bleeds a board's hull past its
edges; the published planes and the candidates use the clipped hulls, the
tracker and the obstacles the detected ones, so that an object on a strip
cut off by a config or TF error is still a box; a plane that the clip would
leave empty is not associated in that detection, with a warning), caps the
obstacle band and blocks the candidates under the board above the level
(the level's room from the config minus the surface's height error) and
serves the request's furniture /
furniture_level: unknown furniture or level is STATUS_ERROR, an object
taller than the room of the level(s) STATUS_NO_SPACE, a level without an
associated surface in the latest detection STATUS_NO_SURFACE. view_point is
the level center from the config in the tracking frame (latest TF). The
level rectangles are published once, latched, on
/aha/perception/furniture/markers (MarkerArray in the config frame: per
level a rectangle (ns "level") and a text "<name> <level>" (ns "label")).

The TF buffer and the surfaces are cleared when a /tf stamp goes back (sim
restart or reset).
"""

import dataclasses
import math
import sys
import threading
import time
from array import array
from collections import Counter

import numpy as np
import rclpy
from aha_msgs.msg import PlaceCandidate as PlaceCandidateMsg
from aha_msgs.msg import PlaceObstacle as PlaceObstacleMsg
from aha_msgs.msg import Plane as PlaneMsg
from aha_msgs.msg import PlaneArray
from aha_msgs.msg import Surface as SurfaceMsg
from aha_msgs.msg import SurfaceArray
from aha_msgs.srv import GetPlaceCandidates
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import (
    Point,
    Point32,
    PointStamped,
    Polygon,
    Pose,
    PoseStamped,
    Quaternion,
    Vector3,
)
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
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.furniture import (
    AssociationOptions,
    FurnitureConfigError,
    associate_surfaces,
    clip_plane,
    level_markers,
    load_config,
    missing_message,
    requested_surfaces,
    resolve_request,
)
from aha_perception.place import (
    ExcludeBox,
    Orientation,
    PlaceCloud,
    PlaceOptions,
    PlaceStatus,
    base_cloud,
    find_place_candidates,
    find_place_obstacles,
    object_size_of,
)
from aha_perception.plane_detection_io import (
    POINT_DTYPE,
    depth_from_buffer,
    depth_to_cloud,
    label_color,
    plane_palette,
    plane_points,
    plane_text,
    quaternion_from_matrix,
    quaternion_from_normal_yaw,
    quaternion_from_z_axis,
    seconds_to_stamp,
    stamp_to_seconds,
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
from aha_perception.surfaces import CameraView, SurfaceTracker, TrackerOptions
from aha_perception.tf_jump import (
    TF_RESET_MIN_INTERVAL_S,
    JumpDetectingTransformListener,
)

NODE_NAME = "aha_perception_plane_detection"
PLANES_TOPIC = "/aha/perception/planes"
CLOUD_TOPIC = "/aha/perception/planes/cloud"
MARKER_TOPIC = "/aha/perception/planes/markers"
SURFACES_TOPIC = "/aha/perception/surfaces"
PLACE_SERVICE = "/aha/perception/get_place_candidates"
PLACE_MARKER_TOPIC = "/aha/perception/place/markers"
FURNITURE_MARKER_TOPIC = "/aha/perception/furniture/markers"

TF_TIMEOUT = Duration(seconds=0.2)
UNTRACKED_COLOR = (0.5, 0.5, 0.5)  # cloud color of planes without a surface id
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
EXECUTOR_THREADS = 2  # the place service beside the subscriptions and TF

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


def make_markers(header, planes, surface_ids):
    markers = [Marker(action=Marker.DELETEALL)]
    up = np.array([0.0, 0.0, 1.0])
    for i, (plane, surface_id) in enumerate(zip(planes, surface_ids)):
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
        if surface_id:
            text.text = f"S{surface_id} {text.text}"
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


def to_time(seconds):
    sec, nanosec = seconds_to_stamp(seconds)
    return TimeMsg(sec=sec, nanosec=nanosec)


def make_surface_array(header, surfaces, furniture=None):
    """SurfaceArray of aha_perception.surfaces.Surface in header.frame_id;
    ``furniture`` maps surface ids to aha_perception.furniture.Match."""
    furniture = furniture or {}
    msgs = []
    for s in surfaces:
        match = furniture.get(s.id)
        plane = PlaneMsg(
            header=header,
            id=s.id,
            label=s.label,
            normal=to_vector(s.normal),
            offset=float(s.offset),
            centroid=to_point(s.centroid),
            area=float(s.area),
            hull=Polygon(
                points=[
                    Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in s.hull
                ]
            ),
            height=s.height,
        )
        msgs.append(
            SurfaceMsg(
                id=s.id,
                label=s.label,
                level=s.level,
                plane=plane,
                object_categories=[],
                last_seen=to_time(s.last_seen),
                observations=s.observations,
                visible=s.visible,
                furniture=match.furniture if match else "",
                furniture_level=match.level if match else 0,
            )
        )
    return SurfaceArray(header=header, surfaces=msgs)


def make_furniture_markers(config):
    """Level rectangles (ns "level") and texts (ns "label") in the config frame."""
    header = Header(frame_id=config.frame)  # zero stamp: the latest TF
    color = ColorRGBA(r=0.9, g=0.7, b=0.2, a=0.9)
    markers = [Marker(action=Marker.DELETEALL)]
    for k, level in enumerate(level_markers(config)):
        outline = Marker(header=header, ns="level", id=k, type=Marker.LINE_STRIP)
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = HULL_WIDTH / 2
        outline.color = color
        outline.points = [to_point(p) for p in level.outline]
        markers.append(outline)
        text = Marker(header=header, ns="label", id=k, type=Marker.TEXT_VIEW_FACING)
        text.action = Marker.ADD
        text.pose.position = to_point(level.center + [0.0, 0.0, PLACE_TEXT_OFFSET])
        text.pose.orientation.w = 1.0
        text.scale.z = PLACE_TEXT_HEIGHT
        text.color = color
        text.text = f"{level.furniture} {level.number}"
        markers.append(text)
    return MarkerArray(markers=markers)


def make_candidate_msg(header, candidate, surface_ids):
    pose = PoseStamped(header=header)
    pose.pose.position = to_point(candidate.position)
    oriented = math.isfinite(candidate.yaw)
    yaw = candidate.yaw if oriented else 0.0
    x, y, z, w = quaternion_from_normal_yaw(candidate.normal, yaw)
    pose.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)
    return PlaceCandidateMsg(
        pose=pose,
        oriented=oriented,
        clearance=float(candidate.clearance),
        plane_id=candidate.plane_index,
        surface_id=surface_ids[candidate.plane_index],
        surface_label=candidate.label,
        surface_height=float(candidate.surface_height),
        score=float(candidate.score),
    )


def surfaces_missing(ids, surfaces, stamp):
    """Message for requested surface ids none of which is in the detection."""
    known = {s.id: s for s in surfaces}
    now = stamp_to_seconds(stamp)
    parts = []
    for surface_id in ids:
        s = known.get(surface_id)
        if s is None:
            parts.append(f"S{surface_id} is not tracked")
        else:
            parts.append(
                f"S{surface_id} ({s.label}) last seen {now - s.last_seen:.1f} s before"
            )
    return "surface not in the latest detection: " + ", ".join(parts)


def obstacle_pose(obstacle):
    x, y, z, w = quaternion_from_matrix(obstacle.rotation)
    orientation = Quaternion(x=x, y=y, z=z, w=w)
    return Pose(position=to_point(obstacle.center), orientation=orientation)


def make_obstacle_msg(obstacle):
    return PlaceObstacleMsg(
        pose=obstacle_pose(obstacle),
        size=to_vector(obstacle.size),
        plane_id=obstacle.plane_index,
    )


def exclude_boxes(regions):
    """aha_perception.place.ExcludeBox of PlaceObstacle messages (target_frame)."""
    boxes = []
    for region in regions:
        p, q = region.pose.position, region.pose.orientation
        size = (region.size.x, region.size.y, region.size.z)
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w, *size)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("exclude region with a non-finite pose or size")
        if q.x == q.y == q.z == q.w == 0.0:
            q = Quaternion(w=1.0)
        matrix = transform_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        boxes.append(ExcludeBox.of(matrix[:3, 3], matrix[:3, :3], size))
    return boxes


def make_place_markers(header, candidates, obstacles=(), excluded=()):
    markers = [Marker(action=Marker.DELETEALL)]
    for k, obstacle in enumerate(obstacles):
        box = Marker(header=header, ns="obstacle", id=k, type=Marker.CUBE)
        box.action = Marker.ADD
        box.pose = obstacle_pose(obstacle)
        box.scale = to_vector(obstacle.size)
        box.color = ColorRGBA(r=1.0, g=0.3, b=0.2, a=0.4)
        markers.append(box)
    for k, region in enumerate(excluded):
        box = Marker(header=header, ns="exclude", id=k, type=Marker.CUBE)
        box.action = Marker.ADD
        box.pose = region.pose
        box.scale = region.size
        box.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.25)
        markers.append(box)
    for k, candidate in enumerate(candidates):
        x, y, z, w = quaternion_from_z_axis(candidate.normal)
        orientation = Quaternion(x=x, y=y, z=z, w=w)
        center = np.asarray(candidate.position)
        normal = np.asarray(candidate.normal)

        disc = Marker(header=header, ns="clearance", id=k, type=Marker.CYLINDER)
        disc.action = Marker.ADD
        disc.pose.position = to_point(center + 0.5 * PLACE_DISC_HEIGHT * normal)
        disc.pose.orientation = orientation
        disc.scale.x = disc.scale.y = 2.0 * float(candidate.clearance)
        disc.scale.z = PLACE_DISC_HEIGHT
        disc.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.25)
        markers.append(disc)

        dot = Marker(header=header, ns="center", id=k, type=Marker.SPHERE)
        dot.action = Marker.ADD
        dot.pose.position = to_point(center)
        dot.pose.orientation.w = 1.0
        dot.scale.x = dot.scale.y = dot.scale.z = PLACE_CENTER_SIZE
        dot.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.9)
        markers.append(dot)
    return MarkerArray(markers=markers)


@dataclasses.dataclass
class Detection:
    """Latest detection kept for the place service."""

    stamp: object
    planes: list  # hulls clipped to the associated furniture footprints
    cloud: np.ndarray  # organized camera-frame cloud the planes index into
    camera_to_base: np.ndarray
    noise: DepthNoiseModel
    wall_time: float  # time.monotonic() when the detection finished
    surface_ids: list  # Surface.id per plane, 0 = not tracked
    surfaces: list  # aha_perception.surfaces.Surface after this detection
    # Surface.id -> aha_perception.furniture.Match; None without association
    furniture: dict | None = None
    # The planes as detected (hulls not clipped), for the obstacle boxes;
    # None: same as planes.
    detected_planes: list | None = None


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
        self.tracking_frame = self.declare_parameter("tracking_frame", "odom").value
        tracker_options = TrackerOptions(
            confirm_hits=int(self.declare_parameter("surface_confirm_hits", 2).value),
            max_misses=int(self.declare_parameter("surface_max_misses", 5).value),
            max_unseen_s=float(
                self.declare_parameter("surface_max_unseen", 120.0).value
            ),
        )
        self.tracker = SurfaceTracker(tracker_options)
        self.tracker_lock = threading.Lock()
        self.tracker_frame = None  # frame the tracker currently works in
        furniture_path = self.declare_parameter("furniture_config", "").value
        self.association = AssociationOptions(
            margin=float(self.declare_parameter("furniture_margin", 0.10).value),
            height_tolerance=float(
                self.declare_parameter("furniture_height_tolerance", 0.05).value
            ),
        )
        # Hulls of planes on a furniture level are clipped to its footprint
        # grown by this (m; negative: not clipped).
        self.clip_margin = float(
            self.declare_parameter("furniture_clip_margin", 0.02).value
        )
        self.furniture = None
        if furniture_path:
            try:
                self.furniture = load_config(furniture_path)
            except FurnitureConfigError as e:
                raise ValueError(str(e)) from e

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
        # Incremented by on_tf_jump: a detection started before is dropped.
        self.reset_epoch = 0

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
        self.surface_pub = self.create_publisher(SurfaceArray, SURFACES_TOPIC, reliable)
        self.place_marker_pub = self.create_publisher(
            MarkerArray, PLACE_MARKER_TOPIC, reliable
        )
        # Own group: the subscriptions and TF keep running while it computes.
        self.create_service(
            GetPlaceCandidates,
            PLACE_SERVICE,
            self.on_get_place_candidates,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # Clear markers left in RViz by a previous run.
        self.clear_markers()
        furniture_text = ""
        if self.furniture is not None:
            latched = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.furniture_marker_pub = self.create_publisher(
                MarkerArray, FURNITURE_MARKER_TOPIC, latched
            )
            self.furniture_marker_pub.publish(make_furniture_markers(self.furniture))
            furniture_text = (
                f", furniture {', '.join(self.furniture.names)} "
                f"in {self.furniture.frame} ({furniture_path})"
            )

        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        rate = f"{max_rate:g} Hz" if max_rate > 0.0 else "unlimited rate"
        self.get_logger().info(
            f"{self.method} on {depth_topic} -> {self.target_frame}, {rate}, "
            f"walls {'on' if self.detect_vertical else 'off'}, stride {self.stride}, "
            f"merge {merge_text}, noise subpixel {self.noise_subpixel:g}, "
            f"surfaces in {self.tracking_frame}{furniture_text}"
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
            self.reset_epoch += 1
            self.clear_markers()
        with self.tracker_lock:
            self.tracker.reset()
            self.tracker_frame = None
        self.get_logger().info(
            "/tf time went back (sim restart or reset): cleared the TF buffer "
            "and the surfaces"
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

    def lookup_base_to_tracking(self, stamp):
        """(frame, target_frame -> frame matrix) for the tracker, or None to skip.

        The TF at the image stamp, else the latest. Without any while the
        tracker is not yet in tracking_frame, target_frame with identity.
        """
        if self.tracking_frame == self.target_frame:
            return self.target_frame, np.eye(4)
        known = self.tracker_frame == self.tracking_frame
        error = None
        for when, timeout in (
            (Time.from_msg(stamp), TF_TIMEOUT if known else Duration()),
            (Time(), Duration()),
        ):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.tracking_frame, self.target_frame, when, timeout=timeout
                )
            except TransformException as e:
                error = e
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            matrix = transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            return self.tracking_frame, matrix
        if known:
            self.get_logger().warn(
                f"TF {self.tracking_frame} <- {self.target_frame} unavailable, "
                f"surfaces not updated: {error}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        self.get_logger().warn(
            f"TF {self.tracking_frame} <- {self.target_frame} unavailable, "
            f"tracking surfaces in {self.target_frame} until it is: {error}",
            once=True,
        )
        return self.target_frame, np.eye(4)

    def track_surfaces(self, stamp, planes, cloud, camera_to_base, info, noise):
        """Update the tracker; (header, surface ids per plane, surfaces) or None."""
        lookup = self.lookup_base_to_tracking(stamp)
        if lookup is None:
            return None
        frame, base_to_tracking = lookup
        view = CameraView(
            camera_to_tracking=base_to_tracking @ camera_to_base,
            fx=info.k[0],
            fy=info.k[4],
            cx=info.k[2],
            cy=info.k[5],
            width=info.width,
            height=info.height,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            depth=cloud[..., 2],
            stride=self.stride,
        )
        with self.tracker_lock:
            if frame != self.tracker_frame:
                if self.tracker_frame is not None:
                    self.get_logger().info(
                        f"tracking surfaces in {frame} (was {self.tracker_frame}), "
                        "surfaces reset"
                    )
                self.tracker.reset()
                self.tracker_frame = frame
            ids = self.tracker.update(
                planes, base_to_tracking, stamp_to_seconds(stamp), view, noise
            )
            surfaces = self.tracker.surfaces()
        return Header(stamp=stamp, frame_id=frame), ids, surfaces

    def lookup_matrix(self, target, source, stamp=None):
        """(4x4 source -> target, None) at ``stamp`` (no wait) or else the
        latest TF (throttled warning); (None, error) without one."""
        if target == source:
            return np.eye(4), None
        error = None
        times = [Time.from_msg(stamp)] if stamp is not None else []
        for when in times + [Time()]:
            try:
                tf = self.tf_buffer.lookup_transform(target, source, when)
            except TransformException as e:
                error = e
                continue
            if error is not None:
                self.get_logger().warn(
                    f"TF {target} <- {source} at the stamp unavailable, using "
                    f"the latest: {error}",
                    throttle_duration_sec=WARN_PERIOD_S,
                )
            t, q = tf.transform.translation, tf.transform.rotation
            return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w)), None
        return None, error

    def associate_furniture(self, tracked, stamp):
        """Surface id -> furniture Match of the tracked surfaces, or None."""
        if self.furniture is None or not tracked:
            return None
        header, _, surfaces = tracked
        frame = self.furniture.frame
        matrix, error = self.lookup_matrix(frame, header.frame_id, stamp)
        if matrix is None:
            self.get_logger().warn(
                f"TF {frame} <- {header.frame_id} unavailable, surfaces not "
                f"associated with the furniture: {error}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return None
        return associate_surfaces(self.furniture, surfaces, matrix, self.association)

    def clip_to_furniture(self, planes, surface_ids, furniture, stamp):
        """(planes, furniture): the planes of surfaces on a furniture level
        with their hull clipped to its footprint grown by
        ``furniture_clip_margin`` (furniture.clip_plane; target_frame <-
        config frame at the stamp). A surface whose plane the clip would
        leave empty (off the footprint: a wrong association) loses its
        association in the returned copy of ``furniture``."""
        if not furniture or self.clip_margin < 0.0:
            return planes, furniture
        matrix, _ = self.lookup_matrix(self.target_frame, self.furniture.frame, stamp)
        if matrix is None:
            return planes, furniture
        out = []
        kept = dict(furniture)
        for plane, surface_id in zip(planes, surface_ids):
            match = furniture.get(surface_id)
            item = self.furniture.get(match.furniture) if match else None
            if item is not None:
                corners = item.corners(self.clip_margin)
                points = np.column_stack([corners, np.zeros(len(corners))])
                xy = (points @ matrix[:3, :3].T + matrix[:3, 3])[:, :2]
                clipped = clip_plane(plane, xy)
                if clipped is None:
                    kept.pop(surface_id, None)
                    self.get_logger().warn(
                        f"S{surface_id} is associated with {match.furniture} "
                        f"level {match.level} but its plane is off the footprint: "
                        "not associated in this detection",
                        throttle_duration_sec=WARN_PERIOD_S,
                    )
                else:
                    plane = clipped
            out.append(plane)
        return out, kept

    def process(self, msg):
        epoch = self.reset_epoch
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
        if epoch != self.reset_epoch:
            return  # TF reset while detecting: the frame is of the old time
        tracked = self.track_surfaces(
            msg.header.stamp, planes, cloud, camera_to_base, info, options.noise
        )
        surface_ids = tracked[1] if tracked else [0] * len(planes)
        furniture = self.associate_furniture(tracked, msg.header.stamp)
        detected_output = planes
        planes, furniture = self.clip_to_furniture(
            planes, surface_ids, furniture, msg.header.stamp
        )

        # Stable colors over detections: by surface id, gray before tracking.
        colors = [plane_palette(s) if s else UNTRACKED_COLOR for s in surface_ids]
        xyz, rgb = plane_points(planes, cloud, camera_to_base, colors, self.max_points)
        header = Header(stamp=msg.header.stamp, frame_id=self.target_frame)
        if self.stopping:
            return
        with self.latest_lock:
            if epoch != self.reset_epoch:
                return
            self.latest = Detection(
                stamp=msg.header.stamp,
                planes=planes,
                cloud=cloud,
                camera_to_base=camera_to_base,
                noise=options.noise,
                wall_time=time.monotonic(),
                surface_ids=surface_ids,
                surfaces=tracked[2] if tracked else [],
                furniture=furniture,
                detected_planes=detected_output,
            )
            # Under the lock, so a TF reset clears the markers after these.
            self.plane_pub.publish(make_plane_array(header, planes))
            if tracked:
                self.surface_pub.publish(
                    make_surface_array(tracked[0], tracked[2], furniture)
                )
            self.cloud_pub.publish(make_cloud(header, xyz, rgb))
            self.marker_pub.publish(make_markers(header, planes, surface_ids))

        total_ms = (time.perf_counter() - start) * 1e3
        detect_ms = (detected - start) * 1e3
        counts = Counter(plane.label.value for plane in planes)
        summary = ", ".join(f"{label} {n}" for label, n in sorted(counts.items()))
        tracked_text = f", {len(tracked[2])} surfaces" if tracked else ""
        if furniture is not None:
            tracked_text += f" ({len(furniture)} on furniture)"
        text = (
            f"{len(planes)} planes ({summary or 'none'}) of {len(detected_planes)} "
            f"detected in {total_ms:.0f} ms "
            f"(detect {detect_ms:.0f} ms){tracked_text}"
        )
        self.get_logger().debug(text)
        self.get_logger().info(text, throttle_duration_sec=SUMMARY_PERIOD_S)

    def point_in_target(self, point, stamp):
        """``point`` (PointStamped) in target_frame with the TF at ``stamp``
        (the detection's: target_frame then, where its cloud is), else the
        latest; None without a TF."""
        p = (point.point.x, point.point.y, point.point.z)
        matrix, error = self.lookup_matrix(
            self.target_frame, point.header.frame_id, stamp
        )
        if matrix is None:
            self.get_logger().warn(f"request point not transformed: {error}")
            return None
        return transform_point(matrix, p)

    def level_ceilings(self, detection) -> dict:
        """Plane index -> room above the plane under the board above its
        furniture level (config: from the level top to the bottom of that
        board, minus the plane's height above the level top) for the planes
        associated with a level that has a board above."""
        if self.furniture is None or not detection.furniture:
            return {}
        out = {}
        for i, surface_id in enumerate(detection.surface_ids):
            match = detection.furniture.get(surface_id)
            furniture = self.furniture.get(match.furniture) if match else None
            level = furniture.level(match.level) if furniture else None
            if level is not None and level.max_object_height is not None:
                out[i] = level.max_object_height - match.height_error
        return out

    def furniture_view_point(self, furniture):
        """Center of a FurnitureRequest's level in tracking_frame (else
        target_frame) with the latest TF, or None."""
        center = furniture.furniture.view_center(furniture.level)
        for frame in dict.fromkeys((self.tracking_frame, self.target_frame)):
            matrix, _ = self.lookup_matrix(frame, self.furniture.frame)
            if matrix is not None:
                point = PointStamped(header=Header(frame_id=frame))
                point.point = to_point(transform_point(matrix, center))
                return point
        return None

    def no_candidates(self, response, status, message, detection=None):
        """A successful response without candidates (NO_SURFACE / NO_SPACE)."""
        response.success = True
        response.status = int(status)
        response.message = message
        header = Header(frame_id=self.target_frame)
        if detection is not None:
            response.stamp = header.stamp = detection.stamp
        self.place_marker_pub.publish(make_place_markers(header, []))
        self.get_logger().warn(f"place candidates: {message}")
        return response

    def on_get_place_candidates(self, request, response):
        def fail(message):
            response.success = False
            response.status = GetPlaceCandidates.Response.STATUS_ERROR
            response.message = message
            # Candidates of an earlier request are no answer to this one.
            self.place_marker_pub.publish(
                make_place_markers(Header(frame_id=self.target_frame), [])
            )
            self.get_logger().warn(f"place candidates: {message}")
            return response

        try:
            size = object_size_of(
                (request.object_size.x, request.object_size.y, request.object_size.z)
            )
        except ValueError as e:
            return fail(str(e))
        furniture = None
        if request.furniture:
            furniture = resolve_request(
                self.furniture,
                request.furniture,
                int(request.furniture_level),
                size[2],
            )
            if furniture.status == PlaceStatus.ERROR:
                return fail(furniture.message)
            if furniture.furniture is not None:
                view_point = self.furniture_view_point(furniture)
                if view_point is not None:
                    response.view_point = view_point
            if furniture.status is not None:
                return self.no_candidates(response, furniture.status, furniture.message)
        elif request.furniture_level:
            return fail("furniture_level given without furniture")

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
        near = None
        if request.near.header.frame_id:
            near = self.point_in_target(request.near, detection.stamp)
            if near is None:
                return fail(
                    f"no TF {self.target_frame} <- {request.near.header.frame_id}"
                )
            if not np.all(np.isfinite(near)):
                return fail("near is not finite")
        orientation = None
        if request.approach_from.header.frame_id:
            origin = self.point_in_target(request.approach_from, detection.stamp)
            if origin is None:
                return fail(
                    f"no TF {self.target_frame} <- "
                    f"{request.approach_from.header.frame_id}"
                )
            offsets = tuple(float(v) for v in request.yaw_offsets) or (0.0,)
            if not (np.all(np.isfinite(origin)) and all(map(math.isfinite, offsets))):
                return fail("approach_from and yaw_offsets must be finite")
            orientation = Orientation((float(origin[0]), float(origin[1])), offsets)
        header = Header(stamp=detection.stamp, frame_id=self.target_frame)
        plane_indices = None
        wanted = None
        if furniture is not None:
            if detection.furniture is None:
                tracking = self.tracker_frame or self.tracking_frame
                return fail(
                    "surfaces not associated with the furniture in the latest "
                    f"detection (no TF {tracking} <- {self.target_frame} to track "
                    f"them or {self.furniture.frame} <- {tracking})"
                )
            ids = requested_surfaces(
                furniture,
                detection.furniture,
                detection.surface_ids,
                request.surface_ids,
            )
            if not ids:
                return self.no_candidates(
                    response,
                    PlaceStatus.NO_SURFACE,
                    missing_message(
                        furniture, detection.furniture, request.surface_ids
                    ),
                    detection,
                )
            wanted = set(ids)
        elif request.surface_ids:
            wanted = set(request.surface_ids)
        if wanted is not None:
            plane_indices = [
                i for i, s in enumerate(detection.surface_ids) if s in wanted
            ]
            if not plane_indices:
                return self.no_candidates(
                    response,
                    PlaceStatus.NO_SURFACE,
                    surfaces_missing(
                        sorted(wanted), detection.surfaces, detection.stamp
                    ),
                    detection,
                )

        surface_height = None
        if request.restrict_height:
            surface_height = float(request.surface_height)

        start = time.perf_counter()
        points, depth = base_cloud(detection.cloud, detection.camera_to_base)
        ceilings = self.level_ceilings(detection)
        try:
            exclude = exclude_boxes(request.exclude_regions)
            # Points behind the exclude boxes, seen from the camera, are edge
            # noise; built once for both calls.
            cloud = PlaceCloud(
                points,
                depth,
                detection.noise,
                self.place_options,
                exclude,
                viewpoint=detection.camera_to_base[:3, 3],
            )
            # First: the candidates keep clear of these boxes too.
            obstacles = find_place_obstacles(
                detection.detected_planes or detection.planes,
                points,
                depth,
                detection.noise,
                self.place_options,
                ceilings=ceilings,
                cloud=cloud,
            )
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
                surface_height=surface_height,
                height_tolerance=float(request.height_tolerance),
                plane_indices=plane_indices,
                clearance=float(request.clearance),
                orientation=orientation,
                boxes=obstacles,
                ceilings=ceilings,
                cloud=cloud,
            )
        except ValueError as e:
            return fail(str(e))
        elapsed_ms = (time.perf_counter() - start) * 1e3

        response.success = True
        response.status = int(result.status)
        response.stamp = detection.stamp
        response.candidates = [
            make_candidate_msg(header, c, detection.surface_ids)
            for c in result.candidates
        ]
        response.planes = make_plane_array(header, detection.planes, result.supports)
        response.obstacles = [make_obstacle_msg(o) for o in obstacles]
        support_ids = sorted({detection.surface_ids[i] for i in result.supports} - {0})
        restriction = "" if surface_height is None else f" at {surface_height:.2f} m"
        if furniture is not None:
            restriction += f" of {furniture.text}"
        if request.clearance > 0.0:
            restriction += f", clearance {request.clearance:.2f} m"
        excluded = f", {len(exclude)} excluded boxes" if exclude else ""
        if result.degenerate:
            excluded += (
                f", {len(result.degenerate)} matching planes without valid inliers"
            )
        response.message = (
            f"{PlaceStatus(result.status).name}: {len(result.candidates)} candidates "
            f"on {len(result.supports)} support planes{restriction} "
            f"(surfaces {support_ids}), {len(obstacles)} obstacles{excluded}, "
            f"detection {age:.1f} s old"
        )
        self.place_marker_pub.publish(
            make_place_markers(
                header, result.candidates, obstacles, request.exclude_regions
            )
        )
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
