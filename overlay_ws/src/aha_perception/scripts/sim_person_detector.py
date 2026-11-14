#!/usr/bin/env python3
"""Simulation only: person_perception's topic and services from Gazebo ground truth.

A stand-in for person_perception.py in debug end-to-end runs (aha_tasks
task.launch.py e2e:=debug), without the DGX. Models whose names match
`models` (fnmatch patterns) are people; one is seen when its body axis is in
the head camera's view and not hidden in the depth image
(aha_perception.sim_person_detection). Like sim_object_detector.py, the pose
is taken relative to the robot model (origin base_footprint) and put into
`target_frame` with the TF at the depth stamp, so it carries the
localization error. Tracks, names and visibility follow person_perception
(TruthTracker): /aha/perception/people at `rate` (Hz, wall clock) on the
latest depth image, once per image; remember_person names the model, and
describe_person answers with the model name instead of a VLM. Nothing of
SAM 3.1, the depth measurement of masks or the VLM is exercised.
"""

import sys
import threading

from aha_msgs.msg import Person, PersonArray
from aha_msgs.srv import DescribePerson, RememberPerson
from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from aha_perception.gazebo_poses import GazeboPoses, tf_matrix
from aha_perception.plane_detection_io import depth_from_buffer
from aha_perception.sim_person_detection import (
    CENTRE_SHARE,
    HEAD_SHARE,
    HIDDEN_BELOW,
    Sighting,
    TruthTracker,
    body_bbox,
    is_person,
    person_visible,
)

NODE_NAME = "sim_person_detector"
PEOPLE_TOPIC = "/aha/perception/people"
REMEMBER_SERVICE = "/aha/perception/remember_person"
DESCRIBE_SERVICE = "/aha/perception/describe_person"
WARN_PERIOD_S = 5.0
WORLD_TIMEOUT_S = 60.0
POSE_MAX_AGE_S = 2.0  # wall clock


def point(vector):
    return Point(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))


class SimPersonDetector(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        p = self.declare_parameter
        self.models = list(p("models", ["guest*"]).value)
        world = p("world", "").value
        self.robot_model = p("robot_model", "aha_robot").value
        self.base_frame = p("base_frame", "base_footprint").value
        self.target_frame = p("target_frame", "map").value
        self.height = float(p("height", 1.75).value)  # m, standing
        self.radius = float(p("body_radius", 0.25).value)
        self.tolerance = float(p("depth_tolerance", 0.1).value)
        self.margin = float(p("image_margin", 4.0).value)
        depth_topic = p("depth_topic", "/camera/depth_registered/image_rect").value
        info_topic = p("info_topic", "/camera/color/camera_info").value
        rate = float(p("rate", 5.0).value)
        self.tracker = TruthTracker(forget_after=float(p("forget_after", 3.0).value))
        self.lock = threading.Lock()
        self.gz = GazeboPoses(world, WORLD_TIMEOUT_S, 2.0 * rate)
        self.camera_info = None
        self.depth_msg = None
        self.done_stamp = None  # of the last depth image processed
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, depth_topic, self.on_depth, qos_profile_sensor_data
        )
        self.people_pub = self.create_publisher(PersonArray, PEOPLE_TOPIC, 10)
        self.create_service(RememberPerson, REMEMBER_SERVICE, self.on_remember)
        self.create_service(DescribePerson, DESCRIBE_SERVICE, self.on_describe)
        self.create_timer(1.0 / max(rate, 0.1), self.on_timer)
        self.get_logger().info(
            f"world {self.gz.world}: ground-truth people {self.models} -> "
            f"{PEOPLE_TOPIC} ({self.target_frame}); no DGX"
        )

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
        if (stamp.sec, stamp.nanosec) == self.done_stamp:
            return  # no new image: like person_perception, publish nothing
        base_cam = self.lookup(self.base_frame, msg.header.frame_id, stamp)
        out_base = self.lookup(self.target_frame, self.base_frame, stamp)
        if base_cam is None or out_base is None:
            self.get_logger().warn(
                f"no TF {self.target_frame} <- {self.base_frame} <- "
                f"{msg.header.frame_id}",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        cam_base = np.linalg.inv(base_cam)
        base_world = np.linalg.inv(world_base)
        sightings = []
        for name in sorted(poses):
            if not is_person(name, self.models) or poses[name][2, 3] < HIDDEN_BELOW:
                continue
            base_model = base_world @ poses[name]
            cam_model = cam_base @ base_model
            if not person_visible(
                depth,
                K,
                cam_model,
                self.height,
                self.radius,
                self.tolerance,
                self.margin,
            ):
                continue
            out_model = out_base @ base_model
            sightings.append(
                Sighting(
                    model=name,
                    centre=out_model
                    @ np.array([0.0, 0.0, CENTRE_SHARE * self.height, 1.0]),
                    head=out_model
                    @ np.array([0.0, 0.0, HEAD_SHARE * self.height, 1.0]),
                    bbox=body_bbox(K, cam_model, self.height, self.radius, depth.shape),
                )
            )
        self.done_stamp = (stamp.sec, stamp.nanosec)
        seconds = stamp.sec + stamp.nanosec * 1e-9
        message = PersonArray()
        message.header.stamp = stamp
        message.header.frame_id = self.target_frame
        with self.lock:
            seen = self.tracker.update(sightings, seconds)
            for track in sorted(self.tracker.tracks.values(), key=lambda t: t.id):
                message.people.append(
                    Person(
                        id=track.id,
                        name=track.name,
                        score=1.0,
                        position=point(track.centre),
                        head=point(track.head),
                        visible=track.id in seen,
                        bbox=[int(v) for v in track.bbox],
                        last_seen=Time(seconds=track.last_seen).to_msg(),
                    )
                )
        self.people_pub.publish(message)

    def on_remember(self, request, response):
        with self.lock:
            error = self.tracker.remember(request.id, request.name.strip())
        response.success = not error
        response.message = error or f"Remembered person {request.id} as {request.name}"
        return response

    def on_describe(self, request, response):
        with self.lock:
            track = self.tracker.by_id(request.id)
        if track is None:
            response.message = f"person {request.id} is not tracked"
            return response
        response.success = True
        response.features = [f"ground truth {track.model}"]
        response.message = "ground truth (no VLM)"
        return response


def main():
    rclpy.init()
    try:
        node = SimPersonDetector()
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
