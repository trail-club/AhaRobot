#!/usr/bin/env python3
"""Point the head at a 3D target.

Subscribes /aha/perception/look_at (geometry_msgs/PointStamped, any TF frame),
computes pan/tilt about the head pivot (link_head_tilt origin, where the pan
and tilt axes intersect) and sends a trajectory to head_controller.
Limits, speed and duration come from config/head.yaml.

The camera body is centered on link_head_tilt's forward axis, but the RGB
optical center (camera_color_optical_frame, sensors.xacro) sits about 32.5 mm
to its left with the optical axis parallel to it. Aiming the pivot ray
therefore leaves a small parallax error (about 1.9 deg at 1 m), ignored here.

sim_view.launch.py remaps RViz's /clicked_point (Publish Point tool) here.
Only the latest transforms are used, so the node runs fine on the wall clock
(no use_sim_time needed, which saves handling /clock at 1 kHz). A target that
arrives before the transforms are known is dropped with a warning.

Without /clock the TF buffer cannot see the sim clock going back, and would
keep the old, later transforms (rejecting new ones as TF_OLD_DATA). So the
buffer is cleared, together with the look-at target and the markers, when
/robot_description is republished (sim restart: robot_state_publisher
starts again) or when a /tf stamp goes back by more than 1 s for a frame
(e.g. a Gazebo world reset; at most once per 5 s).

For RViz, /aha/perception/look_at/markers (visualization_msgs/MarkerArray,
namespace "look_at") shows the last target as a sphere in its own frame and a
line from camera_color_optical_frame to it, refreshed at 5 Hz as the head
moves. Both are orange, or red when the target was clamped to the head range.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.head_config import (
    TRAJECTORY_TOPIC,
    declare_head_config,
    describe,
    head_position,
)

BASE_FRAME = "base_link"
PIVOT_FRAME = "link_head_tilt"
CAMERA_FRAME = "camera_color_optical_frame"

MARKER_TOPIC = "/aha/perception/look_at/markers"
MARKER_NS = "look_at"
MARKER_PERIOD = 0.2
SPHERE_DIAMETER = 0.06
LINE_WIDTH = 0.01
COLOR_OK = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
COLOR_CLAMPED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

TF_JUMP_BACK_S = 1.0
TF_RESET_MIN_INTERVAL_S = 5.0


class JumpDetectingTransformListener(TransformListener):
    """TransformListener that reports /tf stamps going back per child frame.

    on_jump() returns True if it reset the buffer; otherwise the message is
    dropped and the old stamps are kept, so the jump is reported again.
    """

    def __init__(self, buffer, node, on_jump):
        self._on_jump = on_jump
        self._latest = {}
        super().__init__(buffer, node)

    def forget(self):
        self._latest.clear()

    def callback(self, data):
        stamps = [
            (t.child_frame_id, t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
            for t in data.transforms
        ]
        latest = self._latest
        if any(s < latest.get(f, s) - TF_JUMP_BACK_S for f, s in stamps):
            if not self._on_jump():
                return
        latest.update(stamps)
        super().callback(data)


class HeadLookAt(Node):
    def __init__(self):
        super().__init__("aha_perception_head_look_at")
        self.config = declare_head_config(self)
        self.tf_buffer = Buffer()
        self.tf_listener = JumpDetectingTransformListener(
            self.tf_buffer, self, self.on_tf_jump
        )
        self.last_tf_reset = -math.inf
        self.seen_description = False
        self.current = None
        # Last accepted target for the markers: (PointStamped, clamped).
        self.target = None
        self.create_subscription(
            JointState, "/joint_states", self.on_joint_states, qos_profile_sensor_data
        )
        self.create_subscription(
            PointStamped, "/aha/perception/look_at", self.on_target, 10
        )
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
        self.pub = self.create_publisher(JointTrajectory, TRAJECTORY_TOPIC, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            MARKER_TOPIC,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        # Clear markers left in RViz by a previous run.
        self.clear_markers()
        self.create_timer(MARKER_PERIOD, self.publish_markers)

    def clear_markers(self):
        self.marker_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))

    def reset_tf(self, reason):
        """Drop all TF data and the target, which may refer to the old sim."""
        self.tf_buffer.clear()
        self.tf_listener.forget()
        self.last_tf_reset = time.monotonic()
        self.target = None
        self.clear_markers()
        self.get_logger().info(
            f"{reason}: cleared the TF buffer and the look-at target"
        )

    def on_robot_description(self, _msg):
        # The first one is the running robot's (transient local).
        if self.seen_description:
            self.reset_tf("/robot_description republished (sim restart)")
        self.seen_description = True

    def on_tf_jump(self):
        if time.monotonic() - self.last_tf_reset < TF_RESET_MIN_INTERVAL_S:
            return False
        self.reset_tf("/tf time went back (sim restart or reset)")
        return True

    def publish_markers(self):
        if self.target is None:
            return
        target, clamped = self.target
        color = COLOR_CLAMPED if clamped else COLOR_OK
        frame = target.header.frame_id

        def marker(marker_id, marker_type):
            m = Marker(ns=MARKER_NS, id=marker_id, type=marker_type, action=Marker.ADD)
            # Zero stamp: RViz uses the latest TF, so a map/odom target stays put.
            m.header.frame_id = frame
            m.pose.orientation.w = 1.0
            m.color = color
            return m

        sphere = marker(0, Marker.SPHERE)
        sphere.pose.position = target.point
        sphere.scale.x = sphere.scale.y = sphere.scale.z = SPHERE_DIAMETER
        markers = [sphere]

        try:
            cam = self.tf_buffer.lookup_transform(
                frame, CAMERA_FRAME, Time()
            ).transform.translation
        except TransformException:
            cam = None
        if cam is not None:
            line = marker(1, Marker.LINE_LIST)
            line.scale.x = LINE_WIDTH
            line.points = [Point(x=cam.x, y=cam.y, z=cam.z), target.point]
            markers.append(line)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def on_joint_states(self, msg):
        pos = head_position(msg)
        if pos is not None:
            self.current = pos

    def on_target(self, msg):
        # Treat the target as static in its frame: use the latest transforms.
        target = PointStamped(header=msg.header, point=msg.point)
        target.header.stamp = Time().to_msg()
        # No timeout: waiting would only block this single-threaded node, as
        # /tf is handled by the same executor and cannot arrive meanwhile.
        try:
            to_base = self.tf_buffer.lookup_transform(
                BASE_FRAME, target.header.frame_id, Time()
            )
            pivot = self.tf_buffer.lookup_transform(
                BASE_FRAME, PIVOT_FRAME, Time()
            ).transform.translation
        except TransformException as e:
            self.get_logger().warn(f"look_at: TF unavailable: {e}")
            return
        p = do_transform_point(target, to_base).point
        dx, dy, dz = p.x - pivot.x, p.y - pivot.y, p.z - pivot.z
        if math.hypot(dx, dy, dz) < 1e-3:
            self.get_logger().warn("look_at: target is at the head pivot, ignored")
            return

        # +pan turns right (negative yaw), +tilt looks down.
        pan = -math.atan2(dy, dx)
        tilt = math.atan2(-dz, math.hypot(dx, dy))
        cpan, ctilt, clamped = self.config.clamp(pan, tilt)
        if clamped:
            self.get_logger().warn(
                f"look_at: target ({describe(pan, tilt)}) is outside the head range, "
                f"clamped to {describe(cpan, ctilt)}"
            )
        duration = self.config.duration(self.current, (cpan, ctilt))
        self.pub.publish(self.config.trajectory(cpan, ctilt, duration))
        self.target = (target, clamped)
        self.publish_markers()
        self.get_logger().info(
            f"look_at {target.header.frame_id} ({msg.point.x:.2f}, {msg.point.y:.2f}, "
            f"{msg.point.z:.2f}) -> pan {cpan:+.3f} tilt {ctilt:+.3f} rad "
            f"({describe(cpan, ctilt)}) in {duration:.2f} s"
        )


def main():
    rclpy.init()
    node = HeadLookAt()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
