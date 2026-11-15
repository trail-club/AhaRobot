#!/usr/bin/env python3
"""Point the head at a 3D target and keep it there.

Subscribes /aha/perception/look_at (geometry_msgs/PointStamped, any TF frame),
computes pan/tilt about the head pivot (link_head_tilt origin, where the pan
and tilt axes intersect) and sends a trajectory to head_controller.
Limits, speed and duration come from config/head.yaml.

The target is static in its frame. A target in one of head.track_frames
(map, odom) is aimed at again every head.track_period with the latest TF, so
the head keeps looking at it while the base moves. A new trajectory is sent
only when the aim moved by more than head.track_deadband, or when the head
has stopped more than STOPPED_OFF away from the aim (it overshot or was held
back). Each step lasts the time from the measured head angle at max_speed
(at least head.track_period), so a small re-aim never hurries a long move.
A new target in the same tracked frame as the last one (a followed person)
moves the head in the same steps instead of head.min_duration. Targets in
other frames are aimed at once: a point in a camera frame would move with
the head, and one fixed to the base needs no update. A head trajectory from
another node (control panel, teleop) ends the tracking, so they do not fight.

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
A target behind the robot is clamped to the pan limit on the side the head
already is, unless the other limit is SIDE_HYSTERESIS nearer.
"""

from collections import deque
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
from tf2_ros import Buffer, TransformException
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.head_config import (
    TRAJECTORY_TOPIC,
    declare_head_config,
    describe,
    head_position,
    look_angles,
)
from aha_perception.tf_jump import (
    TF_RESET_MIN_INTERVAL_S,
    JumpDetectingTransformListener,
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
LOG_THROTTLE_S = 2.0  # a stream of targets (a followed person) logs this often
STOPPED_OFF = 0.05  # rad between a stopped head and the aim that sends again
STILL_SPEED = 0.02  # rad/s; slower head joints count as stopped
RESEND_AFTER = 1.0  # s (wall) after a command before a stopped head counts
SIDE_HYSTERESIS = math.radians(30.0)  # a target behind keeps the head's side
OWN_COMMANDS = 20  # recent own trajectories, to tell them from other nodes'


class HeadLookAt(Node):
    def __init__(self):
        super().__init__("aha_perception_head_look_at")
        self.config = declare_head_config(self)
        self.track_period = self.declare_parameter("head.track_period", 0.1).value
        self.track_deadband = self.declare_parameter("head.track_deadband", 0.01).value
        self.track_frames = set(
            self.declare_parameter("head.track_frames", ["map", "odom"]).value
        )
        self.tf_buffer = Buffer()
        self.tf_listener = JumpDetectingTransformListener(
            self.tf_buffer, self, self.on_tf_jump
        )
        self.last_tf_reset = -math.inf
        self.seen_description = False
        self.current = None
        # Last accepted target for the markers and tracking: (PointStamped, clamped).
        self.target = None
        self.commanded = None  # (pan, tilt) of the last trajectory sent
        self.sent_at = -math.inf  # monotonic time of the last trajectory
        self.moving = False
        self.own = deque(maxlen=OWN_COMMANDS)
        self.create_subscription(
            JointState, "/joint_states", self.on_joint_states, qos_profile_sensor_data
        )
        self.create_subscription(
            PointStamped, "/aha/perception/look_at", self.on_target, 10
        )
        self.create_subscription(JointTrajectory, TRAJECTORY_TOPIC, self.on_command, 10)
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
        self.create_timer(self.track_period, self.track)

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
            speeds = head_position(
                JointState(name=msg.name, position=list(msg.velocity))
            )
            self.moving = speeds is not None and max(map(abs, speeds)) > STILL_SPEED

    def on_command(self, msg):
        """End tracking when another node moves the head."""
        if self.target is None or not msg.points:
            return
        if tuple(msg.points[-1].positions) in self.own:
            return
        self.target = None
        self.clear_markers()
        self.get_logger().info("Another node moved the head: stopped tracking")

    def aim(self, target):
        """(pan, tilt, clamped, raw pan, raw tilt) for target, or None without TF."""
        # No timeout: waiting would only block this single-threaded node, as
        # /tf is handled by the same executor and cannot arrive meanwhile.
        to_base = self.tf_buffer.lookup_transform(
            BASE_FRAME, target.header.frame_id, Time()
        )
        pivot = self.tf_buffer.lookup_transform(
            BASE_FRAME, PIVOT_FRAME, Time()
        ).transform.translation
        p = do_transform_point(target, to_base).point
        angles = look_angles(p.x - pivot.x, p.y - pivot.y, p.z - pivot.z)
        if angles is None:
            return None
        pan, tilt = angles
        cpan, ctilt, clamped = self.config.clamp(pan, tilt)
        if cpan != pan and self.commanded is not None:
            # Behind the robot the nearer limit flips from side to side as the
            # base turns; stay at the head's side unless the other is clearly
            # nearer, instead of swinging the head across.
            other = (
                self.config.pan_min
                if cpan == self.config.pan_max
                else self.config.pan_max
            )
            if abs(other - self.commanded[0]) < abs(cpan - self.commanded[0]):
                near = abs(math.remainder(pan - cpan, math.tau))
                far = abs(math.remainder(pan - other, math.tau))
                if far < near + SIDE_HYSTERESIS:
                    cpan = other
        return cpan, ctilt, clamped, pan, tilt

    def send(self, pan, tilt, duration):
        trajectory = self.config.trajectory(pan, tilt, duration)
        self.own.append(tuple(trajectory.points[-1].positions))
        self.pub.publish(trajectory)
        self.commanded = (pan, tilt)
        self.sent_at = time.monotonic()

    def on_target(self, msg):
        # Treat the target as static in its frame: use the latest transforms.
        target = PointStamped(header=msg.header, point=msg.point)
        target.header.stamp = Time().to_msg()
        try:
            aimed = self.aim(target)
        except TransformException as e:
            self.get_logger().warn(f"look_at: TF unavailable: {e}")
            return
        if aimed is None:
            self.get_logger().warn("look_at: target is at the head pivot, ignored")
            return
        cpan, ctilt, clamped, pan, tilt = aimed
        if clamped:
            self.get_logger().warn(
                f"look_at: target ({describe(pan, tilt)}) is outside the head range, "
                f"clamped to {describe(cpan, ctilt)}",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
        frame = target.header.frame_id
        if (
            self.target is not None
            and self.target[0].header.frame_id == frame
            and frame in self.track_frames
            and self.current is not None
        ):
            delta = max(abs(cpan - self.current[0]), abs(ctilt - self.current[1]))
            duration = max(delta / self.config.max_speed, self.track_period)
        else:
            duration = self.config.duration(self.current, (cpan, ctilt))
        self.send(cpan, ctilt, duration)
        self.target = (target, clamped)
        self.publish_markers()
        self.get_logger().info(
            f"look_at {target.header.frame_id} ({msg.point.x:.2f}, {msg.point.y:.2f}, "
            f"{msg.point.z:.2f}) -> pan {cpan:+.3f} tilt {ctilt:+.3f} rad "
            f"({describe(cpan, ctilt)}) in {duration:.2f} s",
            throttle_duration_sec=LOG_THROTTLE_S,
        )

    def track(self):
        """Aim at a map/odom target again as the base moves under it."""
        if self.target is None or self.commanded is None:
            return
        target, _ = self.target
        if target.header.frame_id not in self.track_frames:
            return
        try:
            aimed = self.aim(target)
        except TransformException:
            return  # keep the last aim until TF is back
        if aimed is None:
            return
        cpan, ctilt, clamped, _, _ = aimed
        delta = max(abs(cpan - self.commanded[0]), abs(ctilt - self.commanded[1]))
        move = delta
        if self.current is not None:
            move = max(abs(cpan - self.current[0]), abs(ctilt - self.current[1]))
        stopped_off = (
            self.current is not None
            and not self.moving
            and move > STOPPED_OFF
            and time.monotonic() - self.sent_at > RESEND_AFTER
        )
        if delta <= self.track_deadband and not stopped_off:
            return
        if stopped_off:
            self.get_logger().info(
                f"Head stopped {move:.2f} rad from the target: sending it again",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
        # From the measured angle: a small re-aim must not hurry a long move.
        self.send(cpan, ctilt, max(move / self.config.max_speed, self.track_period))
        self.target = (target, clamped)


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
