#!/usr/bin/env python3
"""Show the end-effector error against the goal of the last MoveIt plan.

Every plan that move_group publishes on /display_planned_path (RViz
MotionPlanning, named states, scripts using /move_action) sets the goal: the
last waypoint is converted to a link_ree pose with /compute_fk. The current
pose comes from TF. Markers on MARKER_TOPIC draw the goal axes, a line from
the current pose to the goal and a text with the position / angle error.

Runs on the wall clock and looks up the latest TF, like the perception
helper nodes. The text is ASCII (RViz has no CJK glyphs) without spaces, see
describe_error.
"""

import math
import sys

import rclpy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

MARKER_TOPIC = "/aha/manipulation/eef_goal/markers"
PLAN_TOPIC = "/display_planned_path"
FK_SERVICE = "/compute_fk"
PERIOD_S = 0.1
AXIS_LENGTH = 0.08
NAMESPACE = "eef_goal"


def final_state(display):
    """JointState at the end of a DisplayTrajectory, or None if it is empty.

    Segments are applied in order, so a joint keeps the end of the last
    segment that moves it; joints no segment moves keep trajectory_start.
    """
    start = display.trajectory_start.joint_state
    positions = dict(zip(start.name, start.position))
    moved = False
    for segment in display.trajectory:
        points = segment.joint_trajectory
        if points.points:
            positions.update(zip(points.joint_names, points.points[-1].positions))
            moved = True
    if not moved:
        return None
    return JointState(name=list(positions), position=list(positions.values()))


def pose_error(current, goal):
    """(distance m, rotation angle rad) between two geometry_msgs Poses."""
    distance = math.dist(
        (current.position.x, current.position.y, current.position.z),
        (goal.position.x, goal.position.y, goal.position.z),
    )
    a, b = current.orientation, goal.orientation
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    norm = math.sqrt(
        (a.x**2 + a.y**2 + a.z**2 + a.w**2) * (b.x**2 + b.y**2 + b.z**2 + b.w**2)
    )
    angle = 2.0 * math.acos(min(1.0, dot / norm)) if norm > 0 else 0.0
    return distance, angle


def describe_error(distance, angle):
    # No spaces: RViz drew each space in TEXT_VIEW_FACING several glyphs wide
    # (Jazzy, Xvfb), which scattered the words.
    return f"err={distance * 1000.0:.1f}mm/{math.degrees(angle):.1f}deg"


def _point(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


class EefGoalMonitor:
    def __init__(self, node):
        self.node = node
        self.frame = node.declare_parameter("frame", "base_footprint").value
        self.link = node.declare_parameter("link", "link_ree").value
        self.goal = None
        self._pending = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.fk = node.create_client(GetPositionFK, FK_SERVICE)
        self.marker_pub = node.create_publisher(MarkerArray, MARKER_TOPIC, 1)
        node.create_subscription(DisplayTrajectory, PLAN_TOPIC, self._on_plan, 1)
        node.create_timer(PERIOD_S, self._publish)

    def _on_plan(self, msg):
        state = final_state(msg)
        if state is None:
            return
        if not self.fk.service_is_ready():
            self.node.get_logger().warning(f"{FK_SERVICE} not available")
            self._clear_goal()
            return
        request = GetPositionFK.Request()
        request.header.frame_id = self.frame
        request.fk_link_names = [self.link]
        request.robot_state = RobotState(joint_state=state)
        future = self.fk.call_async(request)
        # Only the newest plan sets the goal.
        self._pending = future
        future.add_done_callback(self._on_fk)

    def _on_fk(self, future):
        if future is not self._pending:
            return
        response = future.result()
        if (
            response is None
            or response.error_code.val != MoveItErrorCodes.SUCCESS
            or not response.pose_stamped
        ):
            self.node.get_logger().warning("FK of the planned goal failed")
            self._clear_goal()
            return
        self.goal = response.pose_stamped[0].pose

    def _clear_goal(self):
        # The error to the previous plan's goal would look like this plan's.
        # Its markers are no longer sent and expire after their lifetime.
        self.goal = None
        self._pending = None

    def _current(self):
        try:
            t = self.tf_buffer.lookup_transform(self.frame, self.link, Time())
        except TransformException:
            return None
        pose = Pose()
        pose.position = _point(
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        )
        pose.orientation = t.transform.rotation
        return pose

    def _marker(self, marker_id, marker_type, color):
        m = Marker()
        m.header.frame_id = self.frame
        m.ns = NAMESPACE
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.color = color
        # Republished every cycle; a marker not sent again (the error line
        # without TF) or a stopped node disappears after the lifetime.
        m.lifetime = Duration(seconds=1.0).to_msg()
        return m

    def markers(self, current, goal):
        """MarkerArray for the goal axes, the error line and the error text."""
        # Points are in the marker pose, so RViz rotates the axes to the goal.
        axes = self._marker(0, Marker.LINE_LIST, _color(1.0, 1.0, 1.0))
        axes.pose = goal
        axes.scale.x = 0.006
        for axis, color in (
            (_point(AXIS_LENGTH, 0, 0), _color(1.0, 0.2, 0.2)),
            (_point(0, AXIS_LENGTH, 0), _color(0.2, 1.0, 0.2)),
            (_point(0, 0, AXIS_LENGTH), _color(0.3, 0.5, 1.0)),
        ):
            axes.points += [_point(0, 0, 0), axis]
            axes.colors += [color, color]
        origin = goal.position
        result = [axes]
        if current is None:
            text = "err=no_TF_" + self.link
            anchor = origin
        else:
            line = self._marker(1, Marker.LINE_STRIP, _color(1.0, 0.85, 0.1))
            line.scale.x = 0.004
            line.points = [current.position, origin]
            result.append(line)
            text = describe_error(*pose_error(current, goal))
            anchor = current.position
        label = self._marker(2, Marker.TEXT_VIEW_FACING, _color(1.0, 1.0, 1.0))
        label.scale.z = 0.04
        label.pose.position = _point(anchor.x, anchor.y, anchor.z + 0.12)
        label.text = text
        result.append(label)
        return MarkerArray(markers=result)

    def _publish(self):
        if self.goal is None:
            return
        self.marker_pub.publish(self.markers(self._current(), self.goal))


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("aha_eef_goal_monitor")
    EefGoalMonitor(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
