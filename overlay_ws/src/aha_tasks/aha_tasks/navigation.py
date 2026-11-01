"""Blocking Nav2 adapter; the node must be spun by an executor in another thread."""

import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

STATUS_NAMES = {
    GoalStatus.STATUS_CANCELED: "canceled",
    GoalStatus.STATUS_ABORTED: "aborted",
}


class Cancelled(Exception):
    pass


class Navigation:
    def __init__(self, node, locations, startup_timeout=300.0, goal_timeout=600.0):
        self.node = node
        self.markers = publish_locations(node, locations)
        self.startup_timeout = startup_timeout
        self.goal_timeout = goal_timeout
        self.client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        self.goal = None
        self.pending = None

    def wait(self, future, timeout, cancelled=None):
        deadline = time.monotonic() + timeout
        while not future.done():
            if not rclpy.ok():
                raise RuntimeError("ROS shut down while waiting for Nav2")
            if cancelled is not None and cancelled.is_set():
                raise Cancelled
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Nav2 response timed out (wall clock)")
            time.sleep(min(0.02, remaining))
        return future.result()

    def ready(self):
        deadline = time.monotonic() + self.startup_timeout
        state = self.node.create_client(GetState, "/bt_navigator/get_state")
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                if state.service_is_ready() and self.client.server_is_ready():
                    reply = self.wait(
                        state.call_async(GetState.Request()),
                        max(0.01, deadline - time.monotonic()),
                    )
                    if reply.current_state.id == State.PRIMARY_STATE_ACTIVE:
                        return
                time.sleep(0.1)
            raise TimeoutError("Nav2 did not become active before startup timeout")
        finally:
            self.node.destroy_client(state)

    def navigate(self, pose, cancelled=None):
        """Return (succeeded, message); the goal is cancelled on timeout or request."""
        target = PoseStamped()
        target.header.frame_id = "map"
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y = map(float, pose[:2])
        target.pose.orientation.z = math.sin(pose[2] / 2)
        target.pose.orientation.w = math.cos(pose[2] / 2)
        self.pending = self.client.send_goal_async(NavigateToPose.Goal(pose=target))
        try:
            self.goal = self.wait(self.pending, 10.0, cancelled)
            self.pending = None
            if not self.goal.accepted:
                self.goal = None
                return False, "Nav2 rejected the goal"
            result = self.wait(
                self.goal.get_result_async(), self.goal_timeout, cancelled
            )
        except (Cancelled, TimeoutError) as error:
            self.cancel()
            return False, "Cancelled" if isinstance(error, Cancelled) else str(error)
        self.goal = None
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            name = STATUS_NAMES.get(result.status, str(result.status))
            return False, f"Nav2 ended with status {name}"
        return True, ""

    def cancel(self):
        # A response arriving after timeout/interruption still needs cancellation.
        if self.pending is not None:
            self.goal = self.wait(self.pending, 5.0)
            self.pending = None
        if self.goal is not None and self.goal.accepted:
            result_future = self.goal.get_result_async()
            response = self.wait(self.goal.cancel_goal_async(), 5.0)
            if response.return_code != 0:
                raise RuntimeError(f"Nav2 cancellation returned {response.return_code}")
            self.wait(result_future, 5.0)
        self.goal = None


def publish_locations(node, locations):
    """Latched arrows and labels of map locations; keep the returned publisher."""
    publisher = node.create_publisher(
        MarkerArray,
        "/aha/tasks/locations",
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    markers = MarkerArray()
    for i, (name, (x, y, yaw)) in enumerate(locations.items()):
        for text in (False, True):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "labels" if text else "poses"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING if text else Marker.ARROW
            marker.pose.position.x, marker.pose.position.y = float(x), float(y)
            marker.pose.position.z = 0.6 if text else 0.1
            marker.pose.orientation.z = math.sin(yaw / 2)
            marker.pose.orientation.w = math.cos(yaw / 2)
            marker.scale.x, marker.scale.y, marker.scale.z = (
                (0.0, 0.0, 0.22) if text else (0.45, 0.08, 0.08)
            )
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                0.2,
                1.0,
                0.3,
                1.0,
            )
            marker.text = name if text else ""
            markers.markers.append(marker)
    publisher.publish(markers)
    return publisher
