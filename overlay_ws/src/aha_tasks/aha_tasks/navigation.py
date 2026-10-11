"""Small blocking Nav2 adapter; spinning remains responsive while awaiting futures."""

import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient


class Navigation:
    def __init__(self, node, startup_timeout=300.0, goal_timeout=600.0):
        self.node = node
        self.startup_timeout = startup_timeout
        self.goal_timeout = goal_timeout
        self.client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        self.goal = None
        self.pending = None

    def wait(self, future, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Nav2 response timed out (wall clock)")
            rclpy.spin_once(self.node, timeout_sec=min(0.1, remaining))
        if not future.done():
            raise RuntimeError("ROS shut down while waiting for Nav2")
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
                    if reply.current_state.id == 3:  # lifecycle ACTIVE
                        return
                rclpy.spin_once(self.node, timeout_sec=0.1)
            raise TimeoutError("Nav2 did not become active before startup timeout")
        finally:
            self.node.destroy_client(state)

    def navigate(self, pose):
        target = PoseStamped()
        target.header.frame_id = "map"
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y = map(float, pose[:2])
        target.pose.orientation.z = math.sin(pose[2] / 2)
        target.pose.orientation.w = math.cos(pose[2] / 2)
        self.pending = self.client.send_goal_async(NavigateToPose.Goal(pose=target))
        self.goal = self.wait(self.pending, 10.0)
        self.pending = None
        if not self.goal.accepted:
            self.goal = None
            self.node.get_logger().error("Nav2 rejected the goal")
            return False
        result = self.wait(self.goal.get_result_async(), self.goal_timeout)
        self.goal = None
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().error(f"Nav2 ended with status {result.status}")
            return False
        return True

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
