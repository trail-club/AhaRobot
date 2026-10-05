#!/usr/bin/env python3
"""Run the scripted apple-pick policy in the staged Gazebo world."""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint


class PickAppleTask(Node):
    APPROACH_DISTANCE = 0.49
    APPROACH_SPEED = 0.12
    APPROACH_LIFT_HEIGHT = 0.12
    GRASP_LIFT_HEIGHT = 0.16
    PICK_LIFT_HEIGHT = 0.28
    GRASP_FINGER_POSITION = 0.035
    PICKED_APPLE_MIN_Z = 0.24
    PICK_CONFIRMATION_TIME = 1.0
    PICK_CONFIRMATION_TIMEOUT = 10.0

    def __init__(self):
        super().__init__("pick_apple_task")
        self._base_command = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self._odom = None
        self._odom_sub = self.create_subscription(
            Odometry,
            "/diff_drive_controller/odom",
            self._on_odom,
            10,
        )
        self._apple_pose = None
        self._logged_apple_frames = False
        self._world_pose_sub = self.create_subscription(
            TFMessage,
            "/model/apple/pose",
            self._on_world_poses,
            10,
        )
        self._gripper = self.create_publisher(
            Float64MultiArray, "/right_gripper_controller/commands", 10
        )
        self._grasp_latch = self.create_publisher(Empty, "/apple/attach", 10)
        self._lift = ActionClient(
            self,
            FollowJointTrajectory,
            "/lift_controller/follow_joint_trajectory",
        )

    def _on_odom(self, message):
        self._odom = message

    def _on_world_poses(self, message):
        for transform in message.transforms:
            child = transform.child_frame_id
            frame_leaf = child.strip("/").replace("::", "/").split("/")[-1]
            translation = transform.transform.translation
            position = (translation.x, translation.y, translation.z)
            if frame_leaf == "apple":
                if not self._logged_apple_frames:
                    self.get_logger().info(
                        "Gazebo apple model world pose: "
                        f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                    )
                    self._logged_apple_frames = True
                self._apple_pose = position

    def _wait_for_controllers(self):
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if (
                self._base_command.get_subscription_count() > 0
                and self._odom_sub.get_publisher_count() > 0
                and self._gripper.get_subscription_count() > 0
                and self._grasp_latch.get_subscription_count() > 0
                and self._lift.server_is_ready()
            ):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            "Timed out waiting for the base, lift, right gripper, "
            "and Gazebo grasp latch"
        )

    def _set_gripper(self, positions):
        message = Float64MultiArray(data=positions)
        # Repeat the command briefly so it is received immediately after discovery.
        for _ in range(4):
            self._gripper.publish(message)
            time.sleep(0.05)
        time.sleep(0.8)

    def _attach_apple(self):
        deadline = time.monotonic() + 5.0
        while (
            self._grasp_latch.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._grasp_latch.get_subscription_count() == 0:
            raise RuntimeError("Timed out waiting for the Gazebo apple grasp latch")

        message = Empty()
        for _ in range(4):
            self._grasp_latch.publish(message)
            time.sleep(0.05)

    @staticmethod
    def _yaw(orientation):
        sin_yaw = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos_yaw = 1.0 - 2.0 * (orientation.y**2 + orientation.z**2)
        return math.atan2(sin_yaw, cos_yaw)

    def _stop_base(self):
        stop = TwistStamped()
        for _ in range(10):
            stop.header.stamp = self.get_clock().now().to_msg()
            self._base_command.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _drive_forward(self, distance):
        deadline = time.monotonic() + 30.0
        while self._odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._odom is None:
            raise RuntimeError("Timed out waiting for wheel odometry")

        start_x = self._odom.pose.pose.position.x
        start_y = self._odom.pose.pose.position.y
        heading = self._yaw(self._odom.pose.pose.orientation)
        command = TwistStamped()
        command.twist.linear.x = self.APPROACH_SPEED
        moved = 0.0
        self.get_logger().info(
            f"Driving forward {distance:.2f} m to approach the apple"
        )

        deadline = time.monotonic() + 30.0
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self._odom is not None:
                    current = self._odom.pose.pose.position
                    moved = math.cos(heading) * (current.x - start_x) + math.sin(
                        heading
                    ) * (current.y - start_y)
                    if moved >= distance:
                        break
                command.header.stamp = self.get_clock().now().to_msg()
                self._base_command.publish(command)
            else:
                raise RuntimeError(
                    f"Forward approach timed out after moving {moved:.2f} m"
                )
        finally:
            self._stop_base()
        return moved

    def _lift_right_arm(self, height):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint_l1", "joint_r1"]
        goal.trajectory.points = [
            JointTrajectoryPoint(
                positions=[0.0, height],
                time_from_start=Duration(sec=2),
            )
        ]

        sent = self._lift.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=15.0)
        if not sent.done():
            raise RuntimeError("Timed out sending the lift trajectory")
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError("The lift controller rejected the apple-pick trajectory")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done():
            raise RuntimeError("Timed out while lifting the apple")
        result = result_future.result()
        if (
            result.status != GoalStatus.STATUS_SUCCEEDED
            or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            raise RuntimeError("Lift failed: " + result.result.error_string)

    def _verify_pick(self):
        deadline = time.monotonic() + self.PICK_CONFIRMATION_TIMEOUT
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._apple_pose is None:
                continue

            apple_z = self._apple_pose[2]
            picked = apple_z >= self.PICKED_APPLE_MIN_Z
            if picked:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= self.PICK_CONFIRMATION_TIME:
                    self.get_logger().info(
                        "Pick succeeded: apple stayed above "
                        f"{self.PICKED_APPLE_MIN_Z:.2f} m for "
                        f"{self.PICK_CONFIRMATION_TIME:.1f} s"
                    )
                    return
            else:
                stable_since = None

        if self._apple_pose is None:
            raise RuntimeError(
                "Could not verify the pick because Gazebo poses for the apple "
                "were not received"
            )
        apple_z = self._apple_pose[2]
        raise RuntimeError(
            "Pick failed: apple did not stay above "
            f"{self.PICKED_APPLE_MIN_Z:.2f} m for "
            f"{self.PICK_CONFIRMATION_TIME:.1f} s "
            f"(apple z={apple_z:.2f} m)"
        )

    def run(self):
        self.get_logger().info("Waiting for the simulation controllers")
        self._wait_for_controllers()

        self.get_logger().info("Opening the right gripper")
        self._set_gripper([0.06, -0.06])

        self.get_logger().info("Raising the lift to clear the table")
        self._lift_right_arm(self.APPROACH_LIFT_HEIGHT)

        moved = self._drive_forward(self.APPROACH_DISTANCE)
        final_position = self._odom.pose.pose.position
        final_heading = self._yaw(self._odom.pose.pose.orientation)
        self.get_logger().info(
            f"Approach complete after {moved:.2f} m; odometry pose "
            f"({final_position.x:.2f}, {final_position.y:.2f}), "
            f"heading {final_heading:.2f} rad"
        )

        self.get_logger().info("Positioning the gripper at the apple")
        self._lift_right_arm(self.GRASP_LIFT_HEIGHT)

        self.get_logger().info("Closing the gripper around the apple")
        self._set_gripper(
            [self.GRASP_FINGER_POSITION, -self.GRASP_FINGER_POSITION]
        )
        self.get_logger().info("Attaching the apple in the Gazebo grasp model")
        self._attach_apple()

        self.get_logger().info("Lifting the apple")
        self._lift_right_arm(self.PICK_LIFT_HEIGHT)
        self._verify_pick()


def main(args=None):
    rclpy.init(args=args)
    task = PickAppleTask()
    try:
        task.run()
    except RuntimeError as error:
        task.get_logger().error(str(error))
        return 1
    finally:
        task.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
