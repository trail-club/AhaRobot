#!/usr/bin/env python3
"""Head tilt from the D435i IMU (aha_perception.imu_tilt).

mode "state" (real_head_camera.launch.py head:=imu, no servos):
  publishes /head_controller/joint_states with the tilt measured by the IMU
  and the pan of the commanded trajectories (as manual_head.py), so the TF of
  a hand-tilted camera follows the real camera. Nothing is published before
  the first IMU sample.
mode "check" (head:=servo imu_check:=true):
  once per second, while the IMU is stationary and the servo tilt has held
  its angle since the previous check, compares the servo tilt on
  /head_controller/joint_states with the IMU tilt and warns when they differ
  by more than check_tolerance.

Both modes warn every WARN_PERIOD when |roll| exceeds roll_tolerance (the
head has no roll joint). The IMU tilt assumes a level base:
joint_head_tilt = camera tilt + tilt_offset. GravityFilter tracks the up
vector with the gyro and corrects its drift toward the accelerometer
(time_constant) while |accel| is near gravity, so the tilt follows the head
smoothly while it moves. The IMU arrives at 200 Hz, so its callback only
updates the filter (angles are computed in the timers) and the node spins in
the EventsExecutor.

  in   /camera/imu (sensor_msgs/Imu, accel + gyro, realsense.launch.py)
       /head_controller/joint_trajectory (state) or
       /head_controller/joint_states (check)
  out  /head_controller/joint_states (state, 30 Hz)
"""

import math
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.experimental import EventsExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory

from aha_perception.head_config import (
    PAN_JOINT,
    TILT_JOINT,
    TRAJECTORY_TOPIC,
    declare_head_config,
    head_position,
)
from aha_perception.head_follow import CommandedHead
from aha_perception.imu_tilt import GravityFilter, joint_tilt, tilt_roll

NODE_NAME = "aha_imu_head"
STATE_TOPIC = "/head_controller/joint_states"
RATE_HZ = 30.0
WARN_PERIOD = 5.0
SETTLED_TOLERANCE = 0.005  # rad


class ImuHead(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.mode = self.declare_parameter("mode", "state").value
        if self.mode not in ("state", "check"):
            raise ValueError(f"mode must be state or check, got {self.mode!r}")
        imu_topic = self.declare_parameter("imu_topic", "/camera/imu").value
        self.offset = float(self.declare_parameter("tilt_offset", 0.0).value)
        self.roll_tolerance = math.radians(
            float(self.declare_parameter("roll_tolerance_deg", 3.0).value)
        )
        self.check_tolerance = math.radians(
            float(self.declare_parameter("check_tolerance_deg", 3.0).value)
        )
        self.filter = GravityFilter(
            time_constant=float(self.declare_parameter("time_constant", 0.5).value),
            gyro_threshold=float(self.declare_parameter("gyro_threshold", 0.05).value),
            accel_tolerance=float(self.declare_parameter("accel_tolerance", 0.8).value),
        )
        self.logged = False
        self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)

        if self.mode == "state":
            head = declare_head_config(self)
            self.follower = CommandedHead(head)
            self.last = time.monotonic()
            self.pub = self.create_publisher(JointState, STATE_TOPIC, 10)
            self.create_subscription(
                JointTrajectory, TRAJECTORY_TOPIC, self.on_trajectory, 10
            )
            self.create_timer(1.0 / RATE_HZ, self.on_timer)
        else:
            self.servo = None
            self.settled = None
            self.create_subscription(JointState, STATE_TOPIC, self.on_servo, 10)
            self.create_timer(1.0, self.on_check)
        self.create_timer(WARN_PERIOD, self.on_report)
        self.get_logger().info(
            f"imu head ({self.mode}): {imu_topic}, tilt_offset {self.offset:+.3f} rad"
        )

    def on_imu(self, msg):
        a = msg.linear_acceleration
        w = msg.angular_velocity
        stamp = msg.header.stamp
        self.filter.update(
            stamp.sec + stamp.nanosec * 1e-9, (a.x, a.y, a.z), (w.x, w.y, w.z)
        )

    def angles(self):
        """(joint tilt, camera roll) in rad from the filter, or None."""
        up = self.filter.estimate
        if up is None:
            return None
        camera_tilt, roll = tilt_roll(up)
        if not self.logged:
            self.logged = True
            ux, uy, uz = up
            self.get_logger().info(
                f"IMU up ({ux:+.3f}, {uy:+.3f}, {uz:+.3f}), camera tilt "
                f"{math.degrees(camera_tilt):+.1f} deg, roll "
                f"{math.degrees(roll):+.1f} deg"
            )
        return joint_tilt(camera_tilt, self.offset), roll

    def on_trajectory(self, msg):
        # Only the pan is taken from the command; the tilt comes from the IMU.
        self.follower.command(msg)

    def on_timer(self):
        now = time.monotonic()
        pan, _ = self.follower.step(now - self.last)
        self.last = now
        angles = self.angles()
        if angles is None:
            self.get_logger().warn(
                "waiting for the IMU; head state not published",
                throttle_duration_sec=WARN_PERIOD,
            )
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [PAN_JOINT, TILT_JOINT]
        msg.position = [pan, angles[0]]
        self.pub.publish(msg)

    def on_servo(self, msg):
        position = head_position(msg)
        if position is not None:
            self.servo = (time.monotonic(), position[1])

    def on_check(self):
        angles = self.angles()
        if angles is None or self.servo is None or not self.filter.stationary:
            self.settled = None
            return
        received, servo_tilt = self.servo
        if time.monotonic() - received > 1.0:
            return
        # Compare only after the servo has held its angle for one check period,
        # so the accelerometer correction has settled.
        previous, self.settled = self.settled, servo_tilt
        if previous is None or abs(servo_tilt - previous) > SETTLED_TOLERANCE:
            return
        imu_tilt = angles[0]
        diff = servo_tilt - imu_tilt
        if abs(diff) > self.check_tolerance:
            self.get_logger().warn(
                f"servo tilt {math.degrees(servo_tilt):+.1f} deg differs from IMU "
                f"tilt {math.degrees(imu_tilt):+.1f} deg by {math.degrees(diff):+.1f}"
                " deg (servo calibration, camera mount or base not level)",
                throttle_duration_sec=WARN_PERIOD,
            )

    def on_report(self):
        angles = self.angles()
        if angles is None:
            return
        roll = angles[1]
        if abs(roll) > self.roll_tolerance:
            self.get_logger().warn(
                f"camera roll {math.degrees(roll):+.1f} deg is not representable "
                "by the head joints (TF ignores it)"
            )


def main():
    rclpy.init()
    try:
        node = ImuHead()
    except ValueError as e:
        rclpy.logging.get_logger(NODE_NAME).fatal(str(e))
        rclpy.try_shutdown()
        return 1
    # The IMU arrives at 200 Hz; the events executor costs about a third of
    # the CPU of the default executor per message.
    executor = EventsExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
