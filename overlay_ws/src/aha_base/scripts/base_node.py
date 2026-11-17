#!/usr/bin/env python3
"""Differential-drive base on one ODrive v3.6 over USB.

Same names and types as the simulation's diff_drive_controller, so teleop
and Nav2 drive the real base unchanged:

  /diff_drive_controller/cmd_vel  geometry_msgs/TwistStamped (sub;
                                  Twist with `stamped:=false`)
  /diff_drive_controller/odom     nav_msgs/Odometry (pub)
  /tf                             odom -> base_footprint (`publish_tf`)

Control follows upstream astra_controller base_controller.py (CAN): position
control with trapezoidal trajectories; every cycle the setpoint moves by
speed x dt and stays within `max_lead_turns` of the measured position.

Behaviour:
  - Start: find the ODrive (exit 1 after `connect_timeout`), clear errors,
    require both axes IDLE with motor calibrated and encoder ready (exit 1
    otherwise; no calibration is run). Setpoint = present position (no
    jump), then closed loop. Gains, Kt and current_lim stored on the device
    are used unless given as parameters. Changed values stay in RAM only.
  - The ODrive watchdog (`watchdog_timeout`) IDLEs the axes if this node
    stops feeding it.
  - cmd_vel older than `cmd_timeout` counts as zero. Wheel speeds are scaled
    together to at most `traj_vel_limit` (keeps the curvature).
  - Odometry pose from wheel positions, twist from vel_estimate.
  - An axis leaving closed loop or a USB error: both axes IDLE, exit 1.
  - SIGINT/SIGTERM: both axes IDLE, RAM values restored, exit 0.
"""

import math
import os
import signal
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry as OdometryMsg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from tf2_ros import TransformBroadcaster

try:
    import aha_base  # noqa: F401
except ModuleNotFoundError:
    # Shell sourced before aha_base was built: find it via the ament index.
    from ament_index_python.packages import get_package_prefix

    sys.path.append(
        os.path.join(
            get_package_prefix("aha_base"),
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
    )

from aha_base.diff_drive import (  # noqa: E402
    Kinematics,
    Odometry,
    advance_setpoint,
    limit_wheel_speeds,
)
from aha_base.odrive_usb import (  # noqa: E402
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    DriveSettings,
    OdriveBase,
    OdriveError,
    find_odrive,
    normalize_serial,
)

PARAMS = {
    "serial_number": "",  # hex, '' = first ODrive found
    "connect_timeout": 10.0,  # [s]
    "left_axis": 0,
    "right_axis": 1,
    "left_sign": -1,  # motor turn direction for forward wheel roll
    "right_sign": 1,
    "wheel_diameter": 0.139,  # [m]
    "wheel_separation": 0.44,  # [m]
    "rate_hz": 20.0,
    "cmd_timeout": 0.5,  # [s]
    "max_lead_turns": 2.0,  # [turn]
    "stamped": True,
    "publish_tf": True,
    "odom_frame": "odom",
    "base_frame": "base_footprint",
    "traj_vel_limit": 2.0,  # [turn/s]
    "traj_accel_limit": 1.0,  # [turn/s^2]
    "traj_decel_limit": 1.0,  # [turn/s^2]
    "pos_gain": -1.0,  # < 0: keep the device value
    "vel_gain": -1.0,
    "vel_integrator_gain": -1.0,
    "current_lim": -1.0,  # [A]
    "watchdog_timeout": 0.5,  # [s], 0 = off
}


class BaseNode(Node):
    def __init__(self, stopping: threading.Event):
        super().__init__("aha_base")
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in PARAMS}
        self.p = p
        self.stopping = stopping
        self.exit_code = 0

        if {p["left_axis"], p["right_axis"]} != {0, 1}:
            raise ValueError("left_axis / right_axis must be 0 and 1")
        if p["left_sign"] not in (-1, 1) or p["right_sign"] not in (-1, 1):
            raise ValueError("left_sign / right_sign must be -1 or 1")
        for name in ("rate_hz", "cmd_timeout", "max_lead_turns", "connect_timeout"):
            if not p[name] > 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("traj_vel_limit", "traj_accel_limit", "traj_decel_limit"):
            if not p[name] > 0:
                raise ValueError(f"{name} must be > 0")
        self.serial = normalize_serial(p["serial_number"])
        self.kin = Kinematics(p["wheel_diameter"], p["wheel_separation"])
        self.signs = (p["left_sign"], p["right_sign"])
        self.settings = DriveSettings(
            traj_vel_limit=p["traj_vel_limit"],
            traj_accel_limit=p["traj_accel_limit"],
            traj_decel_limit=p["traj_decel_limit"],
            pos_gain=p["pos_gain"],
            vel_gain=p["vel_gain"],
            vel_integrator_gain=p["vel_integrator_gain"],
            current_lim=p["current_lim"],
            watchdog_timeout=p["watchdog_timeout"],
        )

        self.base: OdriveBase | None = None
        self.odom = Odometry(self.kin)
        self.setpoints = [0.0, 0.0]
        self.cmd = (0.0, 0.0)
        self.cmd_time = -math.inf
        self.cmd_lock = threading.Lock()
        self.last_tick: float | None = None

        self.odom_pub = self.create_publisher(
            OdometryMsg, "/diff_drive_controller/odom", 10
        )
        self.tf = TransformBroadcaster(self) if p["publish_tf"] else None
        self.create_subscription(
            TwistStamped if p["stamped"] else Twist,
            "/diff_drive_controller/cmd_vel",
            self.on_cmd,
            qos_profile_sensor_data,
        )
        self.timer = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        log = self.get_logger()
        log.info(
            f"looking for ODrive on USB (serial {self.serial or 'any'}, "
            f"{self.p['connect_timeout']:g} s)"
        )
        dev = find_odrive(self.serial, self.p["connect_timeout"])
        log.info(f"ODrive {format(int(dev.serial_number), 'X')} found")
        self.base = OdriveBase(
            dev, (self.p["left_axis"], self.p["right_axis"]), self.settings
        )
        positions = self.base.start()
        for line in self.base.log:
            log.info(line)
        self.setpoints = list(positions)
        self.odom.update(*self.wheel_turns(positions))
        self.last_tick = time.monotonic()
        self.timer = self.create_timer(1.0 / self.p["rate_hz"], self.tick)
        log.info("closed-loop control running")

    def release(self) -> None:
        """IDLE both axes and restore the RAM values (safe to call twice)."""
        if self.timer is not None:
            self.timer.cancel()
        if self.base is None:
            return
        failures = self.base.stop()
        for f in failures:
            self.get_logger().error(f"stop: {f}")
        if not failures:
            self.get_logger().info("axes IDLE, RAM values restored")
        self.base = None

    def fail(self, reason: str) -> None:
        self.get_logger().error(reason)
        self.exit_code = 1
        if self.timer is not None:
            self.timer.cancel()
        self.stopping.set()

    # -- callbacks -------------------------------------------------------------

    def on_cmd(self, msg) -> None:
        twist = msg.twist if isinstance(msg, TwistStamped) else msg
        with self.cmd_lock:
            self.cmd = (twist.linear.x, twist.angular.z)
            self.cmd_time = time.monotonic()

    def wheel_turns(self, motor: list[float]) -> tuple[float, float]:
        """Motor turns (or turn/s) -> wheel turns, forward positive."""
        return motor[0] * self.signs[0], motor[1] * self.signs[1]

    def tick(self) -> None:
        if self.base is None or self.stopping.is_set():
            return
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        try:
            readings = self.base.read()
            bad = [
                f"axis{i}"
                for i, (_, _, st) in zip(self.base.axis_ids, readings)
                if st != AXIS_STATE_CLOSED_LOOP_CONTROL
            ]
            if bad:
                self.fail("left closed-loop control: " + self.base.describe())
                return
            pos = [r[0] for r in readings]
            vel = [r[1] for r in readings]

            with self.cmd_lock:
                linear, angular = self.cmd
                if now - self.cmd_time > self.p["cmd_timeout"]:
                    linear = angular = 0.0
            left, right = limit_wheel_speeds(
                *self.kin.wheel_speeds(linear, angular), self.p["traj_vel_limit"]
            )
            motor_speed = (left * self.signs[0], right * self.signs[1])
            # Cap dt so a stalled cycle cannot make the setpoint jump.
            step = min(dt, 3.0 / self.p["rate_hz"])
            self.setpoints = [
                advance_setpoint(sp, v, step, x, self.p["max_lead_turns"])
                for sp, v, x in zip(self.setpoints, motor_speed, pos)
            ]
            self.base.command(self.setpoints)
        except OdriveError as e:
            self.fail(str(e))
            return
        except Exception as e:  # noqa: BLE001  (USB link lost, etc.)
            self.fail(f"ODrive communication failed: {e!r}")
            return

        self.odom.update(*self.wheel_turns(pos))
        self.publish(*self.kin.body_twist(*self.wheel_turns(vel)))

    def publish(self, linear: float, angular: float) -> None:
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = self.odom.quaternion()
        msg = OdometryMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.p["odom_frame"]
        msg.child_frame_id = self.p["base_frame"]
        msg.pose.pose.position.x = self.odom.x
        msg.pose.pose.position.y = self.odom.y
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = linear
        msg.twist.twist.angular.z = angular
        self.odom_pub.publish(msg)
        if self.tf is not None:
            t = TransformStamped()
            t.header = msg.header
            t.child_frame_id = msg.child_frame_id
            t.transform.translation.x = self.odom.x
            t.transform.translation.y = self.odom.y
            t.transform.rotation = msg.pose.pose.orientation
            self.tf.sendTransform(t)


def ignore_stop_signals():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def main():
    # A signal only requests the stop; the axes are IDLEd in order below.
    # Further signals are ignored (ros2 launch delivers Ctrl-C twice).
    stopping = threading.Event()

    def request_stop(_signum, _frame):
        ignore_stop_signals()
        stopping.set()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        node = BaseNode(stopping)
    except ValueError as e:
        print(f"base_node: {e}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(2)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    code = 0
    try:
        node.start()
        while not stopping.is_set() and node.context.ok():
            executor.spin_once(timeout_sec=0.1)
        code = node.exit_code
    except OdriveError as e:
        node.get_logger().error(str(e))
        code = 1
    finally:
        ignore_stop_signals()
        node.release()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    # libfibre's threads can keep the interpreter from exiting.
    os._exit(code)


if __name__ == "__main__":
    main()
