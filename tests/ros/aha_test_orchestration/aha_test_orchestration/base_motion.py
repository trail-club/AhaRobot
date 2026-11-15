"""Short base moves on the odometry (ROS), e.g. to a place server's
suggested_base_pose in front of a shelf.

A stand-in until Nav2 runs (aha_navigation starts only SLAM): the
differential base turns towards the target, drives straight (backwards when
the target is behind) and turns to the target heading. The odometry lags the
base (in Gazebo about 0.8 s of sim time), so stopping on it overshoots; each
step is therefore a timed command (angle or distance / speed in the node's
clock, i.e. sim time in the simulation), then the base stops, the pose is
read once it has not changed for ``settle_s`` and the remaining error is
driven again (``rounds``). There is no obstacle check, so moves are limited
to ``max_distance``. Commands are geometry_msgs/TwistStamped with a zero
stamp on /diff_drive_controller/cmd_vel (the controller stamps them on
receipt with its own clock, and its cmd_vel_timeout stops the base if this
node stops publishing), like sim_control_panel.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from geometry_msgs.msg import TwistStamped

CMD_VEL_TOPIC = "/diff_drive_controller/cmd_vel"
PERIOD_S = 0.05  # wall s between commands


@dataclass
class MoveOptions:
    max_distance: float = 0.6  # m, longer moves are refused
    linear_speed: float = 0.05  # m/s
    angular_speed: float = 0.2  # rad/s
    xy_tolerance: float = 0.015  # m, final position error accepted
    yaw_tolerance: float = math.radians(1.5)  # final heading error accepted
    min_drive: float = 0.01  # m, shorter drives only turn to the final heading
    rounds: int = 4  # move, settle, measure; again while out of tolerance
    settle_s: float = 1.5  # node-clock s without pose change before measuring
    settle_timeout_s: float = 60.0  # wall s
    step_timeout_s: float = 120.0  # wall s per step (the sim may run slowly)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def plan_steps(start, goal, min_drive: float = 0.01) -> list[tuple[str, float]]:
    """Steps from ``start`` to ``goal`` (x, y, yaw): ("turn", absolute
    heading), ("drive", signed distance along it), ("turn", goal yaw). The
    base drives backwards when the target is behind it."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    distance = math.hypot(dx, dy)
    steps = []
    if distance >= min_drive:
        heading = math.atan2(dy, dx)
        if abs(wrap(heading - start[2])) > math.pi / 2:
            heading, distance = wrap(heading + math.pi), -distance
        steps += [("turn", heading), ("drive", distance)]
    steps.append(("turn", wrap(goal[2])))
    return steps


def pose_of(T) -> tuple[float, float, float]:
    return float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0])


class BaseMover:
    """``transform(target, source)`` returns the 4x4 TF or None (from a
    listener with its own spin thread); ``spin_once()`` processes the node's
    callbacks (the clock in the simulation)."""

    def __init__(
        self,
        node,
        transform,
        spin_once,
        options: MoveOptions | None = None,
        odom_frame: str = "odom",
        base_frame: str = "base_footprint",
    ):
        self.node = node
        self.options = options or MoveOptions()
        self.transform = transform
        self.spin_once = spin_once
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.pub = node.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def pose(self):
        T = self.transform(self.odom_frame, self.base_frame)
        return None if T is None else pose_of(T)

    def command(self, linear: float, angular: float):
        msg = TwistStamped()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.pub.publish(msg)

    def stop(self):
        for _ in range(3):
            self.command(0.0, 0.0)
            time.sleep(PERIOD_S)

    def settled_pose(self):
        """The pose once it has not changed for ``settle_s`` of the node's
        clock (or at the timeout); None without the TF."""
        o = self.options
        deadline = time.monotonic() + o.settle_timeout_s
        last, since = self.pose(), self.now()
        while time.monotonic() < deadline:
            self.spin_once()
            time.sleep(PERIOD_S)
            pose = self.pose()
            if pose is None or last is None:
                last, since = pose, self.now()
                continue
            moved = math.hypot(pose[0] - last[0], pose[1] - last[1]) > 0.001
            if moved or abs(wrap(pose[2] - last[2])) > math.radians(0.1):
                last, since = pose, self.now()
            elif self.now() - since >= o.settle_s:
                return pose
        return last

    def timed(self, linear: float, angular: float, duration: float) -> bool:
        """Send the command for ``duration`` s of the node's clock."""
        if duration <= 0.0:
            return True
        start = self.now()
        deadline = time.monotonic() + self.options.step_timeout_s
        while time.monotonic() < deadline:
            self.spin_once()
            if self.now() - start >= duration:
                return True
            self.command(linear, angular)
            time.sleep(PERIOD_S)
        return False

    def move_to(
        self,
        x: float,
        y: float,
        yaw: float,
        max_distance: float | None = None,
        linear_speed: float | None = None,
    ) -> dict:
        """Move to (x, y, yaw) in odom; report with ok, the steps of every
        round (``rounds``, non-empty whenever the base was commanded) and the
        final error (or the reason it stopped).
        ``max_distance`` / ``linear_speed`` replace the options' for this
        move (e.g. backing off from a furniture to observe it)."""
        o = self.options
        if max_distance is not None or linear_speed is not None:
            o = replace(
                o,
                max_distance=o.max_distance if max_distance is None else max_distance,
                linear_speed=o.linear_speed if linear_speed is None else linear_speed,
            )
        pose = self.settled_pose()
        if pose is None:
            return {
                "ok": False,
                "message": f"no TF {self.odom_frame} <- {self.base_frame}",
            }
        distance = math.hypot(x - pose[0], y - pose[1])
        if distance > o.max_distance:
            return {
                "ok": False,
                "message": f"{distance:.2f} m is beyond max_distance {o.max_distance:.2f} m",
            }
        began, rounds = time.monotonic(), []
        for _ in range(o.rounds):
            error_xy = math.hypot(x - pose[0], y - pose[1])
            error_yaw = abs(wrap(yaw - pose[2]))
            if error_xy <= o.xy_tolerance and error_yaw <= o.yaw_tolerance:
                break
            steps = plan_steps(pose, (x, y, yaw), o.min_drive)
            rounds.append([[kind, round(value, 4)] for kind, value in steps])
            try:
                heading = pose[2]
                for kind, value in steps:
                    if kind == "turn":
                        angle = wrap(value - heading)
                        if abs(angle) < o.yaw_tolerance / 2:
                            continue  # nothing to turn: no command, no settle
                        done = self.timed(
                            0.0,
                            math.copysign(o.angular_speed, angle),
                            abs(angle) / o.angular_speed,
                        )
                        heading = value
                    else:
                        done = self.timed(
                            math.copysign(o.linear_speed, value),
                            0.0,
                            abs(value) / o.linear_speed,
                        )
                    if not done:
                        # rounds: the base may have moved part of the way.
                        return {
                            "ok": False,
                            "message": f"{kind} {value:.3f} timed out",
                            "rounds": rounds,
                        }
                    # Each step from where the previous one really ended.
                    self.stop()
                    if kind == "turn":
                        settled = self.settled_pose()
                        heading = settled[2] if settled else heading
            finally:
                self.stop()
            pose = self.settled_pose() or pose
        error_xy = math.hypot(x - pose[0], y - pose[1])
        error_yaw = abs(wrap(yaw - pose[2]))
        ok = error_xy <= o.xy_tolerance and error_yaw <= o.yaw_tolerance
        return {
            "ok": ok,
            "message": "" if ok else f"{error_xy:.3f} m off after {len(rounds)} rounds",
            "rounds": rounds,
            "error_xy": round(error_xy, 4),
            "error_yaw_deg": round(math.degrees(error_yaw), 2),
            "move_s": round(time.monotonic() - began, 2),
        }
