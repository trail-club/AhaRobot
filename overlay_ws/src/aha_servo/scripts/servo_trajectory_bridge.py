#!/usr/bin/env python3
"""JointTrajectory bridge for one Feetech STS servo bus.

This node mirrors the interface of ros2_control's JointTrajectoryController
for `controller_name`, so the same UIs and scripts (sim_control_panel.py,
head_look_at.py, teleop_head.py) drive the simulation and the real servos:

  /<controller>/joint_trajectory         trajectory_msgs/JointTrajectory (sub)
  /<controller>/follow_joint_trajectory  control_msgs/action/FollowJointTrajectory
  /<controller>/joint_states             sensor_msgs/JointState (pub, this group
                                         only; merge into /joint_states with
                                         joint_state_publisher's source_list)

One node drives one bus with N joints, configured by a parameter file in the
aha_servo/joint_map.py format (config/head.yaml):

  ros2 run aha_servo servo_trajectory_bridge.py --ros-args \\
    --params-file $(ros2 pkg prefix aha_servo)/share/aha_servo/config/head.yaml

Behaviour:
  - Start: ping every servo (exit 1 if one is missing), goal = present
    position (no jump), torque on.
  - Topic: only the final point is used, reached in its time_from_start.
    An empty trajectory holds the current position (read at that moment).
  - Action: points are sent in order at their time_from_start; succeeds when
    every joint is within tolerance of the final point (goal_tolerance, or
    the `goal_tolerance` parameter), aborts `goal_timeout` s (or
    goal_time_tolerance) after the last point. Success needs a fresh state of
    every goal joint (read within `state_timeout` s); while one is stale the
    goal waits and aborts at the deadline. Cancel and the deadline abort hold
    the position read at that moment. A newer topic command or goal preempts
    (aborts) the running goal.
  - Partial joint lists are accepted; unknown joints are rejected.
  - Positions are clamped to the joint min/max (warning). Each move's servo
    speed is |delta| / segment duration, capped by `max_speed`.
  - A joint without a fresh state gets no goal (warning): a segment due
    while its state is stale is skipped for it, and a hold (cancel, abort,
    empty trajectory) leaves a joint that cannot be read at its last goal.
  - header.stamp, velocities and accelerations are ignored (start on receipt).
  - Communication errors are logged (throttled) and the loop keeps running.
    joint_states carries only the joints read in that cycle (nothing while
    the bus is silent).
  - SIGINT/SIGTERM: the state timer and running goals stop first, then torque
    off and port closed, then the node and rclpy shut down. Wall clock only.
"""

import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import rclpy
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.exceptions import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    import aha_servo  # noqa: F401
except ModuleNotFoundError:
    # Shell sourced before aha_servo was built: find it via the ament index.
    from ament_index_python.packages import get_package_prefix

    sys.path.append(
        os.path.join(
            get_package_prefix("aha_servo"),
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
    )

from aha_servo.joint_map import BUS_DEFAULTS, JOINT_DEFAULTS, parse_config
from aha_servo.sts import ADDR_TORQUE_LIMIT, POS_MAX, POS_MIN, StsBus, StsError, le16

FEEDBACK_PERIOD = 0.05
LOG_THROTTLE_S = 2.0
# Consecutive failed cycles (in seconds of the loop) before an error is logged.
FAIL_ERROR_S = 2.0
# How often the main loop checks for a stop request.
SPIN_TIMEOUT_S = 0.1
# Raised by publish() / goal state changes once the context is shut down
# (RCLError) or the entity destroyed (InvalidHandle).
DEAD_CONTEXT_ERRORS = (_rclpy.RCLError, InvalidHandle)

Result = FollowJointTrajectory.Result


class StartupError(Exception):
    pass


@dataclass
class Segment:
    send_at: float  # monotonic time to send the goal
    duration: float  # s to reach targets
    targets: dict  # joint -> rad (already clamped)
    from_present: bool  # speed from the present position (first segment)


class ServoTrajectoryBridge(Node):
    def __init__(self, stopping):
        super().__init__("aha_servo_bridge")
        self.cfg = parse_config(self._read_parameters())
        self.joints = {j.name: j for j in self.cfg.joints}
        self.prefix = f"/{self.cfg.controller_name}"
        self.stopping = stopping  # threading.Event, set by SIGINT/SIGTERM
        self.bus = None
        self.timer = None
        self.lock = threading.Lock()
        self.positions = {}  # rad, latest read
        self.read_times = {}  # monotonic time of the latest successful read
        self.present_steps = {}
        self.goal_steps = {}  # last commanded steps
        self.segments = deque()
        self.plan_seq = 0
        self.failed_cycles = 0
        self.bus_down = False

    def _read_parameters(self):
        """Nested dict in the joint_map layout from this node's parameters."""
        dynamic = ParameterDescriptor(dynamic_typing=True)
        params = {
            key: self.declare_parameter(key, default, dynamic).value
            for key, default in BUS_DEFAULTS.items()
        }
        params["joints"] = list(self.declare_parameter("joints", [""]).value)
        if params["joints"] == [""]:
            raise ValueError(
                "parameter `joints` is not set (pass the config with --params-file)"
            )
        for name in params["joints"]:
            params[name] = {
                key: self.declare_parameter(f"{name}.{key}", default, dynamic).value
                for key, default in JOINT_DEFAULTS.items()
            }
        return params

    # Startup / shutdown ---------------------------------------------------

    def start(self):
        cfg = self.cfg
        log = self.get_logger()
        try:
            self.bus = StsBus(cfg.port, cfg.baud, timeout=cfg.timeout)
        except (OSError, ValueError) as e:
            raise StartupError(
                f"cannot open {cfg.port}: {e}. Check the USB cable, the port "
                "parameter and permissions (dialout / chmod)"
            ) from e

        missing = [
            f"{j.name} (ID{j.id})" for j in cfg.joints if not self.bus.ping(j.id)
        ]
        if missing:
            raise StartupError(
                f"no reply from {', '.join(missing)} on {cfg.port} @ {cfg.baud}. Check the "
                "12 V supply, Serial Forwarding on the Waveshare board, the baud rate and the ids"
            )

        for j in cfg.joints:
            steps, _speed, _load = self.bus.read_state(j.id)
            if cfg.torque_limit > 0:
                self.bus.write(j.id, ADDR_TORQUE_LIMIT, le16(cfg.torque_limit))
            # Goal must equal the present position before torque on, or the servo jumps.
            self.bus.write_goal_position(j.id, steps)
            self.bus.set_torque(j.id, True)
            rad = j.to_rad(steps)
            self.present_steps[j.name] = self.goal_steps[j.name] = steps
            self.positions[j.name] = rad
            self.read_times[j.name] = time.monotonic()
            lo, hi = j.step_range()
            note = (
                ""
                if j.min <= rad <= j.max
                else "  (outside the limits; the next command clamps)"
            )
            log.info(
                f"{j.name}: ID{j.id} at {rad:+.3f} rad ({steps} steps), "
                f"limits [{j.min:+.3f}, {j.max:+.3f}] rad = steps [{lo}, {hi}]{note}"
            )
            if lo < POS_MIN or hi > POS_MAX:
                log.warning(
                    f"{j.name}: limits exceed the servo range {POS_MIN}..{POS_MAX}; "
                    "commands are cut there (re-check zero with servo_calibrate.py)"
                )

        self.state_pub = self.create_publisher(
            JointState, f"{self.prefix}/joint_states", 10
        )
        self.create_subscription(
            JointTrajectory, f"{self.prefix}/joint_trajectory", self._on_trajectory, 10
        )
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            f"{self.prefix}/follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.timer = self.create_timer(
            1.0 / cfg.rate_hz,
            self._tick,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        log.info(
            f"{cfg.controller_name}: {len(cfg.joints)} servos on {cfg.port} @ {cfg.baud}, "
            "torque on"
        )

    def stop(self):
        """First shutdown step: no more state publishing or goal execution.

        Running goals see `stopping` and abort within FEEDBACK_PERIOD; the
        executor shutdown that follows waits for them.
        """
        self.stopping.set()
        if self.timer is not None:
            self.timer.cancel()

    def release(self):
        """Torque off and close the port (idempotent)."""
        if self.bus is None:
            return
        # A logger without /rosout, so the teardown does not depend on the context.
        log = get_logger(f"{self.get_name()}.shutdown")
        with self.lock:
            for j in self.cfg.joints:
                try:
                    self.bus.set_torque(j.id, False)
                except (StsError, OSError) as e:
                    log.error(f"{j.name}: torque off failed: {e}")
            self.bus.close()
            self.bus = None
        log.info("torque off, port closed")

    # Loop -----------------------------------------------------------------

    def _publish(self, publish, msg):
        """Publish unless stopping; a dead context while stopping is ignored."""
        if self.stopping.is_set():
            return
        try:
            publish(msg)
        except DEAD_CONTEXT_ERRORS:
            if not self.stopping.is_set():
                raise

    def _tick(self):
        if self.stopping.is_set():
            return
        with self.lock:
            if self.bus is None:
                return
            msg = self._read_states()
            self._send_due_segments()
        if msg.name:
            self._publish(self.state_pub.publish, msg)

    def _read_joint(self, j):
        """Read one joint (caller holds the lock); its speed in rad/s, or None."""
        log = self.get_logger()
        try:
            steps, speed, _load = self.bus.read_state(j.id)
        except (StsError, OSError) as e:
            log.warning(
                f"{j.name}: read failed: {e}", throttle_duration_sec=LOG_THROTTLE_S
            )
            return None
        if self.bus.status_error:
            log.warning(
                f"{j.name}: servo reports error 0x{self.bus.status_error:02x} "
                "(overload / overheat / voltage)",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
        self.present_steps[j.name] = steps
        self.positions[j.name] = j.to_rad(steps)
        self.read_times[j.name] = time.monotonic()
        return j.speed_to_rad(speed)

    def _is_fresh(self, name, now):
        return now - self.read_times[name] <= self.cfg.state_timeout

    def _read_states(self):
        log = self.get_logger()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for j in self.cfg.joints:
            velocity = self._read_joint(j)
            if velocity is None:
                continue
            msg.name.append(j.name)
            msg.position.append(self.positions[j.name])
            msg.velocity.append(velocity)

        if len(msg.name) == len(self.cfg.joints):
            if self.bus_down:
                log.info("bus responds again")
            self.failed_cycles = 0
            self.bus_down = False
        else:
            self.failed_cycles += 1
            if (
                self.failed_cycles >= FAIL_ERROR_S * self.cfg.rate_hz
                and not self.bus_down
            ):
                self.bus_down = True
                log.error(
                    f"{self.failed_cycles} cycles in a row with failed reads on "
                    f"{self.cfg.port}: check power, cable and Serial Forwarding"
                )
        return msg

    def _send_due_segments(self):
        now = time.monotonic()
        goals = {}
        while self.segments and self.segments[0].send_at <= now:
            seg = self.segments.popleft()
            for name, rad in seg.targets.items():
                if not self._is_fresh(name, now):
                    # The speed (and a hold target) would come from an old position.
                    self.get_logger().warning(
                        f"{name}: no state read within {self.cfg.state_timeout:.2f} s, "
                        "goal not sent",
                        throttle_duration_sec=LOG_THROTTLE_S,
                    )
                    continue
                j = self.joints[name]
                steps = j.to_steps(rad)
                cut = min(max(steps, POS_MIN), POS_MAX)
                if cut != steps:
                    self.get_logger().warning(
                        f"{name}: {rad:+.3f} rad = {steps} steps is outside "
                        f"{POS_MIN}..{POS_MAX}, sent {cut}",
                        throttle_duration_sec=LOG_THROTTLE_S,
                    )
                delta = abs(cut - self.present_steps[name])
                if not seg.from_present:
                    # The servo may still lag behind the previous goal.
                    delta = max(delta, abs(cut - self.goal_steps[name]))
                max_steps = j.speed_to_steps(self.cfg.max_speed)
                speed = delta / seg.duration if seg.duration > 0 else max_steps
                goals[j.id] = (cut, min(speed, max_steps))
                self.goal_steps[name] = cut
        if goals:
            try:
                self.bus.sync_write_goals(goals, self.cfg.acc)
            except (StsError, OSError) as e:
                self.get_logger().warning(
                    f"goal write failed: {e}", throttle_duration_sec=LOG_THROTTLE_S
                )

    # Commands -------------------------------------------------------------

    def _plan(self, joint_names, points):
        """(segments, end time) for points starting now; raises ValueError."""
        unknown = [n for n in joint_names if n not in self.joints]
        if unknown:
            raise ValueError(
                f"unknown joints {unknown} (this bus: {self.cfg.joint_names})"
            )
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"duplicate joint names {list(joint_names)}")
        if not points:
            raise ValueError("trajectory has no points")
        now = time.monotonic()
        segments = []
        prev = 0.0
        for k, point in enumerate(points):
            if len(point.positions) != len(joint_names):
                raise ValueError(
                    f"point {k}: {len(point.positions)} positions "
                    f"for {len(joint_names)} joints"
                )
            t = Duration.from_msg(point.time_from_start).nanoseconds * 1e-9
            if t < prev:
                raise ValueError(f"point {k}: time_from_start goes back")
            targets, clamped = {}, []
            for name, value in zip(joint_names, point.positions):
                if not math.isfinite(value):
                    raise ValueError(f"point {k}: {name} is not finite")
                targets[name], hit = self.joints[name].clamp(value)
                if hit:
                    clamped.append(f"{name} {value:+.3f} -> {targets[name]:+.3f}")
            if clamped:
                self.get_logger().warning(
                    f"clamped to the joint limits: {', '.join(clamped)} rad"
                )
            segments.append(
                Segment(now + prev, t - prev, targets, from_present=(k == 0))
            )
            prev = t
        return segments, now + prev

    def _install(self, segments):
        """Replace the plan; returns the new plan id."""
        with self.lock:
            return self._install_locked(segments)

    def _install_locked(self, segments):
        self.segments = deque(segments)
        self.plan_seq += 1
        return self.plan_seq

    def _hold(self, reason, replace_seq=None):
        """Stop at the position read now; with `replace_seq`, only while that plan is current.

        A joint that cannot be read now gets no goal and keeps its last one:
        the last read position may be old, and the servo would jump back there.
        """
        with self.lock:
            if self.bus is None or (
                replace_seq is not None and self.plan_seq != replace_seq
            ):
                return
            targets, unread = {}, []
            for j in self.cfg.joints:
                if self._read_joint(j) is None:
                    unread.append(j.name)
                else:
                    targets[j.name] = self.positions[j.name]
            # Install even without targets, so the rest of the old plan is dropped.
            self._install_locked(
                [Segment(time.monotonic(), 0.0, targets, from_present=True)]
            )
        if unread:
            self.get_logger().warning(
                f"{reason}: cannot read {', '.join(unread)}, no hold goal sent for "
                "them (they keep their last goal)"
            )
        else:
            self.get_logger().info(f"{reason}: holding the current position")

    def _on_trajectory(self, msg):
        if not msg.points:
            self._hold("empty trajectory")
            return
        try:
            segments, _end = self._plan(msg.joint_names, msg.points[-1:])
        except ValueError as e:
            self.get_logger().error(f"{self.prefix}/joint_trajectory ignored: {e}")
            return
        self._install(segments)

    def _on_goal(self, request):
        try:
            self._plan(request.trajectory.joint_names, request.trajectory.points)
        except ValueError as e:
            self.get_logger().error(
                f"{self.prefix}/follow_joint_trajectory goal rejected: {e}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        result = Result()
        try:
            segments, end = self._plan(
                request.trajectory.joint_names, request.trajectory.points
            )
        except ValueError as e:
            result.error_code = Result.INVALID_GOAL
            result.error_string = str(e)
            goal_handle.abort()
            return result
        seq = self._install(segments)
        final = segments[-1].targets
        names = list(final)
        tolerance = {n: self.cfg.goal_tolerance for n in names}
        for tol in request.goal_tolerance:
            if tol.name in tolerance and tol.position > 0:
                tolerance[tol.name] = tol.position
        extra = Duration.from_msg(request.goal_time_tolerance).nanoseconds * 1e-9
        deadline = end + (extra if extra > 0 else self.cfg.goal_timeout)

        feedback = FollowJointTrajectory.Feedback(joint_names=names)
        feedback.desired.positions = [final[n] for n in names]
        while True:
            time.sleep(FEEDBACK_PERIOD)
            if self.stopping.is_set() or not self.context.ok():
                result.error_string = "shutting down"
                try:
                    goal_handle.abort()
                except DEAD_CONTEXT_ERRORS:
                    pass
                return result
            with self.lock:
                now = time.monotonic()
                preempted = self.plan_seq != seq
                current = {n: self.positions[n] for n in names}
                stale = [n for n in names if not self._is_fresh(n, now)]
            if goal_handle.is_cancel_requested:
                self._hold("goal canceled", seq)
                goal_handle.canceled()
                result.error_string = "canceled"
                return result
            if preempted:
                # Same code as JointTrajectoryController uses for a preempted goal.
                result.error_code = Result.INVALID_GOAL
                result.error_string = "preempted by a newer command"
                goal_handle.abort()
                return result
            errors = [final[n] - current[n] for n in names]
            if not stale:
                feedback.header.stamp = self.get_clock().now().to_msg()
                feedback.actual.positions = [current[n] for n in names]
                feedback.error.positions = errors
                self._publish(goal_handle.publish_feedback, feedback)
                if now >= end and all(
                    abs(e) <= tolerance[n] for n, e in zip(names, errors)
                ):
                    goal_handle.succeed()
                    result.error_code = Result.SUCCESSFUL
                    return result
            if now > deadline:
                result.error_code = Result.GOAL_TOLERANCE_VIOLATED
                if stale:
                    result.error_string = (
                        f"no state read within {self.cfg.state_timeout:.2f} s: "
                        + ", ".join(stale)
                    )
                else:
                    result.error_string = "not within tolerance: " + ", ".join(
                        f"{n} error {e:+.3f} rad" for n, e in zip(names, errors)
                    )
                self.get_logger().warning(f"goal aborted, {result.error_string}")
                # Stop the failed motion, as JointTrajectoryController does.
                self._hold("goal aborted", seq)
                goal_handle.abort()
                return result


def ignore_stop_signals():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def main():
    # rclpy's own SIGINT/SIGTERM handlers shut the context down at once, under
    # callbacks that are still publishing. Here a signal only requests the stop
    # and the context lives until the ordered teardown below. Further signals
    # are ignored (Ctrl-C reaches the node twice under ros2 launch: terminal +
    # launch), so they cannot interrupt the torque off or re-enter Event.set().
    stopping = threading.Event()

    def request_stop(_signum, _frame):
        ignore_stop_signals()
        stopping.set()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        node = ServoTrajectoryBridge(stopping)
    except ValueError as e:
        print(f"servo_trajectory_bridge: {e}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(2)
    executor = MultiThreadedExecutor(num_threads=4)
    code = 0
    try:
        node.start()
        executor.add_node(node)
        while not stopping.is_set() and node.context.ok():
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except (StartupError, StsError, OSError) as e:
        get_logger(node.get_name()).fatal(str(e))
        code = 1
    finally:
        ignore_stop_signals()
        node.stop()
        # Joins the worker threads: the last tick and any aborting goal finish here.
        executor.shutdown(timeout_sec=1.0)
        node.release()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
