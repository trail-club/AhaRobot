#!/usr/bin/env python3
"""JointTrajectory bridge for one real Astra arm.

This node mirrors the interface of ros2_control's JointTrajectoryController
for `controller_name`, so MoveIt and scripts drive the simulation and the real
arm with the same topics (the arm equivalent of aha_servo's
servo_trajectory_bridge.py):

  /<controller>/joint_trajectory         trajectory_msgs/JointTrajectory (sub)
  /<controller>/follow_joint_trajectory  control_msgs/action/FollowJointTrajectory

It drives the upstream astra_controller arm_node, which writes every
JointCommand to the arm board as one goal:

  command_topic  astra_controller_interfaces/JointCommand (pub)
  state_topic    sensor_msgs/JointState (sub, the arm_node's own feedback
                 /<side>/arm/joint_states; on /joint_states
                 joint_state_publisher may put the arm joints at 0)

Configured by config/arm.yaml in the aha_manipulation/arm_map.py format.

Behaviour:
  - Nothing is sent before a trajectory arrives: no goal at startup or while
    idle, so the keyboard teleop can share the arm_node when no trajectory
    runs. Running both at once is not supported.
  - Trajectories are checked and sampled by aha_manipulation/arm_plan.py from
    the measured pose; the plan is sent every 1 / rate_hz s as a goal of all
    five joints. A trajectory outside the position, velocity or acceleration
    limits is rejected as a whole (topic: logged and ignored).
  - Planning needs a fresh state (within `state_timeout`) of every joint.
  - Action: feedback every FEEDBACK_PERIOD s. Aborts when a joint is more
    than `path_tolerance` (or the goal's path_tolerance) from the sample of
    the current time, or when the state goes stale; succeeds when every joint
    is within `goal_tolerance` (or the goal's) of the final point, aborts
    `goal_time` s (or goal_time_tolerance) after the last point.
  - Cancel, an abort for tolerance and an empty trajectory hold the measured
    position. A newer topic command or goal preempts (aborts) the running
    goal. A stale state stops sending, so the arm stays at the last sample.
  - Partial joint lists are accepted; the other joints keep their position.
  - SIGINT/SIGTERM: sampling and goals stop, then the node shuts down. The
    arm_node keeps the last goal. Wall clock only.
"""

import signal
import sys
import threading
import time

import rclpy
from astra_controller_interfaces.msg import JointCommand
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from aha_manipulation.arm_map import BUS_DEFAULTS, JOINT_DEFAULTS, parse_config
from aha_manipulation.arm_plan import ArmPlan, Point, plan_trajectory

Result = FollowJointTrajectory.Result
FEEDBACK_PERIOD = 0.05  # s
SPIN_TIMEOUT_S = 0.1
LOG_THROTTLE_S = 2.0


def seconds(duration):
    return Duration.from_msg(duration).nanoseconds * 1e-9


def points_of(trajectory):
    return [
        Point(
            positions=tuple(p.positions),
            velocities=tuple(p.velocities),
            accelerations=tuple(p.accelerations),
            time=seconds(p.time_from_start),
        )
        for p in trajectory.points
    ]


def tolerances(default, overrides, names):
    """{joint: rad} from a default and JointTolerance overrides (0: default, -1: off)."""
    values = {n: default for n in names}
    for tol in overrides:
        if tol.name in values and tol.position != 0:
            values[tol.name] = tol.position if tol.position > 0 else float("inf")
    return {n: (v if v > 0 else float("inf")) for n, v in values.items()}


class ArmTrajectoryBridge(Node):
    def __init__(self, stopping):
        super().__init__("aha_arm_bridge")
        self.cfg = parse_config(self._read_parameters())
        self.names = self.cfg.joint_names
        self.prefix = f"/{self.cfg.controller_name}"
        self.stopping = stopping  # threading.Event, set by SIGINT/SIGTERM
        self.lock = threading.Lock()
        self.positions = {}  # rad, latest state
        self.state_times = {}  # monotonic time of the latest state per joint
        self.plan = None  # ArmPlan being sampled, None when idle
        self.plan_seq = 0
        self.had_state = False

        self.command_pub = self.create_publisher(
            JointCommand, self.cfg.command_topic, 10
        )
        self.create_subscription(JointState, self.cfg.state_topic, self._on_state, 10)
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
            1.0 / self.cfg.rate_hz,
            self._tick,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        limits = ", ".join(
            f"{j.name} [{j.min:+.2f}, {j.max:+.2f}] {j.max_velocity:.2f} rad/s "
            f"{j.max_acceleration:.2f} rad/s^2"
            for j in self.cfg.joints
        )
        self.get_logger().info(
            f"{self.prefix} -> {self.cfg.command_topic}, waiting for "
            f"{self.cfg.state_topic}; limits {limits}"
        )

    def _read_parameters(self):
        """Nested dict in the arm_map layout from this node's parameters."""
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

    def stop(self):
        """First shutdown step: no more samples; running goals abort."""
        self.stopping.set()
        self.timer.cancel()

    # State and commands ---------------------------------------------------

    def _on_state(self, msg):
        if len(msg.position) != len(msg.name):
            return
        now = time.monotonic()
        with self.lock:
            for name, position in zip(msg.name, msg.position):
                if name in self.names:
                    self.positions[name] = position
                    self.state_times[name] = now
            fresh = self._stale(now) == []
        if fresh and not self.had_state:
            self.had_state = True
            self.get_logger().info(
                "arm state received: "
                + ", ".join(f"{n} {self.positions[n]:+.3f}" for n in self.names)
            )

    def _stale(self, now):
        """Joints without a state within state_timeout (caller holds the lock)."""
        return [
            n
            for n in self.names
            if now - self.state_times.get(n, -float("inf")) > self.cfg.state_timeout
        ]

    def _send(self, pose):
        if self.stopping.is_set():
            return
        self.command_pub.publish(
            JointCommand(
                name=list(self.names), position_cmd=[pose[n] for n in self.names]
            )
        )

    def _tick(self):
        if self.stopping.is_set():
            return
        with self.lock:
            plan = self.plan
            if plan is None:
                return
            now = time.monotonic()
            stale = self._stale(now)
            if stale:
                # The plan started from a pose that may no longer hold.
                self.plan = None
                self.get_logger().error(
                    f"no state of {', '.join(stale)} within {self.cfg.state_timeout} s: "
                    "trajectory stopped at the last sample"
                )
                return
            self._send(plan.sample(now))
            if now >= plan.end_time:
                self.plan = None

    def _plan(self, trajectory):
        """ArmPlan from the measured pose now (caller holds the lock); raises ValueError."""
        now = time.monotonic()
        stale = self._stale(now)
        if stale:
            raise ValueError(
                f"no state of {', '.join(stale)} within {self.cfg.state_timeout} s "
                f"on {self.cfg.state_topic} (is the arm_node running?)"
            )
        return plan_trajectory(
            self.cfg.joints,
            trajectory.joint_names,
            points_of(trajectory),
            dict(self.positions),
            now,
            self.cfg.start_tolerance,
        )

    def _install(self, plan):
        """Replace the plan (caller holds the lock); returns the new plan id."""
        self.plan = plan
        self.plan_seq += 1
        if plan is not None:
            self._send(plan.sample(time.monotonic()))
        return self.plan_seq

    def _hold(self, reason, replace_seq=None):
        """Stop at the measured position; with `replace_seq`, only while that plan is current."""
        with self.lock:
            if replace_seq is not None and self.plan_seq != replace_seq:
                return
            stale = self._stale(time.monotonic())
            if stale:
                self._install(None)
            else:
                pose = dict(self.positions)
                self._install(
                    ArmPlan(
                        names=tuple(self.names),
                        limits={j.name: j for j in self.cfg.joints},
                        times=[time.monotonic()],
                        positions=[pose],
                        velocities=None,
                    )
                )
        if stale:
            self.get_logger().warning(
                f"{reason}: no state of {', '.join(stale)}, sending no hold goal "
                "(the arm keeps its last goal)"
            )
        else:
            self.get_logger().info(f"{reason}: holding the measured position")

    # Commands -------------------------------------------------------------

    def _on_trajectory(self, msg):
        if not msg.points:
            self._hold("empty trajectory")
            return
        try:
            with self.lock:
                self._install(self._plan(msg))
        except ValueError as e:
            self.get_logger().error(f"{self.prefix}/joint_trajectory ignored: {e}")

    def _on_goal(self, request):
        try:
            with self.lock:
                self._plan(request.trajectory)
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
            with self.lock:
                plan = self._plan(request.trajectory)
                seq = self._install(plan)
        except ValueError as e:
            result.error_code = Result.INVALID_GOAL
            result.error_string = str(e)
            goal_handle.abort()
            return result
        names = list(request.trajectory.joint_names)
        final = plan.final
        path_tol = tolerances(self.cfg.path_tolerance, request.path_tolerance, names)
        goal_tol = tolerances(self.cfg.goal_tolerance, request.goal_tolerance, names)
        extra = seconds(request.goal_time_tolerance)
        deadline = plan.end_time + (extra if extra > 0 else self.cfg.goal_time)

        feedback = FollowJointTrajectory.Feedback(joint_names=names)
        while True:
            time.sleep(FEEDBACK_PERIOD)
            if self.stopping.is_set() or not self.context.ok():
                result.error_string = "shutting down"
                try:
                    goal_handle.abort()
                except Exception:  # noqa: BLE001 - the context may be gone
                    pass
                return result
            with self.lock:
                now = time.monotonic()
                preempted = self.plan_seq != seq
                current = {n: self.positions[n] for n in names}
                stale = self._stale(now)
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
            if stale:
                result.error_code = Result.PATH_TOLERANCE_VIOLATED
                result.error_string = (
                    f"no state of {', '.join(stale)} within {self.cfg.state_timeout} s"
                )
                return self._abort(goal_handle, result, seq, hold=False)

            desired = plan.sample(now)
            feedback.header.stamp = self.get_clock().now().to_msg()
            feedback.desired.positions = [desired[n] for n in names]
            feedback.actual.positions = [current[n] for n in names]
            feedback.error.positions = [desired[n] - current[n] for n in names]
            goal_handle.publish_feedback(feedback)

            if now < plan.end_time:
                off = [
                    f"{n} {e:+.3f} rad"
                    for n, e in zip(names, feedback.error.positions)
                    if abs(e) > path_tol[n]
                ]
                if off:
                    result.error_code = Result.PATH_TOLERANCE_VIOLATED
                    result.error_string = "off the path: " + ", ".join(off)
                    return self._abort(goal_handle, result, seq)
                continue
            errors = {n: final[n] - current[n] for n in names}
            if all(abs(e) <= goal_tol[n] for n, e in errors.items()):
                goal_handle.succeed()
                result.error_code = Result.SUCCESSFUL
                return result
            if now > deadline:
                result.error_code = Result.GOAL_TOLERANCE_VIOLATED
                result.error_string = "not within tolerance: " + ", ".join(
                    f"{n} error {e:+.3f} rad"
                    for n, e in errors.items()
                    if abs(e) > goal_tol[n]
                )
                return self._abort(goal_handle, result, seq)

    def _abort(self, goal_handle, result, seq, hold=True):
        self.get_logger().warning(f"goal aborted, {result.error_string}")
        if hold:
            # Stop the failed motion, as JointTrajectoryController does.
            self._hold("goal aborted", seq)
        else:
            with self.lock:
                if self.plan_seq == seq:
                    self._install(None)
        goal_handle.abort()
        return result


def ignore_stop_signals():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def main():
    # As in aha_servo's bridge: a signal only requests the stop, and the
    # context lives until the ordered teardown below.
    stopping = threading.Event()

    def request_stop(_signum, _frame):
        ignore_stop_signals()
        stopping.set()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        node = ArmTrajectoryBridge(stopping)
    except ValueError as e:
        print(f"arm_trajectory_bridge: {e}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(2)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while not stopping.is_set() and node.context.ok():
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except ExternalShutdownException:
        pass
    finally:
        ignore_stop_signals()
        node.stop()
        # Joins the worker threads: running goals see `stopping` and finish here.
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
    get_logger("aha_arm_bridge").info("stopped")


if __name__ == "__main__":
    main()
