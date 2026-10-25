#!/usr/bin/env python3
"""Time-scaled trajectory bridge for a real Astra arm.

Sits between trajectory sources (MoveIt, the sim control UI, eval, teleop) and
the arm's hardware node (upstream ``astra_controller`` ``arm_node`` /
``dry_run_node``) — the arm equivalent of aha_servo's
``servo_trajectory_bridge``. One bridge per arm; wall-clock only.

External ROS interface (mirrors the sim ``JointTrajectoryController``):
    /<controller_name>/joint_trajectory        (JointTrajectory)  fire & forget
    /<controller_name>/follow_joint_trajectory (FollowJointTrajectory action)
    /<controller_name>/joint_states            (JointState)        published

Internal (drives the arm hardware node over its astra topics):
    <command_topic>   astra_controller_interfaces/msg JointCommand   publish
    <state_topic>     sensor_msgs/msg JointState                     subscribe

The upstream ``arm_node`` writes each received goal straight to the servos with
no downstream smoothing — the firmware ramps toward it with its own P-only
controller, so a single goal per trajectory point would make the arm lunge.
ALL motion shaping happens here: ``arm_plan.plan_arm`` turns the trajectory
into speed-bounded keyframes (never faster than ``max_speed``) and the bridge
samples the plan at ``rate_hz``, publishing one interpolated position per
sample. That dense command stream is what keeps the firmware's P-only
controller on the intended path.

Arrival: with ``closed_loop: true`` (the real arm default — ``arm_node``
publishes firmware joint feedback on ``state_topic``) success is reported when
the arm is within ``goal_tolerance`` of the goal within ``goal_timeout`` after
the plan ends; if the state stream goes stale (``state_timeout``) the bridge
falls back to the last commanded pose. With ``closed_loop: false`` success is
reported ``settle_time`` after the plan ends. Cancellation or the ``release``
service stops the plan and re-asserts the last safe position.
"""

import argparse
import logging
import signal
import threading
import time

import rclpy
from astra_controller_interfaces.msg import JointCommand
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory

from aha_manipulation.arm_map import parse_config
from aha_manipulation.arm_plan import ArmPlan, plan_arm

DEFAULTS = {
    "controller_name": "left_arm_controller",
    "command_topic": "joint_command",
    "state_topic": "joint_states",
    "joint_names": ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"],
    "command_joint_names": [],
    "command_hold": 0.0,
    "rate_hz": 30.0,
    "max_speed": 1.0,
    "goal_tolerance": 0.03,
    "goal_timeout": 5.0,
    "settle_time": 1.5,
    "state_timeout": 0.3,
    "closed_loop": True,
    "hold_interval": 1.0,
}
JOINT_DEFAULTS = {"min": -3.14, "max": 3.14}


def _duration_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec / 1e9


def _throttle(message: str, state: dict, key: str, interval: float) -> bool:
    now = time.monotonic()
    if now - state.get(key, 0.0) >= interval:
        state[key] = now
        return True
    return False


class ArmTrajectoryBridge(Node):
    def __init__(self, *, name="arm_trajectory_bridge", context=None, params_file=None):
        overrides = (
            [Parameter("parameters_file", Parameter.Type.STRING, params_file)]
            if params_file
            else []
        )
        super().__init__(name, namespace="", context=context, parameter_overrides=overrides)
        self._log = logging.getLogger(name)
        for key, value in DEFAULTS.items():
            if isinstance(value, list) and not value:
                # An empty-list default would be statically inferred as
                # BYTE_ARRAY and clash with a string-array override from a
                # params file. Declare it dynamically typed instead; arm_map
                # falls back to joint_names when it stays empty.
                self.declare_parameters(
                    "",
                    [(key, value, ParameterDescriptor(dynamic_typing=True))],
                    False,
                )
            else:
                self.declare_parameter(key, value)
        joint_names = self.get_parameter("joint_names").value
        for joint in joint_names:
            self.declare_parameter(f"{joint}__min", JOINT_DEFAULTS["min"])
            self.declare_parameter(f"{joint}__max", JOINT_DEFAULTS["max"])
        self._params = parse_config(
            {
                **{key: self.get_parameter(key).value for key in DEFAULTS},
                **{
                    joint: {
                        "min": self.get_parameter(f"{joint}__min").value,
                        "max": self.get_parameter(f"{joint}__max").value,
                    }
                    for joint in joint_names
                },
            }
        )
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._limits = {j.name: j for j in self._params.joints}
        self._plan = None
        self._goal_handle = None
        self._goal = None
        self._goal_received = None
        self._goal_deadline = None
        self._last = {j: 0.0 for j in self._params.joint_names}
        self._state = {}
        self._state_stamp = None
        self._pending_names = []
        self._pending_points = []
        self._last_hold = 0.0
        self._throttled = {}
        self._pub_state = None
        self._pub_cmd = None
        self._sub_state = None
        self._action_server = None
        self._release_srv = None
        self._timer = None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._pub_state = self.create_publisher(
            JointState, f"{self._params.controller_name}/joint_states", 10
        )
        self._pub_cmd = self.create_publisher(JointCommand, self._params.command_topic, 10)
        self._sub_state = self.create_subscription(
            JointState, self._params.state_topic, self._on_state, 10
        )
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            f"{self._params.controller_name}/follow_joint_trajectory",
            self._execute,
            cancel_callback=self._on_cancel,
        )
        self.create_subscription(
            JointTrajectory,
            f"{self._params.controller_name}/joint_trajectory",
            self._on_trajectory,
            10,
        )
        self._release_srv = self.create_service(
            Empty, f"{self._params.controller_name}/release", self._on_release
        )
        self._timer = self.create_timer(1.0 / self._params.rate_hz, self._tick)
        self._log.info(
            "%s via %s (max_speed %.2f rad/s, %d joints, %s)",
            self._params.controller_name,
            self._params.command_topic,
            self._params.max_speed,
            len(self._params.joint_names),
            "closed_loop" if self._params.closed_loop else "open_loop",
        )
        self._hold()

    def stop(self) -> None:
        self._stopping.set()
        name = self.get_name()
        with self._lock:
            self._plan = None
            self._goal_handle = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer.destroy()
        if self._release_srv is not None:
            self._release_srv.destroy()
        if self._action_server is not None:
            self._action_server.destroy()
        if self._sub_state is not None:
            self._sub_state.destroy()
        self._hold()
        for pub in (self._pub_cmd, self._pub_state):
            if pub is not None:
                pub.destroy()
        self.destroy_node()
        self._log.info("%s stopped", name)

    # -- state / command ----------------------------------------------------
    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                if name in self._limits:
                    self._state[name] = position
            self._state_stamp = self._now()

    def _effective(self, joint: str) -> float:
        if self._params.closed_loop and self._state.get(joint) is not None:
            age = self._now() - (self._state_stamp or 0.0)
            if age <= self._params.state_timeout:
                return self._state[joint]
        return self._last.get(joint, self._params.command_hold)

    def _command(self, targets: dict) -> JointCommand:
        names = list(self._params.command_joint_names)
        positions = []
        for name in names:
            if name in self._limits:
                positions.append(targets.get(name, self._last.get(name, self._params.command_hold)))
            else:
                positions.append(self._params.command_hold)
        cmd = JointCommand()
        cmd.name = names
        cmd.position_cmd = positions
        return cmd

    def _publish_cmd(self, cmd: JointCommand) -> None:
        try:
            self._pub_cmd.publish(cmd)
        except Exception as error:
            if _throttle(f"command publish failed: {error}", self._throttled, "publish", 2.0):
                self._log.warning("command publish failed: %s", error)

    def _publish_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = self._params.joint_names
        msg.name = list(names)
        msg.position = [self._effective(j) for j in names]
        self._pub_state.publish(msg)

    def _hold(self) -> None:
        if self._stopping.is_set() or self._pub_cmd is None:
            return
        targets = {j: self._last.get(j, 0.0) for j in self._params.joint_names}
        self._last.update(targets)
        self._publish_cmd(self._command(targets))

    # -- planning -----------------------------------------------------------
    def _plan_pending(self) -> ArmPlan:
        points = []
        for point in self._pending_points:
            t = _duration_seconds(point.time_from_start)
            for joint, value in zip(self._pending_names, point.positions):
                points.append((joint, float(value), t))
        start_pose = {joint: self._effective(joint) for joint in self._params.joint_names}
        for name in self._params.command_joint_names:
            start_pose.setdefault(name, self._params.command_hold)
        return plan_arm(
            list(self._limits.values()),
            self._params.command_joint_names,
            points,
            now=self._now(),
            max_speed=self._params.max_speed,
            start_pose=start_pose,
        )

    def _final_targets(self) -> dict:
        targets = {}
        if self._pending_points:
            for joint, value in zip(self._pending_names, self._pending_points[-1].positions):
                targets[joint] = float(value)
        return targets

    def _install(self, plan: ArmPlan) -> None:
        with self._lock:
            self._plan = plan
            if plan.segments:
                self._last.update(plan.segments[-1].targets)
            if self._params.closed_loop:
                self._goal_deadline = plan.end_time + self._params.goal_timeout
            else:
                self._goal_deadline = plan.end_time + self._params.settle_time
        if plan.clamped:
            self._log.warning("plan clamped to joint limits: %s", plan.clamped)

    # -- external interface -------------------------------------------------
    def _on_trajectory(self, msg: JointTrajectory) -> None:
        if self._stopping.is_set() or not msg.points:
            return
        self._pending_names = list(msg.joint_names)
        self._pending_points = msg.points
        try:
            plan = self._plan_pending()
        except ValueError as error:
            if _throttle(f"ignoring trajectory: {error}", self._throttled, "bad_traj", 2.0):
                self._log.warning("ignoring trajectory: %s", error)
            return
        self._install(plan)
        if _throttle("trajectory", self._throttled, "traj", 2.0):
            self._log.info(
                "trajectory: %d segments over %.2fs",
                len(plan.segments),
                plan.end_time - self._now(),
            )

    def _on_cancel(self, goal_handle):
        if self._goal_handle is not goal_handle or not goal_handle.is_active:
            return CancelResponse.REJECT
        with self._lock:
            self._goal = None
            self._goal_deadline = None
            self._plan = None  # stop the motion; do not keep driving to the goal
            self._hold()
        # The handle stays registered: the server applies CANCEL_GOAL only after
        # this callback returns, so _check_action completes the transition
        # (EXECUTING->CANCELED directly would be an invalid rcl transition).
        return CancelResponse.ACCEPT

    def _on_release(self, _request, response):
        with self._lock:
            self._plan = None
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            self._hold()
        return response

    def _execute(self, goal_handle) -> None:
        self._goal_handle = goal_handle
        self._goal_received = self._now()
        # goal_handle.request is the FollowJointTrajectory.Goal message; the
        # trajectory lives one level deeper.
        trajectory = goal_handle.request.trajectory
        self._pending_names = list(trajectory.joint_names)
        self._pending_points = trajectory.points
        try:
            plan = self._plan_pending()
        except ValueError as error:
            self._log.warning("rejecting goal: %s", error)
            goal_handle.abort(FollowJointTrajectory.Result())
            self._goal_handle = None
            return
        self._goal = self._final_targets()
        self._install(plan)

    # -- tick ---------------------------------------------------------------
    def _tick(self) -> None:
        if self._stopping.is_set():
            return
        now = self._now()
        with self._lock:
            if self._plan is not None:
                # Fixed-rate sampling: one interpolated position per tick.
                # Past the plan end, position_at() re-asserts the final pose,
                # so the goal pose is always commanded at least once.
                pose = self._plan.position_at(now)
                for joint in self._params.joint_names:
                    self._last[joint] = pose.get(joint, self._params.command_hold)
                self._publish_cmd(self._command(pose))
                if now >= self._plan.end_time:
                    self._plan = None
            elif now - self._last_hold >= self._params.hold_interval:
                self._last_hold = now
                self._hold()
            if self._goal_handle is not None:
                self._check_action(now)
        self._publish_state()

    def _check_action(self, now: float) -> None:
        goal = self._goal
        handle = self._goal_handle
        if handle is None or goal is None:
            return
        if handle.is_cancel_requested:
            # The server applied CANCEL_GOAL (state CANCELING) after the cancel
            # callback returned; complete it so the client observes CANCELED
            # and the result service can resolve.
            handle.canceled(FollowJointTrajectory.Result())
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            return
        if not handle.is_active:
            # Completed out-of-band (e.g. succeeded just before a cancel);
            # the terminal status was already published.
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            return
        tol = self._params.goal_tolerance
        reached = all(abs(self._effective(j) - value) <= tol for j, value in goal.items())
        if reached:
            handle.succeed(FollowJointTrajectory.Result())
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            return
        if self._goal_deadline is not None and now >= self._goal_deadline:
            handle.abort(FollowJointTrajectory.Result())
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            self._log.warning("goal timed out after %.1fs", now - (self._goal_received or now))


def main(args=None) -> int:
    logging.basicConfig(
        level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default=None)
    parsed, _ = parser.parse_known_args(args)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    rclpy.init(args=args)
    bridge = ArmTrajectoryBridge(
        name="aha_arm_bridge", context=rclpy.get_default_context(), params_file=parsed.params_file
    )
    stop = threading.Event()

    def _on_signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        bridge.start()
    except Exception as error:
        logging.getLogger("aha_arm_bridge").error("%s", error)
        rclpy.shutdown()
        return 1

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
    finally:
        bridge.stop()
        executor.remove_node(bridge)
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

