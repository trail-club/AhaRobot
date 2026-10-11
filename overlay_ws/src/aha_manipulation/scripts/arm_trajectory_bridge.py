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

The upstream ``arm_node`` writes each goal straight to the servos with no
downstream smoothing, so ALL motion shaping happens here: every goal is
time-stretched to ``max_speed`` by ``arm_plan.plan_arm`` and issued on a paced
cadence (one ``JointCommand`` per planned segment). Unlike the STS head bridge
there is no servo speed register — the bridge paces the goals and the arm
firmware ramps toward each at its own bounded rate.

Reach is open-loop by default (``closed_loop: false``): success is reported once
the final goal has been paced out, which is safe because a well-paced arm tracks
its goals. Set ``closed_loop: true`` with a real joint-state source to confirm
the arm physically arrived (bounded by ``goal_tolerance`` / ``goal_timeout``).
"""

import argparse
import logging
import signal
import threading
import time

import rclpy
from astra_controller_interfaces.msg import JointCommand
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.action import GoalStatus
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
    "state_timeout": 0.3,
    "closed_loop": False,
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
            self._on_goal,
            self._on_cancel,
            self._on_result,
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
        with self._lock:
            self._plan = None
            self._goal_handle = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer.destroy()
        if self._release_srv is not None:
            self._release_srv.destroy_service()
        if self._action_server is not None:
            self._action_server.destroy_server()
        if self._sub_state is not None:
            self._sub_state.destroy_subscription()
        self._hold()
        for pub in (self._pub_cmd, self._pub_state):
            if pub is not None:
                pub.destroy()
        self.destroy_node()
        self._log.info("%s stopped", self.get_name())

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
        return plan_arm(
            list(self._limits.values()),
            self._params.joint_names,
            points,
            now=self._now(),
            max_speed=self._params.max_speed,
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
            self._goal_deadline = plan.end_time + self._params.goal_timeout

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

    def _on_goal(self, goal_request):
        return self._action_server.accept_new_goal(goal_request)

    def _on_cancel(self, goal_handle):
        with self._lock:
            self._goal_handle = None
            self._goal = None
            self._goal_deadline = None
            self._hold()
        return GoalStatus.ACCEPTED

    def _on_result(self, _goal_handle):
        pass

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
        self._pending_names = list(goal_handle.goal.joint_names)
        self._pending_points = goal_handle.goal.points
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
                i = self._plan.i
                segments = self._plan.segments
                while i < len(segments) and segments[i].send_at <= now:
                    self._publish_cmd(self._command(segments[i].targets))
                    for joint in self._params.joint_names:
                        self._last[joint] = segments[i].targets.get(
                            joint, self._last.get(joint, self._params.command_hold)
                        )
                    self._plan.i = i + 1
                    i += 1
                if i >= len(segments):
                    self._plan = None
                    if self._goal_handle is not None:
                        self._goal_deadline = now + self._params.goal_timeout
            elif now - self._last_hold >= self._params.hold_interval:
                self._last_hold = now
                self._hold()
            if self._goal_handle is not None:
                self._check_action(now)
        self._publish_state()

    def _check_action(self, now: float) -> None:
        goal = self._goal
        if self._goal_handle is None or goal is None:
            return
        tol = self._params.goal_tolerance
        reached = all(abs(self._effective(j) - value) <= tol for j, value in goal.items())
        if reached:
            self._goal_handle.succeed(FollowJointTrajectory.Result())
            self._goal_handle = None
            self._goal = None
            return
        if self._goal_deadline is not None and now >= self._goal_deadline:
            self._goal_handle.abort(FollowJointTrajectory.Result())
            self._goal_handle = None
            self._goal = None
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

