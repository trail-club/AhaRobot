"""Gestures with the right arm: point at a seat to offer it (rulebook 5.1).

The arm and the lift move by FollowJointTrajectory on right_arm_controller and
lift_controller, the same interface as the real arm's trajectory bridge, in
a straight line in joint space without collision checking: the arm starts and
ends at the pose it had. The pointing pose comes from
aha_manipulation.pointing. ROS and aha_manipulation are imported on use, so
task definitions load without them (task_runner --dry-run). Speech is not
part of these skills.
"""

import math
import threading
import time
from types import SimpleNamespace

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.people import People, check_keys, check_name, positive
from aha_tasks.skills import LOCATION

ARM_ACTION = "/right_arm_controller/follow_joint_trajectory"
LIFT_ACTION = "/lift_controller/follow_joint_trajectory"
ARM_SPEED = 1.0  # rad/s, half the joint_limits.yaml limit
LIFT_SPEED = 0.15  # m/s, half the joint_limits.yaml limit
MIN_DURATION = 1.0  # s per motion
SERVER_TIMEOUT = 5.0  # s (wall) for an action server or the first TF / joint state
RESULT_MARGIN = 5.0  # s (node clock) after the motion's duration
CLOCK_STALL_TIMEOUT = 10.0  # s (wall) without the node clock advancing
REAIM = 0.01  # rad; a larger change after the first motion moves the arm again
HOLDING = "robot.holding"


def wait_future(future, timeout, cancelled=None):
    deadline = time.monotonic() + timeout
    while not future.done():
        if cancelled is not None and cancelled.is_set():
            return None
        if time.monotonic() > deadline:
            raise TimeoutError("no answer from the controller")
        time.sleep(0.02)
    return future.result()


def task_sleep(node, seconds, cancelled):
    """Wait `seconds` of the node clock (sim time if active); False if cancelled."""
    clock = node.get_clock()
    if not getattr(clock, "ros_time_is_active", False):
        return not cancelled.wait(seconds)
    start = last = clock.now().nanoseconds * 1e-9
    progress = time.monotonic()
    while (now := clock.now().nanoseconds * 1e-9) - start < seconds:
        if cancelled.is_set():
            return False
        if now > last:
            last, progress = now, time.monotonic()
        elif time.monotonic() - progress > CLOCK_STALL_TIMEOUT:
            return True  # no /clock: do not hang the task
        time.sleep(0.05)
    return True


class Arm:
    """Right arm, lift, joint states and TF for the skills of one runner."""

    def __init__(self, node):
        from action_msgs.msg import GoalStatus
        from aha_manipulation.pointing import ARM_JOINTS, LIFT_JOINT
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self.node = node
        self.arm_joints, self.lift_joint = list(ARM_JOINTS), LIFT_JOINT
        self.GoalStatus = GoalStatus
        self.FollowJointTrajectory = FollowJointTrajectory
        self.JointTrajectory, self.JointTrajectoryPoint = (
            JointTrajectory,
            JointTrajectoryPoint,
        )
        self.arm = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
        self.lift = ActionClient(node, FollowJointTrajectory, LIFT_ACTION)
        self.lock = threading.Lock()
        self.joints = {}
        node.create_subscription(
            JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data
        )
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "arm", None) is None:
            runtime.arm = cls(runtime.node)
        return runtime.arm

    def _on_joint_states(self, message):
        with self.lock:
            self.joints.update(zip(message.name, message.position))

    def positions(self, names):
        """Current joint values, waiting for the first joint state."""
        deadline = time.monotonic() + SERVER_TIMEOUT
        while True:
            with self.lock:
                if all(name in self.joints for name in names):
                    return [self.joints[name] for name in names]
            if time.monotonic() > deadline:
                raise TimeoutError("no /joint_states for the right arm")
            time.sleep(0.05)

    def transform(self, target, source):
        """Latest transform; retried, as a new buffer may lack the older links."""
        from rclpy.time import Time
        from tf2_ros import TransformException

        deadline = time.monotonic() + SERVER_TIMEOUT
        while True:
            try:
                return self.tf.lookup_transform(target, source, Time())
            except TransformException:
                if time.monotonic() > deadline:
                    raise
            time.sleep(0.1)

    def to_base(self, frame, xyz):
        """(x, y, z) of a point in base_footprint with the latest TF."""
        from geometry_msgs.msg import PointStamped
        from tf2_geometry_msgs import do_transform_point

        point = PointStamped()
        point.header.frame_id = frame
        point.point.x, point.point.y, point.point.z = map(float, xyz)
        p = do_transform_point(point, self.transform("base_footprint", frame)).point
        return (p.x, p.y, p.z)

    def robot_pose(self):
        """[x, y, yaw] of base_footprint in map."""
        t = self.transform("map", "base_footprint").transform
        q = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        return [t.translation.x, t.translation.y, yaw]

    def has_lift(self):
        return self.lift.wait_for_server(timeout_sec=SERVER_TIMEOUT)

    def move(self, lift, arm, cancelled):
        """Move to `arm` (ARM_JOINTS) and, unless None, `lift`; (ok, message)."""
        names = [self.lift_joint, *self.arm_joints]
        current = dict(zip(names, self.positions(names)))
        goals = [(ARM_ACTION, self.arm, self.arm_joints, list(arm), ARM_SPEED)]
        if lift is not None:
            goals.append(
                (LIFT_ACTION, self.lift, [self.lift_joint], [lift], LIFT_SPEED)
            )
        duration = MIN_DURATION
        for _, _, joints, values, speed in goals:
            for name, value in zip(joints, values):
                duration = max(duration, abs(value - current[name]) / speed)
        handles = []
        try:
            for name, client, joints, values, _ in goals:
                if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
                    return False, f"{name} is not available"
                handle = wait_future(
                    client.send_goal_async(self._goal(joints, values, duration)),
                    SERVER_TIMEOUT,
                )
                if not handle.accepted:
                    return False, f"{name} rejected the goal"
                handles.append((name, handle))
            for name, handle in handles:
                result = self._wait_result(handle, duration, cancelled)
                if result is None:
                    return False, "Cancelled"
                code = result.result.error_code
                if result.status != self.GoalStatus.STATUS_SUCCEEDED or code != 0:
                    return (
                        False,
                        f"{name} failed ({code}): {result.result.error_string}",
                    )
            handles = []
            return True, ""
        finally:
            for _, handle in handles:  # only after a failure
                handle.cancel_goal_async()

    def _goal(self, joints, values, duration):
        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory = self.JointTrajectory(joint_names=joints)
        point = self.JointTrajectoryPoint(positions=[float(v) for v in values])
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        goal.trajectory.points = [point]
        return goal

    def _wait_result(self, handle, duration, cancelled):
        """The result within the motion's duration + RESULT_MARGIN of node time."""
        future = handle.get_result_async()
        clock = self.node.get_clock()
        deadline = clock.now().nanoseconds * 1e-9 + duration + RESULT_MARGIN
        last, progress = None, time.monotonic()
        while not future.done():
            if cancelled.is_set():
                return None
            now = clock.now().nanoseconds * 1e-9
            if now > deadline:
                raise TimeoutError("the controller did not finish in time")
            if now != last:
                last, progress = now, time.monotonic()
            elif time.monotonic() - progress > CLOCK_STALL_TIMEOUT:
                raise TimeoutError("the clock stopped while the arm moved")
            time.sleep(0.02)
        return future.result()


def finite_point(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(type(v) in (int, float) and math.isfinite(v) for v in value)
    )


class OfferSeat(Skill):
    """Show a guest their seat: look at it and point with the right arm.

    params: {"guest": "guest1", "seat": "chair_left", "position": [x, y, z],
    "height": 0.8, "hold": 3.0}. `position` is the seat in map (m). A seat
    outside the arm's bearings (pointing.MIN_BEARING..MAX_BEARING) turns the
    base first. The wrist rises to `height` m above the floor if the lift
    controller is there, and the arm points for `hold` s, then returns to where
    it was. The pose is computed again once the arm is there, as the base may
    still settle after navigation, and corrected if it changed. While the
    robot holds something (robot.holding), only the head shows the seat. The
    head then follows the guest (by `<guest>.track` and name). Choosing a free
    seat is not part of the skill.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guest", "seat", "position"}, {"height", "hold"})
        check_name(params, "guest")
        check_name(params, "seat")
        if not finite_point(params["position"]):
            raise ValueError(f"position must be [x, y, z] in map: {params}")
        positive(params, "height", 0.8)
        positive(params, "hold", 3.0)

    def ready(self):
        # Fill the TF buffer and joint states from the start, not on first use.
        Arm.of(self.runtime)

    def expected(self, params):
        return {f"{params['guest']}.seat": params["seat"]}

    def execute(self, ctx):
        from aha_manipulation.pointing import (
            ARM_JOINTS,
            LIFT_JOINT,
            MAX_BEARING,
            MIN_BEARING,
            lift_for_height,
            point_at,
            shoulder_bearing,
        )

        guest, seat = ctx.params["guest"], ctx.params["seat"]
        position = ctx.params["position"]
        hold = positive(ctx.params, "hold", 3.0)
        gaze = self.runtime.gaze
        gaze.look_at(
            "map", SimpleNamespace(x=position[0], y=position[1], z=position[2])
        )
        outputs = {f"{guest}.seat": seat}
        holding = ctx.memory.get(HOLDING)
        if holding:
            ctx.log(f"Holding {holding}: showing {seat} with the head only")
            if not task_sleep(self.runtime.node, hold, ctx.cancelled):
                return SkillResult(False, "Cancelled")
            self._look_at_guest(ctx)
            return SkillResult(True, f"Looked at {seat} for {guest}", outputs)

        arm = Arm.of(self.runtime)
        target = arm.to_base("map", position)
        bearing = shoulder_bearing(target)
        if not MIN_BEARING <= bearing <= MAX_BEARING:
            navigation = self.runtime.navigation
            if navigation is None:
                return SkillResult(
                    False, f"{seat} is {math.degrees(bearing):+.0f} deg off; no Nav2"
                )
            x, y, _ = arm.robot_pose()
            pose = [x, y, math.atan2(position[1] - y, position[0] - x)]
            ctx.log(f"Turning to {seat} ({math.degrees(bearing):+.0f} deg)")
            ok, text = navigation.navigate(pose, ctx.cancelled)
            if not ok:
                return SkillResult(False, f"Could not turn to {seat}: {text}")
            outputs[LOCATION] = [round(v, 3) for v in pose]
            target = arm.to_base("map", position)

        start = arm.positions([LIFT_JOINT, *ARM_JOINTS])
        lift = None
        if arm.has_lift():
            lift = lift_for_height(positive(ctx.params, "height", 0.8))
        else:
            ctx.log("No lift controller: pointing from the current height")
        try:
            pointing = point_at(target, start[0] if lift is None else lift)
        except ValueError as error:
            return SkillResult(False, f"Cannot point at {seat}: {error}")
        ok, text = arm.move(lift, pointing.arm, ctx.cancelled)
        if ok:
            # The base may still settle after Nav2 stopped: aim again from here.
            try:
                again = point_at(arm.to_base("map", position), pointing.lift)
            except ValueError as error:
                again, ok, text = None, False, str(error)
            if (
                again is not None
                and max(abs(a - b) for a, b in zip(again.arm, pointing.arm)) > REAIM
            ):
                ok, text = arm.move(lift, again.arm, ctx.cancelled)
        if ok:
            ctx.log(f"Pointing at {seat}")
            if not task_sleep(self.runtime.node, hold, ctx.cancelled):
                ok, text = False, "Cancelled"
        # Back to the start pose also after a failure or cancellation.
        back, back_text = arm.move(
            None if lift is None else start[0], start[1:], threading.Event()
        )
        if not ok:
            return SkillResult(False, f"Pointing at {seat} failed: {text}")
        if not back:
            return SkillResult(False, f"Pointed at {seat}, but not back: {back_text}")
        self._look_at_guest(ctx)
        return SkillResult(True, f"Pointed at {seat} for {guest}", outputs)

    def _look_at_guest(self, ctx):
        guest = ctx.params["guest"]
        People.of(self.runtime).follow(
            track=ctx.memory.get(f"{guest}.track"), name=guest
        )
