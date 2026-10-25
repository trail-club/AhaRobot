"""Time-scaled trajectory planning for the real Astra arms. Pure Python, no ROS.

The real arms are driven by the upstream ``astra_controller`` ``arm_node``, a
black box that writes every received ``JointCommand`` straight to the
servos; the firmware then ramps toward it with its own P-only controller.
Nothing downstream smooths a goal, so one goal per trajectory point would
make the arm jump at whatever rate the firmware allows.

This module is the motion-shaping layer. :func:`plan_arm` turns a
trajectory into an :class:`ArmPlan` of absolute-time keyframes:

* each input point is ``(joint_name, position_rad, time_from_start_s)`` with
  the time relative to ``now``;
* every keyframe carries the **full** command pose: joints present in the
  trajectory move, joints absent from it hold their last value (or the
  ``start_pose`` value before their first point);
* keyframe positions are clamped to the joint limits;
* consecutive keyframes are stretched in time until no joint moves faster
  than ``max_speed`` rad/s, so the piecewise-linear path in between is
  always inside the limits and never faster than ``max_speed``;
* keyframes that would only re-assert the current pose are merged away.

The bridge samples :meth:`ArmPlan.position_at` at a fixed rate and
publishes one position per sample, which is what keeps the firmware's
P-only controller on the intended path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Keyframes whose joints all move by less than this (rad) only re-assert
# the same pose; they are merged into the previous keyframe.
_EPSILON_RAD = 1e-6


@dataclass(frozen=True)
class ArmJoint:
    """One arm joint: name plus its position limits (rad)."""

    name: str
    min: float
    max: float

    def clamp(self, rad: float) -> tuple[float, bool]:
        """Return ``(rad limited to [min, max], whether it was changed)``."""
        if self.min > self.max:
            raise ValueError(f"{self.name}: min {self.min} exceeds max {self.max}")
        value = min(max(rad, self.min), self.max)
        return value, value != rad


@dataclass
class ArmSegment:
    """One keyframe, to be published at ``send_at`` (monotonic seconds).

    ``targets`` carries the full command pose (every joint in the plan's
    ``names``): trajectory joints at their value for this keyframe, the
    rest held at their last value.
    """

    send_at: float
    targets: dict[str, float] = field(repr=False)


@dataclass
class ArmPlan:
    """Result of :func:`plan_arm`.

    Keyframes are ``start_pose`` at ``start_time`` followed by
    ``segments``, in strictly increasing time order. ``clamped`` records
    every ``(joint, requested, sent)`` value that hit a joint limit.
    """

    names: list[str]  # full command joint list, in publish order
    start_time: float
    start_pose: dict[str, float]
    segments: list[ArmSegment] = field(default_factory=list)
    end_time: float = 0.0
    clamped: list[tuple[str, float, float]] = field(default_factory=list)

    def position_at(self, t: float) -> dict[str, float]:
        """The commanded pose at ``t``: piecewise-linear, held at the ends.

        Times before ``start_time`` hold the start pose, times after
        ``end_time`` hold the final pose.
        """
        times = [self.start_time] + [segment.send_at for segment in self.segments]
        poses = [self.start_pose] + [segment.targets for segment in self.segments]
        if t <= times[0]:
            return dict(poses[0])
        if t >= times[-1]:
            return dict(poses[-1])
        i = 0
        while t > times[i + 1]:
            i += 1
        t0, t1 = times[i], times[i + 1]
        if t1 <= t0:
            return dict(poses[i + 1])
        p0, p1 = poses[i], poses[i + 1]
        frac = (t - t0) / (t1 - t0)
        return {name: p0[name] + (p1[name] - p0[name]) * frac for name in self.names}


def plan_arm(
    joints: list[ArmJoint],
    command_joint_names: list[str],
    points: list[tuple[str, float, float]],
    *,
    now: float = 0.0,
    max_speed: float = 1.0,
    start_pose: dict[str, float] | None = None,
) -> ArmPlan:
    """Plan ``points`` into a speed-bounded keyframe sequence.

    Args:
        joints: position limits of the arm's joints (every joint that may
            move); joints absent from this list are never clamped.
        command_joint_names: full ``JointCommand`` joint list, in the order
            the bridge publishes it; every point must name one of these.
        points: trajectory points as ``(joint_name, position_rad,
            time_from_start_s)``, with the times relative to ``now``.
            Unsorted times and repeated joint/time pairs are fine; a joint
            a point does not mention keeps its last value.
        now: monotonic reference for the relative times.
        max_speed: commanded-motion cap in rad/s. Declared times are the
            *minimum* spacing between keyframes; the cap always wins.
        start_pose: the arm's current position per joint (what the bridge
            last commanded or measured); missing command joints start at 0.
            Arm joints are clamped into their limits.

    Raises:
        ValueError: on duplicate or empty joint names, limits with min >
            max, points naming joints outside ``command_joint_names``,
            non-finite or negative values/times, a non-positive
            ``max_speed``, or an empty trajectory.
    """
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise ValueError(f"max_speed must be positive and finite, got {max_speed}")
    if not command_joint_names:
        raise ValueError("command_joint_names is empty")
    if len(set(command_joint_names)) != len(command_joint_names):
        raise ValueError(f"duplicate joint names in {command_joint_names}")
    if not points:
        raise ValueError("trajectory has no points")

    limits: dict[str, ArmJoint] = {}
    for joint in joints:
        if joint.min > joint.max:
            raise ValueError(f"{joint.name}: min {joint.min} exceeds max {joint.max}")
        if joint.name in limits:
            raise ValueError(f"duplicate joint {joint.name}")
        limits[joint.name] = joint

    start = {name: 0.0 for name in command_joint_names}
    if start_pose:
        for name, value in start_pose.items():
            if name not in start:
                raise ValueError(f"start_pose joint {name!r} is not a command joint")
            start[name] = float(value)
    for name, joint in limits.items():
        if name in start:
            start[name], _ = joint.clamp(start[name])

    by_time: dict[float, dict[str, float]] = {}
    clamped: list[tuple[str, float, float]] = []
    for point in points:
        if len(point) != 3:
            raise ValueError(f"trajectory point {point!r} is not (joint, value, time)")
        joint_name, value, t = point
        if joint_name not in command_joint_names:
            raise ValueError(
                f"unknown joint {joint_name!r} (command joints: {command_joint_names})"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{joint_name}: position is not finite")
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"{joint_name}: time_from_start {t} is invalid")
        sent = value
        joint = limits.get(joint_name)
        if joint is not None:
            sent, hit = joint.clamp(value)
            if hit:
                clamped.append((joint_name, value, sent))
        by_time.setdefault(now + t, {})[joint_name] = sent

    # Absolute-time keyframes: the arm's start pose plus one per declared
    # time, each carrying every joint at its latest value so far.
    declared = dict(start)
    keyframes: list[tuple[float, dict[str, float]]] = [(now, dict(start))]
    for t in sorted(by_time):
        declared.update(by_time[t])
        keyframes.append((t, dict(declared)))

    # Stretch in time until no joint exceeds max_speed; drop keyframes that
    # only re-assert the pose. Stretched times are strictly increasing, so
    # the path between consecutive keyframes is the motion actually taken.
    stretched: list[tuple[float, dict[str, float]]] = [keyframes[0]]
    for t, pose in keyframes[1:]:
        prev_t, prev_pose = stretched[-1]
        move = max(abs(pose[name] - prev_pose[name]) for name in command_joint_names)
        if move < _EPSILON_RAD:
            continue
        stretched.append((prev_t + max(t - prev_t, move / max_speed), pose))

    segments = [ArmSegment(send_at=t, targets=pose) for t, pose in stretched[1:]]
    return ArmPlan(
        names=list(command_joint_names),
        start_time=now,
        start_pose=stretched[0][1],
        segments=segments,
        end_time=stretched[-1][0],
        clamped=clamped,
    )
