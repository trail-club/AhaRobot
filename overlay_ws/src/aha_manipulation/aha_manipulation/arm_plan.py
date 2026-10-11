"""Time-scaled planning of an arm trajectory. Pure Python, no ROS.

The real arm is driven by the upstream ``arm_node``, a black box that writes
every received goal to the servos immediately: the servos then ramp to it at
their own rate. Nothing downstream smooths a ``FollowJointTrajectory`` goal,
so a single goal would make the arm jump at full firmware speed.

This module is the interpolation layer. It turns a trajectory (a list of
``(positions, time_from_start)`` points, in the order of ``joint_names``)
into a sequence of position commands (``ArmSegment``) to be published at
specific wall-clock times. Each segment is time-stretched so that no joint
has to move faster than ``max_speed`` rad/s, and two commands are never
closer than the ``time_from_start`` gap asks for. Pacing the goals this way
is all the bridge can do to keep the motion smooth; the arrival of the final
point is confirmed separately against the joint states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# A point whose joints all move by less than this (rad) re-asserts the same
# position; it is dropped so the serial bus is not spammed with duplicates.
_EPSILON_RAD = 1e-6


@dataclass(frozen=True)
class ArmJoint:
    """One arm joint: name plus its position limits (rad)."""

    name: str
    min: float
    max: float

    def clamp(self, rad: float) -> tuple[float, bool]:
        """(rad limited to [min, max], whether it was changed)."""
        if self.min > self.max:
            raise ValueError(f"{self.name}: min {self.min} exceeds max {self.max}")
        value = min(max(rad, self.min), self.max)
        return value, value != rad


@dataclass
class ArmSegment:
    """One position command, to be published at ``send_at`` (monotonic s).

    ``targets`` carries only the joints present in the trajectory (already
    clamped). The node fills the remaining joints with their last commanded
    values before publishing the full vector, so a partial trajectory moves
    the listed joints and holds the rest.
    """

    send_at: float
    targets: dict[str, float] = field(repr=False)


@dataclass
class ArmPlan:
    """The result of :func:`plan_arm`."""

    segments: list[ArmSegment]
    end_time: float  # monotonic s of the last (or only) command
    clamped: list[tuple[str, float, float]]  # (joint, requested, sent)


def _check_joints(joints: dict[str, ArmJoint], joint_names: list[str]) -> None:
    if not joint_names:
        raise ValueError("trajectory has no joints")
    unknown = [name for name in joint_names if name not in joints]
    if unknown:
        raise ValueError(f"unknown joints {unknown} (known: {sorted(joints)})")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"duplicate joint names {list(joint_names)}")


def _check_point(
    k: int, positions: list, t: float, joint_names: list[str]
) -> None:
    if len(positions) != len(joint_names):
        raise ValueError(
            f"point {k}: {len(positions)} positions for {len(joint_names)} joints"
        )
    for name, value in zip(joint_names, positions):
        if not math.isfinite(value):
            raise ValueError(f"point {k}: {name} is not finite")
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"point {k}: time_from_start {t} is invalid")


def plan_arm(
    joints: dict[str, ArmJoint],
    joint_names: list[str],
    points: list[tuple[list[float], float]],
    *,
    now: float,
    max_speed: float,
) -> ArmPlan:
    """Plan ``points`` into time-stretched commands starting at ``now``.

    ``joints`` maps every valid joint name to its :class:`ArmJoint` (limits
    for clamping). ``joint_names`` is the ordered joint list of the
    trajectory; each entry of ``points`` is ``(positions, time_from_start)``
    with ``positions`` aligned to ``joint_names`` and times in seconds.
    ``max_speed`` (rad/s) caps the commanded motion; segments are stretched
    to honour it. Raises ``ValueError`` for malformed input.
    """
    if max_speed <= 0:
        raise ValueError(f"max_speed must be positive, got {max_speed}")
    _check_joints(joints, joint_names)
    if not points:
        raise ValueError("trajectory has no points")

    segments: list[ArmSegment] = []
    clamped: list[tuple[str, float, float]] = []
    elapsed = 0.0
    prev_t = 0.0
    prev_targets: dict[str, float] | None = None

    for k, (positions, t) in enumerate(points):
        _check_point(k, positions, t, joint_names)
        targets: dict[str, float] = {}
        for name, value in zip(joint_names, positions):
            joint = joints[name]
            sent, hit = joint.clamp(float(value))
            targets[name] = sent
            if hit:
                clamped.append((name, float(value), sent))

        if k > 0 and prev_targets is not None and all(
            abs(targets[n] - prev_targets[n]) < _EPSILON_RAD for n in joint_names
        ):
            # A static point only re-asserts the position already commanded:
            # drop it, but keep the time so later gaps are measured from it.
            prev_t = t
            prev_targets = targets
            continue

        if k == 0:
            # Start immediately; the servos ramp from wherever the arm is.
            duration = 0.0
        else:
            gap = max(t - prev_t, 0.0)
            move = max(
                (abs(targets[n] - prev_targets[n]) for n in joint_names),
                default=0.0,
            ) / max_speed
            duration = max(gap, move)

        elapsed += duration
        segments.append(ArmSegment(send_at=now + elapsed, targets=targets))
        prev_t = t
        prev_targets = targets

    return ArmPlan(segments=segments, end_time=now + elapsed, clamped=clamped)
