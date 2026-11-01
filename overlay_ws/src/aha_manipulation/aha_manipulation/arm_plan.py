"""Trajectory checks and sampling for the real Astra arms (no ROS).

The upstream ``arm_node`` writes every ``JointCommand`` to the arm board as
one goal; the board firmware moves there with its own trapezoidal profile.
To follow a trajectory, arm_trajectory_bridge.py samples :class:`ArmPlan` at
a fixed rate and sends each sample as a goal.

:func:`plan_trajectory` checks a trajectory against the joint limits and
builds the plan:

- Knots are the measured start pose at ``now`` and one full pose per point.
  Joints the trajectory does not name keep their start position.
- A point at ``time_from_start`` 0 is the start (MoveIt sends the current
  state there): it must be within ``start_tolerance`` of the measured pose.
  The plan keeps the trajectory's shape and adds the gap to the measured pose
  fading linearly to 0 at the last point, so the arm starts where it is and
  a short first segment does not turn the gap into a jerk. The fade adds
  gap / duration to the velocity and nothing to the acceleration.
- Positions outside ``[min, max]`` and motions faster than ``max_velocity``
  or ``max_acceleration`` are rejected, not clamped or slowed down.
- With velocities in every point, knots are joined by cubic Hermite segments
  (like JointTrajectoryController); otherwise by straight lines. The arm
  starts and ends at rest. For Hermite segments the limits apply to the
  peak velocity and acceleration of the cubic, not only to the given values.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

# Relative allowance on the velocity and acceleration limits for roundoff in
# a trajectory generated at the limit (MoveIt's time parameterization).
LIMIT_ALLOWANCE = 1e-3


@dataclass(frozen=True)
class JointLimits:
    """Position (rad), velocity (rad/s) and acceleration (rad/s^2) limits."""

    name: str
    min: float
    max: float
    max_velocity: float
    max_acceleration: float

    def clamp(self, rad):
        return min(max(rad, self.min), self.max)


@dataclass(frozen=True)
class Point:
    """One trajectory point; velocities / accelerations may be empty."""

    positions: tuple
    velocities: tuple
    accelerations: tuple
    time: float  # s from the trajectory start


@dataclass
class ArmPlan:
    """Knots in absolute time; ``velocities`` is None for linear segments."""

    names: tuple  # every arm joint, in command order
    limits: dict  # name -> JointLimits
    times: list
    positions: list  # one {joint: rad} per knot
    velocities: list | None

    @property
    def end_time(self):
        return self.times[-1]

    @property
    def final(self):
        return dict(self.positions[-1])

    def sample(self, t):
        """Commanded pose at absolute time t, inside the position limits."""
        if t <= self.times[0]:
            pose = self.positions[0]
        elif t >= self.times[-1]:
            pose = self.positions[-1]
        else:
            k = bisect.bisect_right(self.times, t) - 1
            t0, t1 = self.times[k], self.times[k + 1]
            s = (t - t0) / (t1 - t0)
            p0, p1 = self.positions[k], self.positions[k + 1]
            if self.velocities is None:
                pose = {n: p0[n] + (p1[n] - p0[n]) * s for n in self.names}
            else:
                v0, v1 = self.velocities[k], self.velocities[k + 1]
                pose = {
                    n: _hermite(p0[n], v0[n], p1[n], v1[n], t1 - t0, s)
                    for n in self.names
                }
        # A Hermite segment can bulge slightly past a knot at a limit; the
        # arm_node drops a whole command with one value out of range.
        return {n: self.limits[n].clamp(pose[n]) for n in self.names}


def _hermite(p0, v0, p1, v1, dt, s):
    s2, s3 = s * s, s * s * s
    return (
        (2 * s3 - 3 * s2 + 1) * p0
        + (s3 - 2 * s2 + s) * dt * v0
        + (-2 * s3 + 3 * s2) * p1
        + (s3 - s2) * dt * v1
    )


def _hermite_peaks(p0, v0, p1, v1, dt):
    """Peak |velocity| and |acceleration| of a Hermite segment over [0, dt].

    With d = (p1 - p0) / dt the velocity is a s^2 + b s + v0 in s = t / dt,
    and the acceleration is linear in s, so it peaks at an end.
    """
    d = (p1 - p0) / dt
    a = -6 * d + 3 * v0 + 3 * v1
    b = 6 * d - 4 * v0 - 2 * v1
    speeds = [abs(v0), abs(v1)]
    if a != 0 and 0 < -b / (2 * a) < 1:
        s = -b / (2 * a)
        speeds.append(abs(a * s * s + b * s + v0))
    accelerations = (
        abs(6 * d - 4 * v0 - 2 * v1) / dt,
        abs(-6 * d + 2 * v0 + 4 * v1) / dt,
    )
    return max(speeds), max(accelerations)


def _check_finite(values, what):
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{what} is not finite")


def plan_trajectory(
    limits, joint_names, points, start, now, start_tolerance
) -> ArmPlan:
    """ArmPlan for a trajectory from the measured ``start`` pose; raises ValueError.

    limits: [JointLimits] for every arm joint, in command order.
    joint_names: the trajectory's joints (a subset of the arm joints).
    points: [Point] in the trajectory's order.
    start: {joint: rad} measured now for every arm joint.
    """
    order = tuple(j.name for j in limits)
    by_name = {j.name: j for j in limits}
    names = list(joint_names)
    if not names:
        raise ValueError("trajectory has no joint names")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate joint names {names}")
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown joints {unknown} (this arm: {list(order)})")
    if not points:
        raise ValueError("trajectory has no points")

    hermite = all(len(p.velocities) for p in points)
    gap = {}  # measured start - point 0, for a point at time 0
    times = [now]
    positions = [{n: float(start[n]) for n in order}]
    velocities = [{n: 0.0 for n in order}]
    previous = 0.0
    for k, point in enumerate(points):
        if len(point.positions) != len(names):
            raise ValueError(
                f"point {k}: {len(point.positions)} positions for {len(names)} joints"
            )
        for field, values in (
            ("velocities", point.velocities),
            ("accelerations", point.accelerations),
        ):
            if len(values) not in (0, len(names)):
                raise ValueError(
                    f"point {k}: {len(values)} {field} for {len(names)} joints"
                )
            _check_finite(values, f"point {k}: a value in {field}")
        _check_finite(point.positions, f"point {k}: a position")
        _check_finite([point.time], f"point {k}: time_from_start")

        if k == 0 and point.time == 0.0:
            for name, value in zip(names, point.positions):
                off = float(start[name]) - float(value)
                if abs(off) > start_tolerance:
                    raise ValueError(
                        f"point 0 at time 0: {name} {value:+.3f} rad is {abs(off):.3f} "
                        f"rad from the current position (start tolerance {start_tolerance})"
                    )
                positions[0][name] = float(value)
                gap[name] = off
            continue
        if point.time <= previous:
            raise ValueError(f"point {k}: time_from_start does not increase")
        previous = point.time

        pose = dict(positions[-1])
        speed = {n: 0.0 for n in order}
        for i, name in enumerate(names):
            joint = by_name[name]
            value = float(point.positions[i])
            if not joint.min <= value <= joint.max:
                raise ValueError(
                    f"point {k}: {name} {value:+.3f} rad is outside "
                    f"[{joint.min:+.3f}, {joint.max:+.3f}]"
                )
            pose[name] = value
            if point.velocities:
                speed[name] = float(point.velocities[i])
                _check_limit(joint, "velocity", speed[name], joint.max_velocity, k)
            if point.accelerations:
                _check_limit(
                    joint,
                    "acceleration",
                    float(point.accelerations[i]),
                    joint.max_acceleration,
                    k,
                )
        times.append(now + point.time)
        positions.append(pose)
        velocities.append(speed)

    duration = times[-1] - now
    for name, off in gap.items():
        for k, t in enumerate(times):
            positions[k][name] += (
                off * (1 - (t - now) / duration) if duration > 0 else off
            )
            if duration > 0:
                velocities[k][name] -= off / duration
    if len(times) == 1:
        # Only the start point: nothing to move.
        times.append(now)
        positions.append(dict(positions[0]))
        velocities.append(dict(velocities[0]))
    _check_segments(limits, times, positions, velocities if hermite else None)
    return ArmPlan(
        names=order,
        limits=by_name,
        times=times,
        positions=positions,
        velocities=velocities if hermite else None,
    )


def _check_limit(joint, quantity, value, limit, k):
    if abs(value) > limit * (1 + LIMIT_ALLOWANCE):
        raise ValueError(
            f"point {k}: {joint.name} {quantity} {abs(value):.3f} exceeds {limit:.3f}"
        )


def _check_segments(limits, times, positions, velocities):
    """Mean velocity of every segment, then the motion between the knots.

    Hermite segments (``velocities`` given): the peak velocity and
    acceleration of each cubic. Linear segments have no accelerations of their
    own, so the velocity change at each knot (starting and ending at rest)
    over the time between the neighbouring segment midpoints stands for one.
    """
    for joint in limits:
        name = joint.name
        mean = [0.0]  # at rest before the start
        for k in range(1, len(times)):
            dt = times[k] - times[k - 1]
            if dt <= 0:
                mean.append(0.0)
                continue
            v = (positions[k][name] - positions[k - 1][name]) / dt
            if abs(v) > joint.max_velocity * (1 + LIMIT_ALLOWANCE):
                raise ValueError(
                    f"segment {k}: {name} moves at {abs(v):.3f} rad/s, "
                    f"over its limit {joint.max_velocity:.3f}"
                )
            mean.append(v)
            if velocities is None:
                continue
            speed, accel = _hermite_peaks(
                positions[k - 1][name],
                velocities[k - 1][name],
                positions[k][name],
                velocities[k][name],
                dt,
            )
            if speed > joint.max_velocity * (1 + LIMIT_ALLOWANCE):
                raise ValueError(
                    f"segment {k}: {name} reaches {speed:.3f} rad/s between the "
                    f"points, over its limit {joint.max_velocity:.3f} "
                    "(give the trajectory more time)"
                )
            if accel > joint.max_acceleration * (1 + LIMIT_ALLOWANCE):
                raise ValueError(
                    f"segment {k}: {name} needs {accel:.3f} rad/s^2 between the "
                    f"points, over its limit {joint.max_acceleration:.3f} "
                    "(give the trajectory more time)"
                )
        if velocities is not None:
            continue
        mean.append(0.0)  # at rest after the end
        spans = [0.0] + [times[k] - times[k - 1] for k in range(1, len(times))] + [0.0]
        for k in range(len(mean) - 1):
            between = (spans[k] + spans[k + 1]) / 2
            if between <= 0:
                continue
            a = (mean[k + 1] - mean[k]) / between
            if abs(a) > joint.max_acceleration * (1 + LIMIT_ALLOWANCE):
                raise ValueError(
                    f"knot {k}: {name} needs {abs(a):.3f} rad/s^2, over its limit "
                    f"{joint.max_acceleration:.3f} (give the trajectory more time)"
                )
