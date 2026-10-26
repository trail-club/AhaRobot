"""Frozen arm readings from a trajectory controller's feedback (ROS free).

After an arm board reset the arm_node keeps publishing the last joint
readings (docs/context/motor.md). An arm at rest reads constant values too,
so frozen readings only show while the arm is commanded to move: the
FollowJointTrajectory feedback of the arm trajectory bridge carries both the
sampled command (``desired``) and the readings (``actual``). ``FrozenWatch``
flags when every reading stays the same for ``window`` s while the command
moved more than ``motion`` since the readings last changed.
"""

from __future__ import annotations


class FrozenWatch:
    def __init__(self, window: float, motion: float):
        self.window = window  # s
        self.motion = motion  # rad
        self._actual = None  # readings at the last change
        self._since = None  # (time, desired) at the last change of the readings
        self.goal = None  # the caller's id of the watched trajectory goal

    def reset(self):
        self._actual = None
        self._since = None

    def update(self, t: float, desired, actual) -> str | None:
        """Feed one feedback sample (time s, positions); the reason when the
        readings look frozen, else None."""
        actual = tuple(float(v) for v in actual)
        desired = tuple(float(v) for v in desired)
        if actual != self._actual or self._since is None:
            self._actual = actual
            self._since = (t, desired)
            return None
        t0, desired0 = self._since
        if t - t0 <= self.window:
            return None
        moved = max(abs(a - b) for a, b in zip(desired, desired0))
        if moved <= self.motion:
            return None
        return (
            f"every joint read the same value for {t - t0:.1f} s while the "
            f"command moved {moved:.3f} rad: frozen readings (arm board reset or "
            "torque off?), restart the arm_node"
        )
