"""Frozen arm readings from a trajectory controller's feedback (ROS free).

After an arm board reset the arm_node keeps publishing the last joint
readings, of every joint or of the wrist servos only (docs/context/motor.md).
An arm at rest reads constant values too, so frozen readings only show while
the arm is commanded to move: the FollowJointTrajectory feedback of the arm
trajectory bridge carries both the sampled command (``desired``) and the
readings (``actual``). ``FrozenWatch`` flags a joint whose reading stays the
same for ``window`` s while its command moved more than ``motion`` since that
reading last changed.
"""

from __future__ import annotations


class FrozenWatch:
    def __init__(self, window: float, motion: float):
        self.window = window  # s
        self.motion = motion  # rad
        self._since = []  # per joint: (reading, time, command) at its last change
        self.goal = None  # the caller's id of the watched trajectory goal
        self.frozen = []  # indices of the joints flagged by the last update

    def reset(self):
        self._since = []

    def update(self, t: float, desired, actual, names=()) -> str | None:
        """Feed one feedback sample (time s, positions in joint order, optional
        joint names); the reason when a joint's reading looks frozen, else None."""
        desired = [float(v) for v in desired]
        actual = [float(v) for v in actual]
        if len(self._since) != len(actual):
            self._since = [(a, t, d) for a, d in zip(actual, desired)]
            return None
        frozen = []
        self.frozen = []
        for k, (a, d) in enumerate(zip(actual, desired)):
            reading, t0, d0 = self._since[k]
            if a != reading:
                self._since[k] = (a, t, d)
            elif t - t0 > self.window and abs(d - d0) > self.motion:
                name = names[k] if k < len(names) else f"joint {k}"
                frozen.append(
                    f"{name} ({t - t0:.1f} s, command moved {abs(d - d0):.3f} rad)"
                )
                self.frozen.append(k)
        if not frozen:
            return None
        return (
            f"readings unchanged while commanded to move: {', '.join(frozen)}: "
            "frozen readings (arm board reset or torque off?), restart the arm_node"
        )
