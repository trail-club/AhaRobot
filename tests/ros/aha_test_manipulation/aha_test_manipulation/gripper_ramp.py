"""Ramp sequences and contact detection for a gripper handover (ROS free).

Used by ``scripts/gripper_handover.py`` to open/close a real arm_node gripper
(joint_[r|l]7r, position_cmd in m per side) without a single full-speed
command (closing onto an object reset the arm board, docs/context/motor.md):
commands are stepped from the measured finger position instead of jumping,
and a close without a known object width stops advancing once the measured
position stops following the command (the fingers are pushing on something),
holding a small squeeze beyond that contact point rather than driving to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# astra_controller's arm_node position_cmd range for joint_[r|l]7r (real
# hardware; wider than the simulated gripper's place_geometry.FINGER_MAX_OPENING).
FINGER_RANGE = (-0.002, 0.055)

# Contact detection (close without --width): commanded ahead of measured by
# more than this (m) while the measured position closes by less than
# CONTACT_STALL (m) per sample, for this many consecutive samples, counts as
# the fingers touching something. The gap alone is not enough: the real
# fingers lag the 1 s ramp by more than 4 mm while still moving freely.
CONTACT_GAP = 0.004
CONTACT_STALL = 0.0005
CONTACT_SAMPLES = 3

# Staleness detection: measured values identical for at least this long (s)
# while commanding suggests the state topic is frozen, not that the fingers
# stopped there.
STALE_WINDOW = 2.0
STALE_EPSILON = 1e-9


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def close_target(
    width: float, squeeze: float, finger_range: tuple[float, float] = FINGER_RANGE
) -> float:
    """Finger position per side (clamped to ``finger_range``) that grips an
    object of ``width`` (across the fingers) with ``squeeze`` beyond contact."""
    return clamp(width / 2 - squeeze, *finger_range)


def estimated_width(final_position: float, squeeze: float) -> float:
    """Object width implied by a final finger position: the inverse of
    ``close_target`` (also matches a contact-detected close, whose held
    position is the contact position minus ``squeeze``)."""
    return 2 * (final_position + squeeze)


def ramp_steps(duration: float, rate: float) -> int:
    """Number of command steps for a ramp of ``duration`` s at ``rate`` Hz
    (at least 1, so a single command still goes out)."""
    return max(1, round(duration * rate))


def is_stale(
    samples: list[tuple[float, float]],
    window: float = STALE_WINDOW,
    epsilon: float = STALE_EPSILON,
) -> bool:
    """True if ``samples`` (monotonic time, measured position), taken while
    commanding, span at least ``window`` s and are all equal within
    ``epsilon`` — the state looks frozen rather than settled at a new value."""
    if len(samples) < 2:
        return False
    t0 = samples[0][0]
    t1 = samples[-1][0]
    if t1 - t0 < window:
        return False
    first = samples[0][1]
    return all(abs(p - first) <= epsilon for _, p in samples)


class SimpleRamp:
    """Fixed-target ramp from ``start`` to ``target`` over ``duration`` s at
    ``rate`` Hz, with no contact detection (``open``, or ``close`` with a
    known --width)."""

    def __init__(
        self,
        start: float,
        target: float,
        duration: float,
        rate: float,
        finger_range: tuple[float, float] = FINGER_RANGE,
    ):
        self.start = clamp(start, *finger_range)
        self.target = clamp(target, *finger_range)
        self.total_steps = ramp_steps(duration, rate)
        self.finger_range = finger_range
        self._step = 0

    @property
    def done(self) -> bool:
        return self._step >= self.total_steps

    def next_command(self, measured: float | None = None) -> float:
        """The next finger command; call once per tick. ``measured`` is
        accepted (and ignored) so callers can drive SimpleRamp and ContactRamp
        the same way."""
        del measured
        self._step += 1
        n = self.total_steps
        value = (
            self.target
            if self._step >= n
            else self.start + (self.target - self.start) * self._step / n
        )
        return clamp(value, *self.finger_range)


@dataclass
class ContactRamp:
    """Close toward ``target`` (0.0: fully closed) over ``duration`` s at
    ``rate`` Hz, but stop advancing once the measured finger position stops
    following the command down: while closing freely, the state tracks the
    (decreasing) command closely, but once the fingers meet the object they
    cannot close further, so the state stays pinned above the still-dropping
    command. ``samples`` consecutive ticks with ``measured - last command >
    gap`` while the measured position closes by less than ``stall`` per tick
    count as contact (a lagging but moving finger is not). Once that happens
    the command holds at ``measured - squeeze`` (``squeeze`` beyond the
    contact, 0 = no pressing, instead of driving on to the target), clamped
    to the range.
    """

    start: float
    target: float
    duration: float
    rate: float
    squeeze: float
    gap: float = CONTACT_GAP
    stall: float = CONTACT_STALL
    samples: int = CONTACT_SAMPLES
    finger_range: tuple[float, float] = FINGER_RANGE

    _step: int = field(default=0, init=False, repr=False)
    _streak: int = field(default=0, init=False, repr=False)
    _last_command: float = field(default=0.0, init=False, repr=False)
    _last_measured: float | None = field(default=None, init=False, repr=False)
    _held: float | None = field(default=None, init=False, repr=False)
    contact_position: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.start = clamp(self.start, *self.finger_range)
        self.target = clamp(self.target, *self.finger_range)
        self.total_steps = ramp_steps(self.duration, self.rate)
        self._last_command = self.start

    @property
    def done(self) -> bool:
        # A few ticks at the target after the ramp, so that a contact near
        # its end is still seen (and eased off) instead of held at the target.
        settle = self.samples + 1
        return self._held is not None or self._step >= self.total_steps + settle

    @property
    def holding(self) -> bool:
        return self._held is not None

    def next_command(self, measured: float | None) -> float:
        """The next finger command given the latest ``measured`` finger
        position; call once per tick at ``rate`` Hz."""
        if self._held is not None:
            return self._held
        if measured is not None:
            gap = measured - self._last_command
            stalled = (
                self._last_measured is not None
                and self._last_measured - measured < self.stall
            )
            self._last_measured = measured
            self._streak = self._streak + 1 if gap > self.gap and stalled else 0
            if self._streak >= self.samples:
                self.contact_position = measured
                self._held = clamp(measured - self.squeeze, *self.finger_range)
                return self._held
        self._step += 1
        n = self.total_steps
        value = (
            self.target
            if self._step >= n
            else self.start + (self.target - self.start) * self._step / n
        )
        self._last_command = clamp(value, *self.finger_range)
        return self._last_command


def finger_letter(arm: str) -> str:
    return "r" if arm == "right" else "l"


def finger_joint(arm: str) -> str:
    """Name of the commanded/measured finger joint (joint_command's ``name``
    and the one tracked in ``/joint_states``; the mirrored joint_[r|l]7l is
    not used)."""
    return f"joint_{finger_letter(arm)}7r"
