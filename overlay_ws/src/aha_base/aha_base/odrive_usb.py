"""ODrive v3.6 (firmware 0.5.x) over USB with the `odrive` Python package.

ROS-independent. `odrive` is imported only by `find_odrive()`, so the rest
of this module (and its tests) works without it.

All calls on the device object block on USB. Use one thread for them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# odrive.enums of firmware 0.5.x, kept here to avoid importing odrive.
AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
CONTROL_MODE_POSITION_CONTROL = 3
INPUT_MODE_TRAP_TRAJ = 5


class OdriveError(RuntimeError):
    """The device is missing, not ready or left closed-loop control."""


def normalize_serial(serial: str) -> str | None:
    """'0x315536763335' / '315536763335' -> '315536763335'; '' -> None (any).

    odrive.find_any() compares with the upper-case hex without leading zeros.
    """
    serial = serial.strip()
    if not serial:
        return None
    try:
        return format(int(serial, 16), "X")
    except ValueError:
        raise ValueError(f"serial_number is not hex: {serial!r}") from None


def find_odrive(serial_number: str | None, timeout: float):
    """Return the first ODrive on USB (matching serial_number if given).

    odrive 0.5.4's find_any() ignores its timeout argument and waits forever,
    so it runs in a thread that is cancelled after `timeout` seconds.
    """
    try:
        import odrive

        import fibre  # put on sys.path by `import odrive`
    except ImportError as e:
        raise OdriveError(
            f"cannot import the odrive package ({e}); install odrive==0.5.4"
        ) from None

    token = fibre.Event()
    result: dict = {}

    def run():
        try:
            result["dev"] = odrive.find_any(
                serial_number=serial_number, cancellation_token=token
            )
        except Exception as e:  # noqa: BLE001  (reported to the caller)
            result["err"] = e

    t = threading.Thread(target=run, name="odrive_find_any", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        token.set()
        t.join(1.0)
    if result.get("dev") is not None:
        return result["dev"]
    if "err" in result:
        raise OdriveError(f"find_any failed: {result['err']!r}")
    which = f"serial {serial_number}" if serial_number else "any serial"
    raise OdriveError(f"no ODrive on USB ({which}) within {timeout:g} s")


def axis_errors(axis) -> dict[str, int]:
    """Error bit fields of an axis and its sub-objects (0 = none)."""
    return {
        "axis": int(axis.error),
        "motor": int(axis.motor.error),
        "encoder": int(axis.encoder.error),
        "controller": int(axis.controller.error),
    }


def format_errors(errors: dict[str, int]) -> str:
    bad = {k: v for k, v in errors.items() if v}
    if not bad:
        return "no error"
    # Decode with the firmware's enums, e.g. `odrivetool` -> dump_errors(odrv0).
    return " ".join(f"{k}=0x{v:X}" for k, v in bad.items())


@dataclass
class DriveSettings:
    """Values written to RAM for the run and restored afterwards.

    Gains and current_lim < 0 keep the values stored on the device.
    """

    traj_vel_limit: float = 2.0  # [turn/s]
    traj_accel_limit: float = 1.0  # [turn/s^2]
    traj_decel_limit: float = 1.0  # [turn/s^2]
    pos_gain: float = -1.0
    vel_gain: float = -1.0
    vel_integrator_gain: float = -1.0
    current_lim: float = -1.0  # [A]
    watchdog_timeout: float = 0.5  # [s]; 0 disables the watchdog


class OdriveBase:
    """Two axes of one ODrive in position control with trapezoidal trajectories."""

    def __init__(self, dev, axis_ids: tuple[int, int], settings: DriveSettings):
        if len(set(axis_ids)) != 2 or not set(axis_ids) <= {0, 1}:
            raise ValueError(f"axis ids must be 0 and 1 in some order: {axis_ids}")
        self.dev = dev
        self.axis_ids = axis_ids
        self.axes = [getattr(dev, f"axis{i}") for i in axis_ids]
        self.settings = settings
        self._saved: dict[tuple[object, str], object] = {}
        self.log: list[str] = []  # human-readable notes from start()

    # -- setup / teardown ----------------------------------------------------

    def _set(self, obj, name: str, value) -> None:
        """Write a RAM value, remembering the original for restore()."""
        key = (obj, name)
        if key not in self._saved:
            self._saved[key] = getattr(obj, name)
        setattr(obj, name, value)

    def check_ready(self) -> None:
        problems = []
        for i, ax in zip(self.axis_ids, self.axes):
            if int(ax.current_state) != AXIS_STATE_IDLE:
                problems.append(
                    f"axis{i} is in state {int(ax.current_state)}, not IDLE"
                )
            if not ax.motor.is_calibrated:
                problems.append(f"axis{i} motor is not calibrated")
            if not ax.encoder.is_ready:
                problems.append(f"axis{i} encoder is not ready")
        if problems:
            raise OdriveError("; ".join(problems))

    def start(self, timeout: float = 3.0) -> list[float]:
        """Enter closed-loop control holding the present position.

        Returns the present positions (turns) that became the setpoints.
        """
        s = self.settings
        self.dev.clear_errors()
        self.check_ready()
        positions = []
        for i, ax in zip(self.axis_ids, self.axes):
            ctrl = ax.controller
            vel_limit = float(ctrl.config.vel_limit)
            traj_vel = s.traj_vel_limit
            if traj_vel > vel_limit:
                # The controller faults above its own vel_limit (times tolerance).
                self.log.append(
                    f"axis{i}: traj_vel_limit {traj_vel:g} capped to the "
                    f"controller vel_limit {vel_limit:g} turn/s"
                )
                traj_vel = vel_limit
            self._set(ax.trap_traj.config, "vel_limit", traj_vel)
            self._set(ax.trap_traj.config, "accel_limit", s.traj_accel_limit)
            self._set(ax.trap_traj.config, "decel_limit", s.traj_decel_limit)
            for name in ("pos_gain", "vel_gain", "vel_integrator_gain"):
                value = getattr(s, name)
                if value >= 0:
                    self._set(ctrl.config, name, value)
            if s.current_lim > 0:
                self._set(ax.motor.config, "current_lim", s.current_lim)
            self._set(ctrl.config, "control_mode", CONTROL_MODE_POSITION_CONTROL)
            self._set(ctrl.config, "input_mode", INPUT_MODE_TRAP_TRAJ)
            pos = float(ax.encoder.pos_estimate)
            ctrl.input_pos = pos
            positions.append(pos)
            if s.watchdog_timeout > 0:
                self._set(ax.config, "watchdog_timeout", s.watchdog_timeout)
                ax.watchdog_feed()
                self._set(ax.config, "enable_watchdog", True)
            self.log.append(
                f"axis{i}: Kt={float(ax.motor.config.torque_constant):g} "
                f"pos_gain={float(ctrl.config.pos_gain):g} "
                f"vel_gain={float(ctrl.config.vel_gain):g} "
                f"vel_integrator_gain={float(ctrl.config.vel_integrator_gain):g} "
                f"current_lim={float(ax.motor.config.current_lim):g} "
                f"traj vel/accel/decel={traj_vel:g}/{s.traj_accel_limit:g}/"
                f"{s.traj_decel_limit:g}"
            )
        for ax in self.axes:
            ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        deadline = time.monotonic() + timeout
        while True:
            self.feed()
            states = [int(ax.current_state) for ax in self.axes]
            if all(st == AXIS_STATE_CLOSED_LOOP_CONTROL for st in states):
                return positions
            if time.monotonic() > deadline:
                raise OdriveError(
                    "closed-loop control did not start: " + self.describe()
                )
            time.sleep(0.05)

    def feed(self) -> None:
        if self.settings.watchdog_timeout > 0:
            for ax in self.axes:
                ax.watchdog_feed()

    def stop(self) -> list[str]:
        """IDLE both axes, then restore the RAM values. Never raises.

        Returns the failures (e.g. the USB link is gone).
        """
        failures = []
        for i, ax in zip(self.axis_ids, self.axes):
            try:
                ax.requested_state = AXIS_STATE_IDLE
            except Exception as e:  # noqa: BLE001
                failures.append(f"axis{i} IDLE: {e!r}")
        for (obj, name), value in reversed(list(self._saved.items())):
            try:
                setattr(obj, name, value)
            except Exception as e:  # noqa: BLE001
                failures.append(f"restore {name}: {e!r}")
        self._saved.clear()
        return failures

    # -- cyclic --------------------------------------------------------------

    def read(self) -> list[tuple[float, float, int]]:
        """(pos_estimate [turn], vel_estimate [turn/s], current_state) per axis."""
        return [
            (
                float(ax.encoder.pos_estimate),
                float(ax.encoder.vel_estimate),
                int(ax.current_state),
            )
            for ax in self.axes
        ]

    def command(self, setpoints: list[float]) -> None:
        for ax, pos in zip(self.axes, setpoints):
            ax.controller.input_pos = pos
        self.feed()

    def describe(self) -> str:
        parts = []
        for i, ax in zip(self.axis_ids, self.axes):
            try:
                parts.append(
                    f"axis{i} state={int(ax.current_state)} "
                    f"{format_errors(axis_errors(ax))}"
                )
            except Exception as e:  # noqa: BLE001
                parts.append(f"axis{i} unreadable ({e!r})")
        return "; ".join(parts)
