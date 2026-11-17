"""Stand-in for the odrive 0.5.4 package in the aha_base node tests.

One simulated ODrive v3.6 with two axes. Configured by environment variables:

  FAKE_ODRIVE_MODE    ok (default) | missing (never found, find_any blocks
                      like odrive 0.5.4) | uncalibrated (axis1 motor) |
                      fault (axis0 leaves closed loop after FAKE_ODRIVE_FAULT_AFTER s)
  FAKE_ODRIVE_LOG     JSON lines of every attribute write and call
  FAKE_ODRIVE_START   "pos0,pos1": initial pos_estimate (turns)

In closed loop each axis moves toward controller.input_pos at most at
trap_traj.config.vel_limit; the watchdog IDLEs an axis with an error.
"""

import json
import os
import threading
import time

_LOG = os.environ.get("FAKE_ODRIVE_LOG")
_MODE = os.environ.get("FAKE_ODRIVE_MODE", "ok")
_log_lock = threading.Lock()
_lock = threading.RLock()  # the "firmware" state


def _log(path, value):
    if not _LOG:
        return
    with _log_lock, open(_LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"t": time.monotonic(), "path": path, "value": value}) + "\n"
        )


class Obj:
    def __init__(self, path, **attrs):
        object.__setattr__(self, "_path", path)
        for k, v in attrs.items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name, value):
        with _lock:
            object.__setattr__(self, name, value)
        _log(f"{self._path}.{name}", value)


class Axis(Obj):
    def __init__(self, i, pos):
        super().__init__(
            f"axis{i}",
            error=0,
            current_state=1,
            _requested_state=0,
            _last_feed=time.monotonic(),
            config=Obj(f"axis{i}.config", enable_watchdog=False, watchdog_timeout=0.0),
            motor=Obj(
                f"axis{i}.motor",
                error=0,
                is_calibrated=not (_MODE == "uncalibrated" and i == 1),
                config=Obj(
                    f"axis{i}.motor.config", current_lim=10.0, torque_constant=0.374
                ),
                current_control=Obj(f"axis{i}.motor.current_control", Iq_measured=0.0),
            ),
            encoder=Obj(
                f"axis{i}.encoder",
                error=0,
                is_ready=True,
                pos_estimate=pos,
                vel_estimate=0.0,
            ),
            controller=Obj(
                f"axis{i}.controller",
                error=0,
                input_pos=0.0,
                config=Obj(
                    f"axis{i}.controller.config",
                    control_mode=2,
                    input_mode=1,
                    pos_gain=20.0,
                    vel_gain=0.1,
                    vel_integrator_gain=0.0,
                    vel_limit=2.0,
                ),
            ),
            trap_traj=Obj(
                f"axis{i}.trap_traj",
                config=Obj(
                    f"axis{i}.trap_traj.config",
                    vel_limit=2.0,
                    accel_limit=0.5,
                    decel_limit=0.5,
                ),
            ),
        )

    @property
    def requested_state(self):
        return self._requested_state

    @requested_state.setter
    def requested_state(self, value):
        # Obj.__setattr__ logs the write.
        with _lock:
            object.__setattr__(self, "_requested_state", value)
            if value == 8:
                if (
                    self.motor.is_calibrated
                    and self.encoder.is_ready
                    and not self.error
                ):
                    object.__setattr__(self, "current_state", 8)
                    object.__setattr__(self, "_closed_since", time.monotonic())
                else:
                    object.__setattr__(self, "error", 0x40)  # NOT_CALIBRATED-ish
            elif value == 1:
                object.__setattr__(self, "current_state", 1)

    def watchdog_feed(self):
        with _lock:
            object.__setattr__(self, "_last_feed", time.monotonic())

    def step(self, dt, now):
        if self.current_state != 8:
            object.__setattr__(self.encoder, "vel_estimate", 0.0)
            return
        if (
            self.config.enable_watchdog
            and now - self._last_feed > self.config.watchdog_timeout
        ):
            object.__setattr__(self, "error", 0x800)  # WATCHDOG_TIMER_EXPIRED
            object.__setattr__(self, "current_state", 1)
            _log(f"{self._path}.watchdog_expired", True)
            return
        fault_after = float(os.environ.get("FAKE_ODRIVE_FAULT_AFTER", "1.0"))
        if (
            _MODE == "fault"
            and self._path == "axis0"
            and now - self._closed_since > fault_after
        ):
            object.__setattr__(self.motor, "error", 0x1000)
            object.__setattr__(self, "error", 0x40)
            object.__setattr__(self, "current_state", 1)
            return
        pos = self.encoder.pos_estimate
        err = self.controller.input_pos - pos
        vmax = self.trap_traj.config.vel_limit
        move = max(-vmax * dt, min(vmax * dt, err))
        object.__setattr__(self.encoder, "pos_estimate", pos + move)
        # first-order low-pass (~50 ms) like the encoder PLL
        vel = (
            self.encoder.vel_estimate
            + (move / dt - self.encoder.vel_estimate) * dt / 0.05
        )
        object.__setattr__(self.encoder, "vel_estimate", vel)


class Device(Obj):
    def __init__(self):
        start = [
            float(v) for v in os.environ.get("FAKE_ODRIVE_START", "0,0").split(",")
        ]
        super().__init__("dev", serial_number=0x315536763335, vbus_voltage=26.2)
        object.__setattr__(self, "axis0", Axis(0, start[0]))
        object.__setattr__(self, "axis1", Axis(1, start[1]))
        threading.Thread(target=self._run, daemon=True).start()

    def clear_errors(self):
        _log("dev.clear_errors", None)
        with _lock:
            for ax in (self.axis0, self.axis1):
                for o in (ax, ax.motor, ax.encoder, ax.controller):
                    object.__setattr__(o, "error", 0)

    def _run(self):
        dt = 0.005
        while True:
            time.sleep(dt)
            now = time.monotonic()
            with _lock:
                self.axis0.step(dt, now)
                self.axis1.step(dt, now)


_device = None


def find_any(path=None, serial_number=None, cancellation_token=None, timeout=None):
    global _device
    if _MODE == "missing" or serial_number not in (None, "315536763335"):
        done = threading.Event()
        if cancellation_token is not None:
            cancellation_token.subscribe(done.set)
        done.wait()
        return None
    if _device is None:
        _device = Device()
    return _device
