"""OdriveBase / find_odrive against the fake odrive package (no hardware)."""

import os
import sys
import time

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_odrive")
)

import odrive as fake  # noqa: E402  (the fake in test/fake_odrive)

from aha_base import odrive_usb  # noqa: E402
from aha_base.odrive_usb import (  # noqa: E402
    DriveSettings,
    OdriveBase,
    OdriveError,
    find_odrive,
    normalize_serial,
)


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setattr(fake, "_MODE", "ok")
    monkeypatch.setattr(fake, "_LOG", None)
    monkeypatch.setattr(fake, "_device", None)
    return fake.find_any()


def test_normalize_serial():
    assert normalize_serial("") is None
    assert normalize_serial(" 0x315536763335 ") == "315536763335"
    assert normalize_serial("00ab") == "AB"
    with pytest.raises(ValueError):
        normalize_serial("odrive")


def test_start_holds_position_and_keeps_device_gains(dev):
    dev.axis0.encoder.pos_estimate = 3.25
    base = OdriveBase(dev, (0, 1), DriveSettings())
    assert base.start() == [3.25, 0.0]
    ax = dev.axis0
    assert ax.controller.input_pos == 3.25
    assert ax.current_state == 8
    assert (ax.controller.config.control_mode, ax.controller.config.input_mode) == (
        3,
        5,
    )
    assert ax.controller.config.pos_gain == 20.0  # kept
    assert ax.motor.config.current_lim == 10.0  # kept
    assert ax.config.enable_watchdog is True
    assert base.stop() == []
    assert ax.current_state == 1
    assert ax.config.enable_watchdog is False
    assert ax.controller.config.input_mode == 1
    assert ax.trap_traj.config.vel_limit == 2.0


def test_overrides_are_restored_and_traj_capped(dev):
    dev.axis1.controller.config.vel_limit = 1.5
    s = DriveSettings(traj_vel_limit=3.0, pos_gain=5.0, vel_gain=0.2, current_lim=4.0)
    base = OdriveBase(dev, (1, 0), s)  # left = axis1
    base.start()
    assert dev.axis1.trap_traj.config.vel_limit == 1.5
    assert dev.axis0.trap_traj.config.vel_limit == 2.0  # 3.0 capped to 2.0
    assert any("capped" in line for line in base.log)
    assert dev.axis0.controller.config.pos_gain == 5.0
    assert dev.axis0.motor.config.current_lim == 4.0
    base.stop()
    assert dev.axis0.controller.config.pos_gain == 20.0
    assert dev.axis0.controller.config.vel_gain == 0.1
    assert dev.axis0.motor.config.current_lim == 10.0
    assert dev.axis1.trap_traj.config.vel_limit == 2.0


def test_uncalibrated_axis_is_refused(dev):
    dev.axis1.motor.is_calibrated = False
    base = OdriveBase(dev, (0, 1), DriveSettings())
    with pytest.raises(OdriveError, match="axis1 motor is not calibrated"):
        base.start()
    assert dev.axis0.current_state == 1
    assert dev.axis0.requested_state == 0  # closed loop never requested
    base.stop()


def test_axis_ids_validated(dev):
    with pytest.raises(ValueError):
        OdriveBase(dev, (0, 0), DriveSettings())


def test_find_odrive_times_out(monkeypatch):
    monkeypatch.setattr(fake, "_MODE", "missing")
    t0 = time.monotonic()
    with pytest.raises(OdriveError, match="no ODrive on USB"):
        find_odrive(None, 0.3)
    assert time.monotonic() - t0 < 2.0


def test_find_odrive_by_serial(dev):
    assert find_odrive("315536763335", 1.0) is dev
    with pytest.raises(OdriveError):
        find_odrive("1234", 0.2)


def test_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "odrive", None)
    with pytest.raises(OdriveError, match="odrive==0.5.4"):
        odrive_usb.find_odrive(None, 0.1)
