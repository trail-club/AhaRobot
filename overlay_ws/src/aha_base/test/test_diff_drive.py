import math

import pytest

from aha_base.diff_drive import (
    Kinematics,
    Odometry,
    advance_setpoint,
    limit_wheel_speeds,
)

KIN = Kinematics(0.139, 0.44)


def test_wheel_speeds_match_upstream():
    # upstream base_controller.set_vel (before the install direction)
    v, w = 0.3, 0.8
    left = (v - w * 0.44 / 2) / (math.pi * 0.139)
    right = (v + w * 0.44 / 2) / (math.pi * 0.139)
    assert KIN.wheel_speeds(v, w) == pytest.approx((left, right))


def test_body_twist_inverts_wheel_speeds():
    assert KIN.body_twist(*KIN.wheel_speeds(-0.2, 1.3)) == pytest.approx((-0.2, 1.3))


def test_invalid_geometry():
    with pytest.raises(ValueError):
        Kinematics(0.0, 0.44)


def test_limit_keeps_ratio():
    left, right = limit_wheel_speeds(1.0, 4.0, 2.0)
    assert (left, right) == pytest.approx((0.5, 2.0))
    assert limit_wheel_speeds(1.0, -1.5, 2.0) == (1.0, -1.5)


def test_odometry_straight_and_turn():
    odom = Odometry(KIN)
    odom.update(10.0, 20.0)  # first sample only sets the reference
    assert (odom.x, odom.y, odom.theta) == (0.0, 0.0, 0.0)
    odom.update(11.0, 21.0)
    assert odom.x == pytest.approx(math.pi * 0.139)
    assert odom.y == pytest.approx(0.0)
    # turn in place by +90 deg: right forward, left backward
    quarter = (math.pi / 2) * 0.44 / 2 / (math.pi * 0.139)
    odom.update(11.0 - quarter, 21.0 + quarter)
    assert odom.theta == pytest.approx(math.pi / 2)
    assert odom.x == pytest.approx(math.pi * 0.139)
    qx, qy, qz, qw = odom.quaternion()
    assert (qz, qw) == pytest.approx((math.sin(math.pi / 4), math.cos(math.pi / 4)))


def test_odometry_wraps_theta():
    odom = Odometry(KIN)
    odom.update(0.0, 0.0)
    half_turn = math.pi * 0.44 / 2 / (math.pi * 0.139)
    for k in range(1, 4):
        odom.update(-half_turn * k / 1.5, half_turn * k / 1.5)
    assert -math.pi <= odom.theta <= math.pi
    assert odom.theta == pytest.approx(0.0, abs=1e-9)


def test_setpoint_clamped_to_lead():
    assert advance_setpoint(0.0, 1.0, 0.1, 0.0, 2.0) == pytest.approx(0.1)
    assert advance_setpoint(1.95, 1.0, 0.1, 0.0, 2.0) == 2.0
    assert advance_setpoint(-1.95, -1.0, 0.1, 0.0, 2.0) == -2.0
