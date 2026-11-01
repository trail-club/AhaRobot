import math

import numpy as np
import pytest

from aha_perception.imu_tilt import (
    GRAVITY,
    GravityFilter,
    joint_tilt,
    tilt_roll,
    up_vector,
)


def test_level_camera():
    assert tilt_roll((0.0, -GRAVITY, 0.0)) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_looking_down_is_positive_tilt():
    # Optical axis (z) 30 deg below the horizon: up has a -z component.
    tilt, roll = tilt_roll((0.0, -math.cos(math.radians(30)), -0.5))
    assert math.degrees(tilt) == pytest.approx(30.0)
    assert roll == pytest.approx(0.0)


def test_right_side_down_is_positive_roll():
    # Image right side (x) down: up gets a -x component.
    tilt, roll = tilt_roll((-math.sin(0.1), -math.cos(0.1), 0.0))
    assert tilt == pytest.approx(0.0)
    assert roll == pytest.approx(0.1)


@pytest.mark.parametrize("tilt", np.radians([-60, -30, 0, 10, 45, 80]))
@pytest.mark.parametrize("roll", np.radians([-20, -3, 0, 5, 30]))
@pytest.mark.parametrize("scale", [1.0, GRAVITY])
def test_round_trip(tilt, roll, scale):
    assert tilt_roll(scale * up_vector(tilt, roll)) == pytest.approx(
        (tilt, roll), abs=1e-9
    )


@pytest.mark.parametrize("tilt_deg", [-30.0, 0.0, 25.0, 50.0])
@pytest.mark.parametrize("pan_deg", [-60.0, 0.0, 60.0])
def test_head_pose_maps_to_joint_tilt(camera_pose, tilt_deg, pan_deg):
    # camera_pose: head camera optical frame in a level base frame, +tilt down.
    rotation = camera_pose(tilt_deg=tilt_deg, pan_deg=pan_deg)[:3, :3]
    accel = rotation.T @ np.array([0.0, 0.0, GRAVITY])
    tilt, roll = tilt_roll(accel)
    assert math.degrees(joint_tilt(tilt)) == pytest.approx(tilt_deg, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_joint_tilt_offset():
    assert joint_tilt(0.3, offset=-0.05, tilt_at_zero=0.1) == pytest.approx(0.15)


DT = 0.005  # 200 Hz, as the RealSense unite_imu_method output
ZERO = (0.0, 0.0, 0.0)


def level(tilt=0.0):
    return GRAVITY * up_vector(tilt)


def tilt_deg(estimate):
    return math.degrees(tilt_roll(estimate)[0])


def imu_motion(camera_pose, tilt_deg_at, pan_deg_at, times, linear_accel=None):
    """(t, accel, gyro, true tilt deg) of a head moving in a level base.

    The gyro is the body rate from R^T dR/dt of the camera pose, independent of
    the filter's sign convention; ``linear_accel(t)`` (base frame) is added.
    """
    h = 1e-5

    def rotation(t):
        return camera_pose(tilt_deg=tilt_deg_at(t), pan_deg=pan_deg_at(t))[:3, :3]

    for t in times:
        r = rotation(t)
        skew = r.T @ (rotation(t + h) - rotation(t - h)) / (2 * h)
        gyro = (skew[2, 1], skew[0, 2], skew[1, 0])
        specific = np.array([0.0, 0.0, GRAVITY])
        if linear_accel is not None:
            specific = specific + linear_accel(t)
        yield t, r.T @ specific, gyro, tilt_deg_at(t)


def test_filter_initialises_from_first_sample():
    f = GravityFilter()
    assert f.update(0.0, level(0.2), (0.0, 0.3, 0.0)) == pytest.approx(
        tuple(up_vector(0.2))
    )
    assert not f.stationary
    f.update(DT, level(0.2), ZERO)
    assert f.stationary


def test_filter_static_convergence():
    f = GravityFilter(time_constant=0.5)
    f.update(0.0, level(0.0), ZERO)
    t = 0.0
    for _ in range(100):  # one time constant
        t += DT
        estimate = f.update(t, level(0.2), ZERO)
    assert tilt_roll(estimate)[0] / 0.2 == pytest.approx(1 - math.exp(-1), abs=0.01)
    for _ in range(1000):
        t += DT
        estimate = f.update(t, level(0.2), ZERO)
    assert tilt_roll(estimate)[0] == pytest.approx(0.2, abs=1e-5)


def test_filter_gyro_sign(camera_pose):
    # Head tilting down (+tilt) at 1 rad/s; accel rejected so only the gyro acts.
    rate = math.degrees(1.0)
    f = GravityFilter()
    f.update(0.0, level(0.0), ZERO)
    previous = 0.0
    times = np.arange(1, 101) * DT
    for t, accel, gyro, true in imu_motion(
        camera_pose, lambda t: rate * t, lambda t: 0.0, times
    ):
        current = tilt_deg(f.update(t, 2.0 * accel, gyro))
        assert current > previous
        previous = current
    assert previous == pytest.approx(true, abs=0.05)


@pytest.mark.parametrize("bias", [0.0, 0.01])
def test_filter_tracks_constant_rate_rotation(camera_pose, bias):
    # Tilt 0 -> 60 deg at 60 deg/s while panning at 45 deg/s, gyro bias in rad/s.
    f = GravityFilter(time_constant=0.5)
    errors = []
    previous = None
    times = np.arange(0, 201) * DT
    for t, accel, gyro, true in imu_motion(
        camera_pose, lambda t: 60.0 * t, lambda t: 45.0 * t, times
    ):
        current = tilt_deg(f.update(t, accel, np.add(gyro, bias)))
        if previous is not None:
            assert current - previous > 0.1  # moves every sample, no freeze
        previous = current
        errors.append(abs(current - true))
    assert max(errors) < 0.2 + math.degrees(bias * 0.5)


def test_filter_rejects_dynamic_acceleration(camera_pose):
    # Head at rest at 20 deg while the base accelerates at 5 m/s^2 for 0.5 s.
    f = GravityFilter()
    times = np.arange(0, 101) * DT
    for t, accel, gyro, _ in imu_motion(
        camera_pose,
        lambda t: 20.0,
        lambda t: 0.0,
        times,
        linear_accel=lambda t: np.array([5.0, 0.0, 0.0]) if t > 0 else np.zeros(3),
    ):
        estimate = f.update(t, accel, gyro)
    assert not f.stationary
    assert tilt_deg(estimate) == pytest.approx(20.0, abs=1e-6)


def test_filter_noise_averaging():
    rng = np.random.default_rng(0)
    f = GravityFilter(time_constant=0.5)
    true = level(math.radians(25.0))
    t = 0.0
    for _ in range(2000):
        t += DT
        sample = true + rng.normal(0.0, 0.05, 3)
        estimate = f.update(t, sample, rng.normal(0.0, 0.005, 3))
    assert tilt_deg(estimate) == pytest.approx(25.0, abs=0.2)


def test_filter_reinitialises_after_gap_and_ignores_old_stamps():
    f = GravityFilter()
    f.update(0.0, level(0.0), ZERO)
    held = f.update(0.0, level(0.5), (1.0, 0.0, 0.0))
    assert held == pytest.approx(tuple(up_vector(0.0)))
    assert f.update(-0.1, level(0.5), (1.0, 0.0, 0.0)) == held
    assert f.update(1.0, level(0.5), ZERO) == pytest.approx(tuple(up_vector(0.5)))
