"""Head camera tilt and roll from the D435i IMU (ROS-free).

Frames: the RealSense driver reports the IMU in camera_imu_optical_frame,
which librealsense aligns with the depth optical frame of the D435i
(x right, y down, z forward). The color optical frame has the same axes up to
the factory depth-to-color rotation (well under 1 deg), which is ignored, so
the accelerometer is used as if it were in camera_color_optical_frame.

At rest the accelerometer measures the specific force, i.e. the "up" vector
(+9.81 m/s^2 against gravity). With the camera level the optical axis is
horizontal and up = (0, -1, 0).

Angles (from the up vector u, see ``tilt_roll``):
  tilt  rotation of the optical axis below the horizon (+ = down), the same
        sign as joint_head_tilt: atan2(-u_z, hypot(u_x, u_y)).
  roll  rotation about the optical axis (+ = image right side down); the
        head has no roll joint: atan2(-u_x, -u_y).

``GravityFilter`` estimates u from accel and gyro (complementary filter), so
the tilt stays continuous while the head moves.

Joint mapping (aha_description: camera_link on link_head_tilt, optical axis
horizontal at zero head joints, pan axis vertical): with the base level,
joint_head_tilt = tilt - tilt_at_zero, independent of pan.
"""

import math
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665

# Camera tilt relative to gravity at zero head joints (URDF, base level).
URDF_TILT_AT_ZERO = 0.0


def tilt_roll(accel):
    """(tilt, roll) in rad from an accelerometer vector in the optical frame."""
    ux, uy, uz = (float(v) for v in accel)
    tilt = math.atan2(-uz, math.hypot(ux, uy))
    roll = math.atan2(-ux, -uy)
    return tilt, roll


def up_vector(tilt, roll=0.0):
    """Unit up vector in the optical frame for a camera tilt and roll (rad)."""
    c = math.cos(tilt)
    return np.array([-c * math.sin(roll), -c * math.cos(roll), -math.sin(tilt)])


def joint_tilt(tilt, offset=0.0, tilt_at_zero=URDF_TILT_AT_ZERO):
    """joint_head_tilt (rad) for a camera tilt relative to gravity (rad)."""
    return tilt - tilt_at_zero + offset


@dataclass
class GravityFilter:
    """Complementary filter of the unit up vector in the IMU frame.

    Each sample rotates the estimate by the gyro (du/dt = u x w, the opposite
    of the body rotation), renormalises it and blends it toward accel/|accel|
    with alpha = 1 - exp(-dt / time_constant) when |accel| is within
    ``accel_tolerance`` (m/s^2) of gravity; otherwise only the gyro is used.
    The first sample, or one after a gap over ``max_gap`` (s), initialises the
    estimate from the accelerometer. ``stationary`` reports gyro norm below
    ``gyro_threshold`` (rad/s) and accel norm near gravity.
    """

    time_constant: float = 0.5
    gyro_threshold: float = 0.05
    accel_tolerance: float = 0.8
    max_gap: float = 0.5
    estimate: tuple[float, float, float] | None = None
    last_stamp: float | None = None
    stationary: bool = False

    def update(self, stamp, accel, gyro):
        """Add one sample (stamp in s); return the unit up vector or None."""
        ax, ay, az = accel
        wx, wy, wz = gyro
        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        accel_ok = abs(a_norm - GRAVITY) < self.accel_tolerance
        self.stationary = (
            accel_ok and wx * wx + wy * wy + wz * wz < self.gyro_threshold**2
        )
        dt = None if self.last_stamp is None else stamp - self.last_stamp
        if self.estimate is None or dt is None or dt > self.max_gap:
            self.last_stamp = stamp
            if a_norm > 0.0:
                self.estimate = (ax / a_norm, ay / a_norm, az / a_norm)
            return self.estimate
        if dt <= 0.0:
            return self.estimate
        self.last_stamp = stamp
        ux, uy, uz = self.estimate
        # First-order rotation; the renormalisation keeps it on the unit sphere.
        ux, uy, uz = (
            ux + dt * (uy * wz - uz * wy),
            uy + dt * (uz * wx - ux * wz),
            uz + dt * (ux * wy - uy * wx),
        )
        if accel_ok:
            alpha = 1.0 - math.exp(-dt / self.time_constant)
            ux += alpha * (ax / a_norm - ux)
            uy += alpha * (ay / a_norm - uy)
            uz += alpha * (az / a_norm - uz)
        n = math.sqrt(ux * ux + uy * uy + uz * uz)
        if n > 0.0:
            self.estimate = (ux / n, uy / n, uz / n)
        return self.estimate
