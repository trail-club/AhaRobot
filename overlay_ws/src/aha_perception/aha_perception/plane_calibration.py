"""Camera pose correction from a horizontal reference plane (ROS free).

A plane of known height (the floor at 0, a table top measured with a tape)
is detected in the target frame through the modelled camera pose. If the
model is right, the detected plane is horizontal at the known height; if
not, the camera is tilted and offset from where the model puts it:

- ``tilt_correction``: the base-frame rotation C (roll about x, pitch about
  y, no yaw) that turns the detected normal back to +z. The true camera
  orientation is C applied to the modelled one (both in the target frame).
- ``height_correction``: the camera's distance to the plane does not depend
  on its orientation, so the camera is at reference height + distance; the
  difference to the modelled camera height is the z correction.

The plane cannot tell the camera's horizontal position or yaw. The tilt may
also come from the pan / tilt axes upstream of the camera mount; the
corrected mount is then right for the head pose it was measured in, so
calibrate in the pose the head is used in and check the residual in others.
``corrected_mount`` turns both into a new mount transform (parent link ->
camera link) given the modelled transforms at the time of the detection.

``moved_mount``, ``matrix_rpy``, ``transform_matrix`` and ``set_camera`` are
shared with the arm click calibration (camera_arm_calibration.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    """Unit ``normal`` (pointing up) and ``offset`` d of n . p = d."""

    normal: np.ndarray
    offset: float

    @classmethod
    def of(cls, normal, offset) -> PlaneFit:
        n = np.asarray(normal, float)
        length = float(np.linalg.norm(n))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError(f"invalid plane normal {normal}")
        if not math.isfinite(float(offset)):
            raise ValueError(f"invalid plane offset {offset}")
        n, d = n / length, float(offset) / length
        if n[2] < 0.0:
            n, d = -n, -d
        return cls(n, d)


@dataclass(frozen=True)
class Correction:
    roll: float  # rad about the target frame x
    pitch: float  # rad about the target frame y
    dz: float  # m to add to the camera height
    tilt_deg: float  # angle between the detected normal and +z
    distance: float  # m from the camera to the plane


def rotation(roll: float, pitch: float) -> np.ndarray:
    """Rx(roll) @ Ry(pitch)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rx @ ry


def tilt_correction(plane: PlaneFit) -> tuple[float, float]:
    """(roll, pitch) with rotation(roll, pitch) @ normal = +z."""
    nx, ny, nz = plane.normal
    pitch = -math.atan2(nx, nz)
    # After the pitch, the normal is (0, ny, nz') with nz' = hypot(nx, nz).
    roll = math.atan2(ny, math.hypot(nx, nz))
    return roll, pitch


def correction(plane: PlaneFit, camera_position, reference_height: float) -> Correction:
    """Correction of the camera at ``camera_position`` (target frame) seeing
    ``plane`` that is really horizontal at ``reference_height``."""
    c = np.asarray(camera_position, float)
    if not (np.all(np.isfinite(c)) and math.isfinite(reference_height)):
        raise ValueError("camera position and reference height must be finite")
    distance = float(plane.normal @ c - plane.offset)
    if distance <= 0.0:
        raise ValueError("the camera is not above the plane")
    roll, pitch = tilt_correction(plane)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(plane.normal[2])))))
    return Correction(
        roll=roll,
        pitch=pitch,
        dz=reference_height + distance - float(c[2]),
        tilt_deg=tilt,
        distance=distance,
    )


def corrected_mount(
    base_to_parent: np.ndarray, parent_to_camera: np.ndarray, fix: Correction
) -> np.ndarray:
    """New parent -> camera transform (4x4) after ``fix``: the camera turns
    about its own origin by the correction's rotation and rises by dz, all
    in the target frame; the parent link stays where the model puts it."""
    origin = (base_to_parent @ parent_to_camera)[:3, 3]
    motion = np.eye(4)
    motion[:3, :3] = rotation(fix.roll, fix.pitch)
    motion[:3, 3] = origin - motion[:3, :3] @ origin + np.array([0.0, 0.0, fix.dz])
    return moved_mount(base_to_parent, parent_to_camera, motion)


def moved_mount(
    base_to_parent: np.ndarray, parent_to_camera: np.ndarray, motion: np.ndarray
) -> np.ndarray:
    """New parent -> camera transform (4x4) after the camera is moved by
    ``motion`` (4x4, applied in the target frame); the parent link stays."""
    return np.linalg.inv(base_to_parent) @ motion @ base_to_parent @ parent_to_camera


def matrix_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (R = Rz(y) Ry(p) Rx(r)) of a rotation matrix."""
    r = matrix[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    if abs(math.cos(pitch)) < 1e-9:
        return math.atan2(-float(r[1, 2]), float(r[1, 1])), pitch, 0.0
    return (
        math.atan2(float(r[2, 1]), float(r[2, 2])),
        pitch,
        math.atan2(float(r[1, 0]), float(r[0, 0])),
    )


def transform_matrix(transform) -> np.ndarray:
    """4x4 of a geometry_msgs Transform (translation + unit quaternion)."""
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = (t.x, t.y, t.z)
    return m


def set_camera(text: str, xyz, rpy) -> str:
    """Frame config text with ``camera: {xyz, rpy}`` set (replaced or added
    before ``arms:``)."""
    line = (
        "camera: {xyz: ["
        + ", ".join(f"{v:.4f}" for v in xyz)
        + "], rpy: ["
        + ", ".join(f"{v:.4f}" for v in rpy)
        + "]}"
    )
    pattern = re.compile(r"^camera:.*$", re.M)
    if pattern.search(text):
        return pattern.sub(line, text, count=1)
    arms = re.compile(r"^arms:", re.M).search(text)
    if arms is None:
        raise ValueError("no 'arms:' in the frame config")
    return text[: arms.start()] + line + "\n\n" + text[arms.start() :]
