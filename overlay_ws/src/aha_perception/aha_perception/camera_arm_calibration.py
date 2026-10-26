"""Camera horizontal position and yaw from gripper points (ROS free).

A horizontal plane gives the camera's tilt and height (plane_calibration.py)
but not its horizontal position or yaw. Those come from points whose true
position the arm model knows: a point on the gripper as seen in the point
cloud (``seen``, in the target frame through the modelled camera) and the
same point from the arm's TF (``model``). If the camera really sits at C
applied to its modelled pose (C a motion in the target frame), the model
reports every point it sees at C^-1 of the true point, so C maps seen to
model. With the tilt and height already right, C is a rotation by ``dyaw``
about the vertical axis through ``pivot`` followed by a shift (dx, dy):

    model_xy = Rz(dyaw) (seen_xy - pivot) + pivot + (dx, dy)

``horizontal_correction`` fits it in least squares (2D Procrustes, closed
form). C does not depend on the pivot, only its parameters do: with the
camera position as the pivot, (dx, dy) is how far the camera moves. Heights
are not fitted; their differences (``dz``) are reported to check the clicks
and the tilt / height calibration. ``corrected_mount`` applies C to the
camera mount (parent link -> camera link): a rotation about the vertical
keeps the camera's height and its tilt against the horizontal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aha_perception.plane_calibration import moved_mount

MIN_SPREAD = 0.05  # m, RMS distance of the seen points from their centroid


@dataclass(frozen=True)
class HorizontalCorrection:
    dx: float  # m, shift of the pivot along the target frame x
    dy: float  # m, along y
    dyaw: float  # rad about the vertical axis through the pivot
    pivot: np.ndarray  # (x, y) in the target frame
    residuals: np.ndarray  # (N, 2) m, corrected seen minus model
    dz: np.ndarray  # (N,) m, model height minus seen height

    @property
    def rms(self) -> float:
        """RMS horizontal residual, m."""
        return float(np.sqrt(np.mean(np.sum(self.residuals**2, axis=1))))

    def motion(self) -> np.ndarray:
        """C as a 4x4 transform in the target frame."""
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        rotation = np.array([[c, -s], [s, c]])
        m = np.eye(4)
        m[:2, :2] = rotation
        m[:2, 3] = self.pivot + np.array([self.dx, self.dy]) - rotation @ self.pivot
        return m


def _points(values, name: str) -> np.ndarray:
    points = np.asarray(values, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be N x 3 points, got shape {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} has non-finite values")
    return points


def horizontal_correction(
    seen, model, pivot, min_spread: float = MIN_SPREAD
) -> HorizontalCorrection:
    """Least-squares C (see the module doc) mapping ``seen`` to ``model``
    (N x 3 each, target frame) about ``pivot`` (x, y[, z])."""
    s, m = _points(seen, "seen"), _points(model, "model")
    if s.shape != m.shape:
        raise ValueError(f"{len(s)} seen and {len(m)} model points")
    if len(s) < 2:
        raise ValueError("at least 2 point pairs are needed")
    p = np.asarray(pivot, float)[:2]
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError(f"invalid pivot {pivot}")
    a, b = s[:, :2] - p, m[:, :2] - p
    a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - a_mean, b - b_mean
    spread = float(np.sqrt(np.mean(np.sum(ac**2, axis=1))))
    if spread < min_spread:
        raise ValueError(
            f"the points spread {spread:.3f} m around their centroid, less than "
            f"{min_spread} m: the yaw is not observable"
        )
    dyaw = math.atan2(
        float(np.sum(ac[:, 0] * bc[:, 1] - ac[:, 1] * bc[:, 0])),
        float(np.sum(ac[:, 0] * bc[:, 0] + ac[:, 1] * bc[:, 1])),
    )
    c, sn = math.cos(dyaw), math.sin(dyaw)
    rotation = np.array([[c, -sn], [sn, c]])
    shift = b_mean - rotation @ a_mean
    return HorizontalCorrection(
        dx=float(shift[0]),
        dy=float(shift[1]),
        dyaw=dyaw,
        pivot=p,
        residuals=a @ rotation.T + shift - b,
        dz=m[:, 2] - s[:, 2],
    )


def corrected_mount(
    base_to_parent: np.ndarray,
    parent_to_camera: np.ndarray,
    fix: HorizontalCorrection,
) -> np.ndarray:
    """New parent -> camera transform (4x4) after ``fix``; the parent link
    stays where the model puts it."""
    return moved_mount(base_to_parent, parent_to_camera, fix.motion())


def facing_point(point, camera, radius: float) -> np.ndarray:
    """``point`` moved ``radius`` horizontally toward ``camera``: where the
    camera sees the surface of a vertical cylinder of that radius around it."""
    p = np.asarray(point, float).copy()
    if radius == 0.0:
        return p
    direction = np.asarray(camera, float)[:2] - p[:2]
    length = float(np.linalg.norm(direction))
    if length <= radius:
        raise ValueError("the point is within radius of the camera")
    p[:2] += direction * (radius / length)
    return p


def nearest(seen, candidates: dict, max_distance: float) -> str:
    """Name of the only candidate point within ``max_distance`` of ``seen``."""
    s = np.asarray(seen, float)
    close = sorted(
        (float(np.linalg.norm(np.asarray(p, float) - s)), name)
        for name, p in candidates.items()
    )
    close = [(d, name) for d, name in close if d <= max_distance]
    if not close:
        raise ValueError(
            f"no model point within {max_distance * 1000:.0f} mm of the click"
        )
    if len(close) > 1:
        raise ValueError(
            f"{close[0][1]} and {close[1][1]} are both within "
            f"{max_distance * 1000:.0f} mm of the click"
        )
    return close[0][1]
