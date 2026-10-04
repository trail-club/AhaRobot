"""Data types shared by the plane detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class PlaneLabel(str, Enum):
    FLOOR = "floor"
    TABLE = "table"
    SHELF = "shelf"
    WALL = "wall"
    OTHER = "other"


@dataclass(frozen=True)
class DepthNoiseModel:
    """Stereo depth RMS error: sigma(z) = z^2 * subpixel / (focal * baseline).

    Inverse depth noise is constant (``inverse_sigma``) for a stereo camera.
    """

    baseline: float = 0.05
    subpixel: float = 0.08
    focal: float = 465.7

    @property
    def inverse_sigma(self) -> float:
        return self.subpixel / (self.focal * self.baseline)

    def sigma(self, depth):
        return np.square(depth) * self.inverse_sigma


@dataclass(frozen=True)
class ClassifyOptions:
    floor_max_height: float = 0.08
    table_height_range: tuple[float, float] = (0.4, 1.0)
    table_min_area: float = 0.05
    shelf_min_area: float = 0.02
    shelf_min_spacing: float = 0.15
    shelf_min_overlap: float = 0.3
    shelf_min_area_ratio: float = 0.15
    wall_min_area: float = 0.3
    wall_min_extent: float = 0.3


def _identity4() -> np.ndarray:
    return np.eye(4)


def _up() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


@dataclass
class DetectionOptions:
    """Options shared by every detector.

    The input cloud is in the camera optical frame. ``camera_to_base`` maps it
    to the output frame, whose origin is on the floor and whose ``up_axis``
    points against gravity. Planes are reported in the output frame.
    """

    camera_to_base: np.ndarray = field(default_factory=_identity4)
    up_axis: np.ndarray = field(default_factory=_up)
    min_depth: float = 0.3
    max_depth: float = 4.0
    noise: DepthNoiseModel = field(default_factory=DepthNoiseModel)
    detect_vertical: bool = True
    horizontal_tol_deg: float = 10.0
    vertical_tol_deg: float = 10.0
    refine_k: float = 3.0
    refine_min_distance: float = 0.005
    min_inliers: int = 300
    min_area: float = 0.02
    min_camera_distance: float = 0.1
    classify: ClassifyOptions = field(default_factory=ClassifyOptions)

    def unit_up(self) -> np.ndarray:
        up = np.asarray(self.up_axis, dtype=np.float64)
        return up / np.linalg.norm(up)


@dataclass
class Plane:
    """A detected plane: ``normal . p + offset = 0`` in the output frame.

    ``inliers`` are flat indices (row * width + col) into the input organized
    cloud. ``hull`` holds the 2D convex hull vertices (counter-clockwise) in the
    in-plane basis ``axes`` relative to ``centroid``.
    """

    normal: np.ndarray
    offset: float
    inliers: np.ndarray
    centroid: np.ndarray
    axes: np.ndarray
    hull: np.ndarray
    area: float
    rms: float
    label: PlaneLabel = PlaneLabel.OTHER

    def hull_points(self) -> np.ndarray:
        """Hull vertices as (K, 3) points in the output frame."""
        return self.centroid + self.hull @ self.axes
