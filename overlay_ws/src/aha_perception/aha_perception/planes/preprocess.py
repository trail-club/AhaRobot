"""Preprocessing of organized point clouds."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .types import DetectionOptions


def crop_depth(cloud: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Return a float64 copy with points outside the depth range set to NaN."""
    out = np.array(cloud, dtype=np.float64, copy=True)
    z = out[..., 2]
    with np.errstate(invalid="ignore"):
        invalid = ~np.isfinite(out).all(axis=-1) | (z < min_depth) | (z > max_depth)
    out[invalid] = np.nan
    return out


def decimate(cloud: np.ndarray, stride: int) -> np.ndarray:
    """Keep every ``stride``-th row and column of an organized cloud."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return cloud[::stride, ::stride]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (..., 3) points."""
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample(
    points: np.ndarray, voxel_size: float, values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Average (N, 3) points per voxel.

    Returns the voxel centroids (M, 3), the voxel index of every input point
    (N,), and the per-voxel mean of ``values`` (N,) when given.
    """
    if len(points) == 0:
        empty_values = None if values is None else np.empty(0)
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), empty_values
    keys = np.floor(points / voxel_size).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    linear = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inverse, counts = np.unique(linear, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    centroids = np.empty((counts.size, 3))
    for axis in range(3):
        centroids[:, axis] = np.bincount(inverse, points[:, axis]) / counts
    mean_values = None
    if values is not None:
        mean_values = np.bincount(inverse, values) / counts
    return centroids, inverse, mean_values


@dataclass
class Frame:
    """An organized cloud prepared for detection.

    ``points``/``depth``/``valid`` are at the working resolution (decimated by
    ``stride``); the ``full_*`` arrays keep the input resolution. Points are in
    the output frame, depth is the camera optical z.
    """

    points: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    stride: int
    full_points: np.ndarray
    full_depth: np.ndarray
    full_valid: np.ndarray
    camera_origin: np.ndarray

    @property
    def full_shape(self) -> tuple[int, int]:
        return self.full_depth.shape


def prepare_frame(cloud: np.ndarray, options: DetectionOptions, stride: int) -> Frame:
    """Depth-cropped cloud in the output frame; NaN where invalid.

    Same values as transforming ``crop_depth(cloud, ...)``: invalid pixels are
    set to NaN after the transform, and validity is tested per coordinate
    (cheaper than ``isfinite(...).all(axis=-1)``).
    """
    if cloud.ndim != 3 or cloud.shape[2] != 3:
        raise ValueError("cloud must be an organized (H, W, 3) array")
    cloud = np.ascontiguousarray(cloud, dtype=np.float64)
    x, y, z = cloud[..., 0], cloud[..., 1], cloud[..., 2]
    with np.errstate(invalid="ignore"):
        full_valid = (
            np.isfinite(z)
            & np.isfinite(x)
            & np.isfinite(y)
            & (z >= options.min_depth)
            & (z <= options.max_depth)
        )
    invalid = ~full_valid
    full_depth = z.copy()
    full_depth[invalid] = np.nan
    full_points = transform_points(cloud, options.camera_to_base)
    full_points[invalid] = np.nan
    camera_origin = np.asarray(options.camera_to_base, dtype=np.float64)[:3, 3].copy()
    return Frame(
        points=decimate(full_points, stride),
        depth=decimate(full_depth, stride),
        valid=decimate(full_valid, stride),
        stride=stride,
        full_points=full_points,
        full_depth=full_depth,
        full_valid=full_valid,
        camera_origin=camera_origin,
    )


def with_stride(frame: Frame, stride: int) -> Frame:
    """The same prepared frame at another working stride (no recomputation)."""
    return dataclasses.replace(
        frame,
        points=decimate(frame.full_points, stride),
        depth=decimate(frame.full_depth, stride),
        valid=decimate(frame.full_valid, stride),
        stride=stride,
    )


def upsample_labels(labels: np.ndarray, stride: int, shape: tuple[int, int]):
    """Nearest-neighbour upsampling of a decimated label image."""
    if stride == 1:
        return labels[: shape[0], : shape[1]]
    up = np.repeat(np.repeat(labels, stride, axis=0), stride, axis=1)
    return up[: shape[0], : shape[1]]
