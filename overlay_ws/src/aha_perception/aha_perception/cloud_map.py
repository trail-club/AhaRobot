"""Voxel map of the accumulated head camera cloud, for cloud_accumulator.py.

Kept free of rclpy so it can be tested on the host.

``VoxelMap`` keeps one entry per occupied voxel of ``voxel_size`` in the map
frame: the mean position of its points, the color of the last observation,
the number of frames that observed it (``hits``) and the stamp of the last
one. Like a PCL VoxelGrid filter the points are output at the mean, not the
voxel center, so the grid does not quantize them. A frame contributes the
mean of its points in the voxel; the map averages the frames with weight
max(1 / hits, ``MIN_WEIGHT``), a running mean that turns into an
exponential moving average, so that a slowly drifting surface follows.
Voxels are addressed by an int64 key packing the three voxel indices
(``KEY_BITS`` bits each, so +-2^20 voxels, 10 km at 1 cm, per axis); the keys
are kept sorted and merged with ``np.unique``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import minimum_filter

KEY_BITS = 21
KEY_OFFSET = 1 << (KEY_BITS - 1)
KEY_MASK = (1 << KEY_BITS) - 1
MIN_WEIGHT = 0.1


def voxel_indices(points: np.ndarray, size: float) -> np.ndarray:
    """(N, 3) int64 voxel indices (floor) of (N, 3) points."""
    return np.floor(np.asarray(points, dtype=np.float64) / size).astype(np.int64)


def pack_keys(indices: np.ndarray) -> np.ndarray:
    """(N,) int64 keys of (N, 3) voxel indices within +-KEY_OFFSET."""
    shifted = np.asarray(indices, dtype=np.int64) + KEY_OFFSET
    if shifted.size and (shifted.min() < 0 or shifted.max() > KEY_MASK):
        raise ValueError("voxel index out of the key range")
    return (
        (shifted[:, 0] << (2 * KEY_BITS)) | (shifted[:, 1] << KEY_BITS) | shifted[:, 2]
    )


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    """(N, 3) int64 voxel indices of (N,) keys."""
    keys = np.asarray(keys, dtype=np.int64)
    indices = np.stack(
        [
            (keys >> (2 * KEY_BITS)) & KEY_MASK,
            (keys >> KEY_BITS) & KEY_MASK,
            keys & KEY_MASK,
        ],
        axis=1,
    )
    return indices - KEY_OFFSET


def key_centers(keys: np.ndarray, size: float) -> np.ndarray:
    """(N, 3) float64 voxel centers of (N,) keys."""
    return (unpack_keys(keys) + 0.5) * size


def voxelize(
    points: np.ndarray, rgb: np.ndarray, size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sorted unique keys, mean point, mean rgb) of each voxel of a frame.

    Points outside the key range are dropped.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    indices = voxel_indices(points, size)
    in_range = np.all(np.abs(indices) < KEY_OFFSET, axis=1)
    points, rgb = points[in_range], rgb[in_range]
    keys, inverse, counts = np.unique(
        pack_keys(indices[in_range]), return_inverse=True, return_counts=True
    )
    means = np.empty((len(keys), 3))
    colors = np.empty((len(keys), 3))
    for axis in range(3):
        means[:, axis] = np.bincount(inverse, points[:, axis], len(keys)) / counts
        colors[:, axis] = np.bincount(inverse, rgb[:, axis], len(keys)) / counts
    return keys, means, np.rint(colors).astype(np.uint8)


def in_box(points: np.ndarray, lower, upper) -> np.ndarray:
    """(N,) bool: points within the axis-aligned box [lower, upper]."""
    points = np.atleast_2d(points)
    return np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)), axis=1)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def pose_change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(translation in m, rotation in rad) between two 4x4 poses."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    translation = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    relative = a[:3, :3].T @ b[:3, :3]
    cos = (np.trace(relative) - 1.0) / 2.0
    return translation, math.acos(max(-1.0, min(1.0, cos)))


def min_filter_depth(depth: np.ndarray, size: int = 3) -> np.ndarray:
    """Smallest valid depth in each size x size neighborhood (NaN if none).

    The view test compares a voxel with this, so that a voxel at an object
    edge or on a surface seen at a grazing angle is not cleared by the
    background of a neighboring pixel.
    """
    depth = np.asarray(depth, dtype=np.float32)
    filled = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
    filtered = minimum_filter(filled, size=size, mode="nearest")
    return np.where(np.isfinite(filtered), filtered, np.float32(np.nan))


@dataclass
class VoxelMap:
    """Occupied voxels with mean point, color, hits and last-seen stamp."""

    voxel_size: float
    keys: np.ndarray = field(init=False, repr=False)
    points: np.ndarray = field(init=False, repr=False)
    rgb: np.ndarray = field(init=False, repr=False)
    hits: np.ndarray = field(init=False, repr=False)
    last_seen: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not self.voxel_size > 0.0:
            raise ValueError(f"voxel_size must be > 0, got {self.voxel_size}")
        self.clear()

    def __len__(self) -> int:
        return len(self.keys)

    def clear(self):
        self.keys = np.empty(0, dtype=np.int64)
        self.points = np.empty((0, 3), dtype=np.float64)
        self.rgb = np.empty((0, 3), dtype=np.uint8)
        self.hits = np.empty(0, dtype=np.int32)
        self.last_seen = np.empty(0, dtype=np.float64)

    def insert(
        self, keys: np.ndarray, points: np.ndarray, rgb: np.ndarray, stamp: float
    ):
        """Add one frame's unique voxels (keys, mean points, rgb from voxelize).

        hits + 1, the point averaged in (see module doc), color and stamp
        replaced.
        """
        keys = np.asarray(keys, dtype=np.int64)
        if len(keys) == 0:
            return
        if len(np.unique(keys)) != len(keys):
            raise ValueError("keys of one frame must be unique")
        old = len(self.keys)
        merged, inverse = np.unique(
            np.concatenate([self.keys, keys]), return_inverse=True
        )
        hits = np.zeros(len(merged), dtype=np.int32)
        np.add.at(hits, inverse[:old], self.hits)
        hits[inverse[old:]] += 1
        # Zeros: a new voxel has hits 1, weight 1, so it gets the frame's point.
        means = np.zeros((len(merged), 3), dtype=np.float64)
        means[inverse[:old]] = self.points
        new = inverse[old:]
        weight = np.maximum(1.0 / hits[new], MIN_WEIGHT)[:, None]
        means[new] += weight * (np.asarray(points, dtype=np.float64) - means[new])
        colors = np.empty((len(merged), 3), dtype=np.uint8)
        colors[inverse[:old]] = self.rgb
        colors[inverse[old:]] = np.asarray(rgb, dtype=np.uint8)
        seen = np.empty(len(merged), dtype=np.float64)
        seen[inverse[:old]] = self.last_seen
        seen[inverse[old:]] = stamp
        self.keys, self.points, self.rgb = merged, means, colors
        self.hits, self.last_seen = hits, seen

    def keep(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        self.keys = self.keys[mask]
        self.points = self.points[mask]
        self.rgb = self.rgb[mask]
        self.hits = self.hits[mask]
        self.last_seen = self.last_seen[mask]

    def forget_older(self, stamp: float, max_age: float) -> int:
        """Drop voxels last seen more than max_age before stamp; returns the count."""
        if max_age <= 0.0 or len(self.keys) == 0:
            return 0
        keep = self.last_seen >= stamp - max_age
        removed = int(len(keep) - np.count_nonzero(keep))
        if removed:
            self.keep(keep)
        return removed

    def output(self, min_hits: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(mean points, rgb) of the voxels seen in at least min_hits frames."""
        mask = self.hits >= min_hits
        return self.points[mask], self.rgb[mask]
