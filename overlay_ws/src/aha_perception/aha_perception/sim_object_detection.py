"""Simulated object detection from ground-truth poses (ROS free, numpy).

Used by scripts/sim_object_detector.py. An object is detected when its center
or the center of its top face projects into the image and the depth image
there is not closer than that point minus an allowance (``object_visible``).
Misses and mislabels are drawn once per (seed, model name), so the same
objects stay missed or mislabelled while the seed is unchanged
(``corruption``).
"""

from __future__ import annotations

import math
import random

import numpy as np

MIN_Z = 0.05  # m in front of the camera
TOP_INSET = 0.005  # m below the top face for the top point
TOP_FRONT = 0.01  # m allowance of the top point (it is on the visible face)


def project(K, p_cam) -> tuple[float, float] | None:
    """Pixel (u, v) of a point in the optical frame (z forward), None behind."""
    x, y, z = (float(v) for v in p_cam)
    if z <= MIN_Z:
        return None
    return K[0][0] * x / z + K[0][2], K[1][1] * y / z + K[1][2]


def depth_at(depth: np.ndarray, u: float, v: float, radius: int = 2) -> float:
    """Median of the valid depths (m) in a (2 radius + 1)^2 patch, NaN if none."""
    h, w = depth.shape
    col, row = int(round(u)), int(round(v))
    if not (0 <= col < w and 0 <= row < h):
        return math.nan
    patch = depth[
        max(row - radius, 0) : row + radius + 1, max(col - radius, 0) : col + radius + 1
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if valid.size else math.nan


def point_visible(
    depth: np.ndarray,
    K,
    p_cam,
    front: float,
    tolerance: float,
    margin_px: float = 0.0,
) -> bool:
    """Whether a point projects into the image and nothing is in front of it.

    front: how far the visible surface may be in front of the point (m)."""
    uv = project(K, p_cam)
    if uv is None:
        return False
    h, w = depth.shape
    u, v = uv
    if not (margin_px <= u < w - margin_px and margin_px <= v < h - margin_px):
        return False
    d = depth_at(depth, u, v)
    return math.isfinite(d) and d >= float(p_cam[2]) - front - tolerance


def object_visible(
    depth: np.ndarray,
    K,
    T_cam_object: np.ndarray,
    size,
    tolerance: float = 0.03,
    margin_px: float = 0.0,
) -> bool:
    """Whether the center or the top face center of an upright object is seen.

    T_cam_object: object center pose in the optical frame (4x4); size: x, y
    footprint and z height (m)."""
    half_height = size[2] / 2
    center = T_cam_object[:3, 3]
    top = T_cam_object[:3, :3] @ np.array([0.0, 0.0, half_height - TOP_INSET]) + center
    return point_visible(
        depth, K, center, max(size) / 2, tolerance, margin_px
    ) or point_visible(depth, K, top, TOP_FRONT, tolerance, margin_px)


def corruption(
    seed: int,
    model: str,
    label: str,
    labels,
    miss_rate: float,
    mislabel_rate: float,
) -> tuple[bool, str]:
    """(missed, reported label) of a model, fixed for a seed."""
    rng = random.Random(f"{seed}:{model}")
    missed = rng.random() < miss_rate
    others = [x for x in labels if x != label]
    if rng.random() < mislabel_rate and others:
        label = rng.choice(others)
    return missed, label
