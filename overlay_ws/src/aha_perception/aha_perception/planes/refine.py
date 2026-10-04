"""Pixel-level assignment of organized-cloud pixels to known planes."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import label as connected_label

from .preprocess import Frame
from .types import DetectionOptions

EIGHT = np.ones((3, 3), dtype=bool)


def full_resolution(frame: Frame) -> Frame:
    """The same frame with the input resolution as working resolution."""
    return dataclasses.replace(
        frame,
        points=frame.full_points,
        depth=frame.full_depth,
        valid=frame.full_valid,
        stride=1,
    )


def residual_ratio(
    frame: Frame, normal, offset: float, k: float, options: DetectionOptions
) -> np.ndarray:
    """|n.p + d| / max(k sigma(z), min distance) per pixel; inf where invalid."""
    points = frame.points.reshape(-1, 3)
    valid = frame.valid.reshape(-1)
    ratio = np.full(valid.shape, np.inf)
    idx = np.flatnonzero(valid)
    limit = np.maximum(
        k * options.noise.sigma(frame.depth.reshape(-1)[idx]),
        options.refine_min_distance,
    )
    ratio[idx] = np.abs(points[idx] @ normal + offset) / limit
    return ratio.reshape(frame.valid.shape)


def assign_pixels(
    frame: Frame,
    planes,
    candidates,
    k: float,
    options: DetectionOptions,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Give each candidate pixel to the plane with the smallest residual ratio.

    ``planes`` is a list of (normal, offset), ``candidates`` a matching list of
    boolean masks. Only pixels with ratio < 1 are assigned; pixels already
    labeled in ``labels`` are kept.
    """
    shape = frame.valid.shape
    out = np.full(shape, -1, dtype=np.int32) if labels is None else labels.copy()
    free = (out < 0) & frame.valid
    best = np.full(shape, 1.0)
    for index, ((normal, offset), mask) in enumerate(zip(planes, candidates)):
        region = mask & free
        if not region.any():
            continue
        rows, cols = np.nonzero(region)
        sub = frame.points[rows, cols]
        limit = np.maximum(
            k * options.noise.sigma(frame.depth[rows, cols]),
            options.refine_min_distance,
        )
        ratio = np.abs(sub @ normal + offset) / limit
        take = ratio < best[rows, cols]
        best[rows[take], cols[take]] = ratio[take]
        out[rows[take], cols[take]] = index
    return out


def keep_seeded_components(
    labels: np.ndarray, seeds=None, min_size: int = 0
) -> np.ndarray:
    """Drop 8-connected pieces of each label that are small or unseeded.

    ``seeds`` is an optional list of flat pixel indices per label; a piece is
    kept only if it contains one of them.
    """
    out = labels.copy()
    flat = out.reshape(-1)
    for index in range(int(labels.max()) + 1):
        mask = labels == index
        if not mask.any():
            continue
        components, count = connected_label(mask, structure=EIGHT)
        sizes = np.bincount(components.reshape(-1), minlength=count + 1)
        keep = sizes >= min_size
        keep[0] = False
        if seeds is not None and seeds[index] is not None:
            seeded = np.zeros(count + 1, dtype=bool)
            seeded[components.reshape(-1)[seeds[index]]] = True
            keep &= seeded
        flat[(~keep[components.reshape(-1)]) & mask.reshape(-1)] = -1
    return out


def fill_holes(
    frame: Frame,
    labels: np.ndarray,
    planes,
    iterations: int,
    k: float,
    options: DetectionOptions,
) -> np.ndarray:
    """Morphologically close each label, adding only free pixels on its plane."""
    if iterations <= 0:
        return labels
    out = labels.copy()
    h, w = labels.shape
    for index, (normal, offset) in enumerate(planes):
        rows, cols = np.nonzero(labels == index)
        if rows.size == 0:
            continue
        r0, r1 = max(rows.min() - iterations, 0), min(rows.max() + iterations + 1, h)
        c0, c1 = max(cols.min() - iterations, 0), min(cols.max() + iterations + 1, w)
        crop = labels[r0:r1, c0:c1] == index
        closed = binary_closing(crop, EIGHT, iterations=iterations, border_value=0)
        sub = closed & ~crop & (out[r0:r1, c0:c1] < 0) & frame.valid[r0:r1, c0:c1]
        if not sub.any():
            continue
        rr, cc = np.nonzero(sub)
        rr, cc = rr + r0, cc + c0
        limit = np.maximum(
            k * options.noise.sigma(frame.depth[rr, cc]), options.refine_min_distance
        )
        ok = np.abs(frame.points[rr, cc] @ normal + offset) < limit
        out[rr[ok], cc[ok]] = index
    return out


def seeded_support(
    frame: Frame, normal, offset: float, seeds: np.ndarray, k: float, options
) -> np.ndarray:
    """Pixels on the plane (ratio < 1) connected in the image to ``seeds``."""
    candidate = residual_ratio(frame, normal, offset, k, options) < 1.0
    components, count = connected_label(candidate, structure=EIGHT)
    hit = np.zeros(count + 1, dtype=bool)
    hit[components.reshape(-1)[seeds]] = True
    hit[0] = False
    return hit[components]
