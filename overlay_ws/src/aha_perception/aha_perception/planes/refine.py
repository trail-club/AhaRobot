"""Pixel-level assignment of organized-cloud pixels to known planes."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import find_objects, maximum_filter, minimum_filter
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


def limit_image(frame: Frame, k: float, options: DetectionOptions) -> np.ndarray:
    """max(k sigma(z), min distance) per pixel; NaN where invalid."""
    return np.maximum(k * options.noise.sigma(frame.depth), options.refine_min_distance)


def residual_ratio(
    frame: Frame, normal, offset: float, limit: np.ndarray
) -> np.ndarray:
    """|n.p + d| / ``limit`` per pixel; NaN where invalid."""
    shape = frame.valid.shape
    ratio = frame.points.reshape(-1, 3) @ np.asarray(normal, dtype=np.float64)
    ratio += offset
    np.abs(ratio, out=ratio)
    ratio /= limit.reshape(-1)
    return ratio.reshape(shape)


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


def _seeded(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """8-connected pieces of ``mask`` that contain a seed (flat indices)."""
    components, count = connected_label(mask, structure=EIGHT)
    hit = np.zeros(count + 1, dtype=bool)
    hit[components.reshape(-1)[seeds]] = True
    hit[0] = False
    return hit[components]


def assign_seeded(
    frame: Frame,
    planes,
    k: float,
    options: DetectionOptions,
    k_high: float | None = None,
):
    """Label image of ``planes`` (with ``normal``, ``offset``, ``inliers``).

    A plane may take the pixels with ratio < 1 that are connected in the image
    to its inliers; each pixel goes to the plane with the smallest ratio. Same
    result as ``assign_pixels`` with those masks as candidates, but every
    plane's ratio image is computed once for both steps.

    With ``k_high > k`` (hysteresis) a plane may instead take the pixels within
    ``k_high * sigma(z)`` that are connected through such pixels to an inlier
    with ratio < 1, i.e. to a seeded ratio < 1 piece; ``k_high <= k`` gives
    the result above.
    """
    shape = frame.valid.shape
    limit = limit_image(frame, k, options)
    high = max(1.0, (k_high or 0.0) / k)
    out = np.full(shape, -1, dtype=np.int32)
    best = np.full(shape, high)
    for index, plane in enumerate(planes):
        ratio = residual_ratio(frame, plane.normal, plane.offset, limit)
        seeds = plane.inliers
        with np.errstate(invalid="ignore"):
            if high > 1.0:
                # A weak piece needs a strong seed; at high == 1 seeds with
                # ratio >= 1 fall outside the mask anyway.
                seeds = seeds[ratio.reshape(-1)[seeds] < 1.0]
            take = _seeded(ratio < high, seeds)
            take &= ratio < best
        best[take] = ratio[take]
        out[take] = index
    return out


def keep_seeded_components(
    labels: np.ndarray, seeds=None, min_size: int = 0
) -> np.ndarray:
    """Drop 8-connected pieces of each label that are small or unseeded.

    ``seeds`` is an optional list of flat pixel indices per label; a piece is
    kept only if it contains one of them. Each label is processed inside its
    bounding box.
    """
    out = labels.copy()
    w = labels.shape[1]
    for index, box in enumerate(find_objects(labels + 1)):
        if box is None:
            continue
        mask = labels[box] == index
        components, count = connected_label(mask, structure=EIGHT)
        sizes = np.bincount(components.reshape(-1), minlength=count + 1)
        keep = sizes >= min_size
        keep[0] = False
        if seeds is not None and seeds[index] is not None:
            rows, cols = np.divmod(np.asarray(seeds[index]), w)
            r0, r1, c0, c1 = box[0].start, box[0].stop, box[1].start, box[1].stop
            inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            seeded = np.zeros(count + 1, dtype=bool)
            seeded[components[rows[inside] - r0, cols[inside] - c0]] = True
            keep &= seeded
        out[box][~keep[components] & mask] = -1
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
    # n closings with the 3x3 square equal one with the (2n+1) square, done as
    # separable max / min filters (zero outside, like border_value=0).
    size = 2 * iterations + 1
    boxes = find_objects(labels + 1, max_label=len(planes))
    for index, (normal, offset) in enumerate(planes):
        box = boxes[index]
        if box is None:
            continue
        r0, r1 = max(box[0].start - iterations, 0), min(box[0].stop + iterations, h)
        c0, c1 = max(box[1].start - iterations, 0), min(box[1].stop + iterations, w)
        crop = labels[r0:r1, c0:c1] == index
        grown = maximum_filter(crop.view(np.uint8), size=size, mode="constant")
        closed = minimum_filter(grown, size=size, mode="constant").view(bool)
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
