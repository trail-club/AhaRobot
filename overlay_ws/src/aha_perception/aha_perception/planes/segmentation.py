"""Conversion of per-pixel segment labels into refined, classified planes.

Every detector produces a label image at its working resolution and hands it
to ``planes_from_labels``, so refitting, inlier refinement, the size filters
and classification are identical for all of them.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .classify import classify_planes, is_horizontal
from .geometry import convex_hull_2d, face_viewpoint, fit_plane, plane_axes
from .preprocess import Frame, upsample_labels
from .types import DetectionOptions, InlierMoments, Plane

_HULL_MAX_POINTS = 8000


def make_plane(
    points: np.ndarray, inliers: np.ndarray, viewpoint, up: np.ndarray
) -> Plane:
    normal, offset, centroid, _ = fit_plane(points)
    normal, offset = face_viewpoint(normal, offset, viewpoint)
    axes = plane_axes(normal, up)
    step = max(1, len(points) // _HULL_MAX_POINTS)
    hull, area = convex_hull_2d((points[::step] - centroid) @ axes.T)
    rms = float(np.sqrt(np.mean(np.square(points @ normal + offset))))
    return Plane(
        normal=normal,
        offset=offset,
        inliers=inliers,
        centroid=centroid,
        axes=axes,
        hull=hull,
        area=area,
        rms=rms,
    )


def _group_by_label(labels: np.ndarray, valid: np.ndarray) -> dict[int, np.ndarray]:
    """Flat indices of valid pixels per non-negative label."""
    selected = np.flatnonzero((labels >= 0) & valid)
    if selected.size == 0:
        return {}
    keys = labels[selected]
    if int(keys.max()) <= np.iinfo(np.int16).max:
        # Stable argsort of 16-bit keys is a radix sort: same order, faster.
        keys = keys.astype(np.int16)
    selected = selected[np.argsort(keys, kind="stable")]
    bounds = np.flatnonzero(np.diff(labels[selected])) + 1
    return {int(labels[g[0]]): g for g in np.split(selected, bounds)}


def _threshold(
    depth: np.ndarray, options: DetectionOptions, k: float | None = None
) -> np.ndarray:
    k = options.refine_k if k is None else k
    return np.maximum(k * options.noise.sigma(depth), options.refine_min_distance)


def _robust_fit(points: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, float]:
    normal, offset, _, _ = fit_plane(points)
    keep = np.abs(points @ normal + offset) < threshold
    if keep.sum() >= 3:
        normal, offset, _, _ = fit_plane(points[keep])
    return normal, offset


def planes_from_labels(
    frame: Frame,
    labels: np.ndarray,
    options: DetectionOptions,
    refine_k: float | None = None,
) -> list[Plane]:
    """Build planes from a working-resolution label image (-1 = unlabeled).

    The plane of each segment is fitted at the working resolution (with one
    inlier re-fit), then labels are upsampled to the input resolution and
    pixels farther than ``refine_k * sigma(z)`` from that plane are dropped
    before the final fit. ``refine_k`` defaults to ``options.refine_k``.
    """
    work_groups = _group_by_label(labels.reshape(-1), frame.valid.reshape(-1))
    if not work_groups:
        return []
    # At stride 1 the working arrays are the input arrays: group and gather once.
    same = frame.stride == 1
    if same:
        full_groups = work_groups
    else:
        full_labels = upsample_labels(labels, frame.stride, frame.full_shape)
        full_groups = _group_by_label(
            full_labels.reshape(-1), frame.full_valid.reshape(-1)
        )
    work_points = frame.points.reshape(-1, 3)
    work_depth = frame.depth.reshape(-1)
    points = frame.full_points.reshape(-1, 3)
    depth = frame.full_depth.reshape(-1)
    up = options.unit_up()

    planes: list[Plane] = []
    for label, work_idx in work_groups.items():
        idx = full_groups.get(label)
        if idx is None or len(idx) < options.min_inliers or len(work_idx) < 3:
            continue
        pts = points[idx]
        limit = _threshold(depth[idx], options, refine_k)
        if same:
            normal, offset = _robust_fit(pts, limit)
        else:
            normal, offset = _robust_fit(
                work_points[work_idx],
                _threshold(work_depth[work_idx], options, refine_k),
            )
        keep = np.abs(pts @ normal + offset) < limit
        if keep.sum() < options.min_inliers:
            continue
        kept = pts[keep]
        plane = make_plane(kept, idx[keep], frame.camera_origin, up)
        if plane.area < options.min_area:
            continue
        # Planes through the camera are occlusion-boundary artifacts.
        if (
            plane.normal @ frame.camera_origin + plane.offset
            < options.min_camera_distance
        ):
            continue
        if not options.detect_vertical and not is_horizontal(plane.normal, options):
            continue
        plane.moments = InlierMoments(
            count=float(len(kept)),
            total=kept.sum(axis=0),
            outer=kept.T @ kept,
            depth_sum=float(depth[plane.inliers].sum()),
        )
        planes.append(plane)
    planes.sort(key=lambda p: -len(p.inliers))
    classify_planes(planes, options)
    return planes


def grow_labels(
    frame: Frame, labels: np.ndarray, iterations: int, k: float, options
) -> np.ndarray:
    """Grow segments into adjacent unlabeled pixels that lie on their plane.

    Works at the working resolution; each iteration adds one 8-connected ring
    of pixels closer than ``k * sigma(z)`` to the segment's least-squares plane.
    """
    labels = labels.copy()
    count = int(labels.max()) + 1
    if count <= 0 or iterations <= 0:
        return labels
    points = frame.points.reshape(-1, 3)
    depth = frame.depth.reshape(-1)
    normals = np.zeros((count, 3))
    offsets = np.full(count, np.inf)
    groups = _group_by_label(labels.reshape(-1), frame.valid.reshape(-1))
    for label, idx in groups.items():
        if len(idx) >= 3:
            normals[label], offsets[label] = _robust_fit(
                points[idx], _threshold(depth[idx], options)
            )
    for _ in range(iterations):
        neighbour = maximum_filter(labels, size=3, mode="constant", cval=-1)
        cand = np.flatnonzero(((labels < 0) & frame.valid & (neighbour >= 0)).ravel())
        if cand.size == 0:
            break
        lab = neighbour.reshape(-1)[cand]
        dist = np.abs(np.einsum("ij,ij->i", points[cand], normals[lab]) + offsets[lab])
        limit = np.maximum(
            k * options.noise.sigma(depth[cand]), options.refine_min_distance
        )
        ok = dist < limit
        labels.reshape(-1)[cand[ok]] = lab[ok]
    return labels


def label_image(planes: list[Plane], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel plane index (-1 = none) at the input resolution."""
    out = np.full(shape[0] * shape[1], -1, dtype=np.int32)
    for i, plane in enumerate(planes):
        out[plane.inliers] = i
    return out.reshape(shape)
