"""Candidate D: normal clustering on the sphere (after Holz et al. 2011).

Organized normals are clustered by direction with a greedy mode search on the
unit sphere. Each direction cluster is split by plane offset using a histogram
of signed distances, and each offset group by 8-connected components in the
image (bridging gaps of ``component_gap`` pixels left by masked normals).
Segment borders lost to normal masking are regrown along the plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as connected_label

from .normals import organized_normals
from .preprocess import prepare_frame
from .segmentation import grow_labels, planes_from_labels
from .types import DetectionOptions, Plane

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class NormalClusteringParams:
    stride: int = 2
    normal_radius: int = 3
    smoothing_radius: int = 3
    discontinuity_k: float = 6.0
    cluster_angle_deg: float = 10.0
    max_normal_clusters: int = 8
    min_cluster_points: int = 300
    mode_samples: int = 1500
    offset_bin: float = 0.01
    offset_k: float = 3.0
    min_segment_points: int = 80
    component_gap: int = 2
    grow_iterations: int = 10
    seed: int = 0


def _normal_modes(normals, cos_tol, params, rng):
    """Yield (center, member indices) of direction clusters, largest first."""
    remaining = np.ones(len(normals), dtype=bool)
    for _ in range(params.max_normal_clusters):
        rest = np.flatnonzero(remaining)
        if len(rest) < params.min_cluster_points:
            return
        sample = rest
        if len(rest) > params.mode_samples:
            sample = rng.choice(rest, size=params.mode_samples, replace=False)
        votes = normals[sample] @ normals[sample].T > cos_tol
        center = normals[sample[int(np.argmax(votes.sum(axis=1)))]]
        for _ in range(3):
            near = rest[normals[rest] @ center > cos_tol]
            if len(near) == 0:
                break
            center = normals[near].mean(axis=0)
            center /= np.linalg.norm(center)
        members = rest[normals[rest] @ center > cos_tol]
        if len(members) < params.min_cluster_points:
            return
        remaining[members] = False
        yield center, members


def _offset_groups(offsets, thresholds, params):
    """Yield index arrays of points sharing a plane offset, largest peak first."""
    free = np.ones(len(offsets), dtype=bool)
    while free.sum() >= params.min_segment_points:
        idx = np.flatnonzero(free)
        values = offsets[idx]
        low = values.min()
        bins = ((values - low) / params.offset_bin).astype(np.int64)
        hist = np.convolve(np.bincount(bins), [1.0, 2.0, 1.0], mode="same")
        peak = low + (int(np.argmax(hist)) + 0.5) * params.offset_bin
        for _ in range(2):
            near = np.abs(values - peak) < thresholds[idx]
            if near.any():
                peak = float(np.median(values[near]))
        near = np.abs(values - peak) < thresholds[idx]
        if near.sum() < params.min_segment_points:
            return
        free[idx[near]] = False
        yield idx[near]


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: NormalClusteringParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or NormalClusteringParams()
    frame = prepare_frame(cloud, options, params.stride)
    shape = frame.valid.shape
    labels = np.full(shape, -1, dtype=np.int32)
    normals, mask = organized_normals(
        frame.points,
        frame.depth,
        options.noise.inverse_sigma,
        radius=params.normal_radius,
        smoothing_radius=params.smoothing_radius,
        discontinuity_k=params.discontinuity_k,
        viewpoint=frame.camera_origin,
    )
    pixels = np.flatnonzero(mask)
    if pixels.size < params.min_cluster_points:
        return []
    unit_normals = normals.reshape(-1, 3)[pixels]
    points = frame.points.reshape(-1, 3)[pixels]
    sigma = options.noise.sigma(frame.depth.reshape(-1)[pixels])
    thresholds = np.maximum(params.offset_k * sigma, params.offset_bin)

    up = options.unit_up()
    cos_horizontal = np.cos(np.radians(options.horizontal_tol_deg))
    cos_tol = np.cos(np.radians(params.cluster_angle_deg))
    rng = np.random.default_rng(params.seed)
    next_label = 0
    for center, members in _normal_modes(unit_normals, cos_tol, params, rng):
        if not options.detect_vertical and abs(float(center @ up)) < cos_horizontal:
            continue
        offsets = -(points[members] @ center)
        for group in _offset_groups(offsets, thresholds[members], params):
            image = np.zeros(shape, dtype=bool)
            image.reshape(-1)[pixels[members[group]]] = True
            bridged = image
            if params.component_gap > 0:
                bridged = binary_dilation(
                    image, _EIGHT, iterations=params.component_gap
                )
            components, count = connected_label(bridged, structure=_EIGHT)
            components[~image] = 0
            sizes = np.bincount(components.reshape(-1), minlength=count + 1)
            sizes[0] = 0
            big = np.flatnonzero(sizes >= params.min_segment_points)
            lut = np.full(count + 1, -1, dtype=np.int32)
            lut[big] = np.arange(next_label, next_label + len(big), dtype=np.int32)
            next_label += len(big)
            mapped = lut[components]
            labels = np.where(mapped >= 0, mapped, labels)

    labels = grow_labels(
        frame, labels, params.grow_iterations, options.refine_k, options
    )
    return planes_from_labels(frame, labels, options)
