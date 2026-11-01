"""Candidate A: sequential RANSAC with a fixed distance threshold.

The voxelised cloud is searched for the plane with the most inliers, its
inliers are removed and the search repeats. Each plane's inliers are split into
Euclidean connected components.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from .classify import horizontal_mask
from .geometry import point_components, ransac_plane
from .preprocess import prepare_frame, voxel_downsample
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane


@dataclass
class RansacSequentialParams:
    stride: int = 2
    voxel_size: float = 0.02
    distance_threshold: float = 0.02
    iterations: int = 300
    max_rounds: int = 12
    min_points: int = 100
    component_radius: float = 0.05
    min_component_points: int = 40
    seed: int = 0


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: RansacSequentialParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or RansacSequentialParams()
    frame = prepare_frame(cloud, options, params.stride)
    labels = np.full(frame.valid.shape, -1, dtype=np.int32)
    pixels = np.flatnonzero(frame.valid)
    if pixels.size < params.min_points:
        return []
    voxels, inverse, _ = voxel_downsample(
        frame.points.reshape(-1, 3)[pixels], params.voxel_size
    )

    normal_ok = None
    if not options.detect_vertical:
        normal_ok = partial(horizontal_mask, options=options)

    rng = np.random.default_rng(params.seed)
    voxel_labels = np.full(len(voxels), -1, dtype=np.int32)
    remaining = np.arange(len(voxels))
    next_label = 0
    for _ in range(params.max_rounds):
        if len(remaining) < params.min_points:
            break
        result = ransac_plane(
            voxels[remaining],
            params.distance_threshold,
            params.iterations,
            rng,
            normal_ok=normal_ok,
        )
        if result is None:
            break
        _, _, inliers = result
        if inliers.sum() < params.min_points:
            break
        members = remaining[inliers]
        components = point_components(voxels[members], params.component_radius)
        sizes = np.bincount(components)
        for comp in np.flatnonzero(sizes >= params.min_component_points):
            voxel_labels[members[components == comp]] = next_label
            next_label += 1
        remaining = remaining[~inliers]

    labels.reshape(-1)[pixels] = voxel_labels[inverse]
    return planes_from_labels(frame, labels, options)
