"""Candidate B: gravity-aware RANSAC with a depth-dependent threshold.

Normals (kNN PCA on the voxels, or organized-image normals averaged per voxel
with ``normals="organized"``) split the voxelised cloud into horizontal and
vertical candidates using the up axis. RANSAC runs per group with hypotheses constrained to that
orientation and an inlier threshold of ``k * sigma(z)``. Inliers must also
agree in normal direction and are split into connected components. With
``merge=True`` coplanar fragments are merged (``merge.merge_coplanar``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import point_components, ransac_plane
from .merge import merge_coplanar
from .normals import knn_normals, organized_normals
from .preprocess import prepare_frame, voxel_downsample
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane


@dataclass
class RansacGravityParams:
    stride: int = 2
    voxel_size: float = 0.02
    normals: str = "knn"
    normal_k: int = 16
    normal_radius: int = 3
    group_tol_deg: float = 15.0
    inlier_normal_tol_deg: float = 30.0
    threshold_k: float = 3.0
    min_threshold: float = 0.01
    iterations: int = 200
    max_planes_per_group: int = 10
    min_points: int = 60
    component_radius: float = 0.05
    min_component_points: int = 40
    seed: int = 0
    # Coplanar fragment merging (merge.merge_coplanar), off by default.
    merge: bool = False
    merge_angle_deg: float = 10.0
    merge_k: float = 4.0
    merge_gap_px: int = 3
    merge_hull_gap: float = -1.0


def merge_fragments(frame, planes, options, params, refine_k=None) -> list[Plane]:
    """``merge_coplanar`` with the ``merge_*`` fields of ``params`` if enabled."""
    if not params.merge:
        return planes
    return merge_coplanar(
        frame,
        planes,
        options,
        angle_deg=params.merge_angle_deg,
        k=params.merge_k,
        gap_px=params.merge_gap_px,
        hull_gap=params.merge_hull_gap,
        refine_k=refine_k,
    )


def _voxel_normals_from_image(frame, pixels, inverse, count, params, options):
    """Average organized-image normals per voxel; NaN where none are valid."""
    normals, mask = organized_normals(
        frame.points,
        frame.depth,
        options.noise.inverse_sigma,
        radius=params.normal_radius,
        smoothing_radius=params.normal_radius,
        viewpoint=frame.camera_origin,
    )
    pixel_normals = np.where(mask[..., None], normals, 0.0).reshape(-1, 3)[pixels]
    summed = np.stack(
        [np.bincount(inverse, pixel_normals[:, i], minlength=count) for i in range(3)],
        axis=1,
    )
    norm = np.linalg.norm(summed, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm[:, None] > 0, summed / norm[:, None], np.nan)


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: RansacGravityParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or RansacGravityParams()
    frame = prepare_frame(cloud, options, params.stride)
    labels = np.full(frame.valid.shape, -1, dtype=np.int32)
    pixels = np.flatnonzero(frame.valid)
    if pixels.size < params.min_points:
        return []
    voxels, inverse, voxel_depth = voxel_downsample(
        frame.points.reshape(-1, 3)[pixels],
        params.voxel_size,
        frame.depth.reshape(-1)[pixels],
    )
    if params.normals == "organized":
        normals = _voxel_normals_from_image(
            frame, pixels, inverse, len(voxels), params, options
        )
    elif params.normals == "knn":
        normals, _ = knn_normals(voxels, params.normal_k, viewpoint=frame.camera_origin)
    else:
        raise ValueError(f"unknown normals mode: {params.normals}")
    thresholds = np.maximum(
        params.threshold_k * options.noise.sigma(voxel_depth), params.min_threshold
    )

    up = options.unit_up()
    cos_group = np.cos(np.radians(params.group_tol_deg))
    sin_group = np.sin(np.radians(params.group_tol_deg))
    cos_inlier = np.cos(np.radians(params.inlier_normal_tol_deg))
    up_cos = np.abs(normals @ up)
    groups = [(up_cos >= cos_group, lambda n: np.abs(n @ up) >= cos_group)]
    if options.detect_vertical:
        groups.append((up_cos <= sin_group, lambda n: np.abs(n @ up) <= sin_group))

    rng = np.random.default_rng(params.seed)
    voxel_labels = np.full(len(voxels), -1, dtype=np.int32)
    assigned = np.zeros(len(voxels), dtype=bool)
    next_label = 0
    for group, hypothesis_ok in groups:
        for _ in range(params.max_planes_per_group):
            pool = np.flatnonzero(group & ~assigned)
            if len(pool) < params.min_points:
                break
            result = ransac_plane(
                voxels[pool],
                thresholds[pool],
                params.iterations,
                rng,
                normal_ok=hypothesis_ok,
            )
            if result is None:
                break
            normal, offset, inliers = result
            if inliers.sum() < params.min_points:
                break
            free = np.flatnonzero(~assigned)
            close = np.abs(voxels[free] @ normal + offset) < thresholds[free]
            # Voxels without a normal (NaN) are not gated.
            agree = ~(np.abs(normals[free] @ normal) < cos_inlier)
            members = free[close & agree]
            components = point_components(voxels[members], params.component_radius)
            sizes = np.bincount(components, minlength=1)
            for comp in np.flatnonzero(sizes >= params.min_component_points):
                voxel_labels[members[components == comp]] = next_label
                next_label += 1
            assigned[members] = True
            assigned[pool[inliers]] = True

    labels.reshape(-1)[pixels] = voxel_labels[inverse]
    planes = planes_from_labels(frame, labels, options)
    return merge_fragments(frame, planes, options, params)
