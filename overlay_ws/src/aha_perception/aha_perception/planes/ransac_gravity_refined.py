"""Candidate B+: gravity-aware RANSAC followed by full-resolution pixel refinement.

After ``ransac_gravity`` finds the planes, every valid pixel of the input
image is re-assigned to the plane with the smallest residual ratio
``|n.p + d| / (k sigma(z))`` (no normal gating). A plane may only take pixels
connected in the image to its RANSAC inliers, small pieces are dropped, small
holes are closed with pixels that are still on the plane, and the planes are
refitted on the final inliers. With ``merge=True`` coplanar fragments are
merged before and after the refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ransac_gravity
from .preprocess import prepare_frame, with_stride
from .refine import assign_seeded, fill_holes, full_resolution, keep_seeded_components
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane


@dataclass
class RansacGravityRefinedParams(ransac_gravity.RansacGravityParams):
    refine_k: float = 3.0
    min_component_pixels: int = 50
    hole_iterations: int = 2
    hole_k: float = 4.0


def refine_planes(frame, planes, options, params) -> np.ndarray:
    """Full-resolution label image for already detected planes."""
    models = [(p.normal, p.offset) for p in planes]
    labels = assign_seeded(frame, planes, params.refine_k, options)
    labels = keep_seeded_components(
        labels, [p.inliers for p in planes], params.min_component_pixels
    )
    return fill_holes(
        frame, labels, models, params.hole_iterations, params.hole_k, options
    )


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: RansacGravityRefinedParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or RansacGravityRefinedParams()
    # One preparation at the input resolution serves both stages.
    full = prepare_frame(cloud, options, 1)
    planes = ransac_gravity.detect_frame(
        with_stride(full, params.stride), options, params
    )
    if not planes:
        return planes
    frame = full_resolution(full)
    labels = refine_planes(frame, planes, options, params)
    planes = planes_from_labels(frame, labels, options, params.hole_k)
    return ransac_gravity.merge_fragments(frame, planes, options, params, params.hole_k)
