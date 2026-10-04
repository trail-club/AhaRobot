"""Plane detection candidates for organized depth-camera point clouds.

All detectors share the signature ``detect(cloud, options, params)`` where
``cloud`` is an organized (H, W, 3) array in the camera optical frame with NaN
for invalid pixels, and return a list of ``Plane`` in the output frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import (
    cape,
    normal_clustering,
    organized_region_growing,
    peac_ahc,
    ransac_gravity,
    ransac_gravity_cape_hybrid,
    ransac_gravity_refined,
    ransac_sequential,
)
from .classify import select_output_planes
from .types import (
    ClassifyOptions,
    DepthNoiseModel,
    DetectionOptions,
    Plane,
    PlaneLabel,
)


@dataclass(frozen=True)
class Detector:
    name: str
    params_type: type
    detect: Callable[[np.ndarray, DetectionOptions, Any], list[Plane]]


DETECTORS: dict[str, Detector] = {
    "ransac_sequential": Detector(
        "ransac_sequential",
        ransac_sequential.RansacSequentialParams,
        ransac_sequential.detect,
    ),
    "ransac_gravity": Detector(
        "ransac_gravity", ransac_gravity.RansacGravityParams, ransac_gravity.detect
    ),
    "organized_region_growing": Detector(
        "organized_region_growing",
        organized_region_growing.RegionGrowingParams,
        organized_region_growing.detect,
    ),
    "normal_clustering": Detector(
        "normal_clustering",
        normal_clustering.NormalClusteringParams,
        normal_clustering.detect,
    ),
    "ransac_gravity_refined": Detector(
        "ransac_gravity_refined",
        ransac_gravity_refined.RansacGravityRefinedParams,
        ransac_gravity_refined.detect,
    ),
    "ransac_gravity_cape_hybrid": Detector(
        "ransac_gravity_cape_hybrid",
        ransac_gravity_cape_hybrid.HybridParams,
        ransac_gravity_cape_hybrid.detect,
    ),
    "peac_ahc": Detector("peac_ahc", peac_ahc.PeacParams, peac_ahc.detect),
    "cape": Detector("cape", cape.CapeParams, cape.detect),
}


def detect_planes(
    cloud: np.ndarray,
    method: str,
    options: DetectionOptions | None = None,
    params: Any = None,
) -> list[Plane]:
    """Run the detector registered as ``method``."""
    detector = DETECTORS[method]
    if params is None:
        params = detector.params_type()
    return detector.detect(cloud, options or DetectionOptions(), params)


__all__ = [
    "DETECTORS",
    "ClassifyOptions",
    "DepthNoiseModel",
    "DetectionOptions",
    "Detector",
    "Plane",
    "PlaneLabel",
    "detect_planes",
    "select_output_planes",
]
