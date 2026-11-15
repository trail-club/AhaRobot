"""Plane detection candidates for organized depth-camera point clouds.

All detectors share the signature ``detect(cloud, options, params)`` where
``cloud`` is an organized (H, W, 3) array in the camera optical frame with NaN
for invalid pixels, and return a list of labeled ``Plane`` in the output frame
of ``DetectionOptions.camera_to_base`` (floor at 0 along ``up_axis``)::

    params = DETECTORS["ransac_gravity_refined"].params_type(merge=True)
    options = DetectionOptions(camera_to_base=T, noise=DepthNoiseModel(focal=fx))
    planes = detect_planes(cloud, "ransac_gravity_refined", options, params)
    output = select_output_planes(planes, options)  # what the ROS node publishes

Detectors (``DETECTORS``; details in each module):

- ``ransac_sequential``: repeated fixed-threshold RANSAC on a voxel cloud,
  split into connected components.
- ``ransac_gravity``: normals split points into horizontal / vertical
  candidates; orientation-constrained RANSAC with a k * sigma(Z) threshold and
  connected components. Normals from voxel kNN, or ``normals="organized"``.
- ``ransac_gravity_refined``: ``ransac_gravity`` plus full-resolution pixel
  re-assignment, small-piece removal, hole filling and refit (ROS node
  default).
- ``ransac_gravity_cape_hybrid``: ``ransac_gravity`` planes as seeds for cell
  region growing and pixel boundary refinement.
- ``organized_region_growing``: PCA planarity of image cells, region growing,
  coplanar merging and pixel boundary refinement.
- ``normal_clustering``: integral-image normals clustered on the sphere, split
  by offset and image connectivity.
- ``peac_ahc``: own implementation after PEAC (Feng et al., ICRA 2014),
  agglomerative clustering of a node grid with T_MSE(Z).
- ``cape``: reimplementation of the plane part of CAPE (Proenca and Gao,
  IROS 2018).

Shared stages: preprocessing (depth range, decimation, voxels, transform;
``preprocess``), normals (``normals``), cell moments (``cells``), pixel
assignment to planes (``refine``), label image to planes with refit,
k * sigma(Z) pixel selection and size filters (``segmentation``), labeling
(``classify``) and optional coplanar fragment merging for the RANSAC-gravity
detectors (``merge``, ``params.merge``). Depth noise is ``DepthNoiseModel``
(sigma(Z) = Z^2 * subpixel / (f * baseline)).
"""

from __future__ import annotations

import dataclasses
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


def params_for_input_stride(params: Any, stride: int) -> Any:
    """``params`` for a cloud of every ``stride``-th row and column of the image.

    A detector that decimates its input by ``params.stride`` itself (RANSAC
    stage) then keeps its working resolution of the full image:
    ``params.stride`` is divided by ``stride`` (at least 1). Other params and
    the pixel counts of ``DetectionOptions`` stay as they are.
    """
    if stride <= 1 or not hasattr(params, "stride"):
        return params
    return dataclasses.replace(params, stride=max(1, params.stride // stride))


__all__ = [
    "DETECTORS",
    "ClassifyOptions",
    "DepthNoiseModel",
    "DetectionOptions",
    "Detector",
    "Plane",
    "PlaneLabel",
    "detect_planes",
    "params_for_input_stride",
    "select_output_planes",
]
