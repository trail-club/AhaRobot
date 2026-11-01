"""Candidate B+C: gravity-aware RANSAC planes extended by cell region growing.

``ransac_gravity`` provides the planes. On a full-resolution cell grid, planar
cells mostly covered by a plane's inliers seed that plane; it then grows over
4-neighbour planar cells whose points fit it (RMS below
``merge_rms_k * sigma(z)``). Pixels in and around the grown cells (and near
the RANSAC inliers) go to the closest plane, keeping only pieces connected to
the inliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from . import ransac_gravity
from .cells import Moments, cell_grid, tile
from .normals import box_sum
from .organized_region_growing import grow_cells, planar_cells
from .preprocess import prepare_frame
from .refine import EIGHT, assign_pixels, full_resolution, keep_seeded_components
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane


@dataclass
class HybridParams(ransac_gravity.RansacGravityParams):
    cell_size: int = 16
    min_valid_fraction: float = 0.6
    max_discontinuity_fraction: float = 0.05
    discontinuity_k: float = 6.0
    planarity_k: float = 3.0
    merge_rms_k: float = 3.0
    seed_fraction: float = 0.5
    inlier_margin: int = 2
    refine_k: float = 3.0


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: HybridParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or HybridParams()
    planes = ransac_gravity.detect(cloud, options, params)
    if not planes:
        return planes
    frame = full_resolution(prepare_frame(cloud, options, 1))
    grid = cell_grid(frame, params.cell_size, options, params.discontinuity_k)
    shape = frame.valid.shape
    inlier_masks = []
    for plane in planes:
        mask = np.zeros(shape, dtype=bool)
        mask.reshape(-1)[plane.inliers] = True
        inlier_masks.append(mask)

    cell_region = np.full((grid.rows, grid.cols), -1, dtype=np.int32)
    regions = []
    if grid.rows and grid.cols:
        planar = planar_cells(grid, params, options)
        sigma = options.noise.sigma(grid.mean_depth)
        best = np.zeros((grid.rows, grid.cols))
        safe = np.maximum(grid.count, 1.0)
        for index, mask in enumerate(inlier_masks):
            share = tile(mask, grid.size, grid.rows, grid.cols).sum(axis=-1) / safe
            take = planar & (share >= params.seed_fraction) & (share > best)
            cell_region[take] = index
            best[take] = share[take]

        def accept(a, b, region):
            if not planar[a, b]:
                return False
            mse = grid.mse_to_plane(a, b, region.normal, region.offset)
            return mse <= np.square(params.merge_rms_k * sigma[a, b])

        for index, plane in enumerate(planes):
            region = Moments()
            region.normal, region.offset = plane.normal, plane.offset
            for i, j in zip(*np.nonzero(cell_region == index)):
                region.add_cell(grid, int(i), int(j))
            grow_cells(grid, region, cell_region, index, accept, update=False)
            regions.append(region)

    candidates = []
    for index, mask in enumerate(inlier_masks):
        near = box_sum(mask.astype(np.float64), params.inlier_margin) > 0
        if regions:
            cells = binary_dilation(regions[index].mask(grid.rows, grid.cols), EIGHT)
            near |= grid.pixel_mask(cells)
        candidates.append(near)
    models = [(p.normal, p.offset) for p in planes]
    labels = assign_pixels(frame, models, candidates, params.refine_k, options)
    labels = keep_seeded_components(labels, [p.inliers for p in planes])
    return planes_from_labels(frame, labels, options, params.refine_k)
