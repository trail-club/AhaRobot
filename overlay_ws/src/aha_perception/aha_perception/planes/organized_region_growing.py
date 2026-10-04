"""Candidate C: organized cell-grid region growing (simplified CAPE / PEAC).

The organized cloud is tiled into square cells. Cells that pass a PCA
planarity test with a depth-dependent threshold and contain no depth
discontinuity are grown breadth-first over 4-neighbour cells whose points fit
the region plane (RMS below ``merge_rms_k * sigma(z)``). Coplanar regions
separated by a few non-planar cells are merged, then region borders are
refined per pixel.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .cells import CellGrid, Moments, cell_grid
from .preprocess import prepare_frame
from .refine import EIGHT, assign_pixels
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane


@dataclass
class RegionGrowingParams:
    stride: int = 1
    cell_size: int = 16
    min_valid_fraction: float = 0.6
    max_discontinuity_fraction: float = 0.05
    discontinuity_k: float = 6.0
    planarity_k: float = 3.0
    merge_angle_deg: float = 20.0
    normal_conditioning: float = 10.0
    merge_rms_k: float = 3.0
    merge_distance_k: float = 3.0
    min_cells: int = 3
    gap_cells: int = 2
    refine_k: float = 3.0


def planar_cells(grid: CellGrid, params, options: DetectionOptions) -> np.ndarray:
    """Cells with enough data, no depth jump and an MSE below (k sigma(z))^2."""
    area = grid.size * grid.size
    usable = (grid.count >= params.min_valid_fraction * area) & (
        grid.edge_count <= params.max_discontinuity_fraction * area
    )
    sigma = options.noise.sigma(grid.mean_depth)
    return usable & (grid.mse < np.square(params.planarity_k * sigma))


def grow_cells(grid, region, cell_region, rid, accept, update=True):
    """Breadth-first growth of ``region`` over 4-neighbour cells passing ``accept``."""
    queue = deque(region.cells)
    while queue:
        i, j = queue.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if not (0 <= a < grid.rows and 0 <= b < grid.cols):
                continue
            if cell_region[a, b] >= 0 or not accept(a, b, region):
                continue
            cell_region[a, b] = rid
            region.add_cell(grid, a, b)
            if update:
                region.update()
            queue.append((a, b))


def refine_regions(frame, grid, regions, k, options) -> np.ndarray:
    """Assign pixels in and around each region's cells to the closest plane."""
    candidates = [
        grid.pixel_mask(binary_dilation(r.mask(grid.rows, grid.cols), EIGHT))
        for r in regions
    ]
    planes = [(r.normal, r.offset) for r in regions]
    return assign_pixels(frame, planes, candidates, k, options)


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: RegionGrowingParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or RegionGrowingParams()
    frame = prepare_frame(cloud, options, params.stride)
    grid = cell_grid(frame, params.cell_size, options, params.discontinuity_k)
    if grid.rows == 0 or grid.cols == 0:
        return []
    planar = planar_cells(grid, params, options)
    sigma = options.noise.sigma(grid.mean_depth)
    # Far cells can be noisier than they are wide; their normals are not used.
    flatness = grid.mse / np.maximum(grid.eigvals[..., 1], 1e-12)
    reliable = flatness * params.normal_conditioning < 1.0
    cos_merge = np.cos(np.radians(params.merge_angle_deg))

    def accept(a, b, region):
        if not planar[a, b]:
            return False
        if (
            reliable[a, b]
            and abs(float(grid.normals[a, b] @ region.normal)) < cos_merge
        ):
            return False
        mse = grid.mse_to_plane(a, b, region.normal, region.offset)
        return mse <= np.square(params.merge_rms_k * sigma[a, b])

    cell_region = np.full((grid.rows, grid.cols), -1, dtype=np.int32)
    regions: list[Moments] = []
    for flat in np.argsort(np.where(planar, flatness, np.inf), axis=None):
        i0, j0 = divmod(int(flat), grid.cols)
        if not planar[i0, j0]:
            break
        if cell_region[i0, j0] >= 0:
            continue
        region = Moments()
        region.add_cell(grid, i0, j0)
        region.update()
        cell_region[i0, j0] = len(regions)
        grow_cells(grid, region, cell_region, len(regions), accept)
        regions.append(region)

    kept = [r for r in regions if len(r.cells) >= params.min_cells]
    merged = _merge_coplanar(kept, grid, params, options, cos_merge)
    labels = refine_regions(frame, grid, merged, params.refine_k, options)
    return planes_from_labels(frame, labels, options, params.refine_k)


def _merge_coplanar(regions, grid, params, options, cos_merge) -> list[Moments]:
    """Union coplanar regions whose cells are at most ``gap_cells`` apart."""
    n = len(regions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    masks = [r.mask(grid.rows, grid.cols) for r in regions]
    grown = [binary_dilation(m, EIGHT, iterations=params.gap_cells) for m in masks]
    for a in range(n):
        ra = regions[a]
        for b in range(a + 1, n):
            rb = regions[b]
            if abs(float(ra.normal @ rb.normal)) < cos_merge:
                continue
            if not (grown[a] & masks[b]).any():
                continue
            depth = max(ra.mean_depth, rb.mean_depth)
            limit = params.merge_distance_k * float(options.noise.sigma(depth))
            if abs(float(rb.centroid @ ra.normal) + ra.offset) > limit:
                continue
            if abs(float(ra.centroid @ rb.normal) + rb.offset) > limit:
                continue
            parent[find(b)] = find(a)

    merged: dict[int, Moments] = {}
    for i, region in enumerate(regions):
        merged.setdefault(find(i), Moments()).absorb(region)
    for region in merged.values():
        region.update()
    return list(merged.values())
