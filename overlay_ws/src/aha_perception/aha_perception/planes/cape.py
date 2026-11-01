"""Candidate CAPE: cell-wise plane extraction (planes only).

Reimplementation after Proenca and Gao, "Fast cylinder and plane extraction
from depth cameras for visual odometry" (IROS 2018), without the cylinder part.

1. Cells of ``cell_size`` pixels are planar if they have enough valid pixels,
   at most ``max_jumps`` depth jumps along their middle row and column, and a
   plane-fit MSE below ``T_MSE(z) = (sigma z^2 + eps)^2``.
2. Cell normals (camera frame) are binned in a polar/azimuth histogram. The
   most populated bin seeds region growing from its lowest-MSE cell; 4-neighbour
   planar cells join if their normal and centroid agree with the seed plane.
   Cells of a region are removed from the histogram. Small regions are dropped.
3. Adjacent segments with similar planes are merged.
4. Interior (eroded) cells are labeled whole; boundary pixels go to the closest
   plane within ``refine_k * sigma(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .cells import CellGrid, Moments, cell_grid
from .organized_region_growing import grow_cells
from .preprocess import prepare_frame
from .refine import EIGHT, assign_pixels
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class CapeParams:
    cell_size: int = 20
    min_valid_fraction: float = 0.5
    jump_k: float = 4.0
    max_jumps: int = 1
    mse_k: float = 1.5
    mse_eps: float = 0.005
    histogram_bins: int = 20
    min_seed_cells: int = 4
    min_region_cells: int = 4
    grow_angle_deg: float = 15.0
    grow_distance_k: float = 3.0
    merge_angle_deg: float = 15.0
    refine_k: float = 3.0


def _jumps(frame, size, rows, cols, params, options) -> np.ndarray:
    """Depth jumps between consecutive valid pixels on each cell's middle row/column."""
    depth = frame.depth[: rows * size, : cols * size]
    mid = size // 2
    counts = np.zeros((rows, cols), dtype=np.int64)
    lines = (
        depth[mid::size, :].reshape(rows, cols, size),
        depth[:, mid::size].T.reshape(cols, rows, size).swapaxes(0, 1),
    )
    for line in lines:
        diff = np.abs(np.diff(line, axis=-1))
        limit = params.jump_k * np.sqrt(2.0) * options.noise.sigma(line[..., :-1])
        with np.errstate(invalid="ignore"):
            counts += (diff > limit).sum(axis=-1)
    return counts


def _t_mse(depth, params, options):
    sigma = params.mse_k * options.noise.inverse_sigma
    return np.square(sigma * np.square(depth) + params.mse_eps)


def _histogram_bins(grid: CellGrid, rotation: np.ndarray, bins: int) -> np.ndarray:
    """Polar/azimuth bin of each cell normal expressed in the camera frame."""
    normals = grid.normals @ rotation
    flip = normals[..., 2] > 0
    normals[flip] *= -1.0
    polar = np.arccos(np.clip(-normals[..., 2], -1.0, 1.0))
    azimuth = np.arctan2(normals[..., 0], normals[..., 1])
    p = np.minimum((polar / np.pi * bins).astype(int), bins - 1)
    a = np.minimum(((azimuth + np.pi) / (2 * np.pi) * bins).astype(int), bins - 1)
    return p * bins + a


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: CapeParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or CapeParams()
    frame = prepare_frame(cloud, options, 1)
    size = params.cell_size
    grid = cell_grid(frame, size, options)
    if grid.rows == 0 or grid.cols == 0:
        return []
    jumps = _jumps(frame, size, grid.rows, grid.cols, params, options)
    planar = (
        (grid.count >= params.min_valid_fraction * size * size)
        & (jumps <= params.max_jumps)
        & (grid.mse < _t_mse(grid.mean_depth, params, options))
    )
    rotation = np.asarray(options.camera_to_base, dtype=np.float64)[:3, :3]
    bins = _histogram_bins(grid, rotation, params.histogram_bins)
    sigma = options.noise.sigma(grid.mean_depth)
    cos_grow = np.cos(np.radians(params.grow_angle_deg))

    cell_region = np.full((grid.rows, grid.cols), -1, dtype=np.int32)
    available = planar.copy()
    regions: list[Moments] = []
    while available.any():
        counts = np.bincount(bins[available], minlength=params.histogram_bins**2)
        top = int(np.argmax(counts))
        if counts[top] < params.min_seed_cells:
            break
        in_bin = available & (bins == top)
        seed = np.unravel_index(
            int(np.argmin(np.where(in_bin, grid.mse, np.inf))), in_bin.shape
        )
        region = Moments()
        region.add_cell(grid, int(seed[0]), int(seed[1]))
        region.update()
        seed_normal, seed_offset = region.normal, region.offset
        rid = len(regions)

        def accept(a, b, _region, seed_normal=seed_normal, seed_offset=seed_offset):
            if not available[a, b]:
                return False
            if abs(float(grid.normals[a, b] @ seed_normal)) < cos_grow:
                return False
            dist = abs(float(grid.mean[a, b] @ seed_normal) + seed_offset)
            return dist < params.grow_distance_k * sigma[a, b] + params.mse_eps

        cell_region[seed] = rid
        grow_cells(grid, region, cell_region, rid, accept, update=False)
        members = region.mask(grid.rows, grid.cols)
        available &= ~members
        if len(region.cells) < params.min_region_cells:
            cell_region[members] = -2
            continue
        region.update()
        regions.append(region)

    merged = _merge_segments(regions, cell_region, grid, params, options)
    labels = _refine(frame, grid, merged, params, options)
    return planes_from_labels(frame, labels, options, params.refine_k)


def _merge_segments(regions, cell_region, grid, params, options) -> list[Moments]:
    """Merge 4-adjacent segments with similar normals and mutual plane distance."""
    n = len(regions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = set()
    for a, b in (
        (cell_region[1:, :], cell_region[:-1, :]),
        (cell_region[:, 1:], cell_region[:, :-1]),
    ):
        touch = (a >= 0) & (b >= 0) & (a != b)
        pairs |= {(int(min(x, y)), int(max(x, y))) for x, y in zip(a[touch], b[touch])}
    cos_merge = np.cos(np.radians(params.merge_angle_deg))
    for i, j in sorted(pairs):
        ri, rj = regions[i], regions[j]
        if abs(float(ri.normal @ rj.normal)) < cos_merge:
            continue
        depth = max(ri.mean_depth, rj.mean_depth)
        limit = (
            params.grow_distance_k * float(options.noise.sigma(depth)) + params.mse_eps
        )
        if abs(float(rj.centroid @ ri.normal) + ri.offset) > limit:
            continue
        if abs(float(ri.centroid @ rj.normal) + rj.offset) > limit:
            continue
        parent[find(j)] = find(i)
    merged: dict[int, Moments] = {}
    for i, region in enumerate(regions):
        merged.setdefault(find(i), Moments()).absorb(region)
    for region in merged.values():
        region.update()
    return list(merged.values())


def _refine(frame, grid, regions, params, options) -> np.ndarray:
    labels = np.full(frame.valid.shape, -1, dtype=np.int32)
    candidates = []
    for index, region in enumerate(regions):
        mask = region.mask(grid.rows, grid.cols)
        interior = grid.pixel_mask(binary_erosion(mask, _FOUR, border_value=0))
        labels[interior & frame.valid] = index
        candidates.append(grid.pixel_mask(binary_dilation(mask, EIGHT)) & ~interior)
    models = [(r.normal, r.offset) for r in regions]
    return assign_pixels(frame, models, candidates, params.refine_k, options, labels)
