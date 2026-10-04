"""Semantic labels for detected planes from gravity, height and area."""

from __future__ import annotations

import numpy as np

from .geometry import clip_convex_polygons, convex_hull_2d, plane_axes, polygon_area
from .types import DetectionOptions, Plane, PlaneLabel


def horizontal_mask(normals: np.ndarray, options: DetectionOptions) -> np.ndarray:
    """True for (..., 3) normals within ``horizontal_tol_deg`` of the up axis."""
    cos = np.abs(normals @ options.unit_up())
    return cos >= np.cos(np.radians(options.horizontal_tol_deg))


def vertical_mask(normals: np.ndarray, options: DetectionOptions) -> np.ndarray:
    """True for (..., 3) normals within ``vertical_tol_deg`` of the ground plane."""
    cos = np.abs(normals @ options.unit_up())
    return cos <= np.sin(np.radians(options.vertical_tol_deg))


def is_horizontal(normal: np.ndarray, options: DetectionOptions) -> bool:
    return bool(horizontal_mask(np.asarray(normal), options))


def is_vertical(normal: np.ndarray, options: DetectionOptions) -> bool:
    return bool(vertical_mask(np.asarray(normal), options))


def _footprint(plane: Plane, ground_axes: np.ndarray) -> tuple[np.ndarray, float]:
    """Convex footprint of the plane hull projected onto the ground."""
    return convex_hull_2d(plane.hull_points() @ ground_axes.T)


def _stacked_pairs(planes: list[Plane], options: DetectionOptions) -> set[int]:
    """Indices of horizontal planes stacked over the same footprint (shelf boards)."""
    opts = options.classify
    up = options.unit_up()
    ground_axes = plane_axes(up)
    candidates = []
    for i, plane in enumerate(planes):
        height = float(plane.centroid @ up)
        if height <= opts.floor_max_height or plane.area < opts.shelf_min_area:
            continue
        polygon, area = _footprint(plane, ground_axes)
        if area > 0:
            candidates.append((i, height, polygon, area))
    stacked: set[int] = set()
    for a in range(len(candidates)):
        i, hi, pi, ai = candidates[a]
        for b in range(a + 1, len(candidates)):
            j, hj, pj, aj = candidates[b]
            if abs(hi - hj) < opts.shelf_min_spacing:
                continue
            small, large = min(ai, aj), max(ai, aj)
            if small / large < opts.shelf_min_area_ratio:
                continue
            overlap = polygon_area(clip_convex_polygons(pi, pj))
            if overlap >= opts.shelf_min_overlap * small:
                stacked.update((i, j))
    return stacked


def classify_planes(planes: list[Plane], options: DetectionOptions) -> None:
    """Assign ``plane.label`` in place."""
    opts = options.classify
    up = options.unit_up()
    horizontal = [is_horizontal(p.normal, options) for p in planes]
    stacked = _stacked_pairs([p for p, h in zip(planes, horizontal) if h], options)
    horizontal_ids = [i for i, h in enumerate(horizontal) if h]
    stacked_ids = {horizontal_ids[k] for k in stacked}

    for i, plane in enumerate(planes):
        height = float(plane.centroid @ up)
        label = PlaneLabel.OTHER
        if horizontal[i]:
            low, high = opts.table_height_range
            if abs(height) <= opts.floor_max_height:
                label = PlaneLabel.FLOOR
            elif i in stacked_ids:
                label = PlaneLabel.SHELF
            elif low <= height <= high and plane.area >= opts.table_min_area:
                label = PlaneLabel.TABLE
        elif is_vertical(plane.normal, options):
            heights = plane.hull_points() @ up
            extent = float(heights.max() - heights.min()) if len(heights) else 0.0
            if plane.area >= opts.wall_min_area and extent >= opts.wall_min_extent:
                label = PlaneLabel.WALL
        plane.label = label


def select_output_planes(planes: list[Plane], options: DetectionOptions) -> list[Plane]:
    """Horizontal planes and walls, in order; needs labels from ``classify_planes``.

    Oblique planes and vertical planes below the wall size are dropped. Walls are
    kept only with ``options.detect_vertical``.
    """
    return [
        plane
        for plane in planes
        if is_horizontal(plane.normal, options)
        or (options.detect_vertical and plane.label == PlaneLabel.WALL)
    ]
