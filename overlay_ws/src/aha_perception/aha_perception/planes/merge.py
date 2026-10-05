"""Merging of coplanar plane fragments.

Spatially correlated depth noise can split one surface into several nearly
identical planes. ``merge_coplanar`` greedily joins two planes when

- their normals differ by less than ``angle_deg``,
- their centroids differ by less than ``k * sigma(z)`` along the
  inlier-weighted mean normal (z = the larger mean depth of the two),
- if both are horizontal (or both vertical) within the ``DetectionOptions``
  tolerances, the least-squares plane of their union is too, and
- they touch in the image (inlier masks within ``gap_px`` pixels) or, with
  ``hull_gap >= 0``, their convex hulls projected onto the mean plane overlap
  or are less than ``hull_gap`` meters apart.

The pair with the smallest normalized offset is merged first. Merged groups
are refitted with ``planes_from_labels`` (refit, ``refine_k * sigma(z)`` pixel
selection, size filters); untouched planes are kept as they are and all planes
are classified again. This repeats until nothing merges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects, maximum_filter

from .classify import classify_planes, is_horizontal, is_vertical
from .geometry import clip_convex_polygons, convex_hull_2d, plane_axes, polygon_area
from .preprocess import Frame
from .refine import full_resolution
from .segmentation import label_image, planes_from_labels
from .types import DetectionOptions, Plane

_MAX_ROUNDS = 5


@dataclass
class _Group:
    members: list[int]
    count: float
    total: np.ndarray  # sum of inlier points
    outer: np.ndarray  # sum of outer products of inlier points
    depth_sum: float
    normal: np.ndarray  # inlier-weighted mean normal (unit)
    orientation: str | None  # "horizontal" / "vertical" shared by all members
    hull: np.ndarray  # (K, 3) hull points of all members
    neighbours: set[int]

    def centroid(self) -> np.ndarray:
        return self.total / self.count

    def mean_depth(self) -> float:
        return self.depth_sum / self.count


def _orientation(normal: np.ndarray, options: DetectionOptions) -> str | None:
    if is_horizontal(normal, options):
        return "horizontal"
    if is_vertical(normal, options):
        return "vertical"
    return None


def _group(
    index: int,
    plane: Plane,
    points: np.ndarray,
    depth: np.ndarray,
    options: DetectionOptions,
) -> _Group:
    pts = points[plane.inliers]
    return _Group(
        members=[index],
        count=float(len(pts)),
        total=pts.sum(axis=0),
        outer=pts.T @ pts,
        depth_sum=float(depth[plane.inliers].sum()),
        normal=plane.normal,
        orientation=_orientation(plane.normal, options),
        hull=plane.hull_points(),
        neighbours=set(),
    )


def _mean_normal(a: _Group, b: _Group) -> np.ndarray:
    sign = 1.0 if float(a.normal @ b.normal) >= 0 else -1.0
    normal = a.count * a.normal + sign * b.count * b.normal
    return normal / np.linalg.norm(normal)


def _keeps_orientation(a: _Group, b: _Group, options: DetectionOptions) -> bool:
    """False if the union fit leaves the orientation class both groups share."""
    if a.orientation is None or a.orientation != b.orientation:
        return True
    count = a.count + b.count
    centroid = (a.total + b.total) / count
    cov = (a.outer + b.outer) / count - np.outer(centroid, centroid)
    normal = np.linalg.eigh(cov)[1][:, 0]
    return _orientation(normal, options) == a.orientation


def _image_neighbours(labels: np.ndarray, count: int, gap_px: int) -> list[set[int]]:
    """Labels within ``gap_px`` pixels (Chebyshev) of each label's mask."""
    h, w = labels.shape
    out = [set() for _ in range(count)]
    for index, box in enumerate(find_objects(labels + 1, max_label=count)):
        if box is None:
            continue
        r0, r1 = max(box[0].start - gap_px, 0), min(box[0].stop + gap_px, h)
        c0, c1 = max(box[1].start - gap_px, 0), min(box[1].stop + gap_px, w)
        crop = labels[r0:r1, c0:c1]
        near = maximum_filter(
            (crop == index).astype(np.uint8), size=2 * gap_px + 1, mode="constant"
        )
        hits = np.bincount(crop[near.astype(bool)] + 1, minlength=count + 1)[1:]
        out[index] = {int(t) for t in np.flatnonzero(hits) if t != index}
    return out


def _polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two counter-clockwise convex polygons (0 if they overlap)."""
    if len(a) < 3 or len(b) < 3:
        return float("inf")
    if polygon_area(clip_convex_polygons(a, b)) > 0.0:
        return 0.0

    def to_edges(points, polygon):
        edge = np.roll(polygon, -1, axis=0) - polygon
        rel = points[:, None, :] - polygon[None]
        t = np.einsum("pke,ke->pk", rel, edge) / np.maximum(
            np.einsum("ke,ke->k", edge, edge), 1e-12
        )
        t = np.clip(t, 0.0, 1.0)
        return float(np.linalg.norm(rel - t[..., None] * edge[None], axis=2).min())

    return min(to_edges(a, b), to_edges(b, a))


def _hull_gap(a: _Group, b: _Group, normal: np.ndarray, up: np.ndarray) -> float:
    axes = plane_axes(normal, up)
    pa, _ = convex_hull_2d(a.hull @ axes.T)
    pb, _ = convex_hull_2d(b.hull @ axes.T)
    return _polygon_gap(pa, pb)


def _merge_groups(
    groups: list[_Group],
    options: DetectionOptions,
    angle_deg: float,
    k: float,
    hull_gap: float,
) -> bool:
    """Merge pairs of groups in place until none qualifies; True if any merged."""
    cos_angle = np.cos(np.radians(angle_deg))
    up = options.unit_up()
    merged = False
    while True:
        best = None
        alive = [i for i, g in enumerate(groups) if g.members]
        for ai, a in enumerate(alive):
            ga = groups[a]
            for b in alive[ai + 1 :]:
                gb = groups[b]
                if abs(float(ga.normal @ gb.normal)) < cos_angle:
                    continue
                normal = _mean_normal(ga, gb)
                depth = max(ga.mean_depth(), gb.mean_depth())
                limit = max(
                    k * float(options.noise.sigma(depth)), options.refine_min_distance
                )
                score = abs(float(normal @ (ga.centroid() - gb.centroid()))) / limit
                if score >= 1.0 or (best is not None and score >= best[0]):
                    continue
                if b not in ga.neighbours and (
                    hull_gap < 0 or _hull_gap(ga, gb, normal, up) > hull_gap
                ):
                    continue
                if not _keeps_orientation(ga, gb, options):
                    continue
                best = (score, a, b, normal)
        if best is None:
            return merged
        _, a, b, normal = best
        ga, gb = groups[a], groups[b]
        ga.normal = normal
        ga.members += gb.members
        ga.count += gb.count
        ga.total = ga.total + gb.total
        ga.outer = ga.outer + gb.outer
        ga.depth_sum += gb.depth_sum
        if ga.orientation != gb.orientation:
            ga.orientation = None
        ga.hull = np.concatenate([ga.hull, gb.hull])
        ga.neighbours |= gb.neighbours
        gb.members = []
        for g in groups:
            if b in g.neighbours:
                g.neighbours.discard(b)
                g.neighbours.add(a)
        ga.neighbours.discard(a)
        merged = True


def merge_coplanar(
    frame: Frame,
    planes: list[Plane],
    options: DetectionOptions,
    angle_deg: float = 10.0,
    k: float = 4.0,
    gap_px: int = 3,
    hull_gap: float = -1.0,
    refine_k: float | None = None,
) -> list[Plane]:
    """Merge coplanar fragments of ``planes`` (inliers at the input resolution).

    ``frame`` is the frame the planes were detected on (any stride);
    ``refine_k`` is passed to ``planes_from_labels`` for the merged groups.
    Returns ``planes`` unchanged when nothing merges.
    """
    full = full_resolution(frame)
    points = full.points.reshape(-1, 3)
    depth = full.depth.reshape(-1)
    shape = full.valid.shape
    for _ in range(_MAX_ROUNDS):
        if len(planes) < 2:
            return planes
        groups = [_group(i, p, points, depth, options) for i, p in enumerate(planes)]
        neighbours = _image_neighbours(label_image(planes, shape), len(planes), gap_px)
        for group, near in zip(groups, neighbours):
            group.neighbours = near
        if not _merge_groups(groups, options, angle_deg, k, hull_gap):
            return planes
        kept = [planes[g.members[0]] for g in groups if len(g.members) == 1]
        labels = np.full(shape[0] * shape[1], -1, dtype=np.int32)
        merged = [g for g in groups if len(g.members) > 1]
        for index, group in enumerate(merged):
            for member in group.members:
                labels[planes[member].inliers] = index
        refit = planes_from_labels(full, labels.reshape(shape), options, refine_k)
        planes = sorted(kept + refit, key=lambda p: -len(p.inliers))
        classify_planes(planes, options)
    return planes
