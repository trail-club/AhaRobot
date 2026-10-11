"""Place candidates for an object on detected support planes.

For every horizontal support plane a 2D grid is built in the plane's in-plane
basis (``axes``, ``centroid``) over the hull bounding box plus ``grid_margin``.

- Support: cells with inlier pixels (closed by ``support_closing``) inside the
  plane hull. Unobserved cells (e.g. object shadows) are never placeable.
- Obstacles: cells with at least ``obstacle_min_points`` valid points between
  max(``obstacle_min_height``, ``obstacle_k`` * sigma(Z)) and the object
  height + ``height_margin`` above the plane; components smaller than
  ``speckle_cells`` are dropped. sigma(Z) is the detector's noise model.
- Overhead: the hull of any other horizontal plane (object top, upper shelf
  board) whose centroid is ``overhead_min_gap`` to object height +
  ``height_margin`` above the support is blocked.
- Placeable centers keep the circumscribed footprint radius r (valid for any
  yaw) + ``edge_margin`` from the support edge and r + ``obstacle_margin``
  from obstacles. ``clearance`` is the distance to the nearest blocking cell.

Candidates are local clearance maxima, largest first (ties: closer to the
support center; across planes: closer to the frame origin), or with ``near``
all placeable cells by distance to it (clearance may then be minimal); they
are spaced max(r, ``min_spacing``) apart per plane, ``max_candidates`` in
total. ``score`` is the clearance, or minus the distance with ``near``.
``PlaceResult.grids`` holds the rasters of every support.

ROS free (numpy / scipy only); the inputs are the planes of
``aha_perception.planes`` and the organized cloud they were detected in.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    label,
    maximum_filter,
)

from .planes.geometry import convex_hull_2d
from .planes.types import DepthNoiseModel, Plane, PlaneLabel

DEFAULT_SUPPORT_LABELS = (PlaneLabel.TABLE.value, PlaneLabel.SHELF.value)
SUPPORT_LABELS = tuple(
    label.value for label in PlaneLabel if label is not PlaneLabel.WALL
)
_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class PlaceOptions:
    """Grid and margins of ``find_place_candidates`` (lengths in m).

    Defaults were chosen on the synthetic place study
    (tools/perception/plane_bench/results/place.md).
    """

    # Grid cell size and the extent added around the plane hull.
    resolution: float = 0.01
    grid_margin: float = 0.05
    # Closing radius (cells) of the inlier raster; fills gaps between pixel
    # footprints on far or grazing surfaces.
    support_closing: int = 2
    # Obstacle band above the plane: from max(obstacle_min_height,
    # obstacle_k * sigma(Z)) to the object height + height_margin. A low k
    # keeps low objects far away (sigma of the detector's model).
    obstacle_min_height: float = 0.02
    obstacle_k: float = 1.5
    height_margin: float = 0.03
    # A cell is an obstacle with this many band points; obstacle components
    # smaller than speckle_cells cells are noise.
    obstacle_min_points: int = 2
    speckle_cells: int = 3
    # Extra distance from the footprint circle to the support edge (or an
    # unobserved cell) and to obstacles.
    edge_margin: float = 0.02
    obstacle_margin: float = 0.02
    # Another horizontal plane at least this much above the support counts as
    # overhead (shelf board, object top) when lower than the band top.
    overhead_min_gap: float = 0.03
    horizontal_tol_deg: float = 10.0
    # Candidate spacing is max(footprint radius, min_spacing).
    min_spacing: float = 0.05
    max_candidates: int = 10

    def for_input_stride(self, stride: int) -> PlaceOptions:
        """Options for a cloud of every ``stride``-th row and column of the image.

        ``obstacle_min_points`` follows the point density (divided by
        stride², at least 1) and ``support_closing`` grows by stride - 1 cells
        for the wider gaps between pixel footprints (scaling it by stride also
        bridged object shadows in the synthetic place study, run_place.py).
        """
        if stride <= 1:
            return self
        return dataclasses.replace(
            self,
            obstacle_min_points=max(1, math.ceil(self.obstacle_min_points / stride**2)),
            support_closing=self.support_closing + stride - 1,
        )


@dataclass
class PlaceGrid:
    """Rasters of one support plane; cell (i, j) is along (axes[0], axes[1]).

    Cell centers are ``origin + (lower + (i, j) * resolution) @ axes``.
    ``clearance`` is the distance (m) from a cell center to the nearest
    obstacle, overhead or non-support cell (0 outside the support).
    """

    plane_index: int
    label: str
    origin: np.ndarray
    axes: np.ndarray
    normal: np.ndarray
    lower: np.ndarray
    resolution: float
    support: np.ndarray
    observed: np.ndarray
    obstacle: np.ndarray
    overhead: np.ndarray
    placeable: np.ndarray
    clearance: np.ndarray

    def cell_points(self, cells: np.ndarray) -> np.ndarray:
        """(N, 3) points of (N, 2) integer cells."""
        uv = self.lower + np.asarray(cells, dtype=np.float64) * self.resolution
        return self.origin + uv @ self.axes

    def point_cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) cells of (N, 3) points projected on the plane, and in-grid mask."""
        uv = (np.asarray(points, dtype=np.float64) - self.origin) @ self.axes.T
        cells = np.round((uv - self.lower) / self.resolution).astype(np.int64)
        inside = (cells >= 0).all(axis=1) & (cells < self.support.shape).all(axis=1)
        return cells, inside

    def sample(self, raster: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Values of a boolean raster at (N, 3) points; False outside the grid."""
        cells, inside = self.point_cells(points)
        out = np.zeros(len(cells), dtype=bool)
        out[inside] = raster[cells[inside, 0], cells[inside, 1]]
        return out


@dataclass
class PlaceCandidate:
    position: np.ndarray  # on the surface, output frame
    normal: np.ndarray  # surface normal (up)
    plane_index: int
    label: str
    surface_height: float
    clearance: float  # m from the center to the nearest blocking cell
    score: float  # higher first: clearance, or minus the distance to ``near``
    distance: float = float("nan")  # to ``near`` (m)


@dataclass
class PlaceResult:
    candidates: list[PlaceCandidate]
    grids: list[PlaceGrid]
    radius: float  # circumscribed footprint radius (m)
    supports: list[int] = field(default_factory=list)  # plane indices considered


def base_cloud(cloud: np.ndarray, camera_to_base: np.ndarray):
    """Output-frame points and optical depth of an organized camera cloud."""
    cloud = np.asarray(cloud, dtype=np.float64)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    points = cloud @ transform[:3, :3].T + transform[:3, 3]
    return points, cloud[..., 2].copy()


def footprint_radius(object_size: Sequence[float]) -> float:
    """Circumscribed radius of the (x, y) footprint, valid for any yaw."""
    return 0.5 * float(np.hypot(object_size[0], object_size[1]))


def _disk(radius: int) -> np.ndarray:
    r = max(int(radius), 0)
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r + r


def convex_mask(polygon: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(len(u), len(v)) mask of cell centers inside a convex 2D polygon."""
    inside = np.ones((len(u), len(v)), dtype=bool)
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return np.zeros_like(inside)
    x, y = polygon[:, 0], polygon[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    sign = 1.0 if area2 >= 0 else -1.0
    uu, vv = u[:, None], v[None, :]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        cross = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
        inside &= sign * cross >= -1e-12
    return inside


def _up_plane(plane: Plane, up: np.ndarray):
    """(normal, offset) with the normal on the side of ``up``."""
    normal = np.asarray(plane.normal, dtype=np.float64)
    if float(normal @ up) < 0.0:
        return -normal, -float(plane.offset)
    return normal, float(plane.offset)


def _sorted_pick(points, order, spacing, limit) -> list[int]:
    """Greedy non-maximum suppression over cells in ``order``."""
    remaining = np.asarray(order)
    picked: list[int] = []
    while len(remaining) and len(picked) < limit:
        best = int(remaining[0])
        picked.append(best)
        dist = np.linalg.norm(points[remaining] - points[best], axis=1)
        remaining = remaining[dist >= spacing]
    return picked


class _Cloud:
    """Finite points of the organized cloud with their noise thresholds."""

    def __init__(self, points, depth, noise, options):
        flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.flat = flat
        finite = np.isfinite(flat[:, 2])
        z = np.asarray(depth, dtype=np.float64).reshape(-1)[finite]
        self.points = flat[finite]
        sigma = noise.sigma(np.where(np.isfinite(z), z, 0.0))
        self.band_low = np.maximum(
            options.obstacle_min_height, options.obstacle_k * sigma
        )


def _support_grid(plane, index, cloud, horizontal, size, radius, options, up):
    res = options.resolution
    normal, offset = _up_plane(plane, up)
    inlier_points = cloud.flat[np.asarray(plane.inliers, dtype=np.int64)]
    inlier_points = inlier_points[np.isfinite(inlier_points[:, 2])]
    if len(inlier_points) < 3 or len(plane.hull) < 3:
        return None
    origin = np.asarray(plane.centroid, dtype=np.float64)
    axes = np.asarray(plane.axes, dtype=np.float64)
    lower = plane.hull.min(axis=0) - options.grid_margin
    upper = plane.hull.max(axis=0) + options.grid_margin
    shape = tuple(int(n) for n in np.ceil((upper - lower) / res).astype(int) + 1)
    u = lower[0] + res * np.arange(shape[0])
    v = lower[1] + res * np.arange(shape[1])
    shift = origin @ axes.T + lower

    def cell_counts(points):
        uv = (points @ axes.T - shift) / res
        i = np.rint(uv[:, 0]).astype(np.int64)
        j = np.rint(uv[:, 1]).astype(np.int64)
        inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        flat = i[inside] * shape[1] + j[inside]
        return np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)

    observed = cell_counts(inlier_points) > 0
    hull_mask = convex_mask(plane.hull, u, v)
    closed = binary_closing(observed, _disk(options.support_closing))
    support = closed & hull_mask

    top = size[2] + options.height_margin
    height = cloud.points @ normal + offset
    band = (height >= cloud.band_low) & (height <= top)
    obstacle = cell_counts(cloud.points[band]) >= options.obstacle_min_points
    if options.speckle_cells > 1 and obstacle.any():
        components, _ = label(obstacle, structure=_EIGHT)
        sizes = np.bincount(components.ravel())
        keep = sizes >= options.speckle_cells
        keep[0] = False
        obstacle = keep[components]

    overhead = np.zeros(shape, dtype=bool)
    own_height = float(origin @ up)
    for j, other in horizontal:
        if j == index:
            continue
        gap = float(np.asarray(other.centroid) @ up) - own_height
        if not options.overhead_min_gap <= gap <= top:
            continue
        polygon, area = convex_hull_2d((other.hull_points() - origin) @ axes.T)
        if area > 0.0:
            overhead |= convex_mask(polygon, u, v)

    blocked = obstacle | overhead
    edge = distance_transform_edt(support) * res - 0.5 * res
    if blocked.any():
        free = distance_transform_edt(~blocked) * res - 0.5 * res
    else:
        free = np.full(shape, np.inf)
    clearance = np.where(support, np.minimum(edge, free), 0.0)
    placeable = (
        support
        & (edge >= radius + options.edge_margin)
        & (free >= radius + options.obstacle_margin)
    )
    return PlaceGrid(
        plane_index=index,
        label=PlaneLabel(plane.label).value,
        origin=origin,
        axes=axes,
        normal=normal,
        lower=lower,
        resolution=res,
        support=support,
        observed=observed,
        obstacle=obstacle,
        overhead=overhead,
        placeable=placeable,
        clearance=clearance,
    )


def _grid_candidates(grid, radius, near, limit, options, up) -> list[PlaceCandidate]:
    cells = np.argwhere(grid.placeable)
    if len(cells) == 0:
        return []
    points = grid.cell_points(cells)
    clearance = grid.clearance[cells[:, 0], cells[:, 1]]
    if near is None:
        # Local maxima of the clearance (ridges of the free space), largest
        # first; ties go to the cell nearest the support center.
        masked = np.where(grid.placeable, grid.clearance, -np.inf)
        peak = masked >= maximum_filter(masked, size=3, mode="constant", cval=-np.inf)
        keep = peak[cells[:, 0], cells[:, 1]]
        cells, points, clearance = cells[keep], points[keep], clearance[keep]
        center = grid.cell_points(np.argwhere(grid.support).mean(axis=0)[None])[0]
        order = np.lexsort(
            (np.linalg.norm(points - center, axis=1), -np.round(clearance, 4))
        )
        distance = np.full(len(cells), np.nan)
    else:
        distance = np.linalg.norm(points - near, axis=1)
        order = np.lexsort((-clearance, distance))
    spacing = max(radius, options.min_spacing)
    picked = _sorted_pick(points, order, spacing, limit)
    out = []
    for k in picked:
        out.append(
            PlaceCandidate(
                position=points[k],
                normal=grid.normal.copy(),
                plane_index=grid.plane_index,
                label=grid.label,
                surface_height=float(points[k] @ up),
                clearance=float(clearance[k]),
                score=float(clearance[k] if near is None else -distance[k]),
                distance=float(distance[k]),
            )
        )
    return out


def find_place_candidates(
    planes: Sequence[Plane],
    points: np.ndarray,
    depth: np.ndarray,
    object_size: Sequence[float],
    noise: DepthNoiseModel | None = None,
    options: PlaceOptions | None = None,
    labels: Iterable[str] | None = None,
    near: Sequence[float] | None = None,
    max_candidates: int | None = None,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    plane_indices: Iterable[int] | None = None,
) -> PlaceResult:
    """Where an object of ``object_size`` (x, y footprint, z height; m) fits.

    ``planes`` are labeled planes in the output frame (the node's
    ``select_output_planes``); horizontal planes facing up with a label in
    ``labels`` (default table and shelf) are supports, every horizontal plane
    can be overhead. ``points`` is the organized (H, W, 3) cloud the planes
    were detected in, transformed to the output frame (NaN = invalid), and
    ``depth`` its (H, W) optical depth for ``noise`` (the detector's model).

    Candidates are ordered by clearance (largest first) or, with ``near``
    (output frame), by distance to it among all placeable cells. At most
    ``max_candidates`` (default ``options.max_candidates``) are returned, at
    least max(radius, min_spacing) apart on each plane.

    ``plane_indices`` restricts the supports to these indices of ``planes``
    (e.g. the planes of tracked surfaces); ``labels`` then defaults to every
    support label.
    """
    options = options or PlaceOptions()
    noise = noise or DepthNoiseModel()
    size = tuple(float(s) for s in object_size)
    if len(size) != 3 or not all(np.isfinite(size)) or min(size) <= 0.0:
        raise ValueError(f"object size must be 3 positive values, got {object_size}")
    allowed = None if plane_indices is None else {int(i) for i in plane_indices}
    default = DEFAULT_SUPPORT_LABELS if allowed is None else SUPPORT_LABELS
    wanted = tuple(labels or ()) or default
    unknown = [name for name in wanted if name not in SUPPORT_LABELS]
    if unknown:
        raise ValueError(f"unknown support labels {unknown}, use {SUPPORT_LABELS}")
    limit = options.max_candidates if not max_candidates else int(max_candidates)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    near = None if near is None else np.asarray(near, dtype=np.float64)
    radius = footprint_radius(size)

    min_cos = np.cos(np.radians(options.horizontal_tol_deg))
    horizontal = [
        (i, p) for i, p in enumerate(planes) if abs(float(p.normal @ up)) >= min_cos
    ]
    supports = [
        i
        for i, p in horizontal
        if float(p.normal @ up) > 0.0
        and PlaneLabel(p.label).value in wanted
        and (allowed is None or i in allowed)
    ]
    cloud = _Cloud(points, depth, noise, options)
    grids, ranked = [], []
    for i in supports:
        grid = _support_grid(planes[i], i, cloud, horizontal, size, radius, options, up)
        if grid is None:
            continue
        grids.append(grid)
        # Equal clearances (to the grid resolution) go to the plane nearest
        # the origin of the output frame (the robot), then keep the plane order.
        reach = float(np.linalg.norm(grid.origin - (grid.origin @ up) * up))
        for rank, c in enumerate(
            _grid_candidates(grid, radius, near, limit, options, up)
        ):
            if near is None:
                key = (-round(c.clearance / options.resolution), reach, i, rank)
            else:
                key = (c.distance, -c.clearance)
            ranked.append((key, c))
    ranked.sort(key=lambda item: item[0])
    candidates = [c for _, c in ranked[:limit]]
    return PlaceResult(candidates, grids, radius, supports)
