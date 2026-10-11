"""Place candidates and obstacle boxes on detected support planes.

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
  yaw) + ``edge_margin`` from the support edge and r + ``obstacle_margin`` +
  the requested ``clearance`` (free space wanted around the object, e.g. the
  5 cm between items of a breakfast setting) from obstacles and overhead
  cells. The request's clearance does not grow the edge distance: it is
  about the neighbouring objects, and the edge keeps its own margin (a
  narrow shelf board would lose most spots otherwise). The output
  ``clearance`` is the distance to the nearest blocking cell.

Candidates are local clearance maxima, largest first (ties: closer to the
support center; across planes: closer to the frame origin), or with ``near``
all placeable cells by distance to it (clearance may then be minimal); they
are spaced max(r, ``min_spacing``) apart per plane, ``max_candidates`` in
total. ``score`` is the clearance, or minus the distance with ``near``.
``PlaceResult.grids`` holds the rasters of every support.

``find_place_obstacles`` turns what stands on each horizontal plane into boxes
for collision checking: connected components of the cells with points in a
planning band above the plane, each as the minimum-area rectangle of its
points and a high percentile of their heights. Unobserved cells are not
obstacles.

Both take ``exclude`` boxes (``ExcludeBox``: the robot's arms and the held
object): points inside them, or with ``viewpoint`` (the camera) also behind
them on the line of sight (depth edge noise of their silhouettes), are
dropped first, so an object touching the held object stays a separate box.
For the candidates, cells whose column above the support (up to the object
band top) meets an exclude box are unobserved as well. 17 boxes with a
viewpoint added about 0.2 s on a full 640x480 cloud (uniform random points).

ROS free (numpy / scipy only); the inputs are the planes of
``aha_perception.planes`` and the organized cloud they were detected in.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

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
# Inliers of these planes are structure, never part of an obstacle box.
STRUCTURE_LABELS = (
    PlaneLabel.FLOOR.value,
    PlaneLabel.TABLE.value,
    PlaneLabel.SHELF.value,
    PlaneLabel.WALL.value,
)
DEFAULT_HEIGHT_TOLERANCE = 0.03
_EIGHT = np.ones((3, 3), dtype=bool)


class PlaceStatus(IntEnum):
    """Same values as the STATUS_* constants of aha_msgs/srv/GetPlaceCandidates."""

    OK = 0
    NO_SURFACE = 1  # no support plane matches the labels (height, plane indices)
    NO_SPACE = 2  # supports, but no placeable spot for the size
    ERROR = 3


@dataclass(frozen=True)
class PlaceOptions:
    """Grid and margins of ``find_place_candidates`` and ``find_place_obstacles``
    (lengths in m).

    Candidate defaults were chosen on the synthetic place study
    (tools/perception/plane_bench/results/place.md); the box_* / cap_* /
    wall_k defaults were not part of it.
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
    # Obstacle boxes: planes with these labels get boxes (not the floor: the
    # robot base and furniture legs would become boxes). The band starts as
    # the obstacle band and ends at box_max_height, or cap_margin below a
    # horizontal plane above that covers at least cap_min_overlap of the
    # support hull (the board above, with its front edge).
    box_labels: tuple[str, ...] = (
        PlaneLabel.TABLE.value,
        PlaneLabel.SHELF.value,
        PlaneLabel.OTHER.value,
    )
    box_max_height: float = 0.6
    cap_margin: float = 0.05
    cap_min_overlap: float = 0.3
    # Components with fewer cells or points are noise; the box height is this
    # percentile of the point heights above the plane.
    box_min_cells: int = 4
    box_min_points: int = 10
    box_height_percentile: float = 97.0
    # A component filling less than box_min_fill of its rectangle (L / U
    # shapes: shelf panels, objects touching them) is split into tiles of
    # box_split_size along the rectangle, one box per tile.
    box_min_fill: float = 0.5
    box_split_size: float = 0.1
    # Points within max(obstacle_min_height, wall_k * sigma(Z)) of a wall
    # (inside its hull) are wall noise, not boxes.
    wall_k: float = 3.0

    def for_input_stride(self, stride: int) -> PlaceOptions:
        """Options for a cloud of every ``stride``-th row and column of the image.

        ``obstacle_min_points`` and ``box_min_points`` follow the point
        density (divided by stride², at least 1) and ``support_closing`` grows
        by stride - 1 cells for the wider gaps between pixel footprints
        (scaling it by stride also bridged object shadows in the synthetic
        place study, run_place.py).
        """
        if stride <= 1:
            return self

        def per_point(count: int) -> int:
            return max(1, math.ceil(count / stride**2))

        return dataclasses.replace(
            self,
            obstacle_min_points=per_point(self.obstacle_min_points),
            box_min_points=per_point(self.box_min_points),
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
    status: PlaceStatus = PlaceStatus.OK


@dataclass
class PlaceObstacle:
    """Box standing on a support plane, in the output frame."""

    center: np.ndarray  # box center, half the height above the surface
    rotation: np.ndarray  # 3x3, columns: length axis, width axis, up
    size: np.ndarray  # length, width, height (m)
    plane_index: int
    points: int  # cloud points in the component

    @property
    def yaw(self) -> float:
        """Angle of the length axis about +z (output frame with z up)."""
        return math.atan2(float(self.rotation[1, 0]), float(self.rotation[0, 0]))


@dataclass(frozen=True)
class ExcludeBox:
    """Oriented box in the output frame whose points are not used."""

    center: np.ndarray
    rotation: np.ndarray  # 3x3, columns: box axes
    size: np.ndarray  # full extents along the axes (m)

    @classmethod
    def of(cls, center, rotation, size) -> ExcludeBox:
        return cls(
            np.asarray(center, dtype=np.float64).reshape(3),
            np.asarray(rotation, dtype=np.float64).reshape(3, 3),
            np.abs(np.asarray(size, dtype=np.float64).reshape(3)),
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        local = (np.asarray(points, dtype=np.float64) - self.center) @ self.rotation
        return (np.abs(local) <= self.size / 2).all(axis=-1)

    def hits_segments(self, starts: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Whether the segments ``starts`` .. ``starts + delta`` meet the box
        (``starts`` and ``delta`` are (N, 3) or (3,), broadcast)."""
        o = (np.asarray(starts, dtype=np.float64) - self.center) @ self.rotation
        d = np.asarray(delta, dtype=np.float64) @ self.rotation
        o, d = np.broadcast_arrays(np.atleast_2d(o), np.atleast_2d(d))
        half = self.size / 2
        lo = np.zeros(len(o))
        hi = np.ones(len(o))
        hit = np.ones(len(o), dtype=bool)
        for k in range(3):
            flat = np.abs(d[:, k]) < 1e-12
            hit &= ~flat | (np.abs(o[:, k]) <= half[k])
            step = np.where(flat, 1.0, d[:, k])
            t1 = np.where(flat, -np.inf, (-half[k] - o[:, k]) / step)
            t2 = np.where(flat, np.inf, (half[k] - o[:, k]) / step)
            lo = np.maximum(lo, np.minimum(t1, t2))
            hi = np.minimum(hi, np.maximum(t1, t2))
        return hit & (lo <= hi)

    def hides(self, viewpoint, points: np.ndarray) -> np.ndarray:
        """Whether the box is on the line of sight from ``viewpoint`` to each
        point (or contains it): the camera sees such points only through the
        box's silhouette, as depth edge noise (flying pixels)."""
        viewpoint = np.asarray(viewpoint, dtype=np.float64).reshape(3)
        return self.hits_segments(viewpoint, np.asarray(points) - viewpoint)


def base_cloud(cloud: np.ndarray, camera_to_base: np.ndarray):
    """Output-frame points and optical depth of an organized camera cloud."""
    cloud = np.asarray(cloud, dtype=np.float64)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    points = cloud @ transform[:3, :3].T + transform[:3, 3]
    return points, cloud[..., 2].copy()


def footprint_radius(object_size: Sequence[float]) -> float:
    """Circumscribed radius of the (x, y) footprint, valid for any yaw."""
    return 0.5 * float(np.hypot(object_size[0], object_size[1]))


def min_area_rectangle(
    points: np.ndarray,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Minimum-area rectangle of (N, 2) points.

    Returns (center, angle of the first side, (length along it, width)).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("no points")
    hull, _ = convex_hull_2d(pts)
    if len(hull) < 3:
        hull = pts  # collinear: any edge lies on the line
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    edges, lengths = edges[lengths > 1e-12], lengths[lengths > 1e-12]
    if len(edges) == 0:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (lo + hi) / 2, 0.0, (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    along = edges / lengths[:, None]
    across = np.column_stack([-along[:, 1], along[:, 0]])
    u, v = along @ hull.T, across @ hull.T  # (edges, hull points)
    lo = np.column_stack([u.min(axis=1), v.min(axis=1)])
    hi = np.column_stack([u.max(axis=1), v.max(axis=1)])
    k = int(np.argmin(np.prod(hi - lo, axis=1)))
    mid = (lo[k] + hi[k]) / 2
    center = mid[0] * along[k] + mid[1] * across[k]
    angle = math.atan2(along[k, 1], along[k, 0])
    return center, angle, (float(hi[k, 0] - lo[k, 0]), float(hi[k, 1] - lo[k, 1]))


def _disk(radius: int) -> np.ndarray:
    r = max(int(radius), 0)
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r + r


def _inside_convex(polygon, x, y, margin: float = 0.0) -> np.ndarray:
    """Whether points (broadcast ``x``, ``y``) are inside a convex 2D polygon
    grown by ``margin``."""
    polygon = np.asarray(polygon, dtype=np.float64)
    inside = np.ones(np.broadcast(x, y).shape, dtype=bool)
    if len(polygon) < 3:
        return np.zeros_like(inside)
    px, py = polygon[:, 0], polygon[:, 1]
    area2 = float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
    sign = 1.0 if area2 >= 0 else -1.0
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        inside &= sign * cross >= -1e-12 - margin * length
    return inside


def convex_mask(polygon: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(len(u), len(v)) mask of cell centers inside a convex 2D polygon."""
    return _inside_convex(polygon, u[:, None], v[None, :])


def _up_plane(plane: Plane, up: np.ndarray):
    """(normal, offset) with the normal on the side of ``up``."""
    normal = np.asarray(plane.normal, dtype=np.float64)
    if float(normal @ up) < 0.0:
        return -normal, -float(plane.offset)
    return normal, float(plane.offset)


def _horizontal(planes, up, options) -> list[tuple[int, Plane]]:
    min_cos = np.cos(np.radians(options.horizontal_tol_deg))
    return [
        (i, p) for i, p in enumerate(planes) if abs(float(p.normal @ up)) >= min_cos
    ]


def _unit(up) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64)
    return up / np.linalg.norm(up)


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


def _excluded(points, exclude, viewpoint=None) -> np.ndarray:
    """Mask of (N, 3) points inside the boxes, or with ``viewpoint`` hidden by
    them. A cheap test against each box's circumscribed sphere (distance, or
    the cone of view from ``viewpoint``) selects the points tested exactly."""
    out = np.zeros(len(points), dtype=bool)
    if viewpoint is not None:
        viewpoint = np.asarray(viewpoint, dtype=np.float64).reshape(3)
        rays = points - viewpoint
        lengths = np.linalg.norm(rays, axis=1)
        directions = rays / np.maximum(lengths, 1e-9)[:, None]
    for box in exclude:
        radius = float(np.linalg.norm(box.size)) / 2
        if viewpoint is None:
            near = np.einsum("ij,ij->i", points - box.center, points - box.center)
            index = np.flatnonzero(near <= radius * radius)
            out[index[box.contains(points[index])]] = True
            continue
        to_box = box.center - viewpoint
        distance = float(np.linalg.norm(to_box))
        if distance <= radius:
            index = np.arange(len(points))
        else:
            cos_cone = math.sqrt(1.0 - (radius / distance) ** 2)
            index = np.flatnonzero(
                (directions @ (to_box / distance) >= cos_cone)
                & (lengths >= distance - radius)
            )
        out[index[box.hides(viewpoint, points[index])]] = True
    return out


class _Cloud:
    """Finite points of the organized cloud outside the exclude boxes, with
    their noise thresholds."""

    def __init__(self, points, depth, noise, options, exclude=(), viewpoint=None):
        flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.flat = flat
        finite = np.isfinite(flat[:, 2])
        if exclude:
            index = np.flatnonzero(finite)
            finite[index[_excluded(flat[index], exclude, viewpoint)]] = False
        self.valid = finite  # per flat index
        self.index = np.flatnonzero(finite)  # flat index of each point
        z = np.asarray(depth, dtype=np.float64).reshape(-1)[finite]
        self.points = flat[finite]
        self.sigma = noise.sigma(np.where(np.isfinite(z), z, 0.0))
        self.band_low = np.maximum(
            options.obstacle_min_height, options.obstacle_k * self.sigma
        )


@dataclass
class _PlaneFrame:
    """Grid of one plane: cell (i, j) is at ``origin + (lower + (i, j) * res) @ axes``."""

    origin: np.ndarray
    axes: np.ndarray
    normal: np.ndarray  # facing up
    offset: float
    lower: np.ndarray
    shape: tuple[int, int]
    resolution: float

    @classmethod
    def of(cls, plane: Plane, up: np.ndarray, options: PlaceOptions):
        res = options.resolution
        normal, offset = _up_plane(plane, up)
        lower = plane.hull.min(axis=0) - options.grid_margin
        upper = plane.hull.max(axis=0) + options.grid_margin
        shape = tuple(int(n) for n in np.ceil((upper - lower) / res).astype(int) + 1)
        return cls(
            origin=np.asarray(plane.centroid, dtype=np.float64),
            axes=np.asarray(plane.axes, dtype=np.float64),
            normal=normal,
            offset=offset,
            lower=lower,
            shape=shape,
            resolution=res,
        )

    @property
    def u(self) -> np.ndarray:
        return self.lower[0] + self.resolution * np.arange(self.shape[0])

    @property
    def v(self) -> np.ndarray:
        return self.lower[1] + self.resolution * np.arange(self.shape[1])

    def uv(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.axes.T

    def flat_cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat cell index of each point and the in-grid mask."""
        ij = np.rint((self.uv(points) - self.lower) / self.resolution)
        ij = ij.astype(np.int64)
        inside = (ij >= 0).all(axis=1) & (ij < self.shape).all(axis=1)
        return ij[:, 0] * self.shape[1] + ij[:, 1], inside

    def counts(self, points: np.ndarray) -> np.ndarray:
        flat, inside = self.flat_cells(points)
        size = self.shape[0] * self.shape[1]
        return np.bincount(flat[inside], minlength=size).reshape(self.shape)


def _excluded_columns(frame, top, exclude) -> np.ndarray:
    """Cells whose column from the plane up to ``top`` meets an exclude box."""
    cells = np.indices(frame.shape).reshape(2, -1).T
    uv = frame.lower + cells * frame.resolution
    starts = frame.origin + uv @ frame.axes
    starts = starts - np.outer(starts @ frame.normal + frame.offset, frame.normal)
    out = np.zeros(len(starts), dtype=bool)
    for box in exclude:
        out |= box.hits_segments(starts, frame.normal * top)
    return out.reshape(frame.shape)


def _support_grid(
    plane, index, cloud, horizontal, size, radius, options, up, exclude=(), spacing=0.0
):
    res = options.resolution
    inliers = np.asarray(plane.inliers, dtype=np.int64)
    inlier_points = cloud.flat[inliers[cloud.valid[inliers]]]
    if len(inlier_points) < 3 or len(plane.hull) < 3:
        return None
    frame = _PlaneFrame.of(plane, up, options)
    shape, u, v = frame.shape, frame.u, frame.v
    top = size[2] + options.height_margin

    observed = frame.counts(inlier_points) > 0
    hull_mask = convex_mask(plane.hull, u, v)
    closed = binary_closing(observed, _disk(options.support_closing))
    support = closed & hull_mask
    if exclude:
        # Where the robot is, the support is not known to be free.
        hidden = _excluded_columns(frame, top, exclude)
        observed &= ~hidden
        support &= ~hidden

    height = cloud.points @ frame.normal + frame.offset
    band = (height >= cloud.band_low) & (height <= top)
    obstacle = frame.counts(cloud.points[band]) >= options.obstacle_min_points
    if options.speckle_cells > 1 and obstacle.any():
        components, _ = label(obstacle, structure=_EIGHT)
        sizes = np.bincount(components.ravel())
        keep = sizes >= options.speckle_cells
        keep[0] = False
        obstacle = keep[components]

    overhead = np.zeros(shape, dtype=bool)
    own_height = float(frame.origin @ up)
    for j, other in horizontal:
        if j == index:
            continue
        gap = float(np.asarray(other.centroid) @ up) - own_height
        if not options.overhead_min_gap <= gap <= top:
            continue
        polygon, area = convex_hull_2d(frame.uv(other.hull_points()))
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
        & (free >= radius + options.obstacle_margin + spacing)
    )
    return PlaceGrid(
        plane_index=index,
        label=PlaneLabel(plane.label).value,
        origin=frame.origin,
        axes=frame.axes,
        normal=frame.normal,
        lower=frame.lower,
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
    surface_height: float | None = None,
    height_tolerance: float | None = None,
    exclude: Sequence[ExcludeBox] = (),
    viewpoint: Sequence[float] | None = None,
    plane_indices: Iterable[int] | None = None,
    clearance: float = 0.0,
) -> PlaceResult:
    """Where an object of ``object_size`` (x, y footprint, z height; m) fits.

    ``planes`` are labeled planes in the output frame (the node's
    ``select_output_planes``); horizontal planes facing up with a label in
    ``labels`` (default table and shelf) are supports, every horizontal plane
    can be overhead. With ``surface_height`` only those supports whose
    centroid height (along ``up``) is within ``height_tolerance`` (None or
    <= 0: DEFAULT_HEIGHT_TOLERANCE) of it remain. ``points`` is the organized
    (H, W, 3) cloud the planes were detected in, transformed to the output
    frame (NaN = invalid), and ``depth`` its (H, W) optical depth for
    ``noise`` (the detector's model). Points inside ``exclude`` boxes, or
    behind them seen from ``viewpoint`` (the camera position, output frame),
    are dropped, and support cells whose column up to the band top meets one
    are unobserved.

    Candidates are ordered by clearance (largest first) or, with ``near``
    (output frame), by distance to it among all placeable cells. At most
    ``max_candidates`` (default ``options.max_candidates``) are returned, at
    least max(radius, min_spacing) apart on each plane. ``status`` is
    NO_SURFACE without a support, NO_SPACE with supports but no candidate.

    ``plane_indices`` restricts the supports to these indices of ``planes``
    (e.g. the planes of tracked surfaces); ``labels`` then defaults to every
    support label. ``clearance`` (m, >= 0) is free space required between
    the footprint circle and obstacles (or overhead planes) beyond
    ``options.obstacle_margin``; the support edge distance is unchanged.
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
    if surface_height is not None and not np.isfinite(surface_height):
        raise ValueError(f"surface height must be finite, got {surface_height}")
    clearance = float(clearance)
    if not (np.isfinite(clearance) and clearance >= 0.0):
        raise ValueError(f"clearance must be finite and >= 0, got {clearance}")
    tolerance = (
        float(height_tolerance)
        if height_tolerance is not None and height_tolerance > 0.0
        else DEFAULT_HEIGHT_TOLERANCE
    )
    limit = options.max_candidates if not max_candidates else int(max_candidates)
    up = _unit(up)
    near = None if near is None else np.asarray(near, dtype=np.float64)
    radius = footprint_radius(size)

    horizontal = _horizontal(planes, up, options)
    supports = [
        i
        for i, p in horizontal
        if float(p.normal @ up) > 0.0
        and PlaneLabel(p.label).value in wanted
        and (allowed is None or i in allowed)
    ]
    if surface_height is not None:
        supports = [
            i
            for i in supports
            if abs(float(np.asarray(planes[i].centroid) @ up) - surface_height)
            <= tolerance
        ]
    exclude = tuple(exclude)
    cloud = _Cloud(points, depth, noise, options, exclude, viewpoint)
    grids, ranked = [], []
    for i in supports:
        grid = _support_grid(
            planes[i],
            i,
            cloud,
            horizontal,
            size,
            radius,
            options,
            up,
            exclude,
            clearance,
        )
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
    if not grids:
        status = PlaceStatus.NO_SURFACE
    elif not candidates:
        status = PlaceStatus.NO_SPACE
    else:
        status = PlaceStatus.OK
    return PlaceResult(candidates, grids, radius, supports, status)


def _band_top(index, frame, hull_mask, horizontal, options, up) -> float:
    """Top of the planning band: box_max_height, or cap_margin below a plane
    above that covers at least cap_min_overlap of this plane's hull cells."""
    top = options.box_max_height
    own_height = float(frame.origin @ up)
    hull_cells = int(hull_mask.sum())
    for j, other in horizontal:
        if j == index:
            continue
        gap = float(np.asarray(other.centroid) @ up) - own_height
        if gap < options.overhead_min_gap or gap - options.cap_margin >= top:
            continue
        polygon, area = convex_hull_2d(frame.uv(other.hull_points()))
        if area <= 0.0:
            continue
        overlap = int((convex_mask(polygon, frame.u, frame.v) & hull_mask).sum())
        if overlap >= options.cap_min_overlap * hull_cells:
            top = gap - options.cap_margin
    return top


def _plane_obstacles(plane, index, cloud, keep, horizontal, options, up):
    if len(plane.hull) < 3:
        return []
    frame = _PlaneFrame.of(plane, up, options)
    hull_mask = convex_mask(plane.hull, frame.u, frame.v)
    top = _band_top(index, frame, hull_mask, horizontal, options, up)
    height = cloud.points @ frame.normal + frame.offset
    band = keep & (height >= cloud.band_low) & (height <= top)
    if not band.any():
        return []
    points, heights = cloud.points[band], height[band]
    flat, inside = frame.flat_cells(points)
    points, heights, flat = points[inside], heights[inside], flat[inside]
    size = frame.shape[0] * frame.shape[1]
    counts = np.bincount(flat, minlength=size).reshape(frame.shape)
    components, n = label(counts >= options.obstacle_min_points, structure=_EIGHT)
    if n == 0:
        return []
    cells = np.bincount(components.ravel(), minlength=n + 1)
    # Objects stand on the plane: components outside its hull (the robot's
    # hand in front of a table, things beside it) are not boxes of this plane.
    on_hull = np.bincount(components[hull_mask], minlength=n + 1) > 0
    component = components.ravel()[flat]
    order = np.argsort(component, kind="stable")
    bounds = np.searchsorted(component[order], np.arange(n + 2))
    out = []
    cell_area = frame.resolution**2
    for k in range(1, n + 1):
        members = order[bounds[k] : bounds[k + 1]]
        if (
            cells[k] < options.box_min_cells
            or not on_hull[k]
            or len(members) < options.box_min_points
        ):
            continue
        uv, h = frame.uv(points[members]), heights[members]
        center, angle, (length, width) = min_area_rectangle(uv)
        fill = cells[k] * cell_area / max(length * width, cell_area)
        if fill >= options.box_min_fill or max(length, width) <= options.box_split_size:
            out.append(_box(uv, h, frame, index, options, up))
            continue
        c, s = math.cos(angle), math.sin(angle)
        local = (uv - center) @ np.array([[c, -s], [s, c]])
        tiles = np.floor((local - local.min(axis=0)) / options.box_split_size)
        tiles = tiles.astype(np.int64)
        key = tiles[:, 0] * (int(tiles[:, 1].max()) + 1) + tiles[:, 1]
        for tile in np.unique(key):
            part = key == tile
            if part.sum() >= options.box_min_points:
                out.append(_box(uv[part], h[part], frame, index, options, up))
    return out


def _box(uv, heights, frame, index, options, up) -> PlaceObstacle:
    """Box over points (plane coordinates ``uv``, ``heights`` above the plane)."""
    center_uv, angle, (length, width) = min_area_rectangle(uv)
    box_height = float(np.percentile(heights, options.box_height_percentile))
    axis = math.cos(angle) * frame.axes[0] + math.sin(angle) * frame.axes[1]
    axis = axis - (axis @ up) * up
    axis /= np.linalg.norm(axis)
    rotation = np.column_stack([axis, np.cross(up, axis), up])
    base = frame.origin + center_uv @ frame.axes
    return PlaceObstacle(
        center=base + up * box_height / 2,
        rotation=rotation,
        size=np.array([length, width, box_height]),
        plane_index=index,
        points=int(len(uv)),
    )


def find_place_obstacles(
    planes: Sequence[Plane],
    points: np.ndarray,
    depth: np.ndarray,
    noise: DepthNoiseModel | None = None,
    options: PlaceOptions | None = None,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    exclude: Sequence[ExcludeBox] = (),
    viewpoint: Sequence[float] | None = None,
) -> list[PlaceObstacle]:
    """Boxes of what stands on the horizontal planes facing up whose label is
    in ``options.box_labels`` (inputs as ``find_place_candidates``).

    Per plane, points of the cloud whose height above it is in the band from
    max(obstacle_min_height, obstacle_k * sigma(Z)) to the band top
    (``box_max_height``, lowered to ``cap_margin`` under a covering plane
    above) are rasterized on the plane grid; inliers of structure planes
    (floor, table, shelf, wall), points near walls and points inside (or,
    with ``viewpoint``, behind) the ``exclude`` boxes are left out. Each
    8-connected component of cells with
    ``obstacle_min_points`` points that touches the plane hull becomes a box:
    the minimum-area rectangle of its points on the plane and the
    ``box_height_percentile`` of their heights; components filling less than
    ``box_min_fill`` of the rectangle are split into tiles.
    """
    options = options or PlaceOptions()
    noise = noise or DepthNoiseModel()
    up = _unit(up)
    horizontal = _horizontal(planes, up, options)
    cloud = _Cloud(points, depth, noise, options, tuple(exclude), viewpoint)
    structure = np.zeros(len(cloud.flat), dtype=bool)
    for plane in planes:
        if PlaneLabel(plane.label).value in STRUCTURE_LABELS:
            structure[np.asarray(plane.inliers, dtype=np.int64)] = True
    keep = ~structure[cloud.index]
    for plane in planes:
        if PlaneLabel(plane.label) is not PlaneLabel.WALL or len(plane.hull) < 3:
            continue
        distance = np.abs(cloud.points @ plane.normal + plane.offset)
        margin = np.maximum(options.obstacle_min_height, options.wall_k * cloud.sigma)
        near = np.flatnonzero(keep & (distance <= margin))
        uv = (cloud.points[near] - plane.centroid) @ np.asarray(plane.axes).T
        inside = _inside_convex(plane.hull, uv[:, 0], uv[:, 1], margin[near])
        keep[near[inside]] = False
    out = []
    for i, plane in horizontal:
        if float(plane.normal @ up) <= 0.0:
            continue
        if PlaneLabel(plane.label).value not in options.box_labels:
            continue
        out += _plane_obstacles(plane, i, cloud, keep, horizontal, options, up)
    return out
