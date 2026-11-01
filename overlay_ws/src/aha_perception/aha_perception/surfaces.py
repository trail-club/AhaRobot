"""Surface tracking: stable IDs for the detected planes across detections.

``SurfaceTracker.update`` takes the output planes of one detection (the
node's ``select_output_planes``, in the detection's output frame such as
base_footprint), the transform from that frame to the tracking frame (odom by
default; z up, origin on the floor) and optionally the ``CameraView`` of the
detection. Each surface is a track with a stable ``id``; ``update`` returns
the surface id of every input plane and ``surfaces()`` the current surfaces
in the tracking frame.

Association. Planes and tracks of the same orientation class (horizontal:
normal within ``horizontal_tol_deg`` of up; vertical: within it of the
ground plane; other planes are ignored) are assigned one to one with the
Hungarian method (scipy ``linear_sum_assignment``). A pair is feasible when

- the normals differ by less than ``max_angle_deg``,
- the plane centroid is within the gate of the track plane along the track
  normal (the height difference for horizontal surfaces). The gate is
  ``gate_k`` * sigma(Z) of the noise model at the optical depth of the
  centroid, clamped to [``min_gate``, ``max_gate``], and
- the convex hulls, in the track's plane, overlap by at least
  ``min_overlap`` of the smaller one, or are less than ``max_gap`` apart.
  Overlap over the smaller hull keeps the ID of a surface seen only in part
  and the gap test joins the pieces of a wall or a long table seen during a
  pan; two tables at the same height more than ``max_gap`` apart get
  different IDs. Horizontal surfaces at floor height
  (``ClassifyOptions.floor_max_height``) skip this test: the floor is one
  surface even where the detector splits it around furniture.

The cost is angle / ``max_angle_deg`` + distance / gate + (1 - overlap) +
gap / ``max_gap``, plus ``tentative_cost`` for a tentative track so that a
confirmed one wins. A plane left unassigned that is feasible with a track
assigned in the same detection is a fragment of that surface: its hull is
added to the track and no track is started for it (likewise among the
planes that start new tracks).

Track state. Horizontal surfaces keep the normal up (gravity is known; a
slightly tilted fit over a large hull would move the plane by centimeters
far from the seen part), vertical ones a horizontal normal. The height
(offset), and the normal of vertical surfaces, are an exponential moving
average with weight max(1 / observations, ``smoothing``) (a running mean
for the first observations), scaled down by area / (``full_weight_fraction``
* the track's hull area) for a plane covering less of the surface. The hull
is the convex hull of the new observation and of the previous hull's
vertices and edge points (``hull_spacing`` apart) that the detection gives
no evidence against: points outside the image or the depth range, occluded
points (measured depth shorter than expected) and points whose measured
depth matches the plane are kept; points where the camera sees through the
plane (measured depth longer) are dropped. Without a depth image every point
in view is dropped (latest observation in view, previous hull elsewhere);
without a ``CameraView`` the hull is the latest observation. Hull vertices
within ``HULL_TOLERANCE`` of the chord of their neighbours are dropped. The
label is
the majority of the last ``label_window`` observed labels (ties: the most
recent). ``level`` numbers stacked horizontal surfaces (shelf boards, a
table and a large object top) from 1 (lowest) within each group of
confirmed horizontal surfaces whose footprints overlap (the stacking rule of
``planes.classify`` with ``ClassifyOptions``); 0 when not stacked. It is
recomputed every update.

Lifecycle. An unassigned plane starts a tentative track, which gets an ID
(monotonic from 1, not reused, kept across ``reset``) when it has been
observed in ``confirm_hits`` detections and is dropped after
``tentative_max_misses`` consecutive detections without an observation (a
plane detected every other frame is still confirmed). A confirmed track is
deleted after ``max_misses`` consecutive detections in which it was
expected in view but not observed, or ``max_unseen_s`` seconds (detection
stamps) after it was last observed. Expected in view: at least
``min_visible_fraction`` of ``sample_count`` points inside its hull (and
``min_visible_area`` m^2) are observable, i.e. project into the image
(``view_margin_px`` inside) within the depth range and are neither occluded
nor invalid in the depth image. Tracks out of view are kept. When a plane is feasible with a confirmed track that gets
no plane of the detection as well as with the track it went to, both are one
surface (seen before in disjoint parts, or from far apart viewpoints): they
are merged into the older ID, with the union of the hulls. A stamp more than
``max_stamp_back_s`` older than the previous update resets the tracker.

ROS free (numpy / scipy only).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .place import convex_mask
from .planes.geometry import plane_axes, polygon_area
from .planes.types import ClassifyOptions, DepthNoiseModel, Plane, PlaneLabel

UP = np.array([0.0, 0.0, 1.0])
HULL_TOLERANCE = 0.01  # m, track hull vertices closer to a chord are dropped
_INFEASIBLE = 1e6
UNKNOWN, PRESENT, ABSENT = 0, 1, 2


@dataclass(frozen=True)
class TrackerOptions:
    """Association gates and lifecycle of ``SurfaceTracker`` (m, s, deg)."""

    max_angle_deg: float = 15.0
    min_gate: float = 0.03
    gate_k: float = 2.0
    max_gate: float = 0.08
    min_overlap: float = 0.2
    max_gap: float = 0.05
    tentative_cost: float = 0.5
    smoothing: float = 0.3
    full_weight_fraction: float = 0.5
    label_window: int = 10
    confirm_hits: int = 2
    tentative_max_misses: int = 2
    max_misses: int = 5
    max_unseen_s: float = 120.0
    min_visible_fraction: float = 0.5
    min_visible_area: float = 0.03
    view_margin_px: float = 8.0
    # Depth tolerance of the view tests: max(occlusion_tol, occlusion_k * sigma).
    occlusion_tol: float = 0.05
    occlusion_k: float = 3.0
    # Spacing of the previous hull's edge points tested against the view.
    hull_spacing: float = 0.1
    sample_count: int = 64
    max_stamp_back_s: float = 1.0
    horizontal_tol_deg: float = 10.0
    classify: ClassifyOptions = field(default_factory=ClassifyOptions)


@dataclass
class CameraView:
    """The camera of a detection, for the view tests of the tracks.

    ``depth`` is the measured optical depth (NaN = invalid) of every
    ``stride``-th row and column of the image (the node's organized cloud
    z); the intrinsics are those of the full image.
    """

    camera_to_tracking: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    min_depth: float = 0.3
    max_depth: float = 3.0
    depth: np.ndarray | None = None
    stride: int = 1

    def __post_init__(self):
        transform = np.asarray(self.camera_to_tracking, dtype=np.float64)
        self._rotation = transform[:3, :3]
        self._origin = transform[:3, 3]

    def optical_depth(self, points: np.ndarray) -> np.ndarray:
        local = (np.atleast_2d(points) - self._origin) @ self._rotation
        return local[:, 2]

    def states(
        self, points: np.ndarray, noise: DepthNoiseModel, options: TrackerOptions
    ) -> np.ndarray:
        """UNKNOWN / PRESENT / ABSENT per (N, 3) point of a surface.

        PRESENT: the measured depth matches the point; ABSENT: the camera
        sees through it (measured farther), or it is in view and there is no
        depth image; UNKNOWN: out of the image or depth range, occluded or
        invalid.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        states = np.full(len(points), UNKNOWN, dtype=np.int8)
        if len(points) == 0:
            return states
        local = (points - self._origin) @ self._rotation
        z = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * local[:, 0] / z + self.cx
            v = self.fy * local[:, 1] / z + self.cy
        m = options.view_margin_px
        inside = (z >= self.min_depth) & (z <= self.max_depth)
        inside &= (u >= m) & (u <= self.width - 1 - m)
        inside &= (v >= m) & (v <= self.height - 1 - m)
        if self.depth is None:
            states[inside] = ABSENT
            return states
        depth = np.asarray(self.depth)
        rows, cols = depth.shape
        idx = np.flatnonzero(inside)
        r = np.clip(np.rint(v[idx] / self.stride).astype(np.int64), 0, rows - 1)
        c = np.clip(np.rint(u[idx] / self.stride).astype(np.int64), 0, cols - 1)
        measured = depth[r, c].astype(np.float64)
        expected = z[idx]
        tol = np.maximum(
            options.occlusion_tol, options.occlusion_k * noise.sigma(expected)
        )
        valid = np.isfinite(measured)
        states[idx[valid & (np.abs(measured - expected) <= tol)]] = PRESENT
        states[idx[valid & (measured > expected + tol)]] = ABSENT
        return states


@dataclass
class Surface:
    """A tracked surface in the tracking frame (``normal . p + offset = 0``)."""

    id: int
    label: str
    level: int
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    hull: np.ndarray  # (K, 3) convex hull vertices on the plane, in order
    area: float
    observations: int
    first_seen: float
    last_seen: float
    visible: bool
    vertical: bool

    @property
    def height(self) -> float:
        return float(self.centroid[2])


@dataclass
class _Observation:
    index: int
    vertical: bool
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    hull: np.ndarray  # (K, 3) tracking frame
    label: str
    depth: float
    area: float


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return polygon.mean(axis=0)
    return np.array(
        [((x + xn) * cross).sum() / (6 * area), ((y + yn) * cross).sum() / (6 * area)]
    )


def _half_hull(points) -> list:
    out: list = []
    pop, append = out.pop, out.append
    for x, y in points:
        while len(out) > 1:
            (ax, ay), (bx, by) = out[-2], out[-1]
            if (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0.0:
                break
            pop()
        append((x, y))
    return out


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Counter-clockwise hull vertices and area of (N, 2) points.

    Andrew's monotone chain: for the few hundred points of a track update it
    is faster than ``planes.geometry.convex_hull_2d`` (Qhull).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))].tolist()
    lower = _half_hull(ordered)
    upper = _half_hull(reversed(ordered))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64).reshape(-1, 2)
    if len(hull) < 3:
        return np.empty((0, 2)), 0.0
    return hull, abs(polygon_area(hull))


def _simplify(polygon: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop vertices of a convex polygon within ``tolerance`` of the chord of
    their neighbours (keeps hulls of noisy edges small)."""
    while len(polygon) > 3:
        before = np.roll(polygon, 1, axis=0)
        chord = np.roll(polygon, -1, axis=0) - before
        rel = polygon - before
        cross = np.abs(chord[:, 0] * rel[:, 1] - chord[:, 1] * rel[:, 0])
        dist = cross / np.maximum(np.linalg.norm(chord, axis=1), 1e-12)
        drop = np.zeros(len(polygon), dtype=bool)
        for i in np.argsort(dist):
            if dist[i] >= tolerance or len(polygon) - drop.sum() <= 3:
                break
            if not drop[i - 1] and not drop[(i + 1) % len(polygon)]:
                drop[i] = True
        if not drop.any():
            break
        polygon = polygon[~drop]
    return polygon


def _densify(polygon: np.ndarray, spacing: float) -> np.ndarray:
    """Vertices of a closed polygon plus points along its edges ``spacing`` apart."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / spacing)))
        t = np.arange(n)[:, None] / n
        out.append(a + t * (b - a))
    return np.concatenate(out) if out else polygon


def intersection_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Area of the intersection of two counter-clockwise convex polygons.

    ``planes.geometry.clip_convex_polygons`` (Sutherland-Hodgman) with each
    clip edge applied to all subject vertices at once.
    """
    polygon = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if (polygon.max(axis=0) < clip.min(axis=0)).any() or (
        clip.max(axis=0) < polygon.min(axis=0)
    ).any():
        return 0.0
    edges = np.concatenate([clip[1:], clip[:1]]) - clip
    for a, edge in zip(clip, edges):
        side = edge[0] * (polygon[:, 1] - a[1]) - edge[1] * (polygon[:, 0] - a[0])
        inside = side >= 0
        if inside.all():
            continue
        if not inside.any():
            return 0.0
        following = np.concatenate([polygon[1:], polygon[:1]])
        side_next = np.concatenate([side[1:], side[:1]])
        crossing = inside != (side_next >= 0)
        # Crossing edges have sides of opposite sign: no division by zero.
        t = np.zeros(len(polygon))
        t[crossing] = side[crossing] / (side[crossing] - side_next[crossing])
        points = np.empty((2 * len(polygon), 2))
        points[0::2] = polygon
        points[1::2] = polygon + t[:, None] * (following - polygon)
        keep = np.empty(2 * len(polygon), dtype=bool)
        keep[0::2] = inside
        keep[1::2] = crossing
        polygon = points[keep]
    return abs(polygon_area(polygon))


def _overlap(a: np.ndarray, area_a: float, b: np.ndarray, area_b: float) -> float:
    """Intersection of two convex polygons over the smaller one."""
    small = min(area_a, area_b)
    if small <= 0.0 or len(a) < 3 or len(b) < 3:
        return 0.0
    return intersection_area(a, b) / small


def _point_edge_distance(points: np.ndarray, polygon: np.ndarray) -> float:
    """Smallest distance from (N, 2) points to the edges of a closed polygon."""
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - starts
    rel = points[:, None, :] - starts[None]
    lengths = np.maximum((edges * edges).sum(axis=-1), 1e-12)
    t = np.clip((rel * edges[None]).sum(axis=-1) / lengths, 0.0, 1.0)
    closest = starts[None] + t[..., None] * edges[None]
    return float(np.linalg.norm(points[:, None, :] - closest, axis=-1).min())


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two disjoint polygons (their vertices and edges)."""
    if len(a) == 0 or len(b) == 0:
        return math.inf
    return min(_point_edge_distance(a, b), _point_edge_distance(b, a))


class _Track:
    def __init__(self, obs: _Observation, stamp: float, options: TrackerOptions):
        self.id = 0
        self.vertical = obs.vertical
        self.normal = obs.normal.copy()
        self.offset = float(obs.offset)
        self.labels = deque([obs.label], maxlen=max(1, options.label_window))
        self.hits = 1
        self.misses = 0
        self.first_seen = self.last_seen = stamp
        self.visible = True
        self.level = 0
        self._set_hull(obs.hull, obs.hull)

    # Geometry in the track's plane: origin -offset * normal, plane_axes basis.
    def to_2d(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) + self.offset * self.normal) @ self.axes.T

    def lift(self, uv: np.ndarray) -> np.ndarray:
        return -self.offset * self.normal + np.asarray(uv) @ self.axes

    def _set_hull(self, points: np.ndarray, fallback: np.ndarray):
        self.axes = plane_axes(self.normal, UP)
        polygon, area = convex_hull(self.to_2d(points))
        if area <= 0.0:
            polygon, area = convex_hull(self.to_2d(fallback))
        if area > 0.0:
            polygon = _simplify(polygon, HULL_TOLERANCE)
            area = abs(polygon_area(polygon))
        self.polygon, self.area = polygon, float(area)
        self.hull = self.lift(polygon)
        if len(polygon):
            self.centroid = self.lift(_polygon_centroid(polygon))
        else:
            self.centroid = -self.offset * self.normal

    @property
    def height(self) -> float:
        return float(self.centroid[2])

    @property
    def label(self) -> str:
        counts = Counter(self.labels)
        best = max(counts.values())
        return next(label for label in reversed(self.labels) if counts[label] == best)

    def distance(self, point: np.ndarray) -> float:
        return abs(float(self.normal @ point) + self.offset)

    def footprint(self, points: np.ndarray) -> tuple[np.ndarray, float]:
        return convex_hull(self.to_2d(points))

    def overlap(self, points: np.ndarray) -> tuple[float, float]:
        """Overlap over the smaller hull and the gap between the hulls (m)."""
        polygon, area = self.footprint(points)
        if len(polygon) == 0:
            polygon = self.to_2d(points)
        overlap = _overlap(polygon, area, self.polygon, self.area)
        gap = 0.0 if overlap > 0.0 else _gap(polygon, self.polygon)
        return overlap, gap

    def samples(self, count: int) -> np.ndarray:
        """About ``count`` points inside the hull (3D), plus the centroid."""
        if self.area <= 0.0 or len(self.polygon) < 3:
            return self.centroid[None]
        lower, upper = self.polygon.min(axis=0), self.polygon.max(axis=0)
        step = max(math.sqrt(self.area / max(count, 1)), 1e-3)
        u = np.arange(lower[0] + step / 2, upper[0], step)
        v = np.arange(lower[1] + step / 2, upper[1], step)
        inside = convex_mask(self.polygon, u, v)
        i, j = np.nonzero(inside)
        uv = np.column_stack([u[i], v[j]])
        return np.vstack([self.lift(uv), self.centroid[None]])

    def expected(self, view: CameraView, noise: DepthNoiseModel, options) -> bool:
        states = view.states(self.samples(options.sample_count), noise, options)
        fraction = float(np.mean(states != UNKNOWN))
        return (
            fraction >= options.min_visible_fraction
            and fraction * self.area >= options.min_visible_area
        )

    def observe(self, group, stamp, view, noise, options):
        """Update with the observations of one detection (primary first)."""
        primary = group[0]
        weight = max(1.0 / (self.hits + 1), options.smoothing)
        if self.area > 0.0:
            cover = primary.area / (options.full_weight_fraction * self.area)
            weight *= min(1.0, cover)
        normal = (1.0 - weight) * self.normal + weight * primary.normal
        normal /= np.linalg.norm(normal)
        offset = (1.0 - weight) * self.offset + weight * -float(
            normal @ primary.centroid
        )
        observed = np.concatenate([o.hull for o in group])
        kept = np.empty((0, 3))
        if view is not None and len(self.hull):
            old = self.lift(_densify(self.polygon, options.hull_spacing))
            kept = old[view.states(old, noise, options) != ABSENT]
        self.normal, self.offset = normal, offset
        self._set_hull(np.concatenate([observed, kept]), observed)
        self.labels.append(primary.label)
        self.hits += 1
        self.misses = 0
        self.last_seen = stamp
        self.visible = True

    def extend(self, obs: _Observation):
        """Add the hull of a fragment seen in the same detection."""
        self._set_hull(np.concatenate([self.hull, obs.hull]), self.hull)

    def absorb(self, other: _Track, latest: _Track):
        """Merge ``other`` into this track; ``latest`` has the current plane."""
        self.normal, self.offset = latest.normal.copy(), latest.offset
        points = np.concatenate([self.hull, other.hull])
        self._set_hull(points, latest.hull)
        older = other if latest is self else self
        self.labels = deque(
            list(older.labels) + list(latest.labels), maxlen=self.labels.maxlen
        )
        self.hits += other.hits
        self.misses = 0
        self.first_seen = min(self.first_seen, other.first_seen)
        self.last_seen = max(self.last_seen, other.last_seen)
        self.visible = self.visible or other.visible

    def surface(self) -> Surface:
        return Surface(
            id=self.id,
            label=self.label,
            level=self.level,
            normal=self.normal.copy(),
            offset=float(self.offset),
            centroid=self.centroid.copy(),
            hull=self.hull.copy(),
            area=self.area,
            observations=self.hits,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            visible=self.visible,
            vertical=self.vertical,
        )


class SurfaceTracker:
    """Keeps surfaces with stable IDs over detections (see the module doc)."""

    def __init__(self, options: TrackerOptions | None = None):
        self.options = options or TrackerOptions()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._stamp: float | None = None

    def reset(self):
        """Forget every surface; new IDs continue after the last one."""
        self._tracks = []
        self._stamp = None

    def surfaces(self) -> list[Surface]:
        """Confirmed surfaces sorted by id."""
        tracks = sorted((t for t in self._tracks if t.id), key=lambda t: t.id)
        return [t.surface() for t in tracks]

    def update(
        self,
        planes: Sequence[Plane],
        base_to_tracking: np.ndarray,
        stamp: float,
        view: CameraView | None = None,
        noise: DepthNoiseModel | None = None,
    ) -> list[int]:
        """Add one detection; returns the surface id of each plane (0 = none yet).

        ``planes`` are in the frame that ``base_to_tracking`` (4x4) maps to the
        tracking frame, ``stamp`` is the detection time in seconds and
        ``noise`` the detector's depth noise model (gates and view tests).
        """
        options = self.options
        noise = noise or DepthNoiseModel()
        stamp = float(stamp)
        if self._stamp is not None and stamp < self._stamp - options.max_stamp_back_s:
            self.reset()
        self._stamp = stamp
        transform = np.asarray(base_to_tracking, dtype=np.float64)
        observations = [
            obs
            for obs in (
                self._observation(i, plane, transform, view)
                for i, plane in enumerate(planes)
            )
            if obs is not None
        ]
        tracks = self._tracks
        feasible, cost = self._costs(observations, tracks, noise)

        # Track index -> observation rows, the assigned (primary) one first.
        groups: dict[int, list[int]] = {}
        if observations and tracks:
            for r, c in zip(*linear_sum_assignment(cost)):
                if feasible[r, c]:
                    groups[int(c)] = [int(r)]
        assigned = {rows[0] for rows in groups.values()}
        fresh = []
        for r in range(len(observations)):
            if r in assigned:
                continue
            owners = [c for c in groups if feasible[r, c]]
            if owners:
                groups[min(owners, key=lambda c: cost[r, c])].append(r)
            else:
                fresh.append(observations[r])
        for c, rows in groups.items():
            group = [observations[r] for r in rows]
            tracks[c].observe(group, stamp, view, noise, options)
        root, observed = self._merge(tracks, groups, feasible)

        survivors = []
        for c, track in enumerate(tracks):
            if root(c) != c:
                continue
            if c in observed:
                survivors.append(track)
                continue
            track.visible = False
            if track.id == 0:
                track.misses += 1
                if track.misses < options.tentative_max_misses:
                    survivors.append(track)
                continue
            if stamp - track.last_seen > options.max_unseen_s:
                continue
            if view is not None and track.expected(view, noise, options):
                track.misses += 1
            if track.misses < options.max_misses:
                survivors.append(track)
        # Unassigned planes start tracks; fragments of one surface share one.
        new: list[_Track] = []
        new_owner: dict[int, _Track] = {}
        for obs in fresh:
            track = next((t for t in new if self._pair(obs, t, noise)[0]), None)
            if track is None:
                track = _Track(obs, stamp, options)
                new.append(track)
            else:
                track.extend(obs)
            new_owner[obs.index] = track
        self._tracks = survivors + new
        for track in self._tracks:
            if track.id == 0 and track.hits >= options.confirm_hits:
                track.id = self._next_id
                self._next_id += 1
        self._assign_levels()

        owner: dict[int, _Track] = {}
        for c, rows in groups.items():
            for r in rows:
                owner[observations[r].index] = tracks[root(c)]
        owner.update(new_owner)
        return [owner[i].id if i in owner else 0 for i in range(len(planes))]

    def _merge(self, tracks, groups, feasible):
        """Merge unobserved confirmed tracks that an observation was feasible
        with into the track it went to; the older ID is kept.

        Returns (root: track index -> surviving index, observed indices).
        """
        parent = list(range(len(tracks)))

        def root(c):
            while parent[c] != c:
                c = parent[c]
            return c

        observed = set(groups)
        for c, rows in groups.items():
            for r in rows:
                for u in np.flatnonzero(feasible[r]):
                    u = int(u)
                    if u in groups or not tracks[u].id:
                        continue
                    stale, current = root(u), root(c)
                    if stale == current:
                        continue
                    older = tracks[current].id == 0 or (
                        0 < tracks[stale].id < tracks[current].id
                    )
                    keep, drop = (stale, current) if older else (current, stale)
                    tracks[keep].absorb(tracks[drop], latest=tracks[current])
                    parent[drop] = keep
                    observed.add(keep)
        return root, observed

    def _observation(self, index, plane, transform, view) -> _Observation | None:
        rotation, origin = transform[:3, :3], transform[:3, 3]
        normal = rotation @ np.asarray(plane.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        cos = float(normal @ UP)
        tol = math.radians(self.options.horizontal_tol_deg)
        # Gravity is known: horizontal planes face up, vertical ones sideways.
        if abs(cos) >= math.cos(tol):
            vertical = False
            normal = UP.copy()
        elif abs(cos) <= math.sin(tol):
            vertical = True
            normal = normal - cos * UP
            normal /= np.linalg.norm(normal)
        else:
            return None
        centroid = rotation @ np.asarray(plane.centroid, dtype=np.float64) + origin
        hull = plane.hull_points() @ rotation.T + origin
        if view is not None:
            depth = float(view.optical_depth(centroid)[0])
        else:
            depth = float(np.linalg.norm(centroid - origin))
        return _Observation(
            index=index,
            vertical=vertical,
            normal=normal,
            offset=-float(normal @ centroid),
            centroid=centroid,
            hull=hull,
            label=PlaneLabel(plane.label).value,
            depth=depth,
            area=float(plane.area),
        )

    def _gate(self, depth: float, noise: DepthNoiseModel) -> float:
        o = self.options
        return min(o.max_gate, max(o.min_gate, o.gate_k * float(noise.sigma(depth))))

    def _pair(self, obs: _Observation, track: _Track, noise):
        """(feasible, cost) of an observation and a track."""
        o = self.options
        if obs.vertical != track.vertical:
            return False, _INFEASIBLE
        cos = float(np.clip(obs.normal @ track.normal, -1.0, 1.0))
        angle = math.degrees(math.acos(cos))
        if angle > o.max_angle_deg:
            return False, _INFEASIBLE
        gate = self._gate(obs.depth, noise)
        distance = track.distance(obs.centroid)
        if distance > gate:
            return False, _INFEASIBLE
        floor = o.classify.floor_max_height
        if not obs.vertical and max(abs(obs.centroid[2]), abs(track.height)) <= floor:
            overlap, gap = 1.0, 0.0
        else:
            overlap, gap = track.overlap(obs.hull)
            if overlap < o.min_overlap and gap > o.max_gap:
                return False, _INFEASIBLE
        cost = angle / o.max_angle_deg + distance / gate + 1.0 - min(overlap, 1.0)
        cost += gap / o.max_gap
        return True, cost + (o.tentative_cost if track.id == 0 else 0.0)

    def _costs(self, observations, tracks, noise):
        feasible = np.zeros((len(observations), len(tracks)), dtype=bool)
        cost = np.full(feasible.shape, _INFEASIBLE)
        for r, obs in enumerate(observations):
            for c, track in enumerate(tracks):
                feasible[r, c], cost[r, c] = self._pair(obs, track, noise)
        return feasible, cost

    def _assign_levels(self):
        opts = self.options.classify
        for track in self._tracks:
            track.level = 0
        candidates = []
        for track in self._tracks:
            if not track.id or track.vertical:
                continue
            low = track.height <= opts.floor_max_height
            if low or track.area < opts.shelf_min_area:
                continue
            polygon, area = convex_hull(track.hull[:, :2])
            if area > 0.0:
                candidates.append((track, polygon, area))
        parent = list(range(len(candidates)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(len(candidates)):
            ta, pa, aa = candidates[a]
            for b in range(a + 1, len(candidates)):
                tb, pb, ab = candidates[b]
                if abs(ta.height - tb.height) < opts.shelf_min_spacing:
                    continue
                if min(aa, ab) / max(aa, ab) < opts.shelf_min_area_ratio:
                    continue
                if _overlap(pa, aa, pb, ab) >= opts.shelf_min_overlap:
                    parent[root(a)] = root(b)
        groups: dict[int, list[_Track]] = {}
        for i, (track, _, _) in enumerate(candidates):
            groups.setdefault(root(i), []).append(track)
        for members in groups.values():
            if len(members) < 2:
                continue
            for level, track in enumerate(sorted(members, key=lambda t: t.height), 1):
                track.level = level
