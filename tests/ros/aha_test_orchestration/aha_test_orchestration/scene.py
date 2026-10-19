"""Scene summary for the place decision (ROS free, numpy).

Inputs are the tracked surfaces (/aha/perception/surfaces) and the detected
objects (/aha/perception/objects) in one frame (the tracking frame, odom),
plus the target furniture of the perception furniture config. An object is
on a surface when its bottom (center z - height / 2) is within
``height_tolerance`` of the surface height and its center is within
``hull_margin`` of the surface's convex hull (``support_of``). Levels of the
furniture come from the config. The level of an object is, in this order:
the level whose configured height is nearest to the object bottom (within
HEIGHT_GROUP_TOLERANCE) when the object is inside the furniture footprint,
else the level perception associated with the object's support surface
(Surface.furniture / furniture_level); without the footprint (no TF) only
the latter. The first keeps the objects of a
level whose board was not detected (hidden under the objects or the board
above); objects where the two disagree are listed in ``level_conflicts``.

The cabinet only changes when the robot changes it, so a summary can be
kept over several place goals and updated with the robot's own actions
(``with_placed``, ``without_object``) instead of observing again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

SUPPORT_HEIGHT_TOLERANCE = 0.04  # m, object bottom to surface height
HULL_MARGIN = 0.03  # m, object center outside the surface hull
MIN_LEVEL_HEIGHT = 0.1  # m, lower levels (floor height) are not listed
HEIGHT_GROUP_TOLERANCE = 0.05  # m, object bottom to a level height (variant B)
NON_SUPPORT_LABELS = ("wall",)


@dataclass(frozen=True)
class SurfaceInfo:
    id: int
    label: str
    height: float
    centroid: tuple[float, float, float]
    hull: tuple[tuple[float, float], ...]  # convex hull xy, counterclockwise
    furniture: str = ""
    level: int = 0  # furniture level, 0 = not associated
    visible: bool = True

    def extent(self) -> tuple[float, float, float, float]:
        """x min, x max, y min, y max of the hull."""
        if not self.hull:
            x, y = self.centroid[:2]
            return x, x, y, y
        xy = np.asarray(self.hull)
        return (
            float(xy[:, 0].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].min()),
            float(xy[:, 1].max()),
        )

    def distance_xy(self, point) -> float:
        """Horizontal distance of a point to the hull (0 inside)."""
        return polygon_distance(np.asarray(self.hull, dtype=float), point)


@dataclass(frozen=True)
class ObjectInfo:
    id: int
    label: str
    position: tuple[float, float, float]  # center
    size: tuple[float, float, float]  # x, y footprint and z height

    @property
    def bottom(self) -> float:
        return self.position[2] - self.size[2] / 2


@dataclass(frozen=True)
class HeldObject:
    label: str
    size: tuple[float, float, float]
    shape: str = "box"


@dataclass(frozen=True)
class LevelInfo:
    furniture: str
    number: int  # 1 = lowest configured level
    height: float
    max_object_height: float | None  # None: top level
    surface_ids: tuple[int, ...]  # surfaces perception associated with the level
    object_ids: tuple[int, ...]
    # Whether perception found a free spot for the held object: "yes" / "no" /
    # "not seen" / "unknown" (not asked)
    space: str = "unknown"

    @property
    def board_seen(self) -> bool:
        return bool(self.surface_ids)


@dataclass(frozen=True)
class SceneSummary:
    frame: str
    robot_pose: tuple[float, float, float]  # x, y, yaw in frame
    furniture: str
    surfaces: tuple[SurfaceInfo, ...]  # horizontal surfaces (walls excluded)
    objects: tuple[ObjectInfo, ...]
    support: dict = field(default_factory=dict)  # object id -> surface id
    levels: tuple[LevelInfo, ...] = ()
    # Objects inside the furniture footprint grouped by the nearest level
    # height (variant B): level number -> object ids
    by_height: dict = field(default_factory=dict)
    # object id -> (level by height, level of the support surface) where both
    # are known and differ
    level_conflicts: dict = field(default_factory=dict)

    def surface(self, surface_id: int) -> SurfaceInfo | None:
        return next((s for s in self.surfaces if s.id == surface_id), None)

    def object(self, object_id: int) -> ObjectInfo | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def level(self, number: int) -> LevelInfo | None:
        return next((v for v in self.levels if v.number == number), None)

    def level_of_surface(self, surface_id: int) -> int:
        surface = self.surface(surface_id)
        if surface is None or surface.furniture != self.furniture:
            return 0
        return surface.level

    def level_of_object(self, object_id: int) -> int:
        return next((v.number for v in self.levels if object_id in v.object_ids), 0)

    def signature(self) -> tuple:
        """Changes when an object, its label or its level / surface changes."""
        return (
            tuple(
                sorted((o.id, o.label, self.support.get(o.id, 0)) for o in self.objects)
            ),
            tuple((v.number, v.surface_ids, v.object_ids) for v in self.levels),
        )


def with_placed(summary: SceneSummary, obj: ObjectInfo, level: int) -> SceneSummary:
    """The summary after the robot put ``obj`` on ``level`` (0: not on a level);
    the space hints are dropped (they were for the previous held object)."""
    levels = tuple(
        replace(
            v,
            object_ids=v.object_ids + ((obj.id,) if v.number == level else ()),
            space="unknown",
        )
        for v in summary.levels
    )
    by_height = dict(summary.by_height)
    if level in by_height:
        by_height[level] = by_height[level] + (obj.id,)
    return replace(
        summary,
        objects=summary.objects + (obj,),
        support={**summary.support, obj.id: 0},
        levels=levels,
        by_height=by_height,
    )


def without_object(summary: SceneSummary, object_id: int) -> SceneSummary:
    """The summary after the robot took the object away (e.g. the milk)."""
    levels = tuple(
        replace(
            v,
            object_ids=tuple(i for i in v.object_ids if i != object_id),
            space="unknown",
        )
        for v in summary.levels
    )
    return replace(
        summary,
        objects=tuple(o for o in summary.objects if o.id != object_id),
        support={k: v for k, v in summary.support.items() if k != object_id},
        levels=levels,
        by_height={
            k: tuple(i for i in v if i != object_id)
            for k, v in summary.by_height.items()
        },
        level_conflicts={
            k: v for k, v in summary.level_conflicts.items() if k != object_id
        },
    )


def split_groups(
    summary: SceneSummary,
    target_id: int,
    category_of,
    level: int = 0,
    surface_id: int = 0,
) -> tuple[
    tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]
]:
    """(same, other): centers of the other objects next to the target object
    (on ``level``, else on ``surface_id``) in its category (PlaceObject's
    ``group``) and not (its ``avoid``). ``category_of(label)`` is the
    category or None (an unknown label only groups with the same label)."""
    target = summary.object(target_id)
    if target is None:
        return (), ()
    if level:
        ids = next((v.object_ids for v in summary.levels if v.number == level), ())
    else:
        ids = tuple(i for i, s in summary.support.items() if s and s == surface_id)
    group = category_of(target.label)
    same_out, other_out = [], []
    for i in ids:
        obj = summary.object(i)
        if obj is None or i == target_id:
            continue
        category = category_of(obj.label)
        same = category == group if group is not None else obj.label == target.label
        center = tuple(float(v) for v in obj.position)
        (same_out if same else other_out).append(center)
    return tuple(same_out), tuple(other_out)


def other_groups(
    summary: SceneSummary,
    target_id: int,
    category_of,
    level: int = 0,
    surface_id: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """The ``other`` part of ``split_groups``: PlaceObject's ``avoid``."""
    return split_groups(summary, target_id, category_of, level, surface_id)[1]


def next_object_id(summary: SceneSummary) -> int:
    """An id no object of the summary uses (above the detection track ids)."""
    return max((o.id for o in summary.objects), default=0) + 1


def polygon_distance(polygon: np.ndarray, point) -> float:
    """Distance of a 2D point to a convex polygon (0 inside); inf if empty."""
    p = np.asarray(point, dtype=float)[:2]
    if len(polygon) == 0:
        return math.inf
    if len(polygon) < 3:
        return float(np.linalg.norm(polygon - p, axis=1).min())
    inside = True
    best = math.inf
    n = len(polygon)
    area = 0.0
    for k in range(n):
        a, b = polygon[k], polygon[(k + 1) % n]
        area += a[0] * b[1] - b[0] * a[1]
    sign = 1.0 if area >= 0.0 else -1.0
    for k in range(n):
        a, b = polygon[k], polygon[(k + 1) % n]
        edge = b - a
        cross = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        if sign * cross < 0.0:
            inside = False
        t = np.clip((p - a) @ edge / max(float(edge @ edge), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * edge))))
    return 0.0 if inside else best


def support_of(
    obj: ObjectInfo,
    surfaces,
    height_tolerance: float = SUPPORT_HEIGHT_TOLERANCE,
    hull_margin: float = HULL_MARGIN,
) -> int:
    """Id of the surface the object stands on (closest in height), 0 if none."""
    best, best_dz = 0, math.inf
    for s in surfaces:
        if s.label in NON_SUPPORT_LABELS:
            continue
        dz = abs(obj.bottom - s.height)
        if dz > height_tolerance or dz >= best_dz:
            continue
        if s.distance_xy(obj.position) > hull_margin:
            continue
        best, best_dz = s.id, dz
    return best


def nearest_level(height: float, levels, tolerance: float) -> int:
    """Number of the level whose height is nearest to ``height``, 0 if none within."""
    best = min(levels, key=lambda v: abs(v.height - height), default=None)
    if best is None or abs(best.height - height) > tolerance:
        return 0
    return best.number


def build_summary(
    surfaces,
    objects,
    furniture,
    frame: str = "odom",
    robot_pose=(0.0, 0.0, 0.0),
    footprint=None,
    space: dict | None = None,
    min_level_height: float = MIN_LEVEL_HEIGHT,
) -> SceneSummary:
    """Summary of the scene around ``furniture`` (aha_perception.furniture.Furniture).

    footprint: (N, 2) polygon of the furniture in ``frame`` (grown by a
    margin) for the height grouping of variant B; None groups every object.
    space: level number -> LevelInfo.space."""
    surfaces = tuple(s for s in surfaces if s.label not in NON_SUPPORT_LABELS)
    objects = tuple(objects)
    support = {o.id: support_of(o, surfaces) for o in objects}
    space = space or {}
    listed = [v for v in furniture.levels if v.height >= min_level_height]
    surface_level = {
        s.id: s.level
        for s in surfaces
        if s.furniture == furniture.name and any(v.number == s.level for v in listed)
    }
    polygon = None if footprint is None else np.asarray(footprint, dtype=float)
    by_height: dict[int, list[int]] = {v.number: [] for v in listed}
    object_level, conflicts = {}, {}
    for o in objects:
        inside = polygon is None or polygon_distance(polygon, o.position) <= 0.0
        height_level = (
            nearest_level(o.bottom, listed, HEIGHT_GROUP_TOLERANCE) if inside else 0
        )
        if height_level:
            by_height[height_level].append(o.id)
        support_level = surface_level.get(support[o.id], 0)
        if height_level and support_level and height_level != support_level:
            conflicts[o.id] = (height_level, support_level)
        # Without a footprint any object at a level height would count: only
        # the associated support decides then.
        if polygon is None:
            object_level[o.id] = support_level
        else:
            object_level[o.id] = height_level or support_level
    levels = tuple(
        LevelInfo(
            furniture.name,
            level.number,
            level.height,
            level.max_object_height,
            tuple(sid for sid, n in surface_level.items() if n == level.number),
            tuple(o.id for o in objects if object_level[o.id] == level.number),
            space.get(level.number, "unknown"),
        )
        for level in listed
    )
    return SceneSummary(
        frame,
        tuple(float(v) for v in robot_pose),
        furniture.name,
        surfaces,
        objects,
        support,
        levels,
        {k: tuple(v) for k, v in by_height.items()},
        conflicts,
    )
