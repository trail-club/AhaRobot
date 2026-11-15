"""Measured furniture (footprints and level heights) and the association of
tracked horizontal surfaces to (furniture, level).

The YAML config gives, in one ``frame`` (z up, floor at z = 0; ``map`` on the
robot, ``world`` in the simulation)::

    frame: map
    furniture:
      - name: cabinet
        category: shelf          # free text: shelf | table | dishwasher | ...
        pose: {x: 1.2, y: -0.4, yaw: 1.57}  # footprint center, yaw of local x
        size: {x: 0.35, y: 0.80} # footprint along local x (depth) and y (width)
        levels: [0.05, 0.42, 0.78, 1.12]    # top surface heights, lowest first
        board_thickness: 0.02    # optional (DEFAULT_BOARD_THICKNESS)

Level numbers start at 1 for the lowest configured level. A level's
``max_object_height`` is the next level height - board_thickness - its own
height (None for the top level).

Association (``associate_surfaces``): a horizontal surface (not labeled
floor or wall) whose centroid, transformed to the config frame, lies inside
a footprint grown by ``margin`` and within ``height_tolerance`` of one of its
levels gets the nearest such level. If the centroid is inside several grown
footprints, the furniture with the closest footprint center wins. Floor
surfaces are skipped because the tracker merges every floor-height surface
into one floor (a level at floor height is never associated).

``clip_plane`` cuts a detected plane's hull to a footprint (the detection
node does it for the planes of associated surfaces, with a margin).

ROS free (numpy; PyYAML for ``load_config``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .place import PlaceStatus
from .planes.geometry import clip_convex_polygons, convex_hull_2d, polygon_area

DEFAULT_BOARD_THICKNESS = 0.02
SKIPPED_LABELS = ("floor", "wall")
_TOP_KEYS = {"frame", "furniture"}
_FURNITURE_KEYS = {"name", "category", "pose", "size", "levels", "board_thickness"}


class FurnitureConfigError(ValueError):
    """Invalid furniture config."""


@dataclass(frozen=True)
class Level:
    furniture: str
    number: int  # 1 = lowest configured level
    height: float  # top surface, m above the floor
    max_object_height: float | None  # None for the top level


@dataclass(frozen=True)
class Furniture:
    name: str
    category: str
    x: float
    y: float
    yaw: float
    depth: float  # footprint along local x (m)
    width: float  # footprint along local y (m)
    heights: tuple[float, ...]  # level top heights, strictly increasing
    board_thickness: float = DEFAULT_BOARD_THICKNESS

    @property
    def levels(self) -> tuple[Level, ...]:
        out = []
        for k, height in enumerate(self.heights):
            room = None
            if k + 1 < len(self.heights):
                room = self.heights[k + 1] - self.board_thickness - height
            out.append(Level(self.name, k + 1, height, room))
        return tuple(out)

    def level(self, number: int) -> Level | None:
        if 1 <= number <= len(self.heights):
            return self.levels[number - 1]
        return None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def to_local(self, xy) -> np.ndarray:
        """(N, 2) or (2,) config-frame points in the footprint frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(xy, dtype=np.float64) - self.center
        return d @ np.array([[c, -s], [s, c]])

    def contains(self, xy, margin: float = 0.0):
        """Whether config-frame points are inside the footprint grown by ``margin``."""
        local = np.abs(self.to_local(xy))
        return (local[..., 0] <= self.depth / 2 + margin) & (
            local[..., 1] <= self.width / 2 + margin
        )

    def corners(self, margin: float = 0.0) -> np.ndarray:
        """(4, 2) footprint corners in the config frame, counterclockwise."""
        hx, hy = self.depth / 2 + margin, self.width / 2 + margin
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return local @ np.array([[c, s], [-s, c]]) + self.center

    def outline(self, number: int) -> np.ndarray:
        """(5, 3) closed footprint rectangle at a level's height."""
        level = self.level(number)
        if level is None:
            raise ValueError(f"{self.name} has no level {number}")
        corners = self.corners()
        loop = np.vstack([corners, corners[:1]])
        return np.column_stack([loop, np.full(5, level.height)])

    def view_center(self, number: int = 0) -> np.ndarray:
        """Footprint center at a level's height (level 0: the mean of the
        level heights), config frame."""
        level = self.level(number)
        height = level.height if level else float(np.mean(self.heights))
        return np.array([self.x, self.y, height])


@dataclass(frozen=True)
class FurnitureConfig:
    frame: str
    furniture: tuple[Furniture, ...]

    def get(self, name: str) -> Furniture | None:
        return next((f for f in self.furniture if f.name == name), None)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.furniture]


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FurnitureConfigError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise FurnitureConfigError(f"{what} must be finite, got {value}")
    return value


def _mapping(value, what: str, keys: Iterable[str], optional=()) -> dict:
    if not isinstance(value, Mapping):
        raise FurnitureConfigError(f"{what} must be a mapping, got {value!r}")
    keys, optional = set(keys), set(optional)
    unknown = sorted(set(value) - keys - optional)
    if unknown:
        raise FurnitureConfigError(f"{what}: unknown keys {unknown}")
    missing = sorted(keys - set(value))
    if missing:
        raise FurnitureConfigError(f"{what}: missing keys {missing}")
    return dict(value)


def _furniture(item, what: str) -> Furniture:
    item = _mapping(item, what, {"name", "pose", "size", "levels"}, _FURNITURE_KEYS)
    name = item["name"]
    if not isinstance(name, str) or not name:
        raise FurnitureConfigError(f"{what}: name must be a non-empty string")
    what = f"{what} ({name})"
    category = item.get("category", "")
    if not isinstance(category, str):
        raise FurnitureConfigError(f"{what}: category must be a string")
    pose = _mapping(item["pose"], f"{what}: pose", {"x", "y"}, {"yaw"})
    size = _mapping(item["size"], f"{what}: size", {"x", "y"})
    depth = _number(size["x"], f"{what}: size.x")
    width = _number(size["y"], f"{what}: size.y")
    if depth <= 0.0 or width <= 0.0:
        raise FurnitureConfigError(f"{what}: size must be > 0, got {depth}, {width}")
    levels = item["levels"]
    if not isinstance(levels, Sequence) or isinstance(levels, str) or not levels:
        raise FurnitureConfigError(f"{what}: levels must be a non-empty list")
    heights = tuple(_number(h, f"{what}: levels") for h in levels)
    if any(b <= a for a, b in zip(heights, heights[1:])):
        raise FurnitureConfigError(
            f"{what}: levels must be strictly increasing, got {list(heights)}"
        )
    thickness = _number(
        item.get("board_thickness", DEFAULT_BOARD_THICKNESS),
        f"{what}: board_thickness",
    )
    if thickness < 0.0:
        raise FurnitureConfigError(f"{what}: board_thickness must be >= 0")
    if any(b - thickness <= a for a, b in zip(heights, heights[1:])):
        raise FurnitureConfigError(
            f"{what}: levels closer than board_thickness {thickness}"
        )
    return Furniture(
        name=name,
        category=category,
        x=_number(pose["x"], f"{what}: pose.x"),
        y=_number(pose["y"], f"{what}: pose.y"),
        yaw=_number(pose.get("yaw", 0.0), f"{what}: pose.yaw"),
        depth=depth,
        width=width,
        heights=heights,
        board_thickness=thickness,
    )


def parse_config(data, source: str = "furniture config") -> FurnitureConfig:
    """FurnitureConfig of the parsed YAML ``data``; FurnitureConfigError if invalid."""
    data = _mapping(data, source, _TOP_KEYS)
    frame = data["frame"]
    if not isinstance(frame, str) or not frame:
        raise FurnitureConfigError(f"{source}: frame must be a non-empty string")
    items = data["furniture"]
    if not isinstance(items, Sequence) or isinstance(items, str) or not items:
        raise FurnitureConfigError(f"{source}: furniture must be a non-empty list")
    furniture = tuple(
        _furniture(item, f"{source}: furniture[{i}]") for i, item in enumerate(items)
    )
    names = [f.name for f in furniture]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise FurnitureConfigError(f"{source}: duplicate furniture names {duplicates}")
    return FurnitureConfig(frame, furniture)


def load_config(path) -> FurnitureConfig:
    """Read and validate a furniture YAML file."""
    import yaml

    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        raise FurnitureConfigError(f"cannot read furniture config {path}: {error}")
    except yaml.YAMLError as error:
        raise FurnitureConfigError(f"furniture config {path}: {error}")
    return parse_config(data, str(path))


@dataclass(frozen=True)
class AssociationOptions:
    margin: float = 0.10  # m, footprint growth (odometry drift, partial views)
    height_tolerance: float = 0.05  # m between a surface and a level height


@dataclass(frozen=True)
class Match:
    furniture: str
    level: int
    height_error: float  # surface height - level height (m)
    center_distance: float  # horizontal, centroid to footprint center (m)


def match_point(
    config: FurnitureConfig, point, options: AssociationOptions | None = None
) -> Match | None:
    """(furniture, level) of a horizontal surface centroid in the config frame."""
    options = options or AssociationOptions()
    point = np.asarray(point, dtype=np.float64).reshape(3)
    best = None
    for furniture in config.furniture:
        if not furniture.contains(point[:2], options.margin):
            continue
        errors = np.asarray(furniture.heights) - point[2]
        k = int(np.argmin(np.abs(errors)))
        if abs(errors[k]) >= options.height_tolerance:
            continue
        distance = float(np.linalg.norm(point[:2] - furniture.center))
        if best is None or distance < best.center_distance:
            best = Match(furniture.name, k + 1, float(-errors[k]), distance)
    return best


def associate_surfaces(
    config: FurnitureConfig,
    surfaces: Iterable,
    tracking_to_config,
    options: AssociationOptions | None = None,
) -> dict[int, Match]:
    """Surface id -> Match for tracked surfaces (``aha_perception.surfaces.Surface``:
    ``id``, ``label``, ``vertical``, ``centroid`` in the tracking frame);
    ``tracking_to_config`` is the 4x4 transform from the tracking frame."""
    transform = np.asarray(tracking_to_config, dtype=np.float64)
    out = {}
    for surface in surfaces:
        if surface.vertical or surface.label in SKIPPED_LABELS or not surface.id:
            continue
        centroid = transform[:3, :3] @ np.asarray(surface.centroid) + transform[:3, 3]
        match = match_point(config, centroid, options)
        if match is not None:
            out[int(surface.id)] = match
    return out


def clip_plane(plane, corners):
    """``plane`` (aha_perception.planes.types.Plane, not vertical) with its
    hull clipped to a footprint: ``corners`` (N, 2), the footprint's xy in
    the plane's frame (e.g. ``Furniture.corners(margin)`` transformed), taken
    vertically onto the plane. Depth edge noise bleeds a board's hull past
    its edges; the inliers and the centroid stay (it may lie outside the
    clipped hull). Unchanged if the plane is vertical; None if
    nothing would remain (the plane is off the footprint: a wrong
    association)."""
    normal = np.asarray(plane.normal, dtype=np.float64)
    if abs(normal[2]) < 0.5:
        return plane
    corners = np.asarray(corners, dtype=np.float64)
    z = -(corners @ normal[:2] + plane.offset) / normal[2]
    uv = (np.column_stack([corners, z]) - plane.centroid) @ np.asarray(plane.axes).T
    polygon, _ = convex_hull_2d(uv)
    clipped = clip_convex_polygons(np.asarray(plane.hull), polygon)
    area = polygon_area(clipped)
    if len(clipped) < 3 or area <= 0.0:
        return None
    return dataclasses.replace(plane, hull=clipped, area=area)


@dataclass(frozen=True)
class FurnitureRequest:
    """A place request for a furniture (level): the levels the object fits,
    or the status that ends the request (ERROR, NO_SPACE) with its message."""

    furniture: Furniture | None
    level: int  # requested level, 0 = any
    levels: tuple[Level, ...] = ()
    status: PlaceStatus | None = None
    message: str = ""

    @property
    def text(self) -> str:
        name = self.furniture.name if self.furniture else "?"
        return f"{name} level {self.level}" if self.level else name


def resolve_request(
    config: FurnitureConfig | None, name: str, level: int, object_height: float
) -> FurnitureRequest:
    """Check a requested furniture and level (0 = any) against the config and
    the object height (NO_SPACE when no requested level has the room)."""
    if config is None:
        return FurnitureRequest(
            None, level, status=PlaceStatus.ERROR, message="no furniture config loaded"
        )
    furniture = config.get(name)
    if furniture is None:
        return FurnitureRequest(
            None,
            level,
            status=PlaceStatus.ERROR,
            message=f"unknown furniture {name!r} (known: {', '.join(config.names)})",
        )
    count = len(furniture.heights)
    if not 0 <= level <= count:
        return FurnitureRequest(
            furniture,
            level,
            status=PlaceStatus.ERROR,
            message=f"{name} has levels 1-{count}, got {level}",
        )
    levels = furniture.levels if level == 0 else (furniture.level(level),)
    fitting = tuple(
        lv
        for lv in levels
        if lv.max_object_height is None or object_height <= lv.max_object_height
    )
    request = FurnitureRequest(furniture, level, fitting)
    if fitting:
        return request
    rooms = ", ".join(
        f"level {lv.number} {lv.max_object_height:.2f} m" for lv in levels
    )
    return FurnitureRequest(
        furniture,
        level,
        status=PlaceStatus.NO_SPACE,
        message=(
            f"object height {object_height:.2f} m exceeds the room under the "
            f"board above on {request.text} ({rooms})"
        ),
    )


def requested_surfaces(
    request: FurnitureRequest,
    matches: Mapping[int, Match],
    observed: Iterable[int],
    surface_ids: Iterable[int] = (),
) -> list[int]:
    """Observed surface ids associated with the request's levels, intersected
    with ``surface_ids`` when given."""
    numbers = {lv.number for lv in request.levels}
    name = request.furniture.name if request.furniture else None
    ids = {
        sid for sid, m in matches.items() if m.furniture == name and m.level in numbers
    }
    wanted = set(surface_ids)
    if wanted:
        ids &= wanted
    return sorted(ids & set(observed))


def missing_message(
    request: FurnitureRequest,
    matches: Mapping[int, Match],
    surface_ids: Iterable[int] = (),
) -> str:
    """Why ``requested_surfaces`` found nothing in the latest detection."""
    numbers = {lv.number for lv in request.levels}
    name = request.furniture.name if request.furniture else None
    known = sorted(
        sid for sid, m in matches.items() if m.furniture == name and m.level in numbers
    )
    wanted = sorted(set(surface_ids))
    text = f"{request.text} is not in the latest detection"
    if wanted:
        text += f" (with surface_ids {wanted})"
    if known:
        return text + ": associated surfaces " + ", ".join(f"S{s}" for s in known)
    return text + ": no surface associated with it now"


@dataclass(frozen=True)
class LevelMarker:
    """Geometry of one level for display (config frame)."""

    furniture: str
    number: int
    outline: np.ndarray  # (5, 3) closed rectangle at the level height
    center: np.ndarray  # (3,) footprint center at the level height


def level_markers(config: FurnitureConfig) -> list[LevelMarker]:
    """Every level's footprint rectangle and center."""
    return [
        LevelMarker(
            f.name, level.number, f.outline(level.number), f.view_center(level.number)
        )
        for f in config.furniture
        for level in f.levels
    ]
