"""Skills on /aha/perception/objects: what lies on a surface, and the shelf groups.

perceive_objects turns the head over a surface (a piece of furniture of the
furniture config, aha_perception.furniture, or the floor around the robot),
accumulates the detections, keeps the objects on that surface and gives each
role of the subgoal one object by label or category (object list,
aha_perception.object_list). perceive_shelf finds the groups of objects on
the shelf levels and where to put given objects: next to the group of their
category, else in an empty part (rulebook 5.2, Pick and Place).

Detections come from sim_object_detector.py (ground truth) or a camera
detector; both publish aha_msgs/DetectedObjectArray with object-list labels,
the object centre and its box size. Positions are kept in the frame of the
detections (map in task.launch.py); the furniture is transformed there by TF.

An object is a cluster of detections: by track ID, untracked ones by distance,
then tracks of the same label closer than SAME_OBJECT_M are merged (a detector
may give one object a new track). The label is the most frequent one. An
object is on a level at height h when its centre is not below h - BELOW_M and
its bottom (centre - half the box height) is at most STACK_M above h, so that
objects on a plate or lying with an upright box size still count.

The memory record of an object (the contract with pick / place / show):
{"label", "category", "position": [x, y, z] (centre), "frame", "size": [x, y, z]
(along the object's own axes, as in the object list), "track", "confidence",
"yaw" (rad about z of frame, from the detections' orientation)}.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.people import check_keys

OBJECTS_TOPIC = "/aha/perception/objects"
FLOOR = "floor"  # surface name of the floor around the robot
ROBOT_FRAME = "base_footprint"
SAME_OBJECT_M = 0.08  # detections closer than this with one label are one object
BELOW_M = 0.02  # an object's centre is not below the surface it stands on
STACK_M = 0.15  # an object's bottom may be this far above it (on a plate, a box)
SCAN_EXTENT = 1.3  # m; a longer side of a table is looked at in three places
SCAN_INSET = 0.3  # m from the ends of such a side
LOOK_ABOVE = 0.1  # m above a level, where the head looks
FLOOR_LOOK_DISTANCE = 0.8  # m ahead on the floor; the view then spans about 0.25-1.8 m
FLOOR_PANS = (0.0, 0.5, -0.5)  # rad, + = left
LOOK_SETTLE = 1.5  # s (wait clock) after turning the head
DWELL = 2.0  # s (wait clock) of detections per head pose, at least one message
LOOK_TIMEOUT = 20.0  # s (wait clock) at most per head pose for that message
CLOCK_STALL_TIMEOUT = 10.0  # s (wall) without the wait clock advancing
TF_TIMEOUT = 5.0  # s (wall)
HISTORY = 2000  # detection messages kept
# perceive_shelf: an empty part for an object without a group of its category
# is taken among the levels in PLACE_HEIGHTS (m), nearest PREFERRED_HEIGHT.
PLACE_HEIGHTS = (0.2, 1.0)
PREFERRED_HEIGHT = 0.6
SHELF_MARGIN = 0.1  # m beyond the shelf's configured outline
EXCLUDE_MARGIN = 0.05  # m beyond the outline of excluded furniture (exclude)
SIDES = ("left", "right")  # seen from the open side of the shelf
ROLE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
MATCHER_KEYS = ("labels", "category", "category_key")


# -- labels and categories ---------------------------------------------------


def norm_label(label):
    return " ".join(str(label).lower().replace("_", " ").split())


def norm_category(category):
    """Case and a plural "s" of the last word ignored: "Snacks" == "snack"."""
    text = norm_label(category)
    return text[:-1] if text.endswith("s") else text


def same_category(a, b):
    return bool(a) and bool(b) and norm_category(a) == norm_category(b)


# -- clustering --------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    label: str
    track: int  # -1 if untracked
    position: tuple  # x, y, z of the centre
    size: tuple  # x, y, z of the box
    confidence: float
    yaw: float | None = None  # rad about z of the frame; None if not estimated


@dataclass(frozen=True)
class Percept:
    """One object over several detection messages."""

    label: str
    category: str
    position: tuple
    size: tuple
    track: int  # of the latest detection, -1 if untracked
    confidence: float  # mean of the detections with `label`
    count: int  # messages that saw it
    yaw: float | None = None  # circular mean of the detections with `label`


@dataclass
class _Cluster:
    observations: list = field(default_factory=list)  # (message index, Observation)

    def centre(self):
        return tuple(
            median([o.position[k] for _, o in self.observations]) for k in range(3)
        )

    def label(self):
        votes = Counter(o.label for _, o in self.observations)
        return max(votes, key=lambda label: (votes[label], label))


def median(values):
    values = sorted(values)
    n = len(values)
    middle = n // 2
    return values[middle] if n % 2 else 0.5 * (values[middle - 1] + values[middle])


def cluster(messages, category_of, same_object=SAME_OBJECT_M):
    """Percepts from per-message lists of Observation.

    `category_of(label)` gives the object-list category ("" if unknown).
    """
    tracked, untracked = {}, []
    for index, observations in enumerate(messages):
        for observation in observations:
            if observation.track >= 0:
                tracked.setdefault(observation.track, _Cluster()).observations.append(
                    (index, observation)
                )
                continue
            near = [(math.dist(c.centre(), observation.position), c) for c in untracked]
            near = [(d, c) for d, c in near if d < same_object]
            if near:
                min(near, key=lambda item: item[0])[1].observations.append(
                    (index, observation)
                )
            else:
                untracked.append(_Cluster([(index, observation)]))
    clusters = list(tracked.values()) + untracked
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(clusters):
            for b in clusters[i + 1 :]:
                if (
                    a.label() == b.label()
                    and math.dist(a.centre(), b.centre()) < same_object
                ):
                    a.observations += b.observations
                    clusters.remove(b)
                    merged = True
                    break
            if merged:
                break
    percepts = []
    for c in clusters:
        label = c.label()
        same = [o for _, o in c.observations if o.label == label]
        latest = max(c.observations, key=lambda item: item[0])[1]
        percepts.append(
            Percept(
                label=label,
                category=category_of(label),
                position=c.centre(),
                size=tuple(median([o.size[k] for o in same]) for k in range(3)),
                track=latest.track,
                confidence=sum(o.confidence for o in same) / len(same),
                count=len({index for index, _ in c.observations}),
                yaw=mean_angle([o.yaw for o in same if o.yaw is not None]),
            )
        )
    return percepts


def mean_angle(angles):
    if not angles:
        return None
    return math.atan2(
        sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles)
    )


# -- surfaces ----------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A piece of furniture in the detection frame, or the floor (a disc)."""

    name: str
    x: float
    y: float
    yaw: float
    depth: float  # along local x; the floor: diameter
    width: float  # along local y
    heights: tuple  # level tops, lowest first (the floor: (0.0,))
    margin: float  # beyond the outline (the floor: 0)
    floor: bool = False

    def local(self, x, y):
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def world(self, lx, ly, z):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * lx - s * ly, self.y + s * lx + c * ly, z)

    def inside(self, x, y):
        if self.floor:
            return math.dist((x, y), (self.x, self.y)) <= self.depth / 2
        lx, ly = self.local(x, y)
        return (
            abs(lx) <= self.depth / 2 + self.margin
            and abs(ly) <= self.width / 2 + self.margin
        )

    def level_of(self, percept, level=None):
        """Level number (1 = lowest) the object is on, or None."""
        x, y, z = percept.position
        if not self.inside(x, y):
            return None
        bottom = z - percept.size[2] / 2
        found = None
        for number, height in enumerate(self.heights, start=1):
            if z >= height - BELOW_M and bottom <= height + STACK_M:
                found = number  # the highest level that holds it
        if level is not None and found != level:
            return None
        return found

    def look_points(self, robot_xy=None):
        """Head targets that cover the surface: a shelf's levels lowest first,
        the points of a table top nearest `robot_xy` first."""
        if self.floor:
            raise ValueError("look_points of the floor need the robot pose")
        if len(self.heights) > 1:
            return [self.world(0.0, 0.0, h + LOOK_ABOVE) for h in self.heights]
        height = self.heights[0]

        def offsets(extent):
            if extent <= SCAN_EXTENT:
                return [0.0]
            reach = extent / 2 - SCAN_INSET
            return [-reach, 0.0, reach]

        points = [
            self.world(lx, ly, height)
            for lx in offsets(self.depth)
            for ly in offsets(self.width)
        ]
        if robot_xy is not None:
            points.sort(key=lambda p: math.dist(p[:2], robot_xy))
        return points


def furniture_region(furniture, transform, margin):
    """Region of an aha_perception.furniture.Furniture through (tx, ty, tz, yaw)."""
    tx, ty, tz, yaw = transform
    c, s = math.cos(yaw), math.sin(yaw)
    return Region(
        name=furniture.name,
        x=tx + c * furniture.x - s * furniture.y,
        y=ty + s * furniture.x + c * furniture.y,
        yaw=furniture.yaw + yaw,
        depth=furniture.depth,
        width=furniture.width,
        heights=tuple(h + tz for h in furniture.heights),
        margin=margin,
    )


def floor_region(robot_xy, radius, floor_z=0.0):
    return Region(
        name=FLOOR,
        x=robot_xy[0],
        y=robot_xy[1],
        yaw=0.0,
        depth=2 * radius,
        width=2 * radius,
        heights=(floor_z,),
        margin=0.0,
        floor=True,
    )


def floor_look_points(robot_pose, floor_z=0.0):
    x, y, yaw = robot_pose
    return [
        (
            x + FLOOR_LOOK_DISTANCE * math.cos(yaw + pan),
            y + FLOOR_LOOK_DISTANCE * math.sin(yaw + pan),
            floor_z,
        )
        for pan in FLOOR_PANS
    ]


def end_look_point(region, percept, robot_xy):
    """Where perceive_objects leaves the head, given the object of its first
    found role: the floor: the object; a shelf: the look point of its level;
    a table: the look point nearest the robot (its near part)."""
    if region.floor:
        return percept.position
    points = region.look_points(robot_xy)
    if len(region.heights) > 1:
        return points[region.level_of(percept) - 1]
    return points[0]


def on_region(percepts, region, level=None):
    """(percept, level) of the percepts on the region."""
    out = []
    for percept in percepts:
        number = region.level_of(percept, level)
        if number is not None:
            out.append((percept, number))
    return out


# -- roles -------------------------------------------------------------------


def check_matcher(role, matcher):
    if not isinstance(role, str) or not role or set(role) - ROLE_CHARS:
        raise ValueError(f"Role names are identifiers: {role!r}")
    if not isinstance(matcher, dict) or len(matcher) > 1:
        raise ValueError(
            f"Role {role} needs {{}} or one of {list(MATCHER_KEYS)}: {matcher!r}"
        )
    for key, value in matcher.items():
        if key not in MATCHER_KEYS:
            raise ValueError(f"Role {role}: unknown matcher {key!r}")
        if key == "labels":
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(v, str) and v.strip() for v in value)
            ):
                raise ValueError(f"Role {role}: labels must be nonempty names")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"Role {role}: {key} must be a nonempty name")


def matches(matcher, percept, memory):
    if "labels" in matcher:
        return norm_label(percept.label) in {norm_label(v) for v in matcher["labels"]}
    if "category" in matcher:
        return same_category(percept.category, matcher["category"])
    if "category_key" in matcher:
        value = memory.get(matcher["category_key"])
        return isinstance(value, str) and same_category(percept.category, value)
    return False


def assign_roles(percepts, roles, memory, origin):
    """({role: Percept}, [missing roles]).

    Roles with a matcher take, in their order, the best matching object (seen
    in most messages, then most confident, then nearest `origin`). Roles with
    {} take the best remaining objects of the object list that match no other
    role of the subgoal (the dining table's "other" objects), nearest first.
    """

    def rank(p):
        return (-p.count, -p.confidence, math.dist(p.position[:2], origin))

    ranked = sorted(percepts, key=rank)
    specific = [(role, m) for role, m in roles.items() if m]
    rest_roles = [role for role, m in roles.items() if not m]
    taken, assigned = set(), {}
    for role, matcher in specific:
        for i, percept in enumerate(ranked):
            if i not in taken and matches(matcher, percept, memory):
                taken.add(i)
                assigned[role] = percept
                break
    rest = [
        p
        for i, p in enumerate(ranked)
        if i not in taken
        and p.category
        and not any(matches(m, p, memory) for _, m in specific)
    ][: len(rest_roles)]
    rest.sort(key=lambda p: math.dist(p.position[:2], origin))
    assigned.update(zip(rest_roles, rest))
    return assigned, [role for role in roles if role not in assigned]


def record(percept, frame):
    """The memory record; "yaw" when the observations had an orientation."""
    out = {
        "label": percept.label,
        "category": percept.category,
        "position": [round(v, 3) for v in percept.position],
        "frame": frame,
        "size": [round(v, 3) for v in percept.size],
        "track": int(percept.track),
        "confidence": round(float(percept.confidence), 3),
    }
    if percept.yaw is not None:
        out["yaw"] = round(percept.yaw, 3)
    return out


# -- shelf -------------------------------------------------------------------


@dataclass
class Group:
    level: int
    side: str
    category: str
    labels: list
    position: tuple  # centre of the objects, z on the level


def side_of(region, x, y):
    """`left` is local +y: on the left of someone facing the open side (local -x)."""
    return SIDES[0] if region.local(x, y)[1] >= 0 else SIDES[1]


def shelf_groups(placed, region):
    """Groups of (percept, level) by level and side; category by majority."""
    by_part = {}
    for percept, level in placed:
        side = side_of(region, *percept.position[:2])
        by_part.setdefault((level, side), []).append(percept)
    groups = []
    for (level, side), percepts in sorted(
        by_part.items(), key=lambda item: (item[0][0], SIDES.index(item[0][1]))
    ):
        votes = Counter(norm_category(p.category) for p in percepts if p.category)
        if votes:
            top = max(votes, key=lambda c: (votes[c], c))
            category = next(
                p.category for p in percepts if norm_category(p.category) == top
            )
        else:
            category = ""
        x, y, _ = (
            sum(p.position[k] for p in percepts) / len(percepts) for k in range(3)
        )
        groups.append(
            Group(
                level=level,
                side=side,
                category=category,
                labels=sorted(p.label for p in percepts),
                position=(x, y, region.heights[level - 1]),
            )
        )
    return groups


def categories_by_level(groups, levels):
    """Per level, lowest first: the categories of its groups, left first."""
    out = [[] for _ in range(levels)]
    for group in sorted(groups, key=lambda g: (g.level, SIDES.index(g.side))):
        if group.category:
            out[group.level - 1].append(group.category)
    return out


def part_centre(region, level, side):
    ly = region.width / 4 * (1 if side == SIDES[0] else -1)
    return region.world(0.0, ly, region.heights[level - 1])


def placement(items, groups, region):
    """{key: where to put it} for items [(key, label, category)].

    Next to the largest group of the same category (rule 10); otherwise in an
    empty part (a level side without objects) at a height in PLACE_HEIGHTS
    nearest PREFERRED_HEIGHT, which later items of that category then join;
    otherwise in the least occupied part in reach.
    """
    groups = [
        Group(g.level, g.side, g.category, list(g.labels), g.position) for g in groups
    ]
    reach = [
        number
        for number, height in enumerate(region.heights, start=1)
        if PLACE_HEIGHTS[0] <= height <= PLACE_HEIGHTS[1]
    ] or list(range(1, len(region.heights) + 1))

    def nearness(level):
        return abs(region.heights[level - 1] - PREFERRED_HEIGHT)

    out = {}
    for key, label, category in items:
        same = [g for g in groups if same_category(g.category, category)]
        if same:
            group = max(same, key=lambda g: (len(g.labels), -nearness(g.level)))
            reason, level, side, position = (
                "category",
                group.level,
                group.side,
                group.position,
            )
            group.labels.append(label)
        else:
            occupied = {(g.level, g.side) for g in groups}
            empty = [
                (level, side)
                for level in reach
                for side in SIDES
                if (level, side) not in occupied
            ]
            if empty:
                reason = "empty"
                level, side = min(
                    empty, key=lambda part: (nearness(part[0]), SIDES.index(part[1]))
                )
            else:
                reason = "least occupied"
                counts = Counter()
                for g in groups:
                    counts[(g.level, g.side)] += len(g.labels)
                level, side = min(
                    ((lv, sd) for lv in reach for sd in SIDES),
                    key=lambda part: (counts[part], nearness(part[0])),
                )
            position = part_centre(region, level, side)
            groups.append(Group(level, side, category, [label], position))
        out[key] = {
            "level": level,
            "side": side,
            "height": round(region.heights[level - 1], 3),
            "position": [round(v, 3) for v in position],
            "category": category,
            "reason": reason,
        }
    return out


# -- ROS ---------------------------------------------------------------------


class Cancelled(RuntimeError):
    """The skill was aborted."""


def observations_of(message):
    """Observations of a DetectedObjectArray.

    The size is along the axes of the pose's orientation, so the identity
    orientation of a camera detector's axis-aligned box (object_perception.py)
    is yaw 0, not an unknown orientation.
    """
    out = []
    for o in message.objects:
        p, q = o.pose.pose.position, o.pose.pose.orientation
        out.append(
            Observation(
                label=o.class_name,
                track=int(o.track_id),
                position=(p.x, p.y, p.z),
                size=(o.size.x, o.size.y, o.size.z),
                confidence=float(o.confidence),
                yaw=yaw_of(q),
            )
        )
    return out


def stamp_seconds(message):
    """Time of the image of a detection message (its header stamp)."""
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def in_views(seconds, views):
    """Whether an image time lies in one of the (settled, left) intervals."""
    return any(start <= seconds <= end for start, end in views)


def yaw_of(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class Objects:
    """ROS access for the object skills of one runner.

    Runner parameters: `furniture_config` and `objects_config` (empty: the
    simulated arena and the object list of aha_perception).
    """

    look_settle = LOOK_SETTLE
    dwell = DWELL
    look_timeout = LOOK_TIMEOUT

    def __init__(self, node, gaze):
        from aha_msgs.msg import DetectedObjectArray
        from ament_index_python.packages import get_package_share_directory
        from tf2_ros import Buffer, TransformListener

        from aha_perception.furniture import load_config
        from aha_perception.object_list import load_object_list

        share = Path(get_package_share_directory("aha_perception"), "config")
        furniture = self._parameter(node, "furniture_config") or str(
            share / "furniture" / "rcjo2026_arena.yaml"
        )
        objects = self._parameter(node, "objects_config") or str(
            share / "objects" / "robocup_objects.yaml"
        )
        self.node, self.gaze = node, gaze
        self.furniture = load_config(furniture)
        self.object_list = load_object_list(objects)
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY)  # (sequence, wall time, message)
        self.sequence = 0
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        node.create_subscription(
            DetectedObjectArray, OBJECTS_TOPIC, self._on_objects, 10
        )

    @staticmethod
    def _parameter(node, name):
        if not node.has_parameter(name):
            node.declare_parameter(name, "")
        return node.get_parameter(name).value

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "objects", None) is None:
            runtime.objects = cls(runtime.node, runtime.gaze)
        return runtime.objects

    def _on_objects(self, message):
        with self.lock:
            self.sequence += 1
            self.history.append((self.sequence, time.monotonic(), message))

    # -- queries -------------------------------------------------------------

    def category(self, label):
        entry = self.object_list.get(label)
        if entry is None:  # a camera detector may use other spacing or case
            entry = next(
                (
                    o
                    for o in self.object_list.objects
                    if norm_label(o.label) == norm_label(label)
                ),
                None,
            )
        return entry.category if entry else ""

    def mark(self):
        with self.lock:
            return self.sequence

    def since(self, mark):
        with self.lock:
            return [m for s, _, m in self.history if s > mark]

    def image_time(self):
        """Seconds of the node clock, the time base of the detections' stamps."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def fresh(self, mark, taken):
        """The messages after `mark` of images taken at or after `taken`: a slow
        detector (object_perception.py, seconds per image) answers for images
        from before the head turned."""
        with self.lock:
            return [
                m for s, _, m in self.history if s > mark and stamp_seconds(m) >= taken
            ]

    def now(self):
        """Simulation seconds when active; otherwise monotonic wall seconds."""
        clock = self.node.get_clock()
        if not getattr(clock, "ros_time_is_active", False):
            return time.monotonic()
        return clock.now().nanoseconds * 1e-9

    def pause(self, seconds, cancelled, until=None):
        """Wait `seconds` on the wait clock, then until `until()` (if given) up to
        look_timeout. Raises Cancelled; False if the wait clock stalled."""
        start = self.now()
        last, progress = start, time.monotonic()
        while True:
            if cancelled.is_set():
                raise Cancelled("Cancelled")
            now = self.now()
            if now >= start + seconds and (
                until is None or until() or now >= start + self.look_timeout
            ):
                return True
            wall = time.monotonic()
            if now > last:
                last, progress = now, wall
            elif wall - progress >= CLOCK_STALL_TIMEOUT:
                return False
            time.sleep(0.05)

    def transform(self, target, source):
        """(tx, ty, tz, yaw) of `source` in `target` (z up in both), latest TF."""
        if target == source:
            return (0.0, 0.0, 0.0, 0.0)
        from rclpy.time import Time

        deadline = time.monotonic() + TF_TIMEOUT
        while True:
            try:
                t = self.tf.lookup_transform(target, source, Time()).transform
                return (
                    t.translation.x,
                    t.translation.y,
                    t.translation.z,
                    yaw_of(t.rotation),
                )
            except Exception as error:  # tf2 raises several exception types
                if time.monotonic() > deadline:
                    raise RuntimeError(f"No TF {target} -> {source}: {error}") from None
                time.sleep(0.1)

    def robot_pose(self, frame):
        x, y, _, yaw = self.transform(frame, ROBOT_FRAME)
        return (x, y, yaw)

    def region(self, surface, frame, margin, level=None):
        if surface == FLOOR:
            return floor_region(self.robot_pose(frame)[:2], margin)
        furniture = self.furniture.get(surface)
        if furniture is None:
            raise ValueError(
                f"Unknown surface {surface}; furniture: {self.furniture.names}"
            )
        return furniture_region(
            furniture, self.transform(frame, self.furniture.frame), margin
        )

    def look_at(self, frame, xyz):
        from geometry_msgs.msg import Point

        self.gaze.look_at(
            frame, Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        )

    # -- the common scan -------------------------------------------------------

    def scan(self, surface, margin, cancelled, log, exclude=()):
        """(frame, region, percepts on it as (percept, level), percepts seen).

        Objects inside the footprint of a furniture in `exclude` (grown by
        EXCLUDE_MARGIN), at any height, are not on the region: objects put in
        the trash bin are not floor trash.

        Waits for the first detections to learn their frame, then looks at each
        look point for look_settle + dwell, and until a detection of an image
        taken after the head settled arrives. Only detections of images taken
        while the head stood at a look point count: from its settling to the
        turn to the next point (the last point: to the end of the scan). Images
        from before the first turn (the end of the move, the previous view) and
        from the head's motion are left out.
        """
        start = self.mark()
        if not self.pause(0.0, cancelled, until=lambda: self.mark() > start):
            raise RuntimeError(
                f"No /clock progress for {CLOCK_STALL_TIMEOUT:.0f} s; "
                "is simulation running?"
            )
        first = self.since(start)
        if not first:
            raise RuntimeError(
                f"No {OBJECTS_TOPIC} within {self.look_timeout:.0f} s; "
                "is the object detector running (objects:=true)?"
            )
        frame = first[-1].header.frame_id
        region = self.region(surface, frame, margin)
        if region.floor:
            points = floor_look_points(self.robot_pose(frame), region.heights[0])
        else:
            points = region.look_points(self.robot_pose(frame)[:2])
        mark = self.mark()
        settled, left = [], []  # image times of each head pose (seconds)
        for point in points:
            if settled:
                left.append(self.image_time())  # the head leaves the last point
            self.look_at(frame, point)
            if not self.pause(self.look_settle, cancelled):
                raise RuntimeError("The wait clock stopped")
            before, still = self.mark(), self.image_time()
            if not self.pause(
                self.dwell, cancelled, until=lambda: bool(self.fresh(before, still))
            ):
                raise RuntimeError("The wait clock stopped")
            settled.append(still)
        left.append(self.image_time())
        views = list(zip(settled, left))
        messages = self.since(mark)
        if not messages:
            raise RuntimeError(f"No {OBJECTS_TOPIC} while looking at {surface}")
        other = {m.header.frame_id for m in messages} - {frame}
        if other:
            log(f"Ignored detections in {sorted(other)} (not {frame})")
        messages = [
            m
            for m in messages
            if m.header.frame_id == frame and in_views(stamp_seconds(m), views)
        ]
        if not messages:
            raise RuntimeError(
                f"No {OBJECTS_TOPIC} of images taken while looking at {surface}"
            )
        percepts = cluster([observations_of(m) for m in messages], self.category)
        # Seen at least twice, unless the detector is slower than 2 per head pose.
        least = 2 if len(messages) >= 2 * len(points) else 1
        percepts = [p for p in percepts if p.count >= least]
        excluded = [self.region(name, frame, EXCLUDE_MARGIN) for name in exclude]
        candidates = [
            p for p in percepts if not any(r.inside(*p.position[:2]) for r in excluded)
        ]
        if len(candidates) < len(percepts):
            log(
                f"Ignored {len(percepts) - len(candidates)} object(s) in "
                f"{', '.join(exclude)}"
            )
        return frame, region, on_region(candidates, region), percepts


def check_name(params, key):
    if not isinstance(params.get(key), str) or not params[key].strip():
        raise ValueError(f"{key} must be a nonempty name: {params}")


def check_margin(params):
    margin = params["margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError(f"margin must be a number: {margin!r}")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be nonnegative: {margin!r}")


def describe(percept):
    x, y, z = percept.position
    return f"{percept.label} ({x:.2f}, {y:.2f}, {z:.2f})"


class _SurfaceSkill(Skill):
    def ready(self):
        """Load the configs and check the surfaces of this skill's subgoals."""
        objects = Objects.of(self.runtime)
        skills = getattr(self.runtime, "skills", {}) or {}
        for subgoal in self.runtime.spec.subgoals:
            if skills.get(subgoal.skill) is not self:
                continue
            surface = subgoal.params["surface"]
            if surface != FLOOR and objects.furniture.get(surface) is None:
                raise ValueError(
                    f"{subgoal.id}: surface {surface} is not in the furniture config "
                    f"({objects.furniture.names})"
                )
            for name in subgoal.params.get("exclude", []):
                if objects.furniture.get(name) is None:
                    raise ValueError(
                        f"{subgoal.id}: exclude {name} is not in the furniture "
                        f"config ({objects.furniture.names})"
                    )

    def scan(self, ctx, surface, margin, exclude=()):
        """Objects.scan, or a failed SkillResult."""
        try:
            return Objects.of(self.runtime).scan(
                surface, margin, ctx.cancelled, ctx.log, exclude
            )
        except Cancelled:
            return SkillResult(False, "Cancelled")
        except (RuntimeError, ValueError) as error:
            return SkillResult(False, str(error))


class PerceiveObjects(_SurfaceSkill):
    """Give each role an object on a surface; outputs `<output>.<role>` records.

    params: {"surface": "dining_table", "output": "table", "margin": 0.5,
    "roles": {"cutlery": {"category": "cutlery"}, "cup": {"labels": ["mug", "cup"]},
    "trash": {"category_key": "ppc.trash_category"}, "other1": {}, ...}}.
    `surface` is a furniture name or "floor" (within `margin` m of the robot);
    for furniture, objects may be up to `margin` m outside its configured
    outline (the referee may move the dining table by 0.5 m). Objects within
    the footprints of the furniture in `exclude` (e.g. ["trash_box"] for the
    floor trash, which would otherwise take what was put in the bin) are left
    out. A matcher gives
    labels, a category, or a memory key holding the category; {} takes any
    other known object (assign_roles). Succeeds when at least one role was
    found; missing roles get no key. The head then looks towards the object of
    the first found role (end_look_point).
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"surface", "output", "roles", "margin", "exclude"})
        check_name(params, "surface")
        check_name(params, "output")
        check_margin(params)
        exclude = params["exclude"]
        if not isinstance(exclude, list) or not all(
            isinstance(name, str) and name.strip() and name != FLOOR for name in exclude
        ):
            raise ValueError(f"exclude must be furniture names: {exclude!r}")
        roles = params["roles"]
        if not isinstance(roles, dict) or not roles:
            raise ValueError(f"roles must be a nonempty object: {roles!r}")
        for role, matcher in roles.items():
            check_matcher(role, matcher)

    def execute(self, ctx):
        params = ctx.params
        output = params["output"]
        scanned = self.scan(
            ctx, params["surface"], float(params["margin"]), params["exclude"]
        )
        if isinstance(scanned, SkillResult):
            return scanned
        frame, region, placed, seen = scanned
        origin = Objects.of(self.runtime).robot_pose(frame)[:2]
        on = [p for p, _ in placed]
        ctx.log(
            f"On {region.name}: {', '.join(describe(p) for p in on) or 'nothing'}"
            f" ({len(seen) - len(on)} other objects seen)"
        )
        assigned, missing = assign_roles(on, params["roles"], ctx.memory, origin)
        if not assigned:
            return SkillResult(
                False, f"No object for {', '.join(missing)} on {region.name}"
            )
        first = next(assigned[r] for r in params["roles"] if r in assigned)
        Objects.of(self.runtime).look_at(frame, end_look_point(region, first, origin))
        outputs = {f"{output}.{role}": record(p, frame) for role, p in assigned.items()}
        found = ", ".join(f"{role} {p.label}" for role, p in assigned.items())
        message = f"{found}" + (f"; missing {', '.join(missing)}" if missing else "")
        return SkillResult(True, message, outputs)


class PerceiveShelf(_SurfaceSkill):
    """Find the groups on the shelf levels and where to put given objects.

    params: {"surface": "shelf", "objects": ["table.other1", "table.other2"],
    "output": "cabinet"}. `objects` are memory keys of object records (a
    plain label of an old mock works too). Outputs `<output>.categories`
    (per level, lowest first, the group categories left to right),
    `<output>.groups` ([{level, side, category, labels, position, frame}]) and
    `<output>.placement` ({key: {level, side, height, position, frame, category,
    reason}}), see `placement`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"surface", "output", "objects"})
        check_name(params, "surface")
        check_name(params, "output")
        items = params["objects"]
        if not isinstance(items, list) or not all(
            isinstance(k, str) and k.strip() for k in items
        ):
            raise ValueError(f"objects must be memory keys: {items!r}")
        if params["surface"] == FLOOR:
            raise ValueError("perceive_shelf needs a piece of furniture")

    def execute(self, ctx):
        params = ctx.params
        objects = Objects.of(self.runtime)
        scanned = self.scan(ctx, params["surface"], SHELF_MARGIN)
        if isinstance(scanned, SkillResult):
            return scanned
        frame, region, placed, _ = scanned
        groups = shelf_groups(placed, region)
        items = []
        for key in params["objects"]:
            value = ctx.memory.get(key)
            if isinstance(value, dict):
                label = value.get("label", "")
                category = value.get("category") or objects.category(label)
            elif isinstance(value, str):
                label, category = value, objects.category(value)
            else:
                ctx.log(f"No object in {key}; nothing to place")
                continue
            items.append((key, label, category))
        where = placement(items, groups, region)
        for spot in where.values():
            spot["frame"] = frame
        output = params["output"]
        outputs = {
            f"{output}.categories": categories_by_level(groups, len(region.heights)),
            f"{output}.groups": [
                {
                    "level": g.level,
                    "side": g.side,
                    "category": g.category,
                    "labels": g.labels,
                    "position": [round(v, 3) for v in g.position],
                    "frame": frame,
                }
                for g in groups
            ],
            f"{output}.placement": where,
        }
        text = "; ".join(
            f"L{g.level} {g.side}: {g.category or '?'} ({', '.join(g.labels)})"
            for g in groups
        )
        moves = "; ".join(
            f"{k} -> L{v['level']} {v['side']} ({v['reason']})"
            for k, v in where.items()
        )
        return SkillResult(True, f"{text or 'empty shelf'}. {moves}".strip(), outputs)
