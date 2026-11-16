"""Skills that pick up, put down and hand over objects (GPSR, Pick and Place).

pick_object finds the object on its surface with the scan of
aha_tasks.objects, moves the base so that the arm reaches it, and asks the
pick server (aha_msgs/action/PickObject on PICK_ACTION). place_object
chooses a free spot on the surface (away from the objects seen on it; on a
shelf next to its category, aha_tasks.objects.placement), or the inside of a
trash bin, and asks the place server (aha_msgs/action/PlaceObject).
deliver_object goes to the person (`<role>.position` of
find_described_person, or the operator at the instruction point), says what
it brings and asks the handover server (aha_msgs/action/HandOver) to hold
the object out and let go once the person takes it; a touch of the
"Release" button on the screen lets go too.

The motions are the servers' (the manipulation squad's); in e2e:=debug
sim_manipulation (aha_tasks.sim_manipulation_node) serves all three. ROS is
imported on use, so that task definitions load without it.

`object` is a label or category from the command ("cola", "drink") or the
memory key of an object record (perceive_objects, find_object). Memory:
robot.holding is the `object` of the pick, robot.held_object the record of
the picked object (aha_tasks.objects.record with "surface"); both become
null when the object is put down or handed over.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.objects import (
    FLOOR,
    Cancelled,
    Objects,
    check_margin,
    norm_category,
    norm_label,
    placement,
    record,
    same_category,
    shelf_groups,
)
from aha_tasks.people import (
    People,
    approach_pose,
    check_keys,
    check_name,
    planar,
    positive,
)
from aha_tasks.skills import LOCATION, go_to, resolve_pose

PICK_ACTION = "/aha/manipulation/pick"
PLACE_ACTION = "/aha/manipulation/place"
HANDOVER_ACTION = "/aha/manipulation/handover"
HOLDING = "robot.holding"
HELD = "robot.held_object"
OPERATOR = "operator"  # the person who gave the commands
OPERATOR_WORDS = {OPERATOR, "me"}
OPERATOR_LOCATION = "instruction_point"
ARMS = ("right", "left")
# Words of a command that name no particular object: any object matches.
GENERIC = {"object", "objects", "item", "items", "thing", "things", "something"}
ARTICLES = ("a ", "an ", "the ", "some ", "one ")
TRASH = {"trash", "trash bin", "trashbin", "trash can", "bin", "garbage", "waste bin"}
TRASH_CATEGORY = "trash bin"  # furniture category of a bin in the furniture config
RECORD_MATCH = 0.3  # m; an object seen this near a remembered record is that object
SURFACE_DISTANCE = 1.2  # m at most from a location to the furniture it stands for
# m from a furniture edge to the base centre when reaching over it: as the
# task locations, 0.6 m clear of the map's obstacles (Nav2 plans no path
# from a pose in their inflation, e.g. 0.45 m from the dining table).
STANDOFF = 0.65
# m in base_footprint where the right arm takes and puts objects (the left
# arm: y mirrored); about the place test bed's preferred point.
ARM_POINT = (0.45, -0.2)
REACH = 0.75  # m from the base centre; a farther object makes the base move first
MOVE_TOLERANCE = (0.05, 0.1)  # m, rad; a base pose this close is not driven to
PICK_MARGIN = 0.1  # m beyond the furniture outline where objects count
PLACE_CLEARANCE = 0.05  # m to other objects (Pick and Place breakfast: 5 cm)
EDGE_GAP = 0.03  # m from a table edge to the footprint of a placed object
GRID = 0.02  # m between the spots tried on a table
FLOOR_PLACE = 0.5  # m ahead of the base for an object put on the floor
DROP_DEPTH = 0.5  # of a bin's height: the drop point inside it
DEFAULT_SIZE = (0.08, 0.08, 0.12)  # m, an object without a size
HANDOVER_HEIGHT = (0.7, 1.1)  # m; the hand goes out at the person's centre within these
TAKE_TIMEOUT = 20.0  # s for the person to take the object
DELIVER_DISTANCE = 1.0  # m from the person to the base centre
OPERATOR_SEARCH = 3.0  # s (wait clock) to look for the operator
OPERATOR_RADIUS = 2.5  # m around the instruction point
FACE_TOLERANCE = 0.35  # rad; a person farther off the heading makes the base turn
SERVER_TIMEOUT = 5.0  # s (wall) for an action server or its answer to a goal
MOTION_TIMEOUT = 120.0  # s (wall) for a pick or a place
HANDOVER_MARGIN = 60.0  # s (wall) for the handover beyond the take timeout


# -- objects -----------------------------------------------------------------


def query_text(text):
    """The object words of a command, normalized: "the Cola" -> "cola"."""
    out = norm_label(text)
    for article in ARTICLES:
        if out.startswith(article):
            out = out[len(article) :]
    return out


def object_target(value, memory):
    """(record or None, text) of an `object` param: a memory key of an object
    record, else words of the command."""
    found = memory.get(value)
    if isinstance(found, dict) and isinstance(found.get("label"), str):
        return found, found["label"]
    return None, query_text(value)


def matches_text(text, label, category):
    """Whether command words name an object: its label (also plural), its
    category, or a generic word."""
    words = query_text(text)
    if words in GENERIC:
        return True
    name = norm_label(label)
    return (
        words in (name, name + "s", name + "es")
        or same_category(words, category)
        or norm_category(words) == norm_category(name)
    )


def choose_object(percepts, text, remembered, origin):
    """The percept the words name, or None.

    With a remembered record, the matching object within RECORD_MATCH of its
    position; otherwise (or if it moved) the most often seen, then most
    confident, then nearest `origin` (x, y).
    """
    candidates = [p for p in percepts if matches_text(text, p.label, p.category)]
    position = (remembered or {}).get("position")
    if isinstance(position, list) and len(position) >= 2:
        near = [
            p
            for p in candidates
            if math.dist(p.position[:2], position[:2]) <= RECORD_MATCH
        ]
        if near:
            return min(near, key=lambda p: math.dist(p.position[:2], position[:2]))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda p: (-p.count, -p.confidence, math.dist(p.position[:2], origin)),
    )


def object_size(entry, list_size=None):
    """(x, y, z) of an object record, else of the object list, else DEFAULT_SIZE."""
    size = entry.get("size") if isinstance(entry, dict) else None
    if (
        isinstance(size, list)
        and len(size) == 3
        and all(isinstance(v, (int, float)) and v > 0 for v in size)
    ):
        return tuple(float(v) for v in size)
    return tuple(list_size) if list_size else DEFAULT_SIZE


# -- furniture and base poses ------------------------------------------------


def outline_point(region, x, y):
    """Nearest point of a region's outline (inside: the point itself)."""
    lx, ly = region.local(x, y)
    cx = max(-region.depth / 2, min(region.depth / 2, lx))
    cy = max(-region.width / 2, min(region.width / 2, ly))
    return region.world(cx, cy, 0.0)[:2]


def furniture_ahead(pose, regions, max_distance=SURFACE_DISTANCE):
    """The region nearest a location pose [x, y, yaw] whose outline is in
    front of it (within 90 deg of its heading) and at most `max_distance`
    away: the furniture a GPSR placement location stands for. None if none."""
    x, y, yaw = pose
    best = None
    for region in regions:
        if region.floor:
            continue
        px, py = outline_point(region, x, y)
        distance = math.dist((px, py), (x, y))
        if distance > max_distance:
            continue
        if distance > 1e-6:
            bearing = math.atan2(py - y, px - x)
            if abs(math.remainder(bearing - yaw, math.tau)) > math.pi / 2:
                continue
        if best is None or distance < best[0]:
            best = (distance, region)
    return None if best is None else best[1]


def arm_point(arm):
    return (ARM_POINT[0], ARM_POINT[1] if arm == "right" else -ARM_POINT[1])


def reach_pose(robot, target, region=None, arm="right", standoff=STANDOFF):
    """[x, y, yaw] of the base with `target` (x, y) at the arm's point.

    By furniture (`region`): facing the side of it the robot is on, the base
    centre `standoff` from that edge (so the object may still lie farther
    than the arm point). On the floor (`region` None or the floor): facing
    the target from where the robot is.
    """
    ax, ay = arm_point(arm)
    if region is None or region.floor:
        yaw = math.atan2(target[1] - robot[1], target[0] - robot[0])
        forward = ax
    else:
        lx, ly = region.local(*robot[:2])
        if abs(lx) - region.depth / 2 >= abs(ly) - region.width / 2:
            local, edge = (math.copysign(1.0, lx), 0.0), region.depth / 2
        else:
            local, edge = (0.0, math.copysign(1.0, ly)), region.width / 2
        c, s = math.cos(region.yaw), math.sin(region.yaw)
        normal = (c * local[0] - s * local[1], s * local[0] + c * local[1])
        yaw = math.atan2(-normal[1], -normal[0])
        along = (target[0] - region.x) * normal[0] + (target[1] - region.y) * normal[1]
        forward = edge - along + standoff
    f = (math.cos(yaw), math.sin(yaw))
    left = (-f[1], f[0])
    return [
        target[0] - f[0] * forward - left[0] * ay,
        target[1] - f[1] * forward - left[1] * ay,
        yaw,
    ]


def in_reach(robot, target, reach=REACH):
    """Whether (x, y) `target` is within `reach` of the base and in front of it."""
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    ahead = dx * math.cos(robot[2]) + dy * math.sin(robot[2])
    return math.hypot(dx, dy) <= reach and ahead > 0.0


def needs_move(robot, pose, tolerance=MOVE_TOLERANCE):
    return (
        math.dist(robot[:2], pose[:2]) > tolerance[0]
        or abs(math.remainder(robot[2] - pose[2], math.tau)) > tolerance[1]
    )


def free_spot(
    region,
    level,
    obstacles,
    size,
    preferred,
    clearance=PLACE_CLEARANCE,
    edge=EDGE_GAP,
    step=GRID,
):
    """(x, y, z) on a level of a table nearest `preferred` (x, y) where an
    object of `size` stands `clearance` from the obstacles and `edge` inside
    the outline, or None. Objects are circles around their footprint (any yaw)."""
    height = region.heights[level - 1]
    radius = math.hypot(size[0], size[1]) / 2
    half_x = region.depth / 2 - radius - edge
    half_y = region.width / 2 - radius - edge
    if half_x < 0 or half_y < 0:
        return None
    blocked = [
        (o.position[:2], math.hypot(o.size[0], o.size[1]) / 2) for o in obstacles
    ]
    best = None
    nx, ny = int(half_x / step), int(half_y / step)
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            x, y, _ = region.world(i * step, j * step, height)
            if any(
                math.dist((x, y), centre) < radius + other + clearance
                for centre, other in blocked
            ):
                continue
            score = math.dist((x, y), preferred)
            if best is None or score < best[0]:
                best = (score, (x, y, height))
    return None if best is None else best[1]


def arm_target(robot, arm="right"):
    """(x, y) of the arm point of a base pose [x, y, yaw]."""
    ax, ay = arm_point(arm)
    c, s = math.cos(robot[2]), math.sin(robot[2])
    return (robot[0] + c * ax - s * ay, robot[1] + s * ax + c * ay)


def drop_point(region):
    """Inside a bin: its centre at DROP_DEPTH of its height."""
    return (region.x, region.y, region.heights[-1] * DROP_DEPTH)


def spot_from(value, key):
    """(x, y, z, level, frame) of a remembered spot: a dict with "position"
    (perceive_shelf's placement of one object, an object record), a dict of
    such by object key, or [x, y, z] in map. None if there is none."""
    if isinstance(value, dict) and "position" not in value:
        value = value.get(key)
    if isinstance(value, dict):
        position = value.get("position")
        level, frame = value.get("level", 0), value.get("frame", "map")
    else:
        position, level, frame = value, 0, "map"
    if (
        isinstance(position, list)
        and len(position) == 3
        and all(isinstance(v, (int, float)) for v in position)
    ):
        return (*map(float, position), int(level), frame)
    return None


def handover_point(position):
    """Where the hand goes out for a person at `position` [x, y, z] (centre)."""
    low, high = HANDOVER_HEIGHT
    return (position[0], position[1], min(max(position[2], low), high))


def is_operator(person):
    return query_text(person) in OPERATOR_WORDS


def is_trash(name, category=""):
    return query_text(name) in TRASH or same_category(category, TRASH_CATEGORY)


def check_params(params, required, optional):
    check_keys(params, required, optional)
    for key in required:
        check_name(params, key)
    if "surface" in params:
        check_name(params, "surface")
    if params.get("arm", "right") not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}: {params['arm']!r}")
    retries = params.get("retries", 1)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError(f"retries must be 0 to 3: {retries!r}")
    if not isinstance(params.get("approach", True), bool):
        raise ValueError(f"approach must be true or false: {params['approach']!r}")
    if "margin" in params:
        check_margin(params)
    if "clearance" in params:
        check_margin({"margin": params["clearance"]})


# -- ROS ---------------------------------------------------------------------


def wait(future, timeout, cancelled, until=None):
    """The future's result; Cancelled / TimeoutError. None once `until()` holds."""
    deadline = time.monotonic() + timeout
    while not future.done():
        if cancelled.is_set():
            raise Cancelled("Cancelled")
        if until is not None and until():
            return None
        if time.monotonic() > deadline:
            raise TimeoutError
        time.sleep(0.02)
    return future.result()


class Hands:
    """Clients of the pick, place and handover servers of one runner."""

    def __init__(self, node):
        from aha_msgs.action import HandOver, PickObject, PlaceObject
        from aha_msgs.msg import DetectedObject
        from geometry_msgs.msg import PointStamped
        from rclpy.action import ActionClient

        self.node = node
        self.PickObject, self.PlaceObject, self.HandOver = (
            PickObject,
            PlaceObject,
            HandOver,
        )
        self.DetectedObject, self.PointStamped = DetectedObject, PointStamped
        self.clients = {
            name: (action, ActionClient(node, action, name))
            for name, action in (
                (PICK_ACTION, PickObject),
                (PLACE_ACTION, PlaceObject),
                (HANDOVER_ACTION, HandOver),
            )
        }

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "hands", None) is None:
            runtime.hands = cls(runtime.node)
        return runtime.hands

    def point(self, frame, xyz):
        out = self.PointStamped()
        out.header.frame_id = frame
        out.point.x, out.point.y, out.point.z = (float(v) for v in xyz)
        return out

    def target(self, entry):
        """DetectedObject of an object record."""
        out = self.DetectedObject()
        out.header.frame_id = entry["frame"]
        out.class_name = entry["label"]
        out.track_id = int(entry.get("track", -1))
        out.confidence = float(entry.get("confidence", 1.0))
        p = out.pose.pose.position
        p.x, p.y, p.z = (float(v) for v in entry["position"])
        yaw = float(entry.get("yaw", 0.0))
        q = out.pose.pose.orientation
        q.z, q.w = math.sin(yaw / 2), math.cos(yaw / 2)
        out.size.x, out.size.y, out.size.z = (float(v) for v in entry["size"])
        return out

    def start(self, name, goal, cancelled):
        """The accepted goal handle; RuntimeError if not served or rejected."""
        _, client = self.clients[name]
        if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            raise RuntimeError(f"{name} is not served")
        handle = wait(client.send_goal_async(goal), SERVER_TIMEOUT, cancelled)
        if not handle.accepted:
            raise RuntimeError(f"{name} rejected the goal")
        return handle

    def finish(self, handle, timeout, cancelled, until=None):
        """The result, or None once `until()` holds; on cancel or timeout the
        goal is cancelled and Cancelled / TimeoutError raised."""
        future = handle.get_result_async()
        try:
            response = wait(future, timeout, cancelled, until)
        except (Cancelled, TimeoutError):
            self.cancel(handle)
            raise
        return None if response is None else response.result

    def cancel(self, handle):
        """Cancel a goal and wait (a few seconds) until it stopped."""
        future = handle.get_result_async()
        handle.cancel_goal_async()
        try:
            wait(future, SERVER_TIMEOUT, threading.Event())
        except TimeoutError:
            pass

    def run(self, name, goal, timeout, cancelled):
        """(result, "") of a goal, or (None, why)."""
        try:
            handle = self.start(name, goal, cancelled)
            return self.finish(handle, timeout, cancelled), ""
        except Cancelled:
            return None, "Cancelled"
        except TimeoutError:
            return None, f"{name} did not answer in time"
        except RuntimeError as error:
            return None, str(error)


class _HandSkill(Skill):
    def ready(self):
        # Subscriptions and TF from the start, not on the first use.
        Objects.of(self.runtime)
        Hands.of(self.runtime)

    def fail(self, message, outputs=None):
        return SkillResult(False, message, outputs or {})

    def surface(self, objects, name):
        """The furniture (or floor) a surface param means: a furniture name, a
        trash word, or a location of the task in front of furniture."""
        if name == FLOOR or objects.furniture.get(name) is not None:
            return name
        if query_text(name) in TRASH:
            bins = [
                f.name for f in objects.furniture.furniture if is_trash("", f.category)
            ]
            if bins:
                return bins[0]
        if name in self.runtime.spec.locations:
            regions = [
                objects.region(furniture, "map", 0.0)
                for furniture in objects.furniture.names
            ]
            found = furniture_ahead(resolve_pose(self.runtime.spec, name), regions)
            if found is not None:
                return found.name
        raise ValueError(
            f"No furniture for surface {name}; furniture: {objects.furniture.names}"
        )

    def approach(self, ctx, robot, target, region, outputs):
        """Move the base so that the arm reaches `target`, unless it does; the
        new robot pose. A failed move is only logged (the server may reach)."""
        arm = ctx.params.get("arm", "right")
        navigation = self.runtime.navigation
        if (
            not ctx.params.get("approach", True)
            or navigation is None
            or in_reach(robot, target)
        ):
            return robot
        pose = reach_pose(robot, target, region, arm)
        if not needs_move(robot, pose):
            return robot
        ctx.log(f"Moving to [{pose[0]:.2f}, {pose[1]:.2f}] to reach the object")
        ok, text = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            ctx.log(f"Could not move closer: {text}")
            return robot
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return pose

    def look(self, frame, xyz):
        gaze = getattr(self.runtime, "gaze", None)
        if gaze is not None:
            gaze.look_at(frame, SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]))


class PickObject(_HandSkill):
    """Pick up an object from a surface.

    params: {"object": "cola" | "<memory key of a record>", "surface":
    "kitchen_table", "arm": "right", "margin": 0.1, "retries": 1,
    "approach": true}. `surface` is a furniture name, "floor" or a task
    location in front of furniture; it may be left out when the record has a
    "surface". The surface is scanned (objects within `margin` of its
    outline), the object chosen (choose_object), the base moved to reach it
    (approach), and the pick server asked; on failure all of it is tried
    `retries` more times.
    """

    @classmethod
    def validate(cls, params, spec):
        check_params(
            params, {"object"}, {"surface", "arm", "margin", "retries", "approach"}
        )

    def expected(self, params):
        return {HOLDING: params["object"]}

    def execute(self, ctx):
        params = ctx.params
        key = params["object"]
        holding = ctx.memory.get(HOLDING)
        if holding:
            return self.fail(f"Already holding {holding}")
        remembered, text = object_target(key, ctx.memory)
        surface = params.get("surface") or (remembered or {}).get("surface")
        if not isinstance(surface, str) or not surface:
            return self.fail(f"No surface to pick {text} from")
        objects, hands = Objects.of(self.runtime), Hands.of(self.runtime)
        try:
            name = self.surface(objects, surface)
        except (RuntimeError, ValueError) as error:
            return self.fail(str(error))
        outputs, message = {}, ""
        for attempt in range(1 + params.get("retries", 1)):
            if attempt:
                ctx.log(f"Trying again: {message}")
            try:
                frame, region, placed, _ = objects.scan(
                    name,
                    float(params.get("margin", PICK_MARGIN)),
                    ctx.cancelled,
                    ctx.log,
                )
            except Cancelled:
                return self.fail("Cancelled")
            except (RuntimeError, ValueError) as error:
                message = str(error)
                continue
            robot = objects.robot_pose(frame)
            percept = choose_object([p for p, _ in placed], text, remembered, robot[:2])
            if percept is None:
                seen = ", ".join(sorted({p.label for p, _ in placed})) or "nothing"
                message = f"No {text} on {surface} (seen: {seen})"
                continue
            target = {**record(percept, frame), "surface": surface}
            ctx.log(f"Picking {percept.label} at {target['position']} from {name}")
            self.look(frame, target["position"])
            if frame == "map":
                self.approach(ctx, robot, target["position"], region, outputs)
            goal = hands.PickObject.Goal(
                target=hands.target(target), arm=params.get("arm", "right")
            )
            result, why = hands.run(PICK_ACTION, goal, MOTION_TIMEOUT, ctx.cancelled)
            if result is None:
                if why == "Cancelled":
                    return self.fail(why, outputs)
                message = why
                continue
            if not result.success:
                message = f"Pick failed: {result.message}"
                continue
            outputs.update({HOLDING: key, HELD: target})
            return SkillResult(True, f"Picked {percept.label} from {surface}", outputs)
        # The base may have moved: keep the location for the rollback.
        return self.fail(message, {k: v for k, v in outputs.items() if k == LOCATION})


class PlaceObject(_HandSkill):
    """Put the held object on a surface, or into a trash bin.

    params: {"object": "cola" | "<memory key>", "surface": "desk", "spot":
    "<memory key>", "arm": "right", "clearance": 0.05, "retries": 1,
    "approach": true}. A trash bin (furniture of category "trash bin", or
    "trashbin", "trash") gets the object dropped inside. On a shelf the spot
    is next to the objects of its category (aha_tasks.objects.placement), on
    a table the free spot nearest the arm `clearance` from the objects seen,
    on the floor FLOOR_PLACE ahead. `spot` (a placement of perceive_shelf, a
    record or [x, y, z] in map) gives the spot instead.
    """

    @classmethod
    def validate(cls, params, spec):
        check_params(
            params,
            {"object", "surface"},
            {"spot", "arm", "clearance", "retries", "approach"},
        )
        if "spot" in params:
            check_name(params, "spot")

    def expected(self, params):
        return {HOLDING: None, HELD: None}

    def execute(self, ctx):
        params = ctx.params
        holding = ctx.memory.get(HOLDING)
        if not holding:
            return self.fail("Not holding anything")
        held = ctx.memory.get(HELD)
        if not isinstance(held, dict):
            held, _ = object_target(holding, ctx.memory)
        label = (held or {}).get("label") or query_text(params["object"])
        objects, hands = Objects.of(self.runtime), Hands.of(self.runtime)
        listed = objects.object_list.get(label)
        size = object_size(held, listed.size if listed else None)
        try:
            name = self.surface(objects, params["surface"])
        except (RuntimeError, ValueError) as error:
            return self.fail(str(error))
        furniture = objects.furniture.get(name)
        trash = furniture is not None and is_trash(name, furniture.category)
        arm = params.get("arm", "right")
        clearance = float(params.get("clearance", PLACE_CLEARANCE))
        outputs, message = {}, ""
        for attempt in range(1 + params.get("retries", 1)):
            if attempt:
                ctx.log(f"Trying again: {message}")
            try:
                spot = self.spot(ctx, objects, name, trash, label, size, clearance)
            except Cancelled:
                return self.fail("Cancelled", outputs)
            except (RuntimeError, ValueError) as error:
                message = str(error)
                continue
            x, y, z, level, frame, region = spot
            ctx.log(
                f"{'Dropping' if trash else 'Placing'} {label} at "
                f"[{x:.2f}, {y:.2f}, {z:.2f}] on {name}"
            )
            self.look(frame, (x, y, z))
            if frame == "map":
                robot = objects.robot_pose(frame)
                self.approach(ctx, robot, (x, y), region, outputs)
            goal = hands.PlaceObject.Goal(
                arm=arm,
                near=hands.point(frame, (x, y, z)),
                furniture="" if name == FLOOR else name,
                furniture_level=level,
                clearance=clearance,
                drop=trash,
            )
            goal.object_size.x, goal.object_size.y, goal.object_size.z = size
            result, why = hands.run(PLACE_ACTION, goal, MOTION_TIMEOUT, ctx.cancelled)
            if result is None:
                if why == "Cancelled":
                    return self.fail(why, outputs)
                message = why
                continue
            if not result.success:
                message = f"Place failed: {result.message}"
                continue
            outputs.update({HOLDING: None, HELD: None})
            where = "into" if trash else "on"
            return SkillResult(
                True, f"Put {label} {where} {params['surface']}", outputs
            )
        return self.fail(message, {k: v for k, v in outputs.items() if k == LOCATION})

    def spot(self, ctx, objects, name, trash, label, size, clearance):
        """(x, y, z, level, frame, region) to put the object at; raises
        RuntimeError / ValueError (scan) or Cancelled."""
        if trash:
            region = objects.region(name, "map", 0.0)
            return (*drop_point(region), 1, "map", region)
        if "spot" in ctx.params:
            key = ctx.params["spot"]
            found = spot_from(ctx.memory.get(key), ctx.params["object"])
            if found is None:
                raise ValueError(f"No spot for {ctx.params['object']} in {key}")
            x, y, z, level, frame = found
            region = None if name == FLOOR else objects.region(name, frame, 0.0)
            return (x, y, z, level, frame, region)
        if name == FLOOR:
            robot = objects.robot_pose("map")
            c, s = math.cos(robot[2]), math.sin(robot[2])
            x, y = robot[0] + c * FLOOR_PLACE, robot[1] + s * FLOOR_PLACE
            return (x, y, 0.0, 0, "map", None)
        frame, region, placed, _ = objects.scan(name, EDGE_GAP, ctx.cancelled, ctx.log)
        robot = objects.robot_pose(frame)
        arm = ctx.params.get("arm", "right")
        if len(region.heights) > 1:
            where = placement(
                [(label, label, objects.category(label), max(size[:2]))],
                shelf_groups(placed, region),
                region,
            )[label]
            ctx.log(f"Shelf level {where['level']} {where['side']} ({where['reason']})")
            return (*where["position"], where["level"], frame, region)
        found = free_spot(
            region,
            1,
            [p for p, _ in placed],
            size,
            arm_target(robot, arm),
            clearance,
        )
        if found is None:
            raise RuntimeError(f"No free spot for {label} on {name}")
        return (*found, 1, frame, region)


class DeliverObject(_HandSkill):
    """Hand the held object to a person.

    params: {"object": "cola" | "<memory key>", "person": "waving person" |
    "operator", "role": "person", "distance": 1.0, "timeout": 20.0,
    "retries": 1}. The person is at `<role>.position` / `.frame`
    (find_described_person); "operator" (or "me") is the person who gave the
    commands: the robot goes to the instruction point and turns to the
    nearest person there. The base comes `distance` from the person and
    faces them, says what it brings, and the handover server holds the
    object out and lets go when the person takes it (or when the "Release"
    button on the screen is touched), waiting `timeout` s; then it asks once
    more for each retry.
    """

    @classmethod
    def validate(cls, params, spec):
        check_params(
            params,
            {"object", "person"},
            {"role", "distance", "timeout", "retries", "arm"},
        )
        if "role" in params:
            check_name(params, "role")
        positive(params, "distance", DELIVER_DISTANCE)
        positive(params, "timeout", TAKE_TIMEOUT)

    def expected(self, params):
        return {HOLDING: None, HELD: None}

    def execute(self, ctx):
        from aha_tasks.interaction import Interaction

        params = ctx.params
        holding = ctx.memory.get(HOLDING)
        if not holding:
            return self.fail("Not holding anything")
        held = ctx.memory.get(HELD)
        if not isinstance(held, dict):
            held, _ = object_target(holding, ctx.memory)
        label = (held or {}).get("label") or query_text(params["object"])
        outputs = {}
        try:
            found = (
                self.operator(ctx, outputs)
                if is_operator(params["person"])
                else self.person(ctx, outputs)
            )
        except Cancelled:
            return self.fail("Cancelled", outputs)
        if isinstance(found, SkillResult):
            return found
        frame, position = found
        self.look(frame, (position[0], position[1], position[2] + 0.5))
        interaction = Interaction.of(self.runtime)
        timeout = positive(params, "timeout", TAKE_TIMEOUT)
        text = f"Here is the {label}. Please take it from my hand."
        message = ""
        for attempt in range(1 + params.get("retries", 1)):
            if attempt:
                text = f"Please take the {label} from my hand."
            result, message = self.hand_over(
                ctx, interaction, frame, handover_point(position), label, timeout, text
            )
            if result is None:
                if message == "Cancelled":
                    return self.fail(message, outputs)
                break
            if result.success:
                outputs.update({HOLDING: None, HELD: None})
                return SkillResult(
                    True, f"Handed {label} to {params['person']} ({message})", outputs
                )
            message = f"Handover: {result.message}"
            if result.status != type(result).STATUS_TIMEOUT:
                break
        return self.fail(message, outputs)

    def robot(self, memory):
        try:
            return Objects.of(self.runtime).robot_pose("map")
        except RuntimeError:
            x, y, yaw = resolve_pose(self.runtime.spec, memory[LOCATION])
            return [x, y, yaw]

    def face(self, ctx, frame, position, outputs):
        """Come `distance` from a person in map and face them; a failure or
        None (could not move)."""
        navigation = self.runtime.navigation
        if navigation is None or frame != "map":
            return None
        robot = self.robot(ctx.memory)
        distance = positive(ctx.params, "distance", DELIVER_DISTANCE)
        gap = math.dist(robot[:2], position[:2])
        heading = math.atan2(position[1] - robot[1], position[0] - robot[0])
        if gap > distance + 0.3:
            pose = approach_pose(robot[:2], position[:2], distance)
        elif abs(math.remainder(heading - robot[2], math.tau)) > FACE_TOLERANCE:
            pose = [robot[0], robot[1], heading]
        else:
            return None
        ok, text = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            return self.fail(f"Could not go to the person: {text}", outputs)
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return None

    def person(self, ctx, outputs):
        role = ctx.params.get("role", "person")
        position = ctx.memory.get(f"{role}.position")
        frame = ctx.memory.get(f"{role}.frame", "map")
        if not (isinstance(position, list) and len(position) == 3):
            return self.fail(
                f"No {role}.position: find {ctx.params['person']} first "
                "(find_described_person)"
            )
        failed = self.face(ctx, frame, position, outputs)
        return failed or (frame, position)

    def operator(self, ctx, outputs):
        """Go to the instruction point and turn to the nearest person there
        (people perception); without one, hand over straight ahead."""
        if self.runtime.navigation is not None and (
            ctx.memory.get(LOCATION) != OPERATOR_LOCATION
        ):
            moved = go_to(self.runtime, OPERATOR_LOCATION, ctx.cancelled)
            if not moved.success:
                return self.fail(f"Could not go to the operator: {moved.message}")
            outputs[LOCATION] = OPERATOR_LOCATION
        x, y, yaw = resolve_pose(self.runtime.spec, OPERATOR_LOCATION)
        people = People.of(self.runtime)

        def choose(message):
            if message.header.frame_id != "map":
                return None
            near = [
                p
                for p in message.people
                if p.visible
                and math.dist(planar(p.position), (x, y)) <= OPERATOR_RADIUS
            ]
            if not near:
                return None
            p = min(near, key=lambda p: math.dist(planar(p.position), (x, y))).position
            return [p.x, p.y, p.z]

        found, reason = people.wait_for(choose, OPERATOR_SEARCH, ctx.cancelled)
        if ctx.cancelled.is_set():
            raise Cancelled("Cancelled")
        if found is None:
            ctx.log(f"No operator seen ({reason}); handing over straight ahead")
            distance = positive(ctx.params, "distance", DELIVER_DISTANCE)
            return "map", [
                x + distance * math.cos(yaw),
                y + distance * math.sin(yaw),
                HANDOVER_HEIGHT[1],
            ]
        ctx.log(f"Operator at {[round(v, 2) for v in found]}")
        failed = self.face(ctx, "map", found, outputs)
        return failed or ("map", found)

    def hand_over(self, ctx, interaction, frame, point, label, timeout, text):
        """(result, message) of the handover with the screen as a fallback
        cue, or (None, why)."""
        hands = Hands.of(self.runtime)
        goal = hands.HandOver.Goal(
            arm=ctx.params.get("arm", "right"),
            target=hands.point(frame, point),
            release="take",
            timeout=float(timeout),
        )
        try:
            handle = hands.start(HANDOVER_ACTION, goal, ctx.cancelled)
        except Cancelled:
            return None, "Cancelled"
        except (RuntimeError, TimeoutError) as error:
            return None, str(error) or f"{HANDOVER_ACTION} did not answer"
        interaction.say(text, ctx.cancelled)
        stop, touched = threading.Event(), threading.Event()

        def ask():
            index, _ = interaction.ask(
                f"Touch Release when you hold the {label}",
                ["Release"],
                stop,
                timeout=timeout + HANDOVER_MARGIN,
            )
            if index is not None:
                touched.set()

        asking = threading.Thread(target=ask, daemon=True)
        asking.start()
        try:
            result = hands.finish(
                handle, timeout + HANDOVER_MARGIN, ctx.cancelled, until=touched.is_set
            )
            if result is not None:
                return result, "taken"
            ctx.log("Release touched on the screen")
            hands.cancel(handle)  # the arm stays out with the object
            goal.release = "now"
            result, why = hands.run(
                HANDOVER_ACTION, goal, HANDOVER_MARGIN, ctx.cancelled
            )
            return result, why or "released by touch"
        except Cancelled:
            return None, "Cancelled"
        except TimeoutError:
            return None, f"{HANDOVER_ACTION} did not answer in time"
        finally:
            stop.set()
            asking.join(timeout=SERVER_TIMEOUT)
