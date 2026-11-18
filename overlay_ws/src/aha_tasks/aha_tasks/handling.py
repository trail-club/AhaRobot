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

import itertools
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
from aha_tasks.skills import ARM_OUT, LOCATION, go_to, resolve_pose

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
REACH = 0.75  # m from the arm's shoulder; a farther object makes the base move first
SHOULDER = (-0.04, -0.13)  # m, right arm (aha_manipulation.pointing.SHOULDER_XY)
MOVE_TOLERANCE = (0.05, 0.1)  # m, rad; a base pose this close is not driven to
PICK_MARGIN = 0.1  # m beyond the furniture outline where objects count
# m around the robot where objects on the floor count (the floor "margin" of
# aha_tasks.objects, as Pick and Place's floor trash)
FLOOR_RADIUS = 1.5
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


def is_xyz(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def object_size(entry, list_size=None):
    """(x, y, z) of an object record, else of the object list, else DEFAULT_SIZE."""
    size = entry.get("size") if isinstance(entry, dict) else None
    if is_xyz(size) and all(v > 0 for v in size):
        return tuple(float(v) for v in size)
    return tuple(list_size) if list_size else DEFAULT_SIZE


# -- furniture and base poses ------------------------------------------------


def outline_point(region, x, y):
    """Nearest point of a region's outline (inside: the point itself)."""
    lx, ly = region.local(x, y)
    cx = max(-region.depth / 2, min(region.depth / 2, lx))
    cy = max(-region.width / 2, min(region.width / 2, ly))
    return region.world(cx, cy, 0.0)[:2]


def furniture_ahead(pose, regions):
    """The region nearest a location pose [x, y, yaw] whose outline is in
    front of it (within 90 deg of its heading) and at most SURFACE_DISTANCE
    away: the furniture a GPSR placement location stands for. None if none."""
    x, y, yaw = pose
    best = None
    for region in regions:
        if region.floor:
            continue
        px, py = outline_point(region, x, y)
        distance = math.dist((px, py), (x, y))
        if distance > SURFACE_DISTANCE:
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


def reach_pose(robot, target, region=None, arm="right"):
    """[x, y, yaw] of the base with `target` (x, y) at the arm's point.

    By furniture (`region`): facing the side of it the robot is on, the base
    centre STANDOFF from that edge (so the object may still lie farther than
    the arm point). On the floor (`region` None or the floor): facing the
    target from where the robot is.
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
        forward = edge - along + STANDOFF
    f = (math.cos(yaw), math.sin(yaw))
    left = (-f[1], f[0])
    return [
        target[0] - f[0] * forward - left[0] * ay,
        target[1] - f[1] * forward - left[1] * ay,
        yaw,
    ]


def in_reach(robot, target, reach=REACH, arm="right"):
    """Whether (x, y) `target` is within `reach` of the arm's shoulder (the
    right one is right of the base centre) and in front of the base."""
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    c, s = math.cos(robot[2]), math.sin(robot[2])
    ahead, left = dx * c + dy * s, dy * c - dx * s
    side = SHOULDER[1] if arm == "right" else -SHOULDER[1]
    return math.hypot(ahead - SHOULDER[0], left - side) <= reach and ahead > 0.0


def needs_move(robot, pose):
    return (
        math.dist(robot[:2], pose[:2]) > MOVE_TOLERANCE[0]
        or abs(math.remainder(robot[2] - pose[2], math.tau)) > MOVE_TOLERANCE[1]
    )


def free_spot(region, obstacles, size, preferred, clearance=PLACE_CLEARANCE):
    """(x, y, z) on a table nearest `preferred` (x, y) where an object of
    `size` stands `clearance` from the obstacles and EDGE_GAP inside the
    outline, or None. Objects are circles around their footprint (any yaw)."""
    height = region.heights[0]
    radius = math.hypot(size[0], size[1]) / 2
    half_x = region.depth / 2 - radius - EDGE_GAP
    half_y = region.width / 2 - radius - EDGE_GAP
    if half_x < 0 or half_y < 0:
        return None
    blocked = [
        (o.position[:2], math.hypot(o.size[0], o.size[1]) / 2) for o in obstacles
    ]
    best = None
    nx, ny = int(half_x / GRID), int(half_y / GRID)
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            x, y, _ = region.world(i * GRID, j * GRID, height)
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
    """(x, y, z, level, frame) of a remembered spot, z the support height: a
    dict with "position" (perceive_shelf's placement of one object, at the
    level's height; an object record, at the object's centre: half its height
    lower), a dict of such by object key, or [x, y, z] in map. None if there
    is none."""
    if isinstance(value, dict) and "position" not in value:
        value = value.get(key)
    position, level, frame = value, 0, "map"
    if isinstance(value, dict):
        position = value.get("position")
        level, frame = value.get("level", 0), value.get("frame", "map")
        size = value.get("size")
        if "level" not in value and is_xyz(position) and is_xyz(size):
            # an object record: its centre down to its bottom
            position = [*position[:2], position[2] - size[2] / 2]
    if not is_xyz(position):
        return None
    return (*map(float, position), int(level), frame)


def handover_point(position):
    """Where the hand goes out for a person at `position` [x, y, z] (centre)."""
    low, high = HANDOVER_HEIGHT
    return (position[0], position[1], min(max(position[2], low), high))


def is_operator(person):
    return query_text(person) in OPERATOR_WORDS


def is_trash(name, category=""):
    return query_text(name) in TRASH or same_category(category, TRASH_CATEGORY)


def trash_bins(objects):
    """Names of the furniture of the bin category."""
    return [
        f.name
        for f in objects.furniture.furniture
        if same_category(f.category, TRASH_CATEGORY)
    ]


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


class Unstopped(Exception):
    """A goal given up on (timed out) whose stop the server did not confirm:
    it may still move the arm, so no other goal may follow."""


# Statuses a server returns before the arm moves (the actions' contract): the
# arm is where it was, whatever `at_home` says (true by the contract, but the
# message default is false).
BEFORE_MOTION = (
    "STATUS_NO_SURFACE",
    "STATUS_NO_SPACE",
    "STATUS_UNREACHABLE",
    "STATUS_ERROR",
)
# PlaceObject statuses that another spot or base pose may change.
RETRY_PLACE = (
    "STATUS_NO_SURFACE",
    "STATUS_NO_SPACE",
    "STATUS_UNREACHABLE",
    "STATUS_EXECUTION_FAILED",
)
UNANSWERED = itertools.count(1)  # numbers of the goals Hands.pending holds


def has_status(result, names):
    """Whether the result's status is one of the constants `names` of its
    action (an action may lack some)."""
    return any(getattr(type(result), name, None) == result.status for name in names)


def before_motion(result):
    """Whether a failed result may have come before any arm motion: a
    PlaceObject / HandOver status for that, or a failed PickObject with
    `at_home` true (it has no status: a goal that did not move the arm also
    sets it)."""
    if result.success:
        return False
    if not hasattr(result, "status"):  # PickObject
        return result.at_home
    return has_status(result, BEFORE_MOTION)


def arm_why(result):
    """Why the arm may be out after a goal ended with `result` (None: it did
    not confirm that it stopped), or None."""
    if result is None:
        return "a goal did not confirm that it stopped"
    return None if result.at_home else f"not at home: {result.message}"


def arm_text(why):
    """The ARM_OUT value for `why` (None: the arm is at home)."""
    return why and f"the arm may be out ({why}); move it back and restart the runner"


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
    """Clients of the pick, place and handover servers of one runner.

    They keep the runtime's ARM_OUT (aha_tasks.skills), which stops the base,
    in line with how the goals ended: a server ends a goal with the arm at
    home (the SRDF "carry" pose) unless its `at_home` says otherwise, and a
    result from before any motion (before_motion) leaves the arm as earlier
    goals left it. A goal that may still move the arm (its acceptance or the
    stop after a cancel not confirmed) stays in `pending`, and no goal is
    sent until it ends."""

    def __init__(self, runtime):
        from aha_msgs.action import HandOver, PickObject, PlaceObject
        from aha_msgs.msg import DetectedObject
        from geometry_msgs.msg import PointStamped
        from rclpy.action import ActionClient

        self.runtime, self.node = runtime, runtime.node
        self.PickObject, self.PlaceObject, self.HandOver = (
            PickObject,
            PlaceObject,
            HandOver,
        )
        self.DetectedObject, self.PointStamped = DetectedObject, PointStamped
        self.clients = {
            name: ActionClient(self.node, action, name)
            for name, action in (
                (PICK_ACTION, PickObject),
                (PLACE_ACTION, PlaceObject),
                (HANDOVER_ACTION, HandOver),
            )
        }
        self.pending = set()  # numbers of goals that may still move the arm
        # Why no goal may be sent until a restart: a goal succeeded after its
        # skill failed, so the hand and the memory differ.
        self.hold = None

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "hands", None) is None:
            runtime.hands = cls(runtime)
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

    def set_arm(self, why):
        setattr(self.runtime, ARM_OUT, arm_text(why))

    def blocked(self):
        """Why no goal may be sent now, or None."""
        if self.pending:
            return "an earlier goal did not answer and may still move the arm"
        return self.hold

    def start(self, name, goal, cancelled):
        """The accepted goal handle; RuntimeError if not served or rejected.

        A goal still unanswered at a cancel or timeout may be accepted (or
        its answer lost) and move the arm: ARM_OUT is set at once, and the
        goal is pending until a late answer settles it (cancel_late).
        Unstopped on the timeout, and while blocked()."""
        why = self.blocked()
        if why:
            raise Unstopped(f"Not sending a {name} goal: {why}")
        client = self.clients[name]
        if cancelled.is_set():  # e.g. while the base came closer: no goal
            raise Cancelled
        if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            raise RuntimeError(f"{name} is not served")
        if cancelled.is_set():
            raise Cancelled
        sent = client.send_goal_async(goal)
        try:
            handle = wait(sent, SERVER_TIMEOUT, cancelled)
        except (Cancelled, TimeoutError) as error:
            number, earlier = next(UNANSWERED), getattr(self.runtime, ARM_OUT, None)
            self.set_arm(
                f"a goal did not confirm that it stopped: {name} goal {number} "
                "not answered"
            )
            self.pending.add(number)
            sent.add_done_callback(lambda done: self.cancel_late(done, number, earlier))
            if isinstance(error, TimeoutError):
                raise Unstopped(
                    f"{name} did not answer the goal in time; it may still run"
                ) from None
            raise
        if not handle.accepted:
            raise RuntimeError(f"{name} rejected the goal")
        return handle

    def cancel_late(self, sent, number, earlier):
        """Cancel goal `number`, answered after start gave up on it: a
        rejection settles it at once, an acceptance once its result comes."""
        handle = sent.result()
        if handle is None or not handle.accepted:
            self.settle(number, None, earlier, rejected=True)
            return
        self.follow(handle.get_result_async(), number, earlier)
        handle.cancel_goal_async()

    def follow(self, future, number, earlier):
        """Keep goal `number` pending until its result `future` settles it."""
        self.pending.add(number)

        def ended(done):
            response = done.result()
            self.settle(number, None if response is None else response.result, earlier)

        future.add_done_callback(ended)

    def settle(self, number, result, earlier, rejected=False):
        """The late end of goal `number`. A rejection, or a result from before
        any motion, gives back the ARM_OUT of before it (`earlier`); a success
        sets `hold` (check the hand, as its skill failed); another result sets
        ARM_OUT by how the goal ended.

        It runs on an executor thread while the skill thread may be in
        start(): the goal leaves `pending` only after `hold` and ARM_OUT are
        set, so that blocked() never sees it settled before them. No lock is
        needed: while a goal is pending no other goal starts, so nothing else
        writes this state meanwhile."""
        if rejected or (result is not None and before_motion(result)):
            after = earlier
        elif result is not None and result.success:
            after = self.hold = arm_text(
                f"a goal succeeded after the skill gave up ({result.message}): "
                "check the hand"
            )
            self.node.get_logger().warning(after)
        else:
            after = arm_text(arm_why(result))
        setattr(self.runtime, ARM_OUT, after)
        self.pending.discard(number)

    def finish(self, handle, timeout, cancelled, until=None):
        """The result, or None once `until()` holds. On cancel or timeout the
        goal is cancelled; if it succeeded all the same (the motion was done,
        e.g. the object already in the hand), that result is returned, else
        Cancelled / TimeoutError is raised; Unstopped for a timed-out goal
        whose stop is not confirmed."""
        future = handle.get_result_async()
        try:
            response = wait(future, timeout, cancelled, until)
        except (Cancelled, TimeoutError) as error:
            ended = self.cancel(handle)
            if ended is not None and ended.success:
                return ended
            if ended is None and isinstance(error, TimeoutError):
                raise Unstopped(
                    "timed out, and the server did not confirm that the goal stopped"
                ) from None
            raise
        if response is None:  # until(): the goal goes on
            return None
        self.arm_after(response.result)
        return response.result

    def cancel(self, handle):
        """Cancel a goal and wait (a few seconds) until it stopped: its result
        (it may have ended otherwise meanwhile), or None without one; then it
        stays pending until its result comes."""
        future = handle.get_result_async()
        handle.cancel_goal_async()
        try:
            response = wait(future, SERVER_TIMEOUT, threading.Event())
        except TimeoutError:
            response = None
        ended = None if response is None else response.result
        earlier = getattr(self.runtime, ARM_OUT, None)
        self.arm_after(ended)
        if response is None:
            self.follow(future, next(UNANSWERED), earlier)
        return ended

    def arm_after(self, result):
        """Set ARM_OUT by how a goal ended (`result` None: it did not confirm
        that it stopped), unless the result came before any motion or `hold`
        keeps it until a restart."""
        if not self.hold and (result is None or not before_motion(result)):
            self.set_arm(arm_why(result))

    def run(self, name, goal, timeout, cancelled):
        """(result, "") of a goal, or (None, why). Unstopped is raised."""
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

    def fail(self, message):
        return SkillResult(False, message)

    def done(self, ctx, message, outputs):
        """A success. A goal may end well after the task was cancelled (the
        object already in or out of the hand), but the engine then keeps no
        outputs (a cancelled subgoal fails): say so, as the hand and the
        memory's robot.holding differ."""
        if ctx.cancelled.is_set():
            message += (
                f"; the task was cancelled, so its memory does not keep "
                f"{HOLDING}={outputs.get(HOLDING)!r}: check the hand"
            )
            ctx.log(message)
        return SkillResult(True, message, outputs)

    def held(self, ctx):
        """(record or None, label) of the held object, or None if nothing is
        held."""
        holding = ctx.memory.get(HOLDING)
        if not holding:
            return None
        held = ctx.memory.get(HELD)
        if not isinstance(held, dict):
            held, _ = object_target(holding, ctx.memory)
        return held, (held or {}).get("label") or query_text(ctx.params["object"])

    def surface(self, objects, name):
        """The furniture (or floor) a surface param means: a furniture name, a
        trash word, or a location of the task in front of furniture."""
        if name == FLOOR or objects.furniture.get(name) is not None:
            return name
        if query_text(name) in TRASH:
            bins = trash_bins(objects)
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

    def approach(
        self, ctx, robot, target, region, outputs, arm_known=True, force=False
    ):
        """Move the base so that the arm reaches `target`, unless it does (by
        in_reach; `force`: unless the base is at reach_pose already); the new
        robot pose. A failed move (e.g. refused while ARM_OUT is set) is only
        logged: the server may reach.

        Without `arm_known` (a goal was sent before: after a timeout, cancel or
        failed execution the arm may still be out over the furniture) the
        base stays: Nav2's footprint does not include the arm."""
        arm = ctx.params.get("arm", "right")
        navigation = self.runtime.navigation
        if (
            not ctx.params.get("approach", True)
            or navigation is None
            or (in_reach(robot, target, arm=arm) and not force)
        ):
            return robot
        pose = reach_pose(robot, target, region, arm)
        if not needs_move(robot, pose):
            return robot
        if not arm_known:
            ctx.log("Not moving the base: the arm may still be out")
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
    outline; on the floor within `margin`, default FLOOR_RADIUS, of the
    robot), the object chosen (choose_object), the base moved to reach it
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
        margin = float(
            params.get("margin", FLOOR_RADIUS if name == FLOOR else PICK_MARGIN)
        )
        # Objects put into a bin lie near the floor too: no floor objects.
        exclude = tuple(trash_bins(objects)) if name == FLOOR else ()
        outputs, message, sent = {}, "", False
        for attempt in range(1 + params.get("retries", 1)):
            if attempt:
                ctx.log(f"Trying again: {message}")
            try:
                frame, region, placed, _ = objects.scan(
                    name, margin, ctx.cancelled, ctx.log, exclude
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
                self.approach(ctx, robot, target["position"], region, outputs, not sent)
            goal = hands.PickObject.Goal(
                target=hands.target(target), arm=params.get("arm", "right")
            )
            sent = True  # a pick result has no status: the arm may be out now
            try:
                result, why = hands.run(
                    PICK_ACTION, goal, MOTION_TIMEOUT, ctx.cancelled
                )
            except Unstopped as error:
                return self.fail(f"{PICK_ACTION}: {error}")
            if result is None:
                if why == "Cancelled":
                    return self.fail(why)
                message = why
                continue
            if not result.success:
                message = f"Pick failed: {result.message}"
                continue
            outputs.update({HOLDING: key, HELD: target})
            return self.done(ctx, f"Picked {percept.label} from {surface}", outputs)
        return self.fail(message)


class PlaceObject(_HandSkill):
    """Put the held object on a surface, or into a trash bin.

    params: {"object": "cola" | "<memory key>", "surface": "desk", "spot":
    "<memory key>", "arm": "right", "clearance": 0.05, "retries": 1,
    "approach": true}. A failure is tried again only when another spot or
    base pose may help (RETRY_PLACE); the base moves for the next try only
    after a status returned before the arm moved (before_motion; for
    STATUS_UNREACHABLE also within REACH). STATUS_NOT_AT_SPOT (let go, but
    not seen at the spot) counts as done with the object out of the hand. A
    trash bin (furniture of category "trash bin", or "trashbin", "trash")
    gets the object dropped inside. On a shelf the spot is next to the
    objects of its category (aha_tasks.objects.placement), on a table the
    free spot nearest the arm `clearance` from the objects seen, on the floor
    FLOOR_PLACE ahead. `spot` (a placement of perceive_shelf, a record or
    [x, y, z] in map) gives the spot instead.
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
        found = self.held(ctx)
        if found is None:
            return self.fail("Not holding anything")
        held, label = found
        objects, hands = Objects.of(self.runtime), Hands.of(self.runtime)
        listed = objects.object_list.get(label)
        size = object_size(held, listed.size if listed else None)
        try:
            name = self.surface(objects, params["surface"])
        except (RuntimeError, ValueError) as error:
            return self.fail(str(error))
        furniture = objects.furniture.get(name)
        trash = furniture is not None and is_trash(name, furniture.category)
        where = "into" if trash else "on"
        arm = params.get("arm", "right")
        clearance = float(params.get("clearance", PLACE_CLEARANCE))
        outputs, message, sent, force, last = {}, "", False, False, None
        for attempt in range(1 + params.get("retries", 1)):
            if attempt:
                ctx.log(f"Trying again: {message}")
            try:
                spot = self.spot(ctx, objects, name, trash, label, size, clearance)
            except Cancelled:
                return self.fail("Cancelled")
            except (RuntimeError, ValueError) as error:
                message = str(error)
                continue
            x, y, z, level, frame, region = spot
            ctx.log(
                f"{'Dropping' if trash else 'Placing'} {label} at "
                f"[{x:.2f}, {y:.2f}, {z:.2f}] on {name}"
            )
            self.look(frame, (x, y, z))
            moved = False
            if frame == "map":
                robot = objects.robot_pose(frame)
                moved = (
                    self.approach(ctx, robot, (x, y), region, outputs, not sent, force)
                    is not robot
                )
            if force and not moved and math.dist(last, (x, y)) < 0.02:
                # Unreachable, and the base cannot come closer (furniture:
                # STANDOFF from its edge already) to the same spot.
                ctx.log("The base cannot come closer to the spot")
                break
            last = (x, y)
            goal = hands.PlaceObject.Goal(
                arm=arm,
                near=hands.point(frame, (x, y, z)),
                furniture="" if name == FLOOR else name,
                furniture_level=level,
                clearance=clearance,
                drop=trash,
            )
            goal.object_size.x, goal.object_size.y, goal.object_size.z = size
            sent, force = True, False
            try:
                result, why = hands.run(
                    PLACE_ACTION, goal, MOTION_TIMEOUT, ctx.cancelled
                )
            except Unstopped as error:
                return self.fail(f"{PLACE_ACTION}: {error}")
            if result is None:
                if why == "Cancelled":
                    return self.fail(why)
                message = why
                continue
            if result.success:
                outputs.update({HOLDING: None, HELD: None})
                return self.done(
                    ctx, f"Put {label} {where} {params['surface']}", outputs
                )
            status = type(result)
            if result.status == status.STATUS_NOT_AT_SPOT:
                # Released: nothing to place again, and only a success
                # clears robot.holding (a failure rolls memory back).
                outputs.update({HOLDING: None, HELD: None})
                return self.done(
                    ctx,
                    f"Let go of {label} {where} {params['surface']}, but it "
                    f"was not seen there: {result.message}",
                    outputs,
                )
            message = f"Place failed: {result.message}"
            if not has_status(result, RETRY_PLACE):
                break
            if before_motion(result):
                # The base may come closer, also when in_reach thought it
                # close enough.
                sent = False
                force = result.status == status.STATUS_UNREACHABLE
        return self.fail(message)

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
            region, [p for p, _ in placed], size, arm_target(robot, arm), clearance
        )
        if found is None:
            raise RuntimeError(f"No free spot for {label} on {name}")
        return (*found, 1, frame, region)


class DeliverObject(_HandSkill):
    """Hand the held object to a person.

    params: {"object": "cola" | "<memory key>", "person": "waving person" |
    "operator", "role": "person", "distance": 1.0, "timeout": 20.0,
    "retries": 1, "touch": false}. The person is at `<role>.position` / `.frame`
    (find_described_person); "operator" (or "me") is the person who gave the
    commands: the robot goes to the instruction point and turns to the
    nearest person there. The base comes `distance` from the person and
    faces them, says what it brings, and the handover server holds the
    object out and lets go when the person takes it (with `touch`, also when
    the "Release" button on the screen is touched), waiting `timeout` s; then
    it asks once more for each retry.
    """

    @classmethod
    def validate(cls, params, spec):
        check_params(
            params,
            {"object", "person"},
            {"role", "distance", "timeout", "retries", "arm", "touch"},
        )
        if not isinstance(params.get("touch", False), bool):
            raise ValueError(f"touch must be true or false: {params}")
        if "role" in params:
            check_name(params, "role")
        positive(params, "distance", DELIVER_DISTANCE)
        positive(params, "timeout", TAKE_TIMEOUT)

    def expected(self, params):
        return {HOLDING: None, HELD: None}

    def execute(self, ctx):
        from aha_tasks.interaction import Interaction

        params = ctx.params
        found = self.held(ctx)
        if found is None:
            return self.fail("Not holding anything")
        _, label = found
        outputs = {}
        try:
            found = (
                self.operator(ctx, outputs)
                if is_operator(params["person"])
                else self.person(ctx, outputs)
            )
        except Cancelled:
            return self.fail("Cancelled")
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
                break
            if result.success:
                outputs.update({HOLDING: None, HELD: None})
                return self.done(
                    ctx, f"Handed {label} to {params['person']} ({message})", outputs
                )
            message = f"Handover: {result.message}"
            if result.status != type(result).STATUS_TIMEOUT:
                break
        return self.fail(message)

    def robot(self, memory):
        """The base pose in map: from TF, else the remembered robot.location;
        RuntimeError without either."""
        try:
            return Objects.of(self.runtime).robot_pose("map")
        except RuntimeError as error:
            try:
                x, y, yaw = resolve_pose(self.runtime.spec, memory.get(LOCATION))
            except ValueError as why:
                raise RuntimeError(f"{error}; {why}") from None
            return [x, y, yaw]

    def face(self, ctx, frame, position, outputs):
        """Come `distance` from a person in map and face them; a failure or
        None (there, or no navigation). While ARM_OUT is set, navigation
        refuses to move: a person not in reach fails the handover (as go_to
        does for the operator)."""
        navigation = self.runtime.navigation
        if navigation is None or frame != "map":
            return None
        try:
            robot = self.robot(ctx.memory)
        except RuntimeError as error:
            return self.fail(f"Could not go to the person: {error}")
        distance = positive(ctx.params, "distance", DELIVER_DISTANCE)
        heading = math.atan2(position[1] - robot[1], position[0] - robot[0])
        if math.dist(robot[:2], position[:2]) > distance + 0.3:
            pose = approach_pose(robot[:2], position[:2], distance)
        elif abs(math.remainder(heading - robot[2], math.tau)) > FACE_TOLERANCE:
            pose = [robot[0], robot[1], heading]
        else:
            return None
        ok, text = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            return self.fail(f"Could not go to the person: {text}")
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
        except Unstopped as error:  # no answer: no second goal (no retry)
            return None, f"{HANDOVER_ACTION}: {error}"
        except RuntimeError as error:
            return None, str(error)
        try:
            interaction.say(text, ctx.cancelled)
        except Exception:
            hands.cancel(handle)  # the arm may be on its way out
            raise
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

        # The team uses touch input outside HIC only when asked for.
        touch = ctx.params.get("touch", False)
        asking = threading.Thread(target=ask, daemon=True)
        if touch:
            asking.start()
        try:
            result = hands.finish(
                handle, timeout + HANDOVER_MARGIN, ctx.cancelled, until=touched.is_set
            )
            if result is not None:
                return result, "taken"
            ctx.log("Release touched on the screen")
            # The arm stays out with the object, unless the take goal ended
            # meanwhile (e.g. the person pulled it, then touched).
            ended = hands.cancel(handle)
            if ended is None:
                # It may still run, or have let go: a second goal could move
                # the same arm, or be refused.
                return None, f"{HANDOVER_ACTION} did not stop the take goal"
            if ended.success or ended.status != type(ended).STATUS_CANCELED:
                return ended, "taken"
            goal.release = "now"
            result, why = hands.run(
                HANDOVER_ACTION, goal, HANDOVER_MARGIN, ctx.cancelled
            )
            return result, why or "released by touch"
        except Cancelled:
            return None, "Cancelled"
        except TimeoutError:
            return None, f"{HANDOVER_ACTION} did not answer in time"
        except Unstopped as error:
            return None, f"{HANDOVER_ACTION}: {error}"
        finally:
            stop.set()
            if touch:
                asking.join(timeout=SERVER_TIMEOUT)
