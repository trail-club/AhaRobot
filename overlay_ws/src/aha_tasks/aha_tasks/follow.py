"""Follow a person or guide them to a place (GPSR), with Nav2 and person_perception.

follow_person keeps `distance` m behind the person that find_described_person
found (`<role>.track` / `.position`). It sends NavigateToPose goals that replace
each other as the person moves: Nav2 preempts the running goal and keeps
planning around obstacles. A goal is a pose `distance` m from the person facing
them, on the robot's side and clear on the global costmap. (Nav2's follow_point
behaviour tree plans to the followed point itself, which the lidar marks as an
obstacle when it is a person, and its truncated path ends with the planner's
orientation instead of facing the person.) Close by, the robot turns on the
spot to keep the person in front, so that someone walking past it does not get
behind the head's reach. The head stays on the person (aha_tasks.gaze). A lost
track is handed over by name, else to the nearest unnamed person near the last
sighting, in a radius that grows while the person is lost; people seen
together with the person are never taken for them. A person lost for
SEARCH_AFTER s is looked for the way they walked, then by turning the head and
the base. Following ends once the person stood still for `stop_time` s with the
robot next to them, after walking at least MIN_TRAVEL m, and, with a
`location`, within `radius` of it; the robot then completes its last goal.

guide_person asks the person to follow and drives to `location`. Every
`check_every` m, and on arrival, it checks for the person behind it, `gap` m
at most from the robot: first with the head alone (it turns at most HEAD_REACH
to each side), then stopped and turned so that the head reaches the point of
the robot's path where a follower would be. If nobody follows, it calls them
and waits, `calls` times, before giving up.

Speech goes through aha_tasks.interaction (a caption without a TTS).
"""

import itertools
import math
import time
from types import SimpleNamespace

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.interaction import Interaction, cancel_if_accepted
from aha_tasks.people import (
    CLOCK_STALL_TIMEOUT,
    People,
    check_keys,
    check_name,
    person_outputs,
    planar,
    positive,
)
from aha_tasks.skills import LOCATION, resolve_pose

COSTMAP_TOPIC = "/global_costmap/costmap"
# Nav2 sends the full costmap once (and on resize); changes come as updates.
COSTMAP_UPDATES_TOPIC = "/global_costmap/costmap_updates"
TF_TIMEOUT = 5.0  # s (wall clock) for the robot pose
POLL = 0.05  # s (wall clock) between checks
GOAL_PERIOD = 1.0  # s (wait clock) at least between replacing goals
GOAL_MOVED = 0.3  # m; a goal this close to the active one is not sent again
GOAL_TURNED = 0.5  # rad
SLACK = 0.3  # m beyond `distance` before the robot moves up again
TOO_CLOSE = 0.3  # m inside `distance` at which the robot stops
# rad off the front at which the robot turns on the spot to a person close by,
# who would otherwise walk past it out of the head's reach
FACE_TURN = 0.8
MOTION_WINDOW = 1.5  # s of sightings for the direction a person walks
MIN_MOTION = 0.2  # m in MOTION_WINDOW for a direction
LETHAL = 99  # OccupancyGrid value of the inscribed and lethal costs
# Costs below this keep about 0.6 m from obstacles (inflation 0.65 m), so the
# robot does not stop where a small localization error puts it in collision.
CLEAR = 50
APPROACH_STEP = math.radians(30)  # rad between alternative goals about the person
APPROACH_TRIES = 7
ACQUIRE_RANGE = 3.0  # m from the remembered position, or from the robot
ACQUIRE_TIMEOUT = 10.0  # s (wait clock) to see the person at the start
ACQUIRE_PANS = (0.0, 0.6, -0.6)  # rad, + = left
ACQUIRE_DWELL = 2.0  # s per pan
REACQUIRE = 0.5  # m from the last sighting, as aha_tasks.gaze
REACQUIRE_SPEED = 0.8  # m/s the radius grows while the person is lost
REACQUIRE_MAX = 4.0  # m
STILL = 0.35  # m of measured motion that still counts as standing
CAUGHT_UP = 0.8  # m beyond `distance` within which the robot is next to them
MIN_TRAVEL = 1.0  # m the person walks before a stop can end following
SETTLE = 15.0  # s for the last goal once the person stopped
SEARCH_AFTER = 2.0  # s lost before searching
SEARCH_PANS = (0.0, 0.9, -0.9, 1.5, -1.5)  # rad
SEARCH_DWELL = 1.5  # s per pan
LOOK_HEIGHT = 1.4  # m, a standing person's face
HEAD_REACH = 1.4  # rad of head pan used (limit 1.57) to look back
FOLLOWER_GAP = 1.2  # m behind the robot where a follower is expected
TRAIL_STEP = 0.15  # m between remembered robot positions
TRAIL_NEAR = 1.2  # m from the robot's path for a follower
GLANCE = 3.0  # s for a glance back with the head (it turns at 1 rad/s)
LOOK = 5.0  # s for a look back after turning
MAX_NAV_FAILURES = 3  # aborted goals to the destination in a row
NAV_PROGRESS = 1.0  # m driven after an abort before aborts are counted afresh
SUCCEEDED, ACTIVE, FAILED, IDLE = "succeeded", "active", "failed", "idle"
FOLLOW_KEYS = (
    "location",
    "role",
    "distance",
    "radius",
    "stop_time",
    "timeout",
    "lost_timeout",
)
GUIDE_KEYS = ("role", "gap", "check_every", "wait", "calls", "timeout")


# ---- geometry and decisions (ROS-free) --------------------------------------


def bearing(pose, point):
    """Angle (rad, + = left) of `point` from the front of `pose` [x, y, yaw]."""
    angle = math.atan2(point[1] - pose[1], point[0] - pose[0])
    return math.remainder(angle - pose[2], math.tau)


def turn_to_see(pose, point, reach=HEAD_REACH):
    """Base yaw that puts `point` within `reach` of the front (the head does the rest)."""
    relative = bearing(pose, point)
    if abs(relative) <= reach:
        return pose[2]
    return math.remainder(pose[2] + relative - math.copysign(reach, relative), math.tau)


def approach_poses(robot, person, distance, step=APPROACH_STEP, tries=APPROACH_TRIES):
    """Poses `distance` from the person facing them: on the robot's side first.

    Then turned by ±step, ±2·step, ... about the person. When the robot is
    already that close, only its own position, turned to the person.
    """
    if math.dist(robot[:2], person) <= distance:
        return [facing(robot, person)]
    side = math.atan2(robot[1] - person[1], robot[0] - person[0])
    poses = []
    for k in range(tries):
        angle = side + step * ((k + 1) // 2) * (1 if k % 2 else -1)
        x = person[0] + distance * math.cos(angle)
        y = person[1] + distance * math.sin(angle)
        poses.append([x, y, math.atan2(person[1] - y, person[0] - x)])
    return poses


def facing(robot, point):
    """The robot's own position, turned to `point`."""
    return [robot[0], robot[1], math.atan2(point[1] - robot[1], point[0] - robot[0])]


def grid_cost(grid, x, y):
    """OccupancyGrid value (-1 unknown, 0..100) at a map point; None outside or no grid."""
    if grid is None:
        return None
    info = grid.info
    col = math.floor((x - info.origin.position.x) / info.resolution)
    row = math.floor((y - info.origin.position.y) / info.resolution)
    if not (0 <= col < info.width and 0 <= row < info.height):
        return None
    return grid.data[row * info.width + col]


def apply_update(grid, update):
    """Write an OccupancyGridUpdate patch into `grid`; False if it does not fit."""
    if grid is None:
        return False
    width, height = grid.info.width, grid.info.height
    if (
        update.x < 0
        or update.y < 0
        or update.x + update.width > width
        or update.y + update.height > height
        or len(update.data) != update.width * update.height
    ):
        return False
    for row in range(update.height):
        start = (update.y + row) * width + update.x
        patch = update.data[row * update.width : (row + 1) * update.width]
        grid.data[start : start + update.width] = patch
    return True


def free_pose(poses, grid):
    """The first pose clear of obstacles on `grid`, else the first not in collision.

    None if every pose is inscribed, lethal or unknown; the first without a grid.
    """
    if grid is None:
        return poses[0]
    costs = [grid_cost(grid, pose[0], pose[1]) for pose in poses]
    for limit in (CLEAR, LETHAL):
        for pose, cost in zip(poses, costs):
            if cost is not None and 0 <= cost < limit:
                return pose
    return None


def needs_goal(goal, active, sent_at, state, now, period=GOAL_PERIOD):
    """Whether to send `goal` in place of `active` (sent at `sent_at`, in `state`).

    At most one goal per `period`; a running goal is replaced only by one that
    moved or turned enough, an ended one by any.
    """
    if active is None:
        return True
    if now - sent_at < period:
        return False
    return state != ACTIVE or moved(goal, active)


def moved(goal, active):
    return (
        math.dist(goal[:2], active[:2]) >= GOAL_MOVED
        or abs(math.remainder(goal[2] - active[2], math.tau)) >= GOAL_TURNED
    )


class Target:
    """The person followed or guided, across lost tracks."""

    def __init__(self, track=None, name="", position=None):
        self.track, self.name = track, name
        self.position = position  # (x, y) in map of the last sighting
        self.seen = None  # wait-clock time of the last sighting
        self.others = set()  # tracks seen together with the person: someone else
        self.start = None
        self.travelled = 0.0  # m, farthest from where they were first seen
        self.recent = []  # (time, x, y) of the sightings in MOTION_WINDOW

    def reacquire_radius(self, now):
        lost = 0.0 if self.seen is None else max(0.0, now - self.seen)
        return min(REACQUIRE + REACQUIRE_SPEED * lost, REACQUIRE_MAX)

    def find(self, people, now, origin=None, acquire=ACQUIRE_RANGE):
        """The person among `people` (visible ones count), or None; nothing changes.

        By track, then by name, then the nearest person near the last sighting
        (before the first: near the remembered position, else near `origin`)
        who is not someone seen together with the person and has no other name.
        """
        visible = [p for p in people if p.visible]
        for person in visible:
            if self.track is not None and person.id == self.track:
                return person
        if self.name:
            for person in visible:
                if person.name == self.name:
                    return person
        first = self.seen is None
        centre = self.position if self.position is not None else origin
        if centre is None:
            return None
        radius = acquire if first else self.reacquire_radius(now)
        near = [
            (math.dist(planar(p.position), centre), p)
            for p in visible
            if p.id not in self.others and (first or not p.name or p.name == self.name)
        ]
        near = [(d, p) for d, p in near if d <= radius]
        return min(near, key=lambda item: item[0])[1] if near else None

    def saw(self, person, people, now):
        """Take `person` as the target, seen at `now` with `people`."""
        self.track = person.id
        self.name = person.name or self.name
        self.position = planar(person.position)
        self.seen = now
        self.others.discard(person.id)
        self.others.update(p.id for p in people if p.visible and p.id != person.id)
        if self.start is None:
            self.start = self.position
        self.travelled = max(self.travelled, math.dist(self.start, self.position))
        self.recent = [r for r in self.recent if now - r[0] <= MOTION_WINDOW]
        self.recent.append((now, *self.position))

    def ahead(self, distance=1.0):
        """`distance` m past the last sighting the way they walked, else the sighting."""
        (_, x0, y0), (_, x, y) = self.recent[0], self.recent[-1]
        moved = math.hypot(x - x0, y - y0)
        if moved < MIN_MOTION:
            return self.position
        return (x + distance * (x - x0) / moved, y + distance * (y - y0) / moved)

    def update(self, people, now, origin=None):
        """find + saw; the person or None."""
        person = self.find(people, now, origin)
        if person is not None:
            self.saw(person, people, now)
        return person

    def lost_for(self, now):
        return math.inf if self.seen is None else now - self.seen


class Stillness:
    """How long the person has stood still with the robot next to them."""

    def __init__(self, still=STILL):
        self.still = still
        self.anchor = None  # (x, y, since)

    def update(self, position, now, close):
        if not close:
            self.anchor = None
        elif self.anchor is None or math.dist(position, self.anchor[:2]) > self.still:
            self.anchor = (*position, now)
        return 0.0 if self.anchor is None else now - self.anchor[2]


def follow_done(still_for, stop_time, travelled, position, area, min_travel=MIN_TRAVEL):
    """Following ends: stood still long enough after walking, in the area if any."""
    if still_for < stop_time or travelled < min_travel:
        return False
    return area is None or math.dist(position, area[:2]) <= area[2]


def trail_point(trail, back):
    """The point `back` m behind the end of the polyline `trail` (its start if shorter)."""
    left = back
    for (x1, y1), (x0, y0) in zip(reversed(trail), list(reversed(trail))[1:]):
        step = math.hypot(x1 - x0, y1 - y0)
        if step >= left and step > 0.0:
            return (x1 + (x0 - x1) * left / step, y1 + (y0 - y1) * left / step)
        left -= step
    return tuple(trail[0])


def distance_to_trail(point, trail):
    if len(trail) == 1:
        return math.dist(point, trail[0])
    best = math.inf
    for a, b in zip(trail, trail[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = 0.0
        if length > 0.0:
            t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length
            t = min(1.0, max(0.0, t))
        best = min(best, math.dist(point, (a[0] + t * dx, a[1] + t * dy)))
    return best


def follower(people, target, robot, trail, gap, near_trail=TRAIL_NEAR):
    """The visible person following the robot, `gap` m from it at most, or None.

    The target by track or name; else the nearest unnamed person (or one with
    the target's name) near the robot's path who was not seen together with
    the target.
    """
    close = [
        p
        for p in people
        if p.visible and math.dist(planar(p.position), robot[:2]) <= gap
    ]
    for person in close:
        if target.track is not None and person.id == target.track:
            return person
        if target.name and person.name == target.name:
            return person
    candidates = [
        p
        for p in close
        if p.id not in target.others
        and (not p.name or p.name == target.name)
        and distance_to_trail(planar(p.position), trail) <= near_trail
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda p: math.dist(planar(p.position), robot[:2]))


def remembered(memory, role):
    """The target that find_described_person left in memory (track, map position)."""
    track = memory.get(f"{role}.track")
    position = memory.get(f"{role}.position")
    xy = None
    if (
        memory.get(f"{role}.frame") == "map"
        and isinstance(position, list)
        and len(position) >= 2
    ):
        xy = (float(position[0]), float(position[1]))
    return Target(track=track if type(track) is int else None, position=xy)


def spoken(label):
    return label.replace("_", " ")


def address(person):
    """'Charlie, ' for a name, '' for a description or the operator."""
    person = person.strip()
    if person and person[0].isupper() and " " not in person:
        return f"{person}, "
    return ""


def rounded(pose):
    return [round(v, 3) for v in pose]


# ---- ROS access ---------------------------------------------------------------


class ClockWatch:
    """Whether the wait clock stopped (Gazebo stopped), as People.wait_for checks.

    A loop on the wait clock alone would never time out then.
    """

    def __init__(self, timeout=CLOCK_STALL_TIMEOUT):
        self.timeout = timeout
        self.last = None  # wait-clock time last seen to advance
        self.progress = time.monotonic()  # wall time of that

    def stalled(self, now):
        wall = time.monotonic()
        if self.last is None or now > self.last:
            self.last, self.progress = now, wall
            return False
        return wall - self.progress >= self.timeout


STALLED = (
    f"No /clock progress for {CLOCK_STALL_TIMEOUT:.0f} s; is the simulation running?"
)


class Base:
    """Robot pose in map from TF, and Nav2's global costmap, for one runner."""

    def __init__(self, node):
        from map_msgs.msg import OccupancyGridUpdate
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tf2_ros import Buffer, TransformListener

        self.node = node
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        self.grid = None
        node.create_subscription(
            OccupancyGrid,
            COSTMAP_TOPIC,
            self._on_costmap,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        node.create_subscription(
            OccupancyGridUpdate, COSTMAP_UPDATES_TOPIC, self._on_update, 10
        )

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "base", None) is None:
            runtime.base = cls(runtime.node)
        return runtime.base

    def _on_costmap(self, message):
        self.grid = message  # its data is patched in place by the updates

    def _on_update(self, message):
        apply_update(self.grid, message)

    def pose(self):
        """[x, y, yaw] of base_footprint in map with the latest TF."""
        from rclpy.time import Time
        from tf2_ros import TransformException

        deadline = time.monotonic() + TF_TIMEOUT
        while True:
            try:
                t = self.tf.lookup_transform("map", "base_footprint", Time())
                break
            except TransformException:
                if time.monotonic() > deadline:
                    raise TimeoutError("no TF map -> base_footprint") from None
            time.sleep(0.05)
        p, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        return [p.x, p.y, yaw]


class Pursuit:
    """NavigateToPose goals that replace each other without waiting.

    Nav2 preempts the running goal with a new one and keeps going; only the
    latest goal is tracked.
    """

    def __init__(self, navigation, node):
        from nav2_msgs.action import NavigateToPose

        self.navigation = navigation
        self.node = node
        self.Goal = NavigateToPose.Goal
        self.goal = None  # [x, y, yaw] of the latest goal
        self.sent_at = -math.inf
        self.request = None  # future of the goal handle
        self.result = None  # future of the result

    def send(self, pose, now):
        from geometry_msgs.msg import PoseStamped

        target = PoseStamped()
        target.header.frame_id = "map"
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y = map(float, pose[:2])
        target.pose.orientation.z = math.sin(pose[2] / 2)
        target.pose.orientation.w = math.cos(pose[2] / 2)
        self.request = self.navigation.client.send_goal_async(self.Goal(pose=target))
        self.result = None
        self.goal, self.sent_at = list(pose), now

    def state(self):
        """IDLE, ACTIVE, SUCCEEDED or FAILED of the latest goal."""
        from action_msgs.msg import GoalStatus

        if self.request is None:
            return IDLE
        if not self.request.done():
            return ACTIVE
        handle = self.request.result()
        if handle is None or not handle.accepted:
            return FAILED
        if self.result is None:
            self.result = handle.get_result_async()
        if not self.result.done():
            return ACTIVE
        status = self.result.result().status
        return SUCCEEDED if status == GoalStatus.STATUS_SUCCEEDED else FAILED

    def stop(self):
        """Cancel the latest goal without waiting for it."""
        if self.request is not None and self.state() == ACTIVE:
            self.request.add_done_callback(cancel_if_accepted)
        self.request = self.result = self.goal = None

    def halt(self):
        """Cancel the latest goal and wait until Nav2 ended it (best effort, 5 s)."""
        request = self.request
        if request is None or self.state() != ACTIVE:
            self.request = self.result = self.goal = None
            return
        self.request = self.result = self.goal = None
        try:
            handle = self.navigation.wait(request, 5.0)
            if handle.accepted:
                done = handle.get_result_async()
                self.navigation.wait(handle.cancel_goal_async(), 5.0)
                self.navigation.wait(done, 5.0)
        except (TimeoutError, RuntimeError):
            pass

    def go(self, pose, people, cancelled, timeout):
        """Send `pose` and wait for the end: (ok, message)."""
        self.send(pose, people.now())
        return self.finish(people, cancelled, timeout)

    def finish(self, people, cancelled, timeout):
        """Wait for the latest goal to end, cancelling it after `timeout`: (ok, message)."""
        deadline = people.now() + timeout
        clock = ClockWatch()
        while (state := self.state()) == ACTIVE:
            if cancelled.is_set():
                self.halt()
                return False, "Cancelled"
            if clock.stalled(people.now()):
                self.halt()
                return False, STALLED
            if people.now() > deadline:
                self.halt()
                return False, f"Nav2 did not finish within {timeout:.0f} s"
            time.sleep(POLL)
        return state == SUCCEEDED, "" if state == SUCCEEDED else "Nav2 failed"


def check_target_params(params, spec, required, optional):
    check_keys(params, required, optional)
    check_name(params, "person")
    if "role" in params:
        check_name(params, "role")
    if "location" in params:
        if not isinstance(params["location"], str):
            raise ValueError(f"location must be a label: {params}")
        resolve_pose(spec, params["location"])


class _PersonSkill(Skill):
    """Shared start: the person in view and the robot's pose."""

    def ready(self):
        # TF, the costmap and the TTS are found before the first use.
        Base.of(self.runtime)
        Interaction.of(self.runtime).find_tts()

    def _start(self, ctx, timeout=ACQUIRE_TIMEOUT):
        """(people, base, target, robot pose) or a failed SkillResult."""
        if self.runtime.navigation is None:
            return SkillResult(False, "Needs Nav2")
        people = People.of(self.runtime)
        base = Base.of(self.runtime)
        try:
            robot = base.pose()
        except TimeoutError as error:
            return SkillResult(False, str(error))
        target = remembered(ctx.memory, ctx.params.get("role", "person"))
        if target.position is not None and (
            math.dist(robot[:2], target.position) > ACQUIRE_RANGE
        ):
            # find_described_person leaves the robot next to the person: one
            # remembered far away is from an earlier step (e.g. an earlier
            # command) and is not looked for.
            ctx.log("The remembered person is far away: looking around the robot")
            target = Target()
        pans = itertools.cycle(ACQUIRE_PANS)
        next_look = -math.inf
        if target.position is not None:
            people.look_at(
                "map",
                SimpleNamespace(
                    x=target.position[0], y=target.position[1], z=LOOK_HEIGHT
                ),
            )
            next_look = people.now() + ACQUIRE_DWELL

        def choose(message):
            nonlocal next_look
            if message.header.frame_id != "map":
                return None
            person = target.find(message.people, people.now(), origin=robot[:2])
            if person is not None:
                return message, person
            if people.now() >= next_look:
                people.look_towards(next(pans))
                next_look = people.now() + ACQUIRE_DWELL
            return None

        found, reason = people.wait_for(choose, timeout, ctx.cancelled)
        if found is None:
            return SkillResult(False, f"No {ctx.params['person']} in view: {reason}")
        message, person = found
        target.saw(person, message.people, people.now())
        people.follow(track=person.id, name=person.name)
        ctx.log(
            f"Person {person.id} at [{target.position[0]:.2f}, {target.position[1]:.2f}]"
        )
        return people, base, target, robot


class FollowPerson(_PersonSkill):
    """Follow the person found before until they stop (at `location` if given).

    params: {"person": "<name or description, spoken>", "location": "<label>",
    "role": "person", "distance": 1.0, "radius": 2.0, "stop_time": 5.0,
    "timeout": 300.0, "lost_timeout": 30.0}. The target is `<role>.track` /
    `.position` from find_described_person, else the nearest person in view.
    Outputs the robot's pose as robot.location and the person's last sighting
    as `<role>.track` / `.position` / `.frame`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_target_params(params, spec, {"person"}, FOLLOW_KEYS)
        positive(params, "distance", 1.0)
        positive(params, "radius", 2.0)
        positive(params, "stop_time", 5.0)
        positive(params, "timeout", 300.0)
        positive(params, "lost_timeout", 30.0)

    def expected(self, params):
        return {LOCATION: params["location"]} if "location" in params else {}

    def execute(self, ctx):
        started = self._start(ctx)
        if isinstance(started, SkillResult):
            return started
        people, base, target, _ = started
        params = ctx.params
        role = params.get("role", "person")
        to = f" to the {spoken(params['location'])}" if "location" in params else ""
        Interaction.of(self.runtime).say(
            f"{address(params['person'])}I will follow you{to}. Please walk ahead.",
            ctx.cancelled,
        )
        pursuit = Pursuit(self.runtime.navigation, self.runtime.node)
        try:
            ok, text, sighting = self._follow(ctx, people, base, target, pursuit)
        finally:
            pursuit.halt()
        if not ok:
            return SkillResult(False, text)
        outputs = person_outputs(role, *sighting)
        outputs[LOCATION] = rounded(base.pose())
        return SkillResult(True, text, outputs)

    def _follow(self, ctx, people, base, target, pursuit):
        """(ok, message, (people message, person) of the last sighting)."""
        params = ctx.params
        distance = positive(params, "distance", 1.0)
        stop_time = positive(params, "stop_time", 5.0)
        lost_timeout = positive(params, "lost_timeout", 30.0)
        area = None
        if "location" in params:
            x, y, _ = resolve_pose(self.runtime.spec, params["location"])
            area = (x, y, positive(params, "radius", 2.0))
        deadline = people.now() + positive(params, "timeout", 300.0)
        stillness = Stillness()
        last, sighting = None, None
        gazed = target.track
        search = None
        clock = ClockWatch()
        while not ctx.cancelled.is_set():
            now = people.now()
            if clock.stalled(now):
                return False, STALLED, None
            if now >= deadline:
                return (
                    False,
                    f"Timed out; the person walked {target.travelled:.1f} m",
                    None,
                )
            robot = base.pose()
            message = people.latest()
            if message is None or message is last:
                if message is None:
                    # person_perception is silent (stopped, DGX lost): do not
                    # drive on to where the person was.
                    pursuit.stop()
                if target.lost_for(now) > lost_timeout:
                    return False, f"Lost the person for {lost_timeout:.0f} s", None
                time.sleep(POLL)
                continue
            last = message
            if message.header.frame_id != "map":
                return False, "People are not in map", None
            person = target.update(message.people, now)
            if person is None:
                if target.lost_for(now) > lost_timeout:
                    return False, f"Lost the person for {lost_timeout:.0f} s", None
                if target.lost_for(now) > SEARCH_AFTER:
                    search = self._search(
                        search, people, base, target, robot, pursuit, now
                    )
                continue
            sighting = (message, person)
            if search is not None or target.track != gazed:
                if search is not None:
                    ctx.log(f"Found the person again: track {person.id}")
                people.follow(track=person.id, name=person.name)
                gazed, search = target.track, None
            gap = math.dist(robot[:2], target.position)
            still_for = stillness.update(
                target.position, now, gap <= distance + CAUGHT_UP
            )
            if follow_done(
                still_for, stop_time, target.travelled, target.position, area
            ):
                where = f" at the {spoken(params['location'])}" if area else ""
                text = f"The person stopped{where} after {target.travelled:.1f} m"
                # Come up to `distance` from them rather than stop on the way,
                # e.g. in a doorway.
                pursuit.finish(people, ctx.cancelled, SETTLE)
                return True, text, sighting
            goal = None
            if gap > distance + SLACK:
                goal = free_pose(
                    approach_poses(robot, target.position, distance), base.grid
                )
            elif abs(bearing(robot, target.position)) > FACE_TURN:
                goal = facing(robot, target.position)
            elif gap < distance - TOO_CLOSE:
                pursuit.stop()
            if goal is not None and needs_goal(
                goal, pursuit.goal, pursuit.sent_at, pursuit.state(), now
            ):
                pursuit.send(goal, now)
        return False, "Cancelled", None

    def _search(self, search, people, base, target, robot, pursuit, now):
        """One step of looking for the lost person; returns the search state.

        Drives to where they were last seen, facing the way they walked (or
        turns there if it is not free), then turns the head through
        SEARCH_PANS, turns the base around and turns the head again.
        """
        if search is None:
            ahead = target.ahead()
            pose = free_pose(approach_poses(robot, ahead, 1.0), base.grid)
            pursuit.send(pose or facing(robot, ahead), now)
            people.look_at(
                "map", SimpleNamespace(x=ahead[0], y=ahead[1], z=LOOK_HEIGHT)
            )
            return {"stage": "go", "pans": None, "until": -math.inf, "turned": False}
        if search["stage"] in ("go", "turn"):
            if pursuit.state() == ACTIVE:
                return search
            turned = search["stage"] == "turn"
            search.update(stage="scan", pans=iter(SEARCH_PANS), turned=turned)
        if now < search["until"]:
            return search
        pan = next(search["pans"], None)
        if pan is not None:
            people.look_towards(pan, LOOK_HEIGHT)
            search["until"] = now + SEARCH_DWELL
        elif not search["turned"]:
            pursuit.send([robot[0], robot[1], robot[2] + math.pi], now)
            search["stage"] = "turn"
        return search


class GuidePerson(_PersonSkill):
    """Guide the person to `location`, checking that they keep following.

    params: {"person": "<name or description, spoken>", "location": "<label>",
    "role": "person", "gap": 2.5, "check_every": 4.0, "wait": 10.0,
    "calls": 2, "timeout": 300.0}. Outputs robot.location and the follower's
    last sighting as `<role>.track` / `.position` / `.frame`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_target_params(params, spec, {"person", "location"}, GUIDE_KEYS)
        positive(params, "gap", 2.5)
        positive(params, "check_every", 4.0)
        positive(params, "wait", 10.0)
        positive(params, "timeout", 300.0)
        calls = params.get("calls", 2)
        if type(calls) is not int or calls < 0:
            raise ValueError(f"calls must be a count: {params}")

    def expected(self, params):
        return {LOCATION: params["location"]}

    def execute(self, ctx):
        started = self._start(ctx)
        if isinstance(started, SkillResult):
            return started
        people, base, target, robot = started
        params = ctx.params
        role = params.get("role", "person")
        location = params["location"]
        destination = resolve_pose(self.runtime.spec, location)
        interaction = Interaction.of(self.runtime)
        name = address(params["person"])
        interaction.say(
            f"{name}please follow me. I will take you to the {spoken(location)}.",
            ctx.cancelled,
        )
        trail = [target.position, tuple(robot[:2])]
        pursuit = Pursuit(self.runtime.navigation, self.runtime.node)
        try:
            ok, text, sighting = self._guide(
                ctx, people, base, target, pursuit, trail, destination
            )
        finally:
            pursuit.halt()
        if not ok:
            return SkillResult(False, text)
        interaction.say(
            f"{name}we have arrived at the {spoken(location)}.", ctx.cancelled
        )
        outputs = person_outputs(role, *sighting)
        outputs[LOCATION] = rounded(base.pose())
        return SkillResult(True, text, outputs)

    def _guide(self, ctx, people, base, target, pursuit, trail, destination):
        """(ok, message, (people message, follower) at the end)."""
        params = ctx.params
        every = positive(params, "check_every", 4.0)
        deadline = people.now() + positive(params, "timeout", 300.0)
        gaze = self.runtime.gaze
        travelled, failures, checks = 0.0, 0, 0
        failed_at = None  # `travelled` at the last abort
        pursuit.send(destination, people.now())
        gaze.follow_path(destination)
        clock = ClockWatch()
        while not ctx.cancelled.is_set():
            if clock.stalled(people.now()):
                return False, STALLED, None
            if people.now() >= deadline:
                return False, "Timed out on the way", None
            robot = base.pose()
            step = math.dist(trail[-1], robot[:2])
            if step >= TRAIL_STEP:
                trail.append(tuple(robot[:2]))
                travelled += step
            state = pursuit.state()
            if state == SUCCEEDED:
                break
            if state == FAILED:
                failures += 1
                failed_at = travelled
                if failures >= MAX_NAV_FAILURES:
                    return False, "Nav2 could not reach the destination", None
                pursuit.send(destination, people.now())
            elif failed_at is not None and travelled - failed_at >= NAV_PROGRESS:
                # A goal stays ACTIVE until Nav2 aborts it: only driving on
                # shows that the way is clear again.
                failures, failed_at = 0, None
            far = math.dist(robot[:2], destination[:2]) > 1.0
            if travelled >= every * (checks + 1) and far:
                checks += 1
                found = self._check(ctx, people, base, target, pursuit, trail)
                if found is None:
                    return False, f"{params['person']} did not follow", None
                if ctx.cancelled.is_set():
                    break
                pursuit.send(destination, people.now())
                gaze.follow_path(destination)
            time.sleep(POLL)
        if ctx.cancelled.is_set():
            return False, "Cancelled", None
        gaze.hold()
        found = self._check(ctx, people, base, target, pursuit, trail)
        if found is None:
            return False, f"Arrived without {params['person']}", None
        people.follow(track=found[1].id, name=found[1].name)
        text = f"Guided the person {travelled:.1f} m; checks on the way: {checks}"
        return True, text, found

    def _check(self, ctx, people, base, target, pursuit, trail):
        """(message, follower) once someone follows, else None after calling them."""
        gap = positive(ctx.params, "gap", 2.5)
        robot = base.pose()
        expected = trail_point(trail, FOLLOWER_GAP)
        look = SimpleNamespace(x=expected[0], y=expected[1], z=LOOK_HEIGHT)
        if abs(bearing(robot, expected)) <= HEAD_REACH:
            people.look_at("map", look)
            found = self._wait_follower(ctx, people, base, target, trail, gap, GLANCE)
            if found is not None or ctx.cancelled.is_set():
                return found
        pursuit.halt()
        robot = base.pose()
        yaw = turn_to_see(robot, expected)
        if abs(math.remainder(yaw - robot[2], math.tau)) > 0.05:
            ctx.log(f"Turning {math.degrees(yaw - robot[2]):+.0f} deg to look back")
            pursuit.go([robot[0], robot[1], yaw], people, ctx.cancelled, 60.0)
        people.look_at("map", look)
        found = self._wait_follower(ctx, people, base, target, trail, gap, LOOK)
        calls = ctx.params.get("calls", 2)
        wait = positive(ctx.params, "wait", 10.0)
        interaction = Interaction.of(self.runtime)
        for call in range(calls):
            if found is not None or ctx.cancelled.is_set():
                break
            ctx.log(f"Nobody follows within {gap:.1f} m: calling ({call + 1}/{calls})")
            interaction.say(
                f"{address(ctx.params['person'])}I am here. Please follow me.",
                ctx.cancelled,
            )
            found = self._wait_follower(ctx, people, base, target, trail, gap, wait)
        if found is not None:
            message, person = found
            ctx.log(
                f"Person {person.id} follows "
                f"{math.dist(base.pose()[:2], planar(person.position)):.1f} m behind"
            )
        return found

    def _wait_follower(self, ctx, people, base, target, trail, gap, timeout):
        def choose(message):
            if message.header.frame_id != "map":
                return None
            robot = base.pose()
            person = follower(message.people, target, robot, trail, gap)
            if person is None:
                return None
            target.saw(person, message.people, people.now())
            return message, person

        found, _ = people.wait_for(choose, timeout, ctx.cancelled)
        return found
