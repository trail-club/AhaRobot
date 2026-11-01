"""Skills on aha_perception person_perception: find, memorize and locate people.

Tracks come from /aha/perception/people. A skill outputs `<role>.track` (the
track ID) and `<role>.position` ([x, y, z] in the people frame) so that later
skills can address the same person; a remembered name survives lost tracks.
Speech is not part of these skills.
"""

import math
import threading
import time

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.skills import LOCATION, resolve_pose

PEOPLE_TOPIC = "/aha/perception/people"
REMEMBER_SERVICE = "/aha/perception/remember_person"
DESCRIBE_SERVICE = "/aha/perception/describe_person"
LOOK_AT_TOPIC = "/aha/perception/look_at"
STALE_AFTER = 2.0  # s (wall clock) without a people message


def planar(point):
    return (point.x, point.y)


def choose_person(people, origin, taken=()):
    """The nearest visible person whose name is not taken, or None."""
    candidates = [p for p in people if p.visible and p.name not in taken]
    if not candidates:
        return None
    return min(candidates, key=lambda p: math.dist(planar(p.position), origin))


def approach_pose(origin, target, distance):
    """[x, y, yaw] `distance` short of `target` on the way from `origin`, facing it."""
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    yaw = math.atan2(dy, dx)
    gap = math.hypot(dx, dy)
    if gap <= distance:  # already close enough: turn on the spot
        return [origin[0], origin[1], yaw]
    return [
        target[0] - distance * math.cos(yaw),
        target[1] - distance * math.sin(yaw),
        yaw,
    ]


def robot_xy(runtime, memory):
    value = memory.get(LOCATION)
    if value is None:
        return (0.0, 0.0)
    return tuple(resolve_pose(runtime.spec, value)[:2])


def positive(params, key, default):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number: {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive: {value!r}")
    return float(value)


def check_keys(params, required, optional=()):
    unknown = set(params) - set(required) - set(optional)
    missing = set(required) - set(params)
    if unknown or missing:
        raise ValueError(
            f"needs {sorted(required)}, may have {sorted(optional)}: {params}"
        )


def check_name(params, key):
    if not isinstance(params.get(key), str) or not params[key].strip():
        raise ValueError(f"{key} must be a nonempty name: {params}")


class People:
    """ROS access to person_perception, shared by the skills of one runner."""

    def __init__(self, node):
        from aha_msgs.msg import PersonArray
        from aha_msgs.srv import DescribePerson, RememberPerson
        from geometry_msgs.msg import PointStamped

        self.node = node
        self.lock = threading.Lock()
        self.message = None
        self.received = -math.inf
        node.create_subscription(PersonArray, PEOPLE_TOPIC, self._on_people, 10)
        self.remember = node.create_client(RememberPerson, REMEMBER_SERVICE)
        self.describe = node.create_client(DescribePerson, DESCRIBE_SERVICE)
        self.look_at_pub = node.create_publisher(PointStamped, LOOK_AT_TOPIC, 10)
        self.PointStamped = PointStamped
        self.RememberPerson, self.DescribePerson = RememberPerson, DescribePerson

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "people", None) is None:
            runtime.people = cls(runtime.node)
        return runtime.people

    def _on_people(self, message):
        with self.lock:
            self.message, self.received = message, time.monotonic()

    def latest(self):
        with self.lock:
            if time.monotonic() - self.received > STALE_AFTER:
                return None
            return self.message

    def wait_for(self, choose, timeout, cancelled):
        """Poll the latest people until `choose(message)` returns a value."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancelled.is_set():
                return None, "Cancelled"
            message = self.latest()
            if message is not None:
                found = choose(message)
                if found is not None:
                    return found, ""
            time.sleep(0.1)
        if self.latest() is None:
            return None, f"No {PEOPLE_TOPIC}; is person_perception running?"
        return None, f"Not found within {timeout:.0f} s"

    def call(self, client, request, timeout, cancelled):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{client.srv_name} is not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if cancelled.is_set():
                future.cancel()
                raise RuntimeError("Cancelled")
            if time.monotonic() > deadline:
                future.cancel()
                raise TimeoutError(f"{client.srv_name} did not answer")
            time.sleep(0.05)
        return future.result()

    def look_at(self, frame, point):
        target = self.PointStamped()
        target.header.frame_id = frame
        target.point = point
        self.look_at_pub.publish(target)


def person_outputs(role, message, person):
    p = person.position
    return {
        f"{role}.track": int(person.id),
        f"{role}.position": [round(p.x, 3), round(p.y, 3), round(p.z, 3)],
        f"{role}.frame": message.header.frame_id,
    }


class FindPerson(Skill):
    """Find the nearest visible person without a name, look at them and face them.

    params: {"role": "guest1", "distance": 1.0, "timeout": 30.0}. Without
    navigation, or when the people are not in `map`, the robot only looks.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"role"}, {"distance", "timeout"})
        check_name(params, "role")
        positive(params, "distance", 1.0)
        positive(params, "timeout", 30.0)

    def execute(self, ctx):
        people = People.of(self.runtime)
        role = ctx.params["role"]
        origin = robot_xy(self.runtime, ctx.memory)

        def choose(message):
            # Remembered people (the first guest, the host) are not new arrivals.
            person = choose_person([p for p in message.people if not p.name], origin)
            return None if person is None else (message, person)

        found, reason = people.wait_for(
            choose, positive(ctx.params, "timeout", 30.0), ctx.cancelled
        )
        if found is None:
            return SkillResult(False, f"No person: {reason}")
        message, person = found
        frame = message.header.frame_id
        people.look_at(frame, person.head)
        outputs = person_outputs(role, message, person)
        ctx.log(f"Person {person.id} at {outputs[f'{role}.position']} in {frame}")
        navigation = self.runtime.navigation
        if navigation is None or frame != "map":
            return SkillResult(True, f"Looking at person {person.id}", outputs)
        pose = approach_pose(
            origin, planar(person.position), ctx.params.get("distance", 1.0)
        )
        ok, text = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            return SkillResult(False, f"Could not approach person {person.id}: {text}")
        people.look_at(frame, person.head)
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return SkillResult(True, f"Facing person {person.id}", outputs)


class MemorizePerson(Skill):
    """Remember a guest's appearance by name and ask the VLM for features.

    params: {"guest": "guest1", "timeout": 10.0}. Uses `<guest>.track` from
    find_person, or the nearest unnamed visible person if that track is gone.
    Outputs `<guest>.appearance` (feature phrases) for describe_person.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guest"}, {"timeout", "describe_timeout"})
        check_name(params, "guest")
        positive(params, "timeout", 10.0)
        positive(params, "describe_timeout", 180.0)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guest = ctx.params["guest"]
        track = ctx.memory.get(f"{guest}.track")
        origin = robot_xy(self.runtime, ctx.memory)

        def choose(message):
            for person in message.people:
                if person.visible and person.id == track:
                    return message, person
            unnamed = [p for p in message.people if not p.name]
            person = choose_person(unnamed, origin)
            return None if person is None else (message, person)

        found, reason = people.wait_for(
            choose, positive(ctx.params, "timeout", 10.0), ctx.cancelled
        )
        if found is None:
            return SkillResult(False, f"No person to memorize: {reason}")
        message, person = found
        reply = people.call(
            people.remember,
            people.RememberPerson.Request(id=person.id, name=guest),
            10.0,
            ctx.cancelled,
        )
        if not reply.success:
            return SkillResult(False, reply.message)
        ctx.log(f"{reply.message}; asking the VLM for features")
        reply = people.call(
            people.describe,
            people.DescribePerson.Request(id=person.id),
            positive(ctx.params, "describe_timeout", 180.0),
            ctx.cancelled,
        )
        if not reply.success:
            return SkillResult(False, f"Remembered, but no features: {reply.message}")
        features = list(reply.features)
        outputs = person_outputs(guest, message, person)
        outputs[f"{guest}.appearance"] = features
        return SkillResult(
            True, f"{guest}: {', '.join(features) or 'no features'}", outputs
        )


class LocateGuests(Skill):
    """Wait until every remembered guest is visible and output where each is.

    params: {"guests": ["guest1", "guest2"], "timeout": 30.0}. Names come from
    appearance, so guests who swapped seats are still told apart.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guests"}, {"timeout"})
        guests = params["guests"]
        if (
            not isinstance(guests, list)
            or not guests
            or not all(isinstance(g, str) and g.strip() for g in guests)
            or len(set(guests)) != len(guests)
        ):
            raise ValueError(f"guests must be distinct nonempty names: {guests!r}")
        positive(params, "timeout", 30.0)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guests = ctx.params["guests"]

        def choose(message):
            named = {p.name: p for p in message.people if p.visible and p.name}
            if all(g in named for g in guests):
                return message, named
            return None

        found, reason = people.wait_for(
            choose, positive(ctx.params, "timeout", 30.0), ctx.cancelled
        )
        if found is None:
            return SkillResult(False, f"Not all of {', '.join(guests)} seen: {reason}")
        message, named = found
        outputs = {}
        for guest in guests:
            outputs.update(person_outputs(guest, message, named[guest]))
        where = "; ".join(f"{g} {outputs[f'{g}.position']}" for g in guests)
        return SkillResult(True, where, outputs)
