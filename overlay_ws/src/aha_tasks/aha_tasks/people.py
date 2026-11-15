"""Skills on aha_perception person_perception: find, memorize, locate and count people.

Tracks come from /aha/perception/people. A skill outputs `<role>.track` (the
track ID) and `<role>.position` ([x, y, z] in the people frame) so that later
skills can address the same person; a remembered name survives lost tracks.
Described people (GPSR: gesture, posture, clothing) are checked with
/aha/perception/classify_person. The head goes through the runner's
aha_tasks.gaze.Gaze: find_person and memorize_person leave it following the
person, also through the subgoals that come next (the conversation). Speech is
not part of these skills.
"""

import itertools
import math
import threading
import time

from aha_tasks.appearance import consensus
from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.gaze import LOOK_HEIGHT
from aha_tasks.person_description import parse as parse_description
from aha_tasks.skills import LOCATION, resolve_pose

PEOPLE_TOPIC = "/aha/perception/people"
REMEMBER_SERVICE = "/aha/perception/remember_person"
DESCRIBE_SERVICE = "/aha/perception/describe_person"
CLASSIFY_SERVICE = "/aha/perception/classify_person"
STALE_AFTER = 2.0  # s (wall clock) without a people message
CLOCK_STALL_TIMEOUT = 10.0  # s (wall clock) without the wait clock advancing
MIN_GUEST_SEPARATION = 0.4  # m; closer sightings may be the same person
SCAN_PANS = (0.0, 0.4, -0.4)  # rad, + = the robot's left
# Sightings closer than this (m) are one person when counting across head poses.
SAME_PERSON_M = 0.5
SEARCH_RADIUS = 3.0  # m around the location for described people
# s (wait clock) after turning the head to a person before judging them, so
# that new images show them in the middle, raised hands included.
LOOK_SETTLE = 1.5
SKIP_VLM_TIMEOUT = 10.0  # s for ClassifyPerson without the VLM
# s more than describe_timeout for ClassifyPerson with the VLM: the server
# first waits for motion (up to 20 s wall clock) and then up to its vlm_timeout.
SERVER_MARGIN = 30.0
# Judgements of one person; one that fails (lost track, VLM error, timeout)
# is tried again.
MAX_JUDGEMENTS = 2


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


class Cancelled(RuntimeError):
    """The skill was aborted while waiting for person_perception."""


class People:
    """ROS access to person_perception, shared by the skills of one runner."""

    look_settle = LOOK_SETTLE

    def __init__(self, node, gaze):
        from aha_msgs.msg import PersonArray
        from aha_msgs.srv import ClassifyPerson, DescribePerson, RememberPerson

        self.node = node
        self.gaze = gaze
        self.lock = threading.Lock()
        self.message = None
        self.received = -math.inf
        node.create_subscription(PersonArray, PEOPLE_TOPIC, self._on_people, 10)
        self.remember = node.create_client(RememberPerson, REMEMBER_SERVICE)
        self.describe = node.create_client(DescribePerson, DESCRIBE_SERVICE)
        self.classify = node.create_client(ClassifyPerson, CLASSIFY_SERVICE)
        self.RememberPerson, self.DescribePerson = RememberPerson, DescribePerson
        self.ClassifyPerson = ClassifyPerson

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "people", None) is None:
            runtime.people = cls(runtime.node, runtime.gaze)
        return runtime.people

    def _on_people(self, message):
        with self.lock:
            self.message, self.received = message, time.monotonic()

    def latest(self):
        with self.lock:
            if time.monotonic() - self.received > STALE_AFTER:
                return None
            return self.message

    def now(self):
        """Simulation seconds when active; otherwise monotonic wall seconds."""
        clock = self.node.get_clock()
        if not getattr(clock, "ros_time_is_active", False):
            return time.monotonic()
        return clock.now().nanoseconds * 1e-9

    def wait_for(self, choose, timeout, cancelled):
        """Poll the latest people until `choose(message)` returns a value."""
        last_time = self.now()
        deadline = last_time + timeout
        last_progress = time.monotonic()
        while (now := self.now()) < deadline:
            if cancelled.is_set():
                return None, "Cancelled"
            wall_now = time.monotonic()
            if now > last_time:
                last_time, last_progress = now, wall_now
            elif wall_now - last_progress >= CLOCK_STALL_TIMEOUT:
                return None, (
                    f"No /clock progress for {CLOCK_STALL_TIMEOUT:.0f} s; "
                    "is simulation running?"
                )
            message = self.latest()
            if message is not None:
                found = choose(message)
                if found is not None:
                    return found, ""
            time.sleep(0.1)
        if self.latest() is None:
            return None, f"No {PEOPLE_TOPIC}; is person_perception running?"
        return None, f"Not found within {timeout:.0f} s"

    def call(self, client, request, timeout, cancelled, deadline=math.inf):
        """The reply; TimeoutError after `timeout` wall seconds or at `deadline` (now())."""
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{client.srv_name} is not available")
        future = client.call_async(request)
        wall_deadline = time.monotonic() + timeout
        while not future.done():
            if cancelled.is_set():
                future.cancel()
                raise Cancelled("Cancelled")
            if time.monotonic() > wall_deadline or self.now() >= deadline:
                future.cancel()
                raise TimeoutError(f"{client.srv_name} did not answer")
            time.sleep(0.05)
        return future.result()

    def look_at(self, frame, point):
        """Look at a point and stop following (see Gaze)."""
        self.gaze.look_at(frame, point)

    def look_towards(self, pan, height=LOOK_HEIGHT):
        """Turn the head `pan` rad from the base's front, towards `height` at 2 m."""
        self.gaze.look_towards(pan, height)

    def follow(self, track=None, name=""):
        """Keep the head on a person by track, else by name (see Gaze)."""
        self.gaze.follow_person(self, track=track, name=name)

    def following(self):
        """(track, name) of the person the head follows, or None (see Gaze)."""
        return self.gaze.following()

    def pause(self, seconds, cancelled):
        """Wait on the wait clock; gives up after CLOCK_STALL_TIMEOUT more wall seconds."""
        end = self.now() + seconds
        wall_end = time.monotonic() + seconds + CLOCK_STALL_TIMEOUT
        while self.now() < end and time.monotonic() < wall_end:
            if cancelled.is_set():
                return
            time.sleep(0.05)

    def attributes(
        self, person, description, cancelled, vlm_timeout, frame=None, deadline=math.inf
    ):
        """ClassifyPerson answers for the person, or None if they could not be judged.

        The head first turns to the person (when `frame` is given). Depth and
        keypoints answer at once; the VLM is asked only when the description
        still cannot be decided (clothing, unclear posture or gesture). None
        when the track is gone, the service failed or timed out, or `deadline`
        (now()) passed meanwhile; Cancelled is raised when the skill is aborted.
        """
        if frame is not None and self.look_settle > 0:
            # The head point is the top of the mask, which is too low while the
            # head is cut off at the image edge: look twice.
            target = person.head
            for share in (0.5, 0.5):
                self.look_at(frame, target)
                self.pause(
                    min(share * self.look_settle, deadline - self.now()), cancelled
                )
                message = self.latest()
                same = [
                    p
                    for p in (message.people if message is not None else [])
                    if p.id == person.id and p.visible
                ]
                if same:
                    target = same[0].head
        answers = None
        for skip_vlm in (True, False):
            try:
                reply = self.call(
                    self.classify,
                    self.ClassifyPerson.Request(id=person.id, skip_vlm=skip_vlm),
                    # The server waits for motion and then for the VLM up to
                    # its own vlm_timeout; give it time to answer first.
                    SKIP_VLM_TIMEOUT if skip_vlm else vlm_timeout + SERVER_MARGIN,
                    cancelled,
                    deadline,
                )
            except Cancelled:
                raise
            except (TimeoutError, RuntimeError):
                return None
            if not reply.success:
                return None
            answers = {
                "posture": reply.posture,
                "gesture": reply.gesture,
                "garments": list(reply.garments),
                "reasons": getattr(reply, "message", ""),
            }
            if description.matches(name=person.name, **seen(answers)) is not None:
                break
        return answers


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
        people.follow(track=person.id)
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
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return SkillResult(True, f"Facing person {person.id}", outputs)


def positive_int(params, key, default):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer: {value!r}")
    return value


def stamp_of(message):
    """Image time (s) of a people message."""
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


class MemorizePerson(Skill):
    """Remember a guest's appearance by name and ask the VLM for features.

    params: {"guest": "guest1", "timeout": 10.0, "views": 3, "view_interval": 0.5,
    "describe_timeout": 180.0}. Uses `<guest>.track` from find_person, or the
    nearest unnamed visible person if that track is gone. The VLM describes
    the person in `views` images at least `view_interval` s (image time) apart,
    all within `describe_timeout` (wall clock), and `<guest>.appearance` keeps
    the feature phrases most of the views asked for agree on
    (aha_tasks.appearance.consensus), for describe_person.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(
            params,
            {"guest"},
            {"timeout", "describe_timeout", "views", "view_interval"},
        )
        check_name(params, "guest")
        positive(params, "timeout", 10.0)
        positive(params, "describe_timeout", 180.0)
        positive_int(params, "views", 3)
        positive(params, "view_interval", 0.5)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guest = ctx.params["guest"]
        track = ctx.memory.get(f"{guest}.track")
        origin = robot_xy(self.runtime, ctx.memory)
        timeout = positive(ctx.params, "timeout", 10.0)

        def choose(message):
            for person in message.people:
                if person.visible and person.id == track:
                    return message, person
            unnamed = [p for p in message.people if not p.name]
            person = choose_person(unnamed, origin)
            return None if person is None else (message, person)

        found, reason = people.wait_for(choose, timeout, ctx.cancelled)
        if found is None:
            return SkillResult(False, f"No person to memorize: {reason}")
        message, person = found
        people.follow(track=person.id, name=guest)
        reply = people.call(
            people.remember,
            people.RememberPerson.Request(id=person.id, name=guest),
            10.0,
            ctx.cancelled,
        )
        if not reply.success:
            return SkillResult(False, reply.message)
        ctx.log(f"{reply.message}; asking the VLM for features")
        asked = positive_int(ctx.params, "views", 3)
        views, failure = self.describe(ctx, people, message, person, guest, timeout)
        described = sum(view is not None for view in views)
        if not described:
            return SkillResult(False, f"Remembered, but no features: {failure}")
        if described < 2 <= asked:
            ctx.log(
                f"Only {described} of {asked} views described {guest}; "
                "no feature is agreed on"
            )
        features = consensus(views, asked)
        outputs = person_outputs(guest, message, person)
        outputs[f"{guest}.appearance"] = features
        return SkillResult(
            True,
            f"{guest}: {', '.join(features) or 'no features'} ({described} of "
            f"{asked} views described; {asked // 2 + 1} must agree)",
            outputs,
        )

    def describe(self, ctx, people, message, person, guest, timeout):
        """(feature lists of the views, None for a failed one; last failure).

        A later view waits for a newer image with the person by track or by
        name, and asks for the track seen there (a lost guest gets a new one).
        """
        interval = positive(ctx.params, "view_interval", 0.5)
        end = time.monotonic() + positive(ctx.params, "describe_timeout", 180.0)
        views, failure, last, track = [], "", stamp_of(message), person.id

        def newer(message):
            if stamp_of(message) < last + interval:
                return None
            for seen in message.people:
                if seen.visible and (seen.id == track or seen.name == guest):
                    return message, seen
            return None

        for view in range(positive_int(ctx.params, "views", 3)):
            left = end - time.monotonic()
            if left <= 0:
                ctx.log(f"View {view + 1}: describe_timeout passed")
                break
            if view:
                found, reason = people.wait_for(
                    newer, min(timeout, left), ctx.cancelled
                )
                if found is None:
                    ctx.log(f"No new view of {guest}: {reason}")
                    break
                message, seen = found
                last, track = stamp_of(message), seen.id
            try:
                reply = people.call(
                    people.describe,
                    people.DescribePerson.Request(id=track),
                    end - time.monotonic(),
                    ctx.cancelled,
                )
            except TimeoutError as error:
                failure = f"{error} within describe_timeout"
                ctx.log(f"View {view + 1}: {failure}")
                views.append(None)
                break
            if reply.success:
                views.append(list(reply.features))
                ctx.log(f"View {view + 1}: {', '.join(reply.features) or '-'}")
            else:
                failure = reply.message
                views.append(None)
                ctx.log(f"View {view + 1}: {failure}")
        return views, failure


class LocateGuests(Skill):
    """Look around until every remembered guest was seen and output where each is.

    params: {"guests": ["guest1", "guest2"], "timeout": 60.0, "dwell": 10.0}. The
    head stays `dwell` s at each of SCAN_PANS, long enough for the VLM to name a
    track, because the seats need not fit in one image. Names come from
    appearance, so guests who swapped seats are still told apart.
    Sightings are accumulated only in map, with at least 0.4 m between guests.
    Output tracks may already be forgotten; later skills should use names or
    positions to reacquire guests.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guests"}, {"timeout", "dwell"})
        guests = params["guests"]
        if (
            not isinstance(guests, list)
            or not guests
            or not all(isinstance(g, str) and g.strip() for g in guests)
            or len(set(guests)) != len(guests)
        ):
            raise ValueError(f"guests must be distinct nonempty names: {guests!r}")
        positive(params, "timeout", 60.0)
        positive(params, "dwell", 10.0)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guests = ctx.params["guests"]
        dwell = positive(ctx.params, "dwell", 10.0)
        pans = itertools.cycle(SCAN_PANS)
        next_look = -math.inf
        seen = {}

        def choose(message):
            nonlocal next_look
            now = people.now()
            if now >= next_look:
                people.look_towards(next(pans))
                next_look = now + dwell
            # Camera-frame positions from different head poses are not comparable.
            if message.header.frame_id != "map" or any(
                m.header.frame_id != "map" for m, _ in seen.values()
            ):
                seen.clear()
            for person in message.people:
                if person.visible and person.name in guests:
                    for name, (_, previous) in list(seen.items()):
                        if name == person.name:
                            continue
                        if message.header.frame_id == "map":
                            distance = math.dist(
                                planar(previous.position), planar(person.position)
                            )
                        else:
                            distance = math.dist(
                                (
                                    previous.position.x,
                                    previous.position.y,
                                    previous.position.z,
                                ),
                                (
                                    person.position.x,
                                    person.position.y,
                                    person.position.z,
                                ),
                            )
                        if previous.id == person.id or distance < MIN_GUEST_SEPARATION:
                            del seen[name]
                    seen[person.name] = (message, person)
            return seen if all(g in seen for g in guests) else None

        found, reason = people.wait_for(
            choose, positive(ctx.params, "timeout", 60.0), ctx.cancelled
        )
        people.look_towards(0.0)
        if found is None:
            return SkillResult(False, f"Not all of {', '.join(guests)} seen: {reason}")
        outputs = {}
        for guest in guests:
            outputs.update(person_outputs(guest, *found[guest]))
        where = "; ".join(f"{g} {outputs[f'{g}.position']}" for g in guests)
        return SkillResult(True, where, outputs)


def check_description_params(params, required=(), optional=()):
    keys = {"description", "gesture", "pose", "clothing", "name"}
    check_keys(params, set(required), set(optional) | keys)
    return parse_description(params)


def search_area(runtime, memory, params):
    """(x, y, radius) in map: around `location` if given, else around the robot."""
    if "location" in params:
        centre = resolve_pose(runtime.spec, params["location"])[:2]
    else:
        centre = robot_xy(runtime, memory)
    return (
        float(centre[0]),
        float(centre[1]),
        positive(params, "radius", SEARCH_RADIUS),
    )


def in_area(message, person, area):
    """People outside map cannot be placed; they count as inside."""
    if message.header.frame_id != "map":
        return True
    return math.dist(planar(person.position), area[:2]) <= area[2]


def check_search_params(params, spec):
    if "location" in params:
        resolve_pose(spec, params["location"])
    positive(params, "radius", SEARCH_RADIUS)
    positive(params, "dwell", 6.0)
    positive(params, "look_height", 1.0)
    positive(params, "describe_timeout", 180.0)
    pans = params.get("pans", list(SCAN_PANS))
    if (
        not isinstance(pans, list)
        or not pans
        or not all(
            isinstance(p, (int, float)) and not isinstance(p, bool) for p in pans
        )
    ):
        raise ValueError(f"pans must be a list of angles (rad): {pans!r}")


SEARCH_KEYS = (
    "location",
    "radius",
    "pans",
    "dwell",
    "look_height",
    "timeout",
    "describe_timeout",
)


def seen(answers):
    """The answers that a Description matches against."""
    return {key: answers[key] for key in ("posture", "gesture", "garments")}


def where(person):
    p = person.position
    return f"Person {person.id} at [{p.x:.2f}, {p.y:.2f}]"


def describe_answers(answers):
    if not answers:
        return "unknown"
    parts = [answers["posture"] or "?", answers["gesture"] or "?"]
    parts += answers["garments"]
    text = ", ".join(parts)
    if answers.get("reasons"):
        text += f" ({answers['reasons']})"
    return text


class FindDescribedPerson(Skill):
    """Find a person by GPSR description in the area, look at them and face them.

    params: {"gesture" | "pose" | "clothing" | "description": "<phrase>",
    "name": "<remembered name>", "location": "<area centre>", "radius": 3.0,
    "role": "person", "approach": true, "distance": 1.0, "pans": [0.0, 0.4, -0.4],
    "dwell": 6.0, "look_height": 1.0, "timeout": 60.0}. Phrases are those of the GPSR command
    generator, e.g. "waving person", "person pointing to the left", "sitting
    person", "person wearing a red t shirt"; they may be combined. The head
    scans `pans`; each visible person in the area is checked once, nearest
    first (up to MAX_JUDGEMENTS times while checks fail). A name matches only
    people person_perception remembered.
    Outputs `<role>.track` / `.position` / `.frame` and what was seen:
    `<role>.pose` / `.gesture` / `.garments`.
    """

    @classmethod
    def validate(cls, params, spec):
        description = check_description_params(
            params, optional=("role", "approach", "distance", *SEARCH_KEYS)
        )
        if not isinstance(params.get("approach", True), bool):
            raise ValueError(f"approach must be true or false: {params['approach']!r}")
        if description.empty:
            raise ValueError(f"needs a gesture, posture, clothing or name: {params}")
        if "role" in params:
            check_name(params, "role")
        positive(params, "distance", 1.0)
        positive(params, "timeout", 60.0)
        check_search_params(params, spec)

    def unsupported(self, params):
        # person_perception names only people remembered with remember_person;
        # nothing in a GPSR plan asks a person's name yet.
        if params.get("name"):
            return "finding a person by name needs their name asked first (not implemented)"
        return None

    def execute(self, ctx):
        people = People.of(self.runtime)
        params = ctx.params
        description = parse_description(params)
        role = params.get("role", "person")
        area = search_area(self.runtime, ctx.memory, params)
        origin = robot_xy(self.runtime, ctx.memory)
        pans = itertools.cycle(params.get("pans", SCAN_PANS))
        dwell = positive(params, "dwell", 6.0)
        height = positive(params, "look_height", 1.0)
        vlm_timeout = positive(params, "describe_timeout", 180.0)
        timeout = positive(params, "timeout", 60.0)
        deadline = people.now() + timeout
        checked = {}  # track ID -> answers (None: not the named person)
        failures = {}  # track ID -> judgements that failed
        checked_at = []  # (map position, track ID) of people already judged
        pan, next_look = None, -math.inf  # the head's pan, end of its dwell

        def judged(message, person):
            if person.id in checked or failures.get(person.id, 0) >= MAX_JUDGEMENTS:
                return True
            if message.header.frame_id != "map":
                return False
            # A track lost while the head turns comes back with a new ID nearby.
            # A judged track still in view is someone else: person_perception
            # keeps people 0.4 m apart as separate tracks.
            in_view = {p.id for p in message.people if p.visible}
            return any(
                math.dist(planar(person.position), xy) < SAME_PERSON_M
                for xy, track in checked_at
                if track not in in_view
            )

        def look_next():
            nonlocal pan, next_look
            pan = next(pans)
            people.look_towards(pan, height)
            next_look = people.now() + dwell

        def choose(message):
            # One person per call, so that the timeout is checked between people
            # and each person is judged on a fresh message. The head moves on
            # only when nobody visible is left to judge this way.
            nonlocal next_look
            if pan is None:
                look_next()
            candidates = sorted(
                (
                    p
                    for p in message.people
                    if p.visible
                    and not judged(message, p)
                    and in_area(message, p, area)
                    # Names come seconds after a track starts: wait for one.
                    and not (description.name and not p.name)
                ),
                key=lambda p: math.dist(planar(p.position), origin),
            )
            if not candidates:
                if people.now() >= next_look:
                    look_next()
                return None
            person = candidates[0]
            if description.name and person.name != description.name:
                checked[person.id] = None
                return None
            if description.name and not (
                description.gesture or description.posture or description.clothing
            ):
                answers = {"posture": "", "gesture": "", "garments": []}
            else:
                try:
                    answers = people.attributes(
                        person,
                        description,
                        ctx.cancelled,
                        vlm_timeout,
                        message.header.frame_id,
                        deadline,
                    )
                except Cancelled:
                    return None  # wait_for reports the abort
            match = (
                None
                if answers is None
                else description.matches(name=person.name, **seen(answers))
            )
            if match is None:
                # Not judged, or the answers cannot decide (e.g. the clothing
                # was hidden): try again, as count_people does.
                failures[person.id] = failures.get(person.id, 0) + 1
            else:
                checked[person.id] = answers
                checked_at.append((planar(person.position), person.id))
            ctx.log(f"{where(person)}: {describe_answers(answers)}")
            if match:
                return message, person, answers
            # The head turned to this person: look this way again for a full
            # dwell, so that nobody at the edge of the view is missed.
            people.look_towards(pan, height)
            next_look = people.now() + dwell
            return None

        found, reason = people.wait_for(choose, timeout, ctx.cancelled)
        if found is None:
            people.look_towards(0.0)
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            unchecked = len(set(failures) - set(checked))
            return SkillResult(
                False,
                f"No {description.text()} person among {len(checked)} checked"
                + (f" ({unchecked} could not be checked)" if unchecked else "")
                + f": {reason}",
            )
        message, person, answers = found
        frame = message.header.frame_id
        people.look_at(frame, person.head)
        outputs = person_outputs(role, message, person)
        outputs.update(
            {
                f"{role}.pose": answers["posture"],
                f"{role}.gesture": answers["gesture"],
                f"{role}.garments": answers["garments"],
            }
        )
        ctx.log(
            f"Person {person.id} ({description.text()}) at {outputs[f'{role}.position']}"
        )
        navigation = self.runtime.navigation
        if navigation is None or frame != "map" or not params.get("approach", True):
            return SkillResult(True, f"Looking at person {person.id}", outputs)
        pose = approach_pose(
            origin, planar(person.position), params.get("distance", 1.0)
        )
        ok, text = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            return SkillResult(False, f"Could not approach person {person.id}: {text}")
        people.look_at(frame, person.head)
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return SkillResult(True, f"Facing person {person.id}", outputs)


class CountPeople(Skill):
    """Count the people in the area that fit a GPSR description, scanning once.

    params: {"output": "<memory key>", "description" | "gesture" | "pose" |
    "clothing": "<phrase>" (optional: everyone), "location": "<area centre>",
    "radius": 3.0, "pans": [0.0, 0.4, -0.4], "dwell": 6.0, "look_height": 1.0,
    "timeout": 300.0}. The head stays `dwell` s at each pan, again after turning
    to judge a person; sightings less than 0.5 m apart in map are one person.
    Each person is judged once while visible, because tracks are dropped soon
    after they leave the view; a track lost meanwhile is tried up to
    MAX_JUDGEMENTS times.
    People who could not be judged, or whose answers cannot decide the
    description, do not count. Outputs `<output>` (the count) and
    `<output>.positions`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_description_params(params, required=("output",), optional=SEARCH_KEYS)
        if not isinstance(params["output"], str) or not params["output"].strip():
            raise ValueError(f"output must be a memory key: {params['output']!r}")
        positive(params, "timeout", 300.0)
        check_search_params(params, spec)

    def execute(self, ctx):
        people = People.of(self.runtime)
        params = ctx.params
        description = parse_description(params)
        area = search_area(self.runtime, ctx.memory, params)
        pans = list(params.get("pans", SCAN_PANS))
        dwell = positive(params, "dwell", 6.0)
        height = positive(params, "look_height", 1.0)
        vlm_timeout = positive(params, "describe_timeout", 180.0)
        timeout = positive(params, "timeout", 300.0)
        deadline = people.now() + timeout
        # {"position", "ids", "judgements", "done", "match"}; match None: unknown
        found = []
        step = {"index": -1, "until": -math.inf}  # pan on show, end of its dwell

        def entry_for(message, person):
            # Its own ID first: two people may come closer than SAME_PERSON_M
            # later on, and the other one's entry must not take this track.
            for entry in found:
                if person.id in entry["ids"]:
                    return entry
            if message.header.frame_id != "map":
                return None
            # A lost track comes back with a new ID near its entry; an entry
            # whose track is still in view is someone else: person_perception
            # keeps people 0.4 m apart as separate tracks.
            in_view = {p.id for p in message.people if p.visible}
            near = [
                (math.dist(planar(person.position), entry["position"][:2]), i)
                for i, entry in enumerate(found)
                if not entry["ids"] & in_view
            ]
            near = [(d, i) for d, i in near if d < SAME_PERSON_M]
            return found[min(near)[1]] if near else None

        def choose(message):
            # At most one judgement per call, so that the timeout is checked
            # between people and each person is judged on a fresh message. The
            # head moves on only when nobody visible is left to judge here.
            if step["index"] < 0:
                step["index"] = 0
                people.look_towards(pans[0], height)
                step["until"] = people.now() + dwell
            for person in message.people:
                if not person.visible or not in_area(message, person, area):
                    continue
                p = person.position
                entry = entry_for(message, person)
                if entry is None:
                    entry = {
                        "ids": set(),
                        "judgements": 0,
                        "done": False,
                        "match": None,
                    }
                    found.append(entry)
                entry["ids"].add(person.id)
                entry["position"] = [round(p.x, 3), round(p.y, 3), round(p.z, 3)]
                if entry["done"]:
                    continue
                if description.empty:
                    entry["match"], entry["done"] = True, True
                    continue
                try:
                    answers = people.attributes(
                        person,
                        description,
                        ctx.cancelled,
                        vlm_timeout,
                        message.header.frame_id,
                        deadline,
                    )
                except Cancelled:
                    return None  # wait_for reports the abort
                entry["judgements"] += 1
                if answers is not None:
                    # None when the answers cannot decide (e.g. no clothing seen).
                    entry["match"] = description.matches(
                        name=person.name, **seen(answers)
                    )
                # Judged, or a lost track tried MAX_JUDGEMENTS times: move on.
                entry["done"] = (
                    answers is not None or entry["judgements"] >= MAX_JUDGEMENTS
                )
                ctx.log(f"{where(person)}: {describe_answers(answers)}")
                # The head turned to this person: look this way again for a
                # full dwell, so that nobody at the edge of the view is missed.
                people.look_towards(pans[step["index"]], height)
                step["until"] = people.now() + dwell
                return None
            if people.now() < step["until"]:
                return None
            step["index"] += 1
            if step["index"] >= len(pans):
                return found
            people.look_towards(pans[step["index"]], height)
            step["until"] = people.now() + dwell
            return None

        result, reason = people.wait_for(choose, timeout, ctx.cancelled)
        people.look_towards(0.0)
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        if result is None:
            return SkillResult(False, f"Counting did not finish: {reason}")
        matching = [entry for entry in found if entry["match"]]
        unknown = sum(1 for entry in found if entry["match"] is None)
        output = params["output"]
        message = f"{len(matching)} of {len(found)} people are {description.text()}"
        if unknown:
            message += f"; {unknown} could not be checked"
        return SkillResult(
            True,
            message,
            {
                output: len(matching),
                f"{output}.positions": [entry["position"] for entry in matching],
            },
        )
