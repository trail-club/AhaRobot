"""Find a free seat for a guest (rulebook 5.1: offer a free seat to the guest).

The candidate seats are fixed points in map, one per place to sit (a couch has
one per place), measured on the setup days like the locations:
person_perception reports people, not furniture. They are in the subgoal's
params or, shared by the guests, in the task memory (`initial_state`). The
head looks at each seat in the listed order, which is the preference, and a
visible person of /aha/perception/people occupies the nearest seat within
`radius`. The first seat that the camera saw without anyone there is free.
People seen at any time of the search count, so a seat taken in the image of
another seat is skipped without looking at it. The guest being seated (by
name and track) occupies no seat. If no seat was seen free, the first seat not
seen taken (also one not seen at all) is offered, rather than a fixed mock.
"""

import math
from types import SimpleNamespace

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.gestures import finite_point
from aha_tasks.people import People, check_keys, check_name, positive
from aha_tasks.skills import LOCATION, resolve_pose

TORSO = 0.4  # m above the seat point: where the head looks, a sitting torso
# m from a seat point to a person on it. person_perception measures the visible
# front of the body: in the sim, SAM 3.1 put seated people 0.03-0.24 m from
# their seat and someone standing 0.45 m in front of one at 0.67 m, who blocks
# it too. Each person takes only the nearest seat, so seats 0.8 m apart stay
# apart.
RADIUS = 0.8
SETTLE = 2.0  # s for the head to turn (90 deg at 1 rad/s, and the start)
HEAD_REACH = 1.57  # rad from the base's front; config/head.yaml pan_limits


def check_seats(seats):
    if not isinstance(seats, dict) or not seats:
        raise ValueError(f"seats must be a nonempty object: {seats!r}")
    for name, point in seats.items():
        if not name.strip() or not finite_point(point):
            raise ValueError(f"seat {name!r} must be [x, y, z] in map: {point!r}")
    return seats


def seats_of(params, memory):
    """The seats of params: an object, or the memory key of one."""
    seats = params.get("seats")
    if isinstance(seats, str):
        if seats not in memory:
            raise ValueError(f"No seats in the task memory: {seats}")
        seats = memory[seats]
    return check_seats(seats)


def nearest_seat(seats, xy, radius):
    """Name of the seat nearest to `xy` within `radius` in the plane, or None."""
    distance, name = min(
        (math.dist(xy, point[:2]), name) for name, point in seats.items()
    )
    return name if distance <= radius else None


def occupants(message, seats, radius, guest, track=None):
    """Seat name -> person, of the visible people of a map message."""
    if message.header.frame_id != "map":
        return {}
    taken = {}
    for person in message.people:
        if not person.visible or person.name == guest or person.id == track:
            continue
        seat = nearest_seat(seats, (person.position.x, person.position.y), radius)
        if seat is not None:
            taken.setdefault(seat, person)
    return taken


def bearing(pose, point):
    """Angle (rad, + = left) of a map point from the front of [x, y, yaw]."""
    x, y, yaw = pose
    return math.remainder(math.atan2(point[1] - y, point[0] - x) - yaw, math.tau)


def who(person):
    return f"{person.name or 'person'} {person.id} sits there"


def listing(reasons):
    return "; ".join(f"{seat}: {reason}" for seat, reason in reasons.items())


class FindFreeSeat(Skill):
    """Look at the seats in order and output the first one nobody occupies.

    params: {"guest": "guest1", "seats": {"chair_left": [x, y, z], ...},
    "radius": RADIUS, "dwell": 1.5}. `seats` are map points (m) in order of
    preference, or the memory key of them ("living_room.seats", set in
    initial_state). The head looks TORSO above each for SETTLE + `dwell` s
    (node clock), less if someone is seen there; the seat counts as seen once
    a people message in map arrives after the one at SETTLE s. A seat more
    than HEAD_REACH from the front of robot.location is not looked at. Outputs
    `<guest>.seat` (the name) and `<guest>.seat_position` for offer_seat and
    leaves the head on the seat it saw free. If none was seen free, outputs
    the first seat not seen taken (logged; when person_perception has nothing
    in map, without looking at the rest). Fails only if every seat was seen
    taken, naming each seat's occupant.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guest", "seats"}, {"radius", "dwell"})
        check_name(params, "guest")
        seats_of(params, spec.initial_state)
        positive(params, "radius", RADIUS)
        positive(params, "dwell", 1.5)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guest = ctx.params["guest"]
        try:
            seats = seats_of(ctx.params, ctx.memory)
        except ValueError as error:
            return SkillResult(False, str(error))
        radius = positive(ctx.params, "radius", RADIUS)
        dwell = positive(ctx.params, "dwell", 1.5)
        track = ctx.memory.get(f"{guest}.track")
        pose = None
        if ctx.memory.get(LOCATION) is not None:
            pose = resolve_pose(self.runtime.spec, ctx.memory[LOCATION])
        taken = {}  # seat -> person, from every message of the search
        reasons = {}
        down = None  # why person_perception gives nothing in map

        def chosen(name, text):
            outputs = {
                f"{guest}.seat": name,
                f"{guest}.seat_position": list(seats[name]),
            }
            for seat, person in taken.items():
                reasons[seat] = who(person)
            if reasons:
                text += f" ({listing(reasons)})"
            return SkillResult(True, text, outputs)

        for name, point in seats.items():
            if name in taken:
                reasons[name] = who(taken[name])
                continue
            if down is not None:
                reasons[name] = f"not looked at ({down})"
                continue
            if pose is not None and abs(bearing(pose, point)) > HEAD_REACH:
                angle = math.degrees(bearing(pose, point))
                reasons[name] = f"out of the head's reach ({angle:+.0f} deg)"
                continue
            self.runtime.gaze.look_at(
                "map", SimpleNamespace(x=point[0], y=point[1], z=point[2] + TORSO)
            )
            settled = people.now() + SETTLE
            state = {"first": None, "seen": False}

            def choose(message, name=name, settled=settled, state=state):
                for seat, person in occupants(
                    message, seats, radius, guest, track
                ).items():
                    taken.setdefault(seat, person)
                if people.now() >= settled and message.header.frame_id == "map":
                    if state["first"] is None:
                        state["first"] = message  # may show the head turning
                    elif message is not state["first"]:
                        state["seen"] = True
                return name if name in taken else None

            found, reason = people.wait_for(choose, SETTLE + dwell, ctx.cancelled)
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            latest = people.latest()
            if found is not None:
                reasons[name] = who(taken[name])
            elif latest is None:
                down = reason
                reasons[name] = f"not seen ({reason})"
            elif latest.header.frame_id != "map":
                down = f"people are in {latest.header.frame_id}, not map"
                reasons[name] = f"not seen ({down})"
            elif not state["seen"]:
                reasons[name] = f"not seen ({reason})"
            else:
                ctx.log(f"{name} is free")
                return chosen(name, f"{name} for {guest}")
        unseen = [name for name in seats if name not in taken]
        if not unseen:
            return SkillResult(False, f"No free seat: {listing(reasons)}")
        name = unseen[0]
        ctx.log(f"No seat seen free: offering {name}, not seen taken")
        return chosen(name, f"{name} for {guest}, not seen free")
