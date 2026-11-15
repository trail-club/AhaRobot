"""Find a free seat for a guest (rulebook 5.1: offer a free seat to the guest).

The candidate seats are fixed points in map from the task definition, one per
place to sit (a couch has one per place), measured on the setup days like the
locations: person_perception reports people, not furniture. The head looks at
each seat in the listed order, which is the preference, and a visible person
of /aha/perception/people occupies the nearest seat within `radius`. The
first seat that the camera saw without anyone there is free. People seen at
any time of the search count, so a seat taken in the image of another seat is
skipped without looking at it. The guest being seated (by name and track)
occupies no seat.
"""

import math
from types import SimpleNamespace

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.gestures import finite_point
from aha_tasks.people import People, check_keys, check_name, positive
from aha_tasks.skills import LOCATION, resolve_pose

TORSO = 0.4  # m above the seat point: where the head looks, a sitting torso
SETTLE = 2.0  # s for the head to turn (90 deg at 1 rad/s, and the start)
HEAD_REACH = 1.57  # rad from the base's front; config/head.yaml pan_limits


def check_seats(params):
    seats = params.get("seats")
    if not isinstance(seats, dict) or not seats:
        raise ValueError(f"seats must be a nonempty object: {params}")
    for name, point in seats.items():
        if not name.strip() or not finite_point(point):
            raise ValueError(f"seat {name!r} must be [x, y, z] in map: {point!r}")
    return seats


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
    "radius": 0.6, "dwell": 1.5}. `seats` are map points (m) in order of
    preference. The head looks TORSO above each for SETTLE + `dwell` s (node
    clock), less if someone is seen there; the seat counts as seen once a
    people message in map arrives after the one at SETTLE s. A seat more than
    HEAD_REACH from the front of robot.location is not looked at. Outputs
    `<guest>.seat` (the name) and `<guest>.seat_position` for offer_seat and
    leaves the head on that seat. Fails if no seat was seen free, naming each
    seat's occupant or why it was not seen.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"guest", "seats"}, {"radius", "dwell"})
        check_name(params, "guest")
        check_seats(params)
        positive(params, "radius", 0.6)
        positive(params, "dwell", 1.5)

    def execute(self, ctx):
        people = People.of(self.runtime)
        guest, seats = ctx.params["guest"], ctx.params["seats"]
        radius = positive(ctx.params, "radius", 0.6)
        dwell = positive(ctx.params, "dwell", 1.5)
        track = ctx.memory.get(f"{guest}.track")
        pose = None
        if ctx.memory.get(LOCATION) is not None:
            pose = resolve_pose(self.runtime.spec, ctx.memory[LOCATION])
        taken = {}  # seat -> person, from every message of the search
        reasons = {}
        for name, point in seats.items():
            if name in taken:
                reasons[name] = who(taken[name])
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
            if found is None and latest is None:
                return SkillResult(False, f"No free seat: {reason}")
            if found is None and latest.header.frame_id != "map":
                frame = latest.header.frame_id
                return SkillResult(False, f"People are in {frame}, not map")
            if found is not None:
                reasons[name] = who(taken[name])
            elif not state["seen"]:
                reasons[name] = f"not seen ({reason})"
            else:
                ctx.log(f"{name} is free")
                outputs = {
                    f"{guest}.seat": name,
                    f"{guest}.seat_position": list(point),
                }
                text = f"{name} for {guest}"
                if reasons:
                    text += f" ({listing(reasons)})"
                return SkillResult(True, text, outputs)
        return SkillResult(False, f"No free seat: {listing(reasons)}")
