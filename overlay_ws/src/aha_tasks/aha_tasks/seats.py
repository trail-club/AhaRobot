"""Find a free seat for a guest (rulebook 5.1: offer a free seat to the guest).

person_perception reports people, not furniture, so the candidate seats are
fixed map points measured on the setup days, like the locations.
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


def seats_of(params, memory):
    """The seats of params: an object, or the memory key of one."""
    seats = params.get("seats")
    if isinstance(seats, str):
        if seats not in memory:
            raise ValueError(f"No seats in the task memory: {seats}")
        seats = memory[seats]
    if not isinstance(seats, dict) or not seats:
        raise ValueError(f"seats must be a nonempty object: {seats!r}")
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


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def who(person):
    return f"{person.name or 'person'} {person.id} sits there"


class FindFreeSeat(Skill):
    """Look at the seats in order and output the first one nobody occupies.

    params: {"guest": "guest1", "seats": {"chair_left": [x, y, z], ...},
    "radius": RADIUS, "dwell": 1.5}. `seats` are map points (m) in order of
    preference, or the memory key of them ("living_room.seats", set in
    initial_state). A visible person occupies the nearest seat within
    `radius`, except the guest being seated (by name and `<guest>.track`).
    People seen at any time of the search count, so a seat taken in the image
    of another one is not looked at. The head looks TORSO above each seat for
    SETTLE + `dwell` s (the wait clock of People), less if someone is seen
    there. The seat counts as seen once a people message in map arrives whose
    image was taken SETTLE s or more after the head was turned (header.stamp
    on the node clock), as person_perception publishes later than it takes the
    image. A seat more than HEAD_REACH from the front of robot.location
    is not looked at. Outputs `<guest>.seat` (the name) and
    `<guest>.seat_position` for offer_seat and leaves the head on the seat
    seen free. If none was, outputs the first seat not seen taken (when
    person_perception has nothing in map, without looking at the rest). Fails
    only if every seat was seen taken. The message gives each seat's reason.
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
        location = ctx.memory.get(LOCATION)
        pose = None if location is None else resolve_pose(self.runtime.spec, location)
        taken = {}  # seat -> person, from every message of the search
        reasons = {}  # seat -> why it is not chosen, if not taken
        down = None  # why person_perception gives nothing in map
        clock = self.runtime.node.get_clock()  # of the image stamps

        def listing():
            return "; ".join(
                f"{seat}: {who(taken[seat]) if seat in taken else reasons[seat]}"
                for seat in seats
                if seat in taken or seat in reasons
            )

        def chosen(name, text):
            if taken or reasons:
                text += f" ({listing()})"
            outputs = {
                f"{guest}.seat": name,
                f"{guest}.seat_position": list(seats[name]),
            }
            return SkillResult(True, text, outputs)

        for name, point in seats.items():
            if name in taken:
                continue
            if down is not None:
                reasons[name] = f"not looked at ({down})"
                continue
            angle = 0.0 if pose is None else bearing(pose, point)
            if abs(angle) > HEAD_REACH:
                degrees = math.degrees(angle)
                reasons[name] = f"out of the head's reach ({degrees:+.0f} deg)"
                continue
            people.look_at(
                "map", SimpleNamespace(x=point[0], y=point[1], z=point[2] + TORSO)
            )
            settled = clock.now().nanoseconds * 1e-9 + SETTLE
            seen = False

            def choose(message):
                nonlocal seen
                for seat, person in occupants(
                    message, seats, radius, guest, track
                ).items():
                    taken.setdefault(seat, person)
                if (
                    message.header.frame_id == "map"
                    and stamp_seconds(message.header.stamp) >= settled
                ):
                    seen = True
                return name if name in taken else None

            found, reason = people.wait_for(choose, SETTLE + dwell, ctx.cancelled)
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            if found is not None:
                continue
            latest = people.latest()
            if latest is None:
                down = reason
            elif latest.header.frame_id != "map":
                down = f"people are in {latest.header.frame_id}, not map"
            elif seen:
                ctx.log(f"{name} is free")
                return chosen(name, f"{name} for {guest}")
            reasons[name] = f"not seen ({down or reason})"
        unseen = [name for name in seats if name not in taken]
        if not unseen:
            return SkillResult(False, f"No free seat: {listing()}")
        ctx.log(f"No seat seen free: offering {unseen[0]}, not seen taken")
        return chosen(unseen[0], f"{unseen[0]} for {guest}, not seen free")
