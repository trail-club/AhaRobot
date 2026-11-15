"""FindFreeSeat against fake people messages and gaze (no ROS graph)."""

import math
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.people import People  # noqa: E402
from aha_tasks.seats import (  # noqa: E402
    TORSO,
    FindFreeSeat,
    bearing,
    nearest_seat,
    occupants,
)

SEATS = {
    "chair_left": [1.75, 3.5, 0.5],
    "chair_right": [0.75, 3.5, 0.5],
    "chair_extra": [2.55, 3.5, 0.5],
}
VIEW = 0.8  # m around the point the head looks at


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def person(id, x, y, name="", visible=True):
    return SimpleNamespace(id=id, name=name, visible=visible, position=point(x, y, 0.9))


def message(people, frame="map"):
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame), people=people)


class FakeGaze:
    def __init__(self):
        self.looks = []

    def look_at(self, frame, target):
        self.looks.append((frame, (target.x, target.y, target.z)))


class FakePeople(People):
    """A new message at every poll, of the people within VIEW of the head's target.

    `fresh=False` repeats one message (no new image), `alive=False` has none.
    """

    now = staticmethod(time.monotonic)

    def __init__(self, gaze, people, frame="map", fresh=True, alive=True):
        self.gaze, self.people, self.frame = gaze, people, frame
        self.fresh, self.alive = fresh, alive
        self.cached = None

    def latest(self):
        if not self.alive:
            return None
        if not self.fresh and self.cached is not None:
            return self.cached
        target = self.gaze.looks[-1][1] if self.gaze.looks else (0.0, 0.0)
        seen = [
            SimpleNamespace(
                **{
                    **vars(p),
                    "visible": p.visible
                    and math.dist(target[:2], (p.position.x, p.position.y)) <= VIEW,
                }
            )
            for p in self.people
        ]
        self.cached = message(seen, self.frame)
        return self.cached


def runtime(people_list, **kwargs):
    gaze = FakeGaze()
    spec = SimpleNamespace(
        locations={
            "living_room": [1.25, 2.5, math.pi / 2],
            "facing_away": [1.25, 2.5, -math.pi / 2],
        }
    )
    people = FakePeople(gaze, people_list, **kwargs)
    return SimpleNamespace(node=None, spec=spec, gaze=gaze, people=people)


def context(memory=None, **params):
    return SkillContext(
        subgoal="guest2.find_seat",
        params={"guest": "guest2", "seats": SEATS, "dwell": 0.3, **params},
        memory={"robot.location": "living_room", **(memory or {})},
        cancelled=threading.Event(),
        log=lambda text: None,
    )


def looked_at(rt):
    return [
        name
        for _, (x, y, _) in rt.gaze.looks
        for name, seat in SEATS.items()
        if (x, y) == tuple(seat[:2])
    ]


HOST = person(3, 1.75, 3.05)  # stands in front of chair_left, like the sim
GUEST1 = person(5, 0.75, 3.05, name="guest1")


class GeometryTests(unittest.TestCase):
    def test_a_person_occupies_only_the_nearest_seat_within_the_radius(self):
        self.assertEqual(nearest_seat(SEATS, (1.75, 3.05), 0.6), "chair_left")
        self.assertEqual(nearest_seat(SEATS, (2.2, 3.4), 0.6), "chair_extra")
        self.assertIsNone(nearest_seat(SEATS, (1.25, 2.5), 0.6))

    def test_the_guest_and_people_not_seen_now_occupy_nothing(self):
        people = [
            person(1, 1.75, 3.4, name="guest2"),
            person(2, 0.75, 3.4),  # the guest's track before it had a name
            person(4, 2.55, 3.4, visible=False),
        ]
        self.assertEqual(occupants(message(people), SEATS, 0.6, "guest2", 2), {})
        taken = occupants(message([HOST, GUEST1]), SEATS, 0.6, "guest2")
        self.assertEqual(
            {k: v.id for k, v in taken.items()}, {"chair_left": 3, "chair_right": 5}
        )

    def test_people_outside_map_have_no_seat(self):
        self.assertEqual(occupants(message([HOST], "camera"), SEATS, 0.6, "g"), {})

    def test_bearing_from_the_front_of_the_robot(self):
        self.assertAlmostEqual(bearing([0.0, 0.0, 0.0], [1.0, 1.0]), math.pi / 4)
        self.assertAlmostEqual(abs(bearing([0.0, 0.0, math.pi], [1.0, 0.0])), math.pi)
        # Across the -x axis (yaw 172 deg, point at -169 deg): wrapped, 19 deg left.
        self.assertAlmostEqual(bearing([0.0, 0.0, 3.0], [-1.0, -0.2]), 0.339, places=3)


@patch("aha_tasks.seats.SETTLE", 0.05)
class FindFreeSeatTests(unittest.TestCase):
    def test_skips_the_host_and_outputs_the_next_seat(self):
        rt = runtime([HOST])
        result = FindFreeSeat(rt).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            result.outputs,
            {"guest2.seat": "chair_right", "guest2.seat_position": [0.75, 3.5, 0.5]},
        )
        self.assertEqual(looked_at(rt), ["chair_left", "chair_right"])
        self.assertIn("chair_left: person 3 sits there", result.message)
        # The head looks above the seat, where a sitting person is seen.
        self.assertEqual(rt.gaze.looks[0], ("map", (1.75, 3.5, 0.5 + TORSO)))

    def test_the_first_guest_also_occupies_a_seat(self):
        rt = runtime([HOST, GUEST1])
        result = FindFreeSeat(rt).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest2.seat"], "chair_extra")

    def test_a_seat_taken_in_an_earlier_image_is_not_looked_at(self):
        rt = runtime([HOST, GUEST1])
        with patch(f"{__name__}.VIEW", 10.0):  # every image shows every seat
            result = FindFreeSeat(rt).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(looked_at(rt), ["chair_left", "chair_extra"])

    def test_the_guest_being_seated_does_not_take_a_seat(self):
        rt = runtime([person(7, 1.75, 3.05, name="guest2")])
        result = FindFreeSeat(rt).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest2.seat"], "chair_left")
        rt = runtime([person(7, 1.75, 3.05)])
        result = FindFreeSeat(rt).execute(context({"guest2.track": 7}))
        self.assertEqual(result.outputs["guest2.seat"], "chair_left")

    def test_fails_when_every_seat_is_taken(self):
        rt = runtime([HOST, GUEST1, person(8, 2.55, 3.1)])
        result = FindFreeSeat(rt).execute(context())
        self.assertFalse(result.success)
        self.assertEqual(result.message.count("sits there"), len(SEATS))
        self.assertIn("chair_right: guest1 5 sits there", result.message)

    def test_a_seat_without_a_new_image_is_not_free(self):
        rt = runtime([], fresh=False)
        result = FindFreeSeat(rt).execute(context())
        self.assertFalse(result.success)
        self.assertIn("chair_left: not seen", result.message)

    def test_fails_without_person_perception(self):
        rt = runtime([HOST], alive=False)
        result = FindFreeSeat(rt).execute(context())
        self.assertFalse(result.success)
        self.assertIn("/aha/perception/people", result.message)
        self.assertEqual(len(rt.gaze.looks), 1)

    def test_fails_when_people_are_not_in_map(self):
        rt = runtime([], frame="camera_color_optical_frame")
        result = FindFreeSeat(rt).execute(context())
        self.assertFalse(result.success)
        self.assertIn("not map", result.message)

    def test_seats_behind_the_robot_are_not_looked_at(self):
        rt = runtime([])
        result = FindFreeSeat(rt).execute(context({"robot.location": "facing_away"}))
        self.assertFalse(result.success)
        self.assertIn("out of the head's reach", result.message)
        self.assertEqual(rt.gaze.looks, [])

    def test_cancel_stops_the_search(self):
        rt = runtime([HOST])
        ctx = context()
        ctx.cancelled.set()
        result = FindFreeSeat(rt).execute(ctx)
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Cancelled")

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        FindFreeSeat.validate({"guest": "g", "seats": SEATS, "radius": 0.5}, spec)
        for params in (
            {"guest": "g"},
            {"guest": "g", "seats": {}},
            {"guest": "g", "seats": ["chair_left"]},
            {"guest": "g", "seats": {"a": [1.0, 2.0]}},
            {"guest": "g", "seats": {" ": [1.0, 2.0, 0.5]}},
            {"guest": "", "seats": SEATS},
            {"guest": "g", "seats": SEATS, "dwell": 0},
            {"guest": "g", "seats": SEATS, "x": 1},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                FindFreeSeat.validate(params, spec)

    def test_nothing_is_expected_before_it_looks(self):
        self.assertEqual(
            FindFreeSeat(None).expected({"guest": "g", "seats": SEATS}), {}
        )


if __name__ == "__main__":
    unittest.main()
