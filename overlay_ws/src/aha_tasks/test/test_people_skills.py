"""Person skills against fake people messages and services (no ROS graph)."""

import math
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.people import (  # noqa: E402
    FindPerson,
    LocateGuests,
    MemorizePerson,
    People,
    approach_pose,
    choose_person,
)


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def person(id, x, y, name="", visible=True):
    return SimpleNamespace(
        id=id,
        name=name,
        visible=visible,
        position=point(x, y, 1.0),
        head=point(x, y, 1.6),
    )


def people_message(*people, frame="map"):
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame), people=list(people))


class FakeClient:
    def __init__(self, name, answer):
        self.srv_name = name
        self.answer = answer
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        self.requests.append(request)
        future = SimpleNamespace(done=lambda: True, cancel=lambda: None)
        future.result = lambda: self.answer(request)
        return future


class FakePeople(People):
    """People without ROS: a settable latest message and recorded calls."""

    def __init__(self, message=None, features=("glasses",)):
        self.lock = threading.Lock()
        self.message, self.received = message, time.monotonic()
        self.looked = []
        self.remember = FakeClient(
            "remember", lambda r: SimpleNamespace(success=True, message="ok")
        )
        self.describe = FakeClient(
            "describe",
            lambda r: SimpleNamespace(
                success=True, message="", features=list(features)
            ),
        )
        service = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
        self.RememberPerson = self.DescribePerson = service

    def look_at(self, frame, point):
        self.looked.append((frame, point))


class FakeNavigation:
    def __init__(self, ok=True):
        self.ok = ok
        self.goals = []

    def navigate(self, pose, cancelled=None):
        self.goals.append(pose)
        return self.ok, "" if self.ok else "blocked"


def runtime(people, navigation=None):
    spec = SimpleNamespace(locations={"entrance": [0.0, 0.0, 0.0]})
    return SimpleNamespace(node=None, spec=spec, navigation=navigation, people=people)


def context(params, memory=None):
    return SkillContext(
        subgoal="test",
        params=params,
        memory={"robot.location": "entrance", **(memory or {})},
        cancelled=threading.Event(),
        log=lambda text: None,
    )


class GeometryTests(unittest.TestCase):
    def test_nearest_free_visible_person(self):
        people = [
            person(1, 0.5, 0.0, name="guest1"),
            person(2, 3.0, 0.0),
            person(3, 1.0, 0.0, visible=False),
            person(4, 2.0, 0.0),
        ]
        self.assertEqual(choose_person(people, (0.0, 0.0), taken={"guest1"}).id, 4)
        self.assertEqual(choose_person(people, (0.0, 0.0)).id, 1)
        self.assertIsNone(choose_person([], (0.0, 0.0)))

    def test_approach_stops_short_and_faces_the_person(self):
        x, y, yaw = approach_pose((0.0, 0.0), (0.0, 3.0), 1.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(yaw, math.pi / 2)
        self.assertEqual(approach_pose((1.0, 1.0), (1.5, 1.0), 1.0), [1.0, 1.0, 0.0])


class FindPersonTests(unittest.TestCase):
    def test_approaches_the_unnamed_person_and_outputs_the_track(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0, "guest1"), person(2, 3.0, 0.0))
        )
        navigation = FakeNavigation()
        skill = FindPerson(runtime(people, navigation))
        result = skill.execute(context({"role": "guest2"}))
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest2.track"], 2)
        self.assertEqual(result.outputs["robot.location"], [2.0, 0.0, 0.0])
        self.assertEqual(navigation.goals, [[2.0, 0.0, 0.0]])
        self.assertEqual(len(people.looked), 2)

    def test_camera_frame_people_are_only_looked_at(self):
        people = FakePeople(people_message(person(5, 0.0, 0.0), frame="camera"))
        navigation = FakeNavigation()
        result = FindPerson(runtime(people, navigation)).execute(
            context({"role": "guest1"})
        )
        self.assertTrue(result.success)
        self.assertNotIn("robot.location", result.outputs)
        self.assertEqual(navigation.goals, [])

    def test_no_people_topic_fails_with_a_reason(self):
        people = FakePeople(None)
        result = FindPerson(runtime(people)).execute(
            context({"role": "guest1", "timeout": 0.2})
        )
        self.assertFalse(result.success)
        self.assertIn("person_perception", result.message)

    def test_navigation_failure_fails_the_skill(self):
        people = FakePeople(people_message(person(1, 3.0, 0.0)))
        result = FindPerson(runtime(people, FakeNavigation(ok=False))).execute(
            context({"role": "guest1"})
        )
        self.assertFalse(result.success)
        self.assertIn("blocked", result.message)

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        FindPerson.validate({"role": "guest1", "distance": 0.8}, spec)
        for params in (
            {},
            {"role": ""},
            {"role": "g", "distance": -1},
            {"role": "g", "x": 1},
        ):
            with self.assertRaises(ValueError):
                FindPerson.validate(params, spec)


class MemorizePersonTests(unittest.TestCase):
    def test_remembers_the_found_track_and_outputs_features(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(7, 2.0, 0.0)),
            features=("glasses", "red t-shirt"),
        )
        result = MemorizePerson(runtime(people)).execute(
            context({"guest": "guest1"}, {"guest1.track": 7})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            result.outputs["guest1.appearance"], ["glasses", "red t-shirt"]
        )
        self.assertEqual(people.remember.requests[0].id, 7)
        self.assertEqual(people.remember.requests[0].name, "guest1")
        self.assertEqual(people.describe.requests[0].id, 7)

    def test_lost_track_falls_back_to_the_nearest_unnamed_person(self):
        people = FakePeople(
            people_message(person(1, 0.5, 0.0, "guest1"), person(9, 2.0, 0.0))
        )
        result = MemorizePerson(runtime(people)).execute(
            context({"guest": "guest2"}, {"guest2.track": 4})
        )
        self.assertTrue(result.success)
        self.assertEqual(people.remember.requests[0].id, 9)

    def test_failed_description_fails_the_skill(self):
        people = FakePeople(people_message(person(1, 0.5, 0.0)))
        people.describe.answer = lambda r: SimpleNamespace(
            success=False, message="VLM timeout", features=[]
        )
        result = MemorizePerson(runtime(people)).execute(context({"guest": "guest1"}))
        self.assertFalse(result.success)
        self.assertIn("VLM timeout", result.message)


class LocateGuestsTests(unittest.TestCase):
    def test_outputs_each_named_guest_wherever_they_sit(self):
        people = FakePeople(
            people_message(
                person(11, 1.75, 2.5, "guest2"),
                person(12, 0.75, 2.5, "guest1"),
                person(13, 2.7, 1.8),
            )
        )
        result = LocateGuests(runtime(people)).execute(
            context({"guests": ["guest1", "guest2"]})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest1.track"], 12)
        self.assertEqual(result.outputs["guest2.position"], [1.75, 2.5, 1.0])

    def test_missing_guest_times_out(self):
        people = FakePeople(people_message(person(11, 1.0, 2.0, "guest1")))
        result = LocateGuests(runtime(people)).execute(
            context({"guests": ["guest1", "guest2"], "timeout": 0.2})
        )
        self.assertFalse(result.success)

    def test_guest_list_is_validated(self):
        spec = SimpleNamespace(locations={})
        for guests in ([], ["a", "a"], [""], "guest1"):
            with self.assertRaises(ValueError):
                LocateGuests.validate({"guests": guests}, spec)


if __name__ == "__main__":
    unittest.main()
