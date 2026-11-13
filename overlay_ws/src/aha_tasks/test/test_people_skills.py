"""Person skills against fake people messages and services (no ROS graph)."""

import itertools
import math
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.people import (  # noqa: E402
    FindPerson,
    LocateGuests,
    MemorizePerson,
    People,
    CLOCK_STALL_TIMEOUT,
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

    now = staticmethod(time.monotonic)

    def __init__(self, message=None, features=("glasses",)):
        self.lock = threading.Lock()
        self.message, self.received = message, time.monotonic()
        self.panned = []
        self.followed = []
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

    def look_towards(self, pan):
        self.panned.append(pan)

    def follow(self, track=None, name=""):
        self.followed.append((track, name))


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
        # The head stays on the guest during the approach and afterwards.
        self.assertEqual(people.followed, [(2, "")])

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
        self.assertEqual(people.followed, [(7, "guest1")])

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


class WaitTests(unittest.TestCase):
    def test_stopped_clock_fails_even_without_people_messages(self):
        for message in (None, people_message()):
            with self.subTest(has_people=message is not None):
                people = FakePeople(message)
                people.now = lambda: 5.0
                people.latest = lambda: message
                wall_time = 0.0

                def advance(seconds):
                    nonlocal wall_time
                    wall_time += seconds

                with (
                    patch("aha_tasks.people.time.monotonic", lambda: wall_time),
                    patch("aha_tasks.people.time.sleep", side_effect=advance),
                ):
                    found, reason = people.wait_for(
                        lambda m: None, 60.0, threading.Event()
                    )
                self.assertIsNone(found)
                self.assertIn("/clock", reason)
                self.assertGreaterEqual(wall_time, CLOCK_STALL_TIMEOUT)
                self.assertLess(wall_time, CLOCK_STALL_TIMEOUT + 0.2)

    def test_slow_clock_keeps_waiting_while_it_advances(self):
        people = FakePeople(people_message())
        wall_time = 0.0
        people.now = lambda: wall_time * 0.08
        people.latest = lambda: people.message

        def advance(seconds):
            nonlocal wall_time
            wall_time += seconds

        with (
            patch("aha_tasks.people.time.monotonic", lambda: wall_time),
            patch("aha_tasks.people.time.sleep", side_effect=advance),
        ):
            found, reason = people.wait_for(lambda m: None, 2.0, threading.Event())
        self.assertIsNone(found)
        self.assertIn("Not found", reason)
        self.assertAlmostEqual(wall_time, 25.0, delta=0.11)

    def test_stopped_clock_can_be_cancelled(self):
        people = FakePeople(people_message())
        people.now = lambda: 5.0
        cancelled = threading.Event()
        cancelled.set()
        found, reason = people.wait_for(lambda m: None, 60.0, cancelled)
        self.assertIsNone(found)
        self.assertEqual(reason, "Cancelled")

    def test_timeout_ignores_system_clock_jumps_without_sim_time(self):
        for ros_time_is_active in (False, None):
            with self.subTest(ros_time_is_active=ros_time_is_active):
                people = FakePeople(people_message())
                clock = SimpleNamespace(now=Mock())
                if ros_time_is_active is not None:
                    clock.ros_time_is_active = ros_time_is_active
                people.node = SimpleNamespace(get_clock=lambda: clock)
                people.now = People.now.__get__(people)
                people.latest = lambda: people.message
                elapsed = 0.0

                def advance(seconds):
                    nonlocal elapsed
                    elapsed += seconds
                    # Simulate alternating forward and backward clock corrections.
                    clock.now.return_value = SimpleNamespace(
                        nanoseconds=(-1 if elapsed < 0.2 else 1) * 3600 * 10**9
                    )

                with (
                    patch("aha_tasks.people.time.monotonic", lambda: elapsed),
                    patch("aha_tasks.people.time.sleep", side_effect=advance),
                ):
                    found, reason = people.wait_for(
                        lambda m: None, 0.3, threading.Event()
                    )
                self.assertIsNone(found)
                self.assertIn("Not found", reason)
                self.assertAlmostEqual(elapsed, 0.3)
                clock.now.assert_not_called()

    def test_timeout_waits_for_sim_time_to_advance(self):
        people = FakePeople(people_message())
        sim_times = iter((0.0, 0.0, 1.0, 2.0))
        clock = SimpleNamespace(
            ros_time_is_active=True,
            now=lambda: SimpleNamespace(nanoseconds=next(sim_times) * 10**9),
        )
        people.node = SimpleNamespace(get_clock=lambda: clock)
        people.now = People.now.__get__(people)
        people.latest = lambda: people.message
        with (
            patch("aha_tasks.people.time.monotonic", return_value=10000.0),
            patch("aha_tasks.people.time.sleep") as sleep,
        ):
            found, reason = people.wait_for(lambda m: None, 2.0, threading.Event())
        self.assertIsNone(found)
        self.assertIn("Not found", reason)
        self.assertEqual(sleep.call_count, 2)

    def test_timeout_counts_the_node_clock(self):
        people = FakePeople(people_message())
        people.now = itertools.count(0.0, 100.0).__next__  # a fast simulation clock
        started = time.monotonic()
        found, reason = people.wait_for(lambda m: None, 1000.0, threading.Event())
        self.assertIsNone(found)
        self.assertIn("Not found", reason)
        self.assertLess(time.monotonic() - started, 5.0)


class LocateGuestsTests(unittest.TestCase):
    def locate_messages(self, *messages):
        people = FakePeople(messages[0])
        people.latest = Mock(
            side_effect=itertools.chain(messages, itertools.repeat(messages[-1]))
        )
        people.now = itertools.count(0.0, 0.1).__next__
        with patch("aha_tasks.people.time.sleep"):
            result = LocateGuests(runtime(people)).execute(
                context({"guests": ["guest1", "guest2"], "timeout": 2.0})
            )
        self.assertEqual(people.panned[-1], 0.0)
        return result

    def test_same_person_reidentified_as_another_guest_is_not_success(self):
        result = self.locate_messages(
            people_message(person(11, 0.75, 2.5, "guest1")),
            people_message(person(13, 0.85, 2.5, "guest2")),
        )
        self.assertFalse(result.success)

    def test_renamed_track_is_not_two_guests_even_if_it_moved(self):
        result = self.locate_messages(
            people_message(person(11, 0.75, 2.5, "guest1")),
            people_message(person(11, 1.75, 2.5, "guest2")),
        )
        self.assertFalse(result.success)

    def test_height_noise_does_not_make_a_duplicate_a_different_guest(self):
        first = person(11, 0.75, 2.5, "guest1")
        second = person(13, 0.85, 2.5, "guest2")
        second.position.z += 0.5
        result = self.locate_messages(people_message(first), people_message(second))
        self.assertFalse(result.success)

    def test_duplicate_observation_is_replaced_and_search_continues(self):
        result = self.locate_messages(
            people_message(person(11, 0.75, 2.5, "guest1")),
            people_message(person(13, 0.85, 2.5, "guest2")),
            people_message(person(14, 1.75, 2.5, "guest1")),
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest1.track"], 14)
        self.assertEqual(result.outputs["guest2.track"], 13)

    def test_camera_observations_are_not_accumulated_across_head_poses(self):
        result = self.locate_messages(
            people_message(person(11, 0.75, 2.5, "guest1"), frame="camera"),
            people_message(person(13, 1.75, 2.5, "guest2"), frame="camera"),
        )
        self.assertFalse(result.success)

    def test_different_coordinate_frames_are_not_combined(self):
        result = self.locate_messages(
            people_message(person(11, 0.75, 2.5, "guest1"), frame="camera"),
            people_message(person(13, 1.75, 2.5, "guest2")),
        )
        self.assertFalse(result.success)

    def test_camera_message_with_both_guests_still_succeeds(self):
        result = self.locate_messages(
            people_message(
                person(11, 0.75, 2.5, "guest1"),
                person(13, 1.75, 2.5, "guest2"),
                frame="camera",
            ),
        )
        self.assertTrue(result.success, result.message)

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

    def test_guests_seen_at_different_head_poses_are_all_located(self):
        people = FakePeople(people_message(person(12, 0.75, 2.5, "guest1")))

        def second_guest_comes_into_view():
            time.sleep(0.25)
            people._on_people(people_message(person(13, 1.75, 2.5, "guest2")))

        thread = threading.Thread(target=second_guest_comes_into_view)
        thread.start()
        result = LocateGuests(runtime(people)).execute(
            context({"guests": ["guest1", "guest2"], "timeout": 5.0, "dwell": 0.1})
        )
        thread.join()
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest1.track"], 12)
        self.assertEqual(result.outputs["guest2.track"], 13)

    def test_head_scans_and_ends_straight(self):
        people = FakePeople(people_message(person(12, 0.75, 2.5, "guest1")))
        LocateGuests(runtime(people)).execute(
            context({"guests": ["guest1", "guest2"], "timeout": 0.35, "dwell": 0.1})
        )
        self.assertEqual(people.panned[:3], [0.0, 0.4, -0.4])
        self.assertEqual(people.panned[-1], 0.0)

    def test_guest_list_is_validated(self):
        spec = SimpleNamespace(locations={})
        for guests in ([], ["a", "a"], [""], "guest1"):
            with self.assertRaises(ValueError):
                LocateGuests.validate({"guests": guests}, spec)


if __name__ == "__main__":
    unittest.main()
