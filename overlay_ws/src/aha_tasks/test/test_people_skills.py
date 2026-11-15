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
    MAX_JUDGEMENTS,
    SERVER_MARGIN,
    SKIP_VLM_TIMEOUT,
    Cancelled,
    CountPeople,
    FindDescribedPerson,
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


def people_message(*people, frame="map", stamp=0.0):
    time = SimpleNamespace(sec=int(stamp), nanosec=round(stamp % 1 * 1e9))
    header = SimpleNamespace(frame_id=frame, stamp=time)
    return SimpleNamespace(header=header, people=list(people))


class FakeClient:
    def __init__(self, name, answer):
        self.srv_name = name
        self.answer = answer
        self.requests = []
        self.delay = lambda request: 0.0  # s until the reply

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        self.requests.append(request)
        ready = time.monotonic() + self.delay(request)
        future = SimpleNamespace(
            done=lambda: time.monotonic() >= ready, cancel=lambda: None
        )
        future.result = lambda: self.answer(request)
        return future


class FakePeople(People):
    """People without ROS: a settable latest message and recorded calls."""

    now = staticmethod(time.monotonic)
    look_settle = 0.0

    def __init__(
        self, message=None, features=("glasses",), attributes=None, vlm_failures=None
    ):
        self.lock = threading.Lock()
        self.message, self.received = message, time.monotonic()
        self.looked = []
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
        # id -> {"rules": (posture, gesture), "vlm": (posture, gesture, garments)}
        self.attributes_by_id = attributes or {}
        # id -> calls with the VLM that fail before it answers (like a timeout)
        self.vlm_failures = dict(vlm_failures or {})
        self.classify = FakeClient("classify", self._classify)
        service = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
        self.RememberPerson = self.DescribePerson = self.ClassifyPerson = service

    def _classify(self, request):
        known = self.attributes_by_id.get(request.id)
        if known is None:
            return SimpleNamespace(success=False, message="not tracked")
        if request.skip_vlm:
            posture, gesture = known.get("rules", ("", ""))
            garments = []
        elif self.vlm_failures.get(request.id, 0) > 0:
            self.vlm_failures[request.id] -= 1
            posture, gesture = known.get("rules", ("", ""))
            return SimpleNamespace(
                success=False,
                message="VLM failed: the DGX worker did not answer",
                posture=posture,
                gesture=gesture,
                garments=[],
            )
        else:
            posture, gesture, garments = known["vlm"]
        return SimpleNamespace(
            success=True, posture=posture, gesture=gesture, garments=list(garments)
        )

    def look_at(self, frame, point):
        self.looked.append((frame, point))

    def look_towards(self, pan, height=1.2):
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
            context({"guest": "guest1", "views": 1}, {"guest1.track": 7})
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
            context({"guest": "guest2", "views": 1}, {"guest2.track": 4})
        )
        self.assertTrue(result.success)
        self.assertEqual(people.remember.requests[0].id, 9)

    def test_failed_description_fails_the_skill(self):
        people = FakePeople(people_message(person(1, 0.5, 0.0)))
        people.describe.answer = lambda r: SimpleNamespace(
            success=False, message="VLM timeout", features=[]
        )
        result = MemorizePerson(runtime(people)).execute(
            context({"guest": "guest1", "views": 1})
        )
        self.assertFalse(result.success)
        self.assertIn("VLM timeout", result.message)

    def memorize_views(self, answers, messages, params=None, delay=0.0):
        """MemorizePerson with one describe answer per view, in order."""
        people = FakePeople(messages[0])
        people.latest = Mock(
            side_effect=itertools.chain(messages, itertools.repeat(messages[-1]))
        )
        people.now = itertools.count(0.0, 0.1).__next__
        replies = iter(answers)
        people.describe.answer = lambda r: next(replies)
        people.describe.delay = lambda r: delay
        ctx = context({"guest": "guest1", "timeout": 1.0, **(params or {})})
        people.logs = []
        ctx.log = people.logs.append
        with patch("aha_tasks.people.time.sleep"):
            result = MemorizePerson(runtime(people)).execute(ctx)
        return result, people

    def test_features_most_views_agree_on_are_kept(self):
        def answer(*features):
            return SimpleNamespace(success=True, message="", features=list(features))

        result, people = self.memorize_views(
            [
                answer("green t-shirt", "short blond hair"),
                answer("green t-shirt", "short hair", "glasses"),
                answer("green t-shirt", "short blond hair"),
            ],
            [people_message(person(1, 0.5, 0.0), stamp=t) for t in (0.0, 0.6, 1.2)],
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            result.outputs["guest1.appearance"], ["green t-shirt", "short blond hair"]
        )
        self.assertEqual(len(people.describe.requests), 3)

    def test_views_wait_for_newer_images(self):
        def answer():
            return SimpleNamespace(success=True, message="", features=["red t-shirt"])

        result, people = self.memorize_views(
            [answer(), answer()],
            [people_message(person(1, 0.5, 0.0), stamp=t) for t in (0.0, 0.2, 0.7)],
            {"views": 2},
        )
        self.assertTrue(result.success, result.message)
        # Views of the images at 0.0 and 0.7 s; the one at 0.2 s is too close.
        self.assertEqual(people.latest.call_count, 3)
        self.assertEqual(len(people.describe.requests), 2)

    def test_failed_views_count_against_the_features(self):
        def answer(*features):
            return SimpleNamespace(success=True, message="", features=list(features))

        result, _ = self.memorize_views(
            [
                answer("glasses", "red t-shirt"),
                SimpleNamespace(success=False, message="VLM timeout", features=[]),
                answer("red shirt"),
            ],
            [people_message(person(1, 0.5, 0.0), stamp=t) for t in (0.0, 0.6, 1.2)],
        )
        self.assertTrue(result.success, result.message)
        # 2 of 3 views must agree: the glasses of one view are dropped.
        self.assertEqual(result.outputs["guest1.appearance"], ["red shirt"])

    def test_one_described_view_of_three_keeps_nothing(self):
        result, people = self.memorize_views(
            [SimpleNamespace(success=True, message="", features=["glasses"])],
            [people_message(person(1, 0.5, 0.0)), people_message()],
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["guest1.appearance"], [])
        self.assertEqual(len(people.describe.requests), 1)
        self.assertIn("Only 1 of 3 views described guest1", "\n".join(people.logs))

    def test_later_views_ask_for_the_track_the_guest_has_now(self):
        answer = SimpleNamespace(success=True, message="", features=["glasses"])
        result, people = self.memorize_views(
            [answer, answer],
            [
                people_message(person(1, 0.5, 0.0), stamp=0.0),
                people_message(person(5, 0.5, 0.0, "guest1"), stamp=0.6),
            ],
            {"views": 2},
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual([r.id for r in people.describe.requests], [1, 5])
        self.assertEqual(result.outputs["guest1.appearance"], ["glasses"])

    def test_describe_timeout_bounds_all_views(self):
        answer = SimpleNamespace(success=True, message="", features=["glasses"])
        result, people = self.memorize_views(
            [answer, answer, answer],
            [people_message(person(1, 0.5, 0.0), stamp=t) for t in (0.0, 0.6, 1.2)],
            {"describe_timeout": 0.5},
            delay=0.3,
        )
        # The second view runs out of the 0.5 s; one view keeps nothing.
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(people.describe.requests), 2)
        self.assertEqual(result.outputs["guest1.appearance"], [])
        self.assertIn("within describe_timeout", "\n".join(people.logs))

    def test_view_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        MemorizePerson.validate({"guest": "g", "views": 5, "view_interval": 1}, spec)
        for params in (
            {"guest": "g", "views": 0},
            {"guest": "g", "views": 1.5},
            {"guest": "g", "view_interval": 0},
        ):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    MemorizePerson.validate(params, spec)


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


def classified(people):
    return [(r.id, r.skip_vlm) for r in people.classify.requests]


def resight_after_judging(people, track, message):
    """Switch to `message` once the VLM has judged `track`."""
    answer = people.classify.answer

    def then_switch(request):
        reply = answer(request)
        if request.id == track and not request.skip_vlm:
            people.message, people.received = message, time.monotonic()
        return reply

    people.classify.answer = then_switch


class FindDescribedPersonTests(unittest.TestCase):
    def test_looks_at_each_person_before_judging_them(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0)),
            attributes={
                1: {"rules": ("standing", "none")},
                2: {"rules": ("standing", "waving")},
            },
        )
        people.look_settle = 0.05
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person"})
        )
        self.assertTrue(result.success, result.message)
        heads = [(p.x, p.y) for _, p in people.looked]
        # Each candidate twice (again where its head is seen then), then the one found.
        self.assertEqual(heads, [(1.0, 0.0)] * 2 + [(2.0, 0.0)] * 3)

    def test_the_head_moves_on_only_when_everyone_in_view_is_judged(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0)),
            attributes={
                1: {"rules": ("standing", "none")},
                2: {"rules": ("standing", "none")},
            },
        )
        # Each judgement takes longer than the dwell.
        people.classify.delay = lambda request: 0.05
        answer = people.classify.answer
        pans_judged = []

        def recording(request):
            pans_judged.append(people.panned[-1])
            return answer(request)

        people.classify.answer = recording
        result = FindDescribedPerson(runtime(people)).execute(
            context(
                {
                    "gesture": "waving person",
                    "pans": [0.0, 0.4],
                    "dwell": 0.01,
                    "timeout": 0.5,
                }
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(pans_judged, [0.0, 0.0])
        # After each judgement the head looks the same way again.
        self.assertEqual(people.panned[:3], [0.0, 0.0, 0.0])

    def test_a_person_whose_clothing_was_unclear_is_judged_again(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={1: {"vlm": ("standing", "none", ["red t shirt"])}},
        )
        answer = people.classify.answer
        unclear = [True]

        def first_unclear(request):
            reply = answer(request)
            if not request.skip_vlm and unclear:
                unclear.pop()
                reply.garments = []  # the upper body was hidden
            return reply

        people.classify.answer = first_unclear
        result = FindDescribedPerson(runtime(people)).execute(
            context(
                {
                    "clothing": "person wearing a red t shirt",
                    "timeout": 5.0,
                    "approach": False,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        vlm = [r.id for r in people.classify.requests if not r.skip_vlm]
        self.assertEqual(vlm, [1, 1])

    def test_a_person_next_to_a_judged_one_is_judged_too(self):
        # 0.45 m apart: person_perception keeps them as two tracks.
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 1.0, 0.45)),
            attributes={
                1: {"rules": ("standing", "none")},
                2: {"rules": ("standing", "waving")},
            },
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 5.0, "approach": False})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 2)

    def test_a_judged_person_seen_again_is_not_judged_again(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("standing", ""), "vlm": ("standing", "none", [])},
                3: {"rules": ("standing", "waving")},
            },
        )
        resight_after_judging(
            people,
            1,
            people_message(person(1, 1.0, 0.0, visible=False), person(3, 1.1, 0.1)),
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.5, "approach": False})
        )
        self.assertFalse(result.success)
        self.assertNotIn(3, [r.id for r in people.classify.requests])

    def test_finds_the_waving_person_asking_the_vlm_only_when_unclear(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0)),
            attributes={
                1: {"rules": ("standing", "none"), "vlm": ("standing", "none", [])},
                2: {
                    "rules": ("standing", ""),
                    "vlm": ("standing", "waving", ["red t shirt"]),
                },
            },
        )
        navigation = FakeNavigation()
        result = FindDescribedPerson(runtime(people, navigation)).execute(
            context({"gesture": "waving person", "location": "entrance"})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 2)
        self.assertEqual(result.outputs["person.gesture"], "waving")
        self.assertEqual(result.outputs["person.garments"], ["red t shirt"])
        self.assertEqual(classified(people), [(1, True), (2, True), (2, False)])
        self.assertEqual(navigation.goals, [[1.0, 0.0, 0.0]])
        self.assertEqual(people.panned[0], 0.0)

    def test_clothing_needs_the_vlm_and_people_outside_the_area_are_skipped(self):
        people = FakePeople(
            people_message(person(1, 5.0, 0.0), person(2, 1.0, 1.0)),
            attributes={
                1: {"vlm": ("standing", "none", ["blue jacket"])},
                2: {"vlm": ("sitting", "none", ["black coat", "blue jacket"])},
            },
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context(
                {
                    "clothing": "person wearing a blue jacket",
                    "role": "target",
                    "radius": 2.0,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["target.track"], 2)
        self.assertEqual(result.outputs["target.pose"], "sitting")
        self.assertEqual(classified(people), [(2, True), (2, False)])

    def test_without_approach_the_robot_only_looks(self):
        people = FakePeople(
            people_message(person(1, 2.0, 0.0)),
            attributes={1: {"rules": ("lying", "none")}},
        )
        navigation = FakeNavigation()
        result = FindDescribedPerson(runtime(people, navigation)).execute(
            context({"pose": "lying person", "approach": False})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(navigation.goals, [])
        self.assertNotIn("robot.location", result.outputs)

    def test_nobody_fits(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("sitting", "none"), "vlm": ("sitting", "none", [])}
            },
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"pose": "lying person", "timeout": 0.3})
        )
        self.assertFalse(result.success)
        self.assertIn("lying", result.message)
        self.assertEqual(classified(people), [(1, True)])  # checked once

    def test_a_new_track_where_someone_was_judged_is_not_judged_again(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("standing", "none")},
                7: {"rules": ("standing", "none")},
            },
        )

        def track_renumbered():
            time.sleep(0.2)
            people._on_people(people_message(person(7, 1.1, 0.1)))

        thread = threading.Thread(target=track_renumbered)
        thread.start()
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.6})
        )
        thread.join()
        self.assertFalse(result.success)
        self.assertEqual(classified(people), [(1, True)])

    def test_name_matches_remembered_people_without_classifying(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0, "Jules"))
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"name": "Jules"})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 2)
        self.assertEqual(people.classify.requests, [])

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={"office": [0.0, 0.0, 0.0]})
        FindDescribedPerson.validate(
            {"gesture": "waving person", "location": "office", "pans": [0.0, 0.5]},
            spec,
        )
        for params in (
            {},
            {"location": "office"},
            {"gesture": "dancing person"},
            {"gesture": "waving person", "location": "garage"},
            {"gesture": "waving person", "pans": []},
            {"gesture": "waving person", "radius": 0},
            {"gesture": "waving person", "x": 1},
            {"gesture": "waving person", "approach": "no"},
        ):
            with self.assertRaises(ValueError):
                FindDescribedPerson.validate(params, spec)


class FailureTests(unittest.TestCase):
    """One person who cannot be judged must not stop or fail the search."""

    def test_a_failing_classification_skips_only_that_person(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0)),
            attributes={2: {"rules": ("standing", "waving")}},
        )
        answer = people.classify.answer

        def flaky(request):
            if request.id == 1:
                raise TimeoutError("classify_person did not answer")
            return answer(request)

        people.classify.answer = flaky
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person"})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 2)

    def test_the_client_waits_longer_than_the_server(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("standing", ""), "vlm": ("standing", "waving", [])}
            },
        )
        timeouts = []
        call = people.call

        def spy(client, request, timeout, cancelled, deadline):
            timeouts.append(timeout)
            return call(client, request, timeout, cancelled, deadline)

        people.call = spy
        FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "describe_timeout": 180.0})
        )
        self.assertEqual(timeouts[0], SKIP_VLM_TIMEOUT)
        self.assertEqual(timeouts[1], 180.0 + SERVER_MARGIN)

    def test_an_abort_is_reported_as_cancelled(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={1: {"rules": ("standing", "none")}},
        )
        ctx = context({"gesture": "waving person", "timeout": 5.0})
        answer = people.classify.answer

        def abort(request):
            ctx.cancelled.set()
            raise Cancelled("Cancelled")

        people.classify.answer = abort
        result = FindDescribedPerson(runtime(people)).execute(ctx)
        self.assertEqual((result.success, result.message), (False, "Cancelled"))
        people.classify.answer = answer
        ctx = context({"output": "n", "gesture": "waving persons", "pans": [0.0]})
        people.classify.answer = abort
        result = CountPeople(runtime(people)).execute(ctx)
        self.assertEqual((result.success, result.message), (False, "Cancelled"))

    def test_slow_judgements_still_end_near_the_timeout(self):
        people = FakePeople(
            people_message(*(person(i, 1.0 + i, 0.0) for i in range(1, 6))),
            attributes={i: {"rules": ("standing", "none")} for i in range(1, 6)},
        )
        answer = people.classify.answer

        def slow(request):
            time.sleep(0.3)
            return answer(request)

        people.classify.answer = slow
        started = time.monotonic()
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.5})
        )
        self.assertFalse(result.success)
        # One person per check of the deadline: not all five (1.5 s).
        self.assertLess(time.monotonic() - started, 1.2)

    def test_a_person_the_vlm_failed_on_is_judged_again(self):
        unclear = {
            1: {"rules": ("standing", ""), "vlm": ("standing", "waving", ["red coat"])}
        }
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes=unclear,
            vlm_failures={1: 1},
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 5.0})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(classified(people), [(1, True), (1, False)] * 2)

        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes=unclear,
            vlm_failures={1: 1},
        )
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "n",
                    "gesture": "waving persons",
                    "pans": [0.0],
                    "dwell": 0.05,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["n"], 1)
        self.assertEqual(classified(people), [(1, True), (1, False)] * 2)

    def test_a_person_the_vlm_always_fails_on_is_tried_a_limited_number_of_times(
        self,
    ):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("standing", ""), "vlm": ("standing", "waving", [])}
            },
            vlm_failures={1: 100},
        )
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.5})
        )
        self.assertFalse(result.success)
        self.assertIn("1 could not be checked", result.message)
        self.assertEqual(classified(people), [(1, True), (1, False)] * MAX_JUDGEMENTS)

        people.classify.requests.clear()
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "n",
                    "gesture": "waving persons",
                    "pans": [0.0],
                    "dwell": 0.05,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["n"], 0)
        self.assertIn("1 could not be checked", result.message)
        self.assertEqual(classified(people), [(1, True), (1, False)] * MAX_JUDGEMENTS)

    def test_a_vlm_that_does_not_answer_is_cut_at_the_timeout(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={
                1: {"rules": ("standing", ""), "vlm": ("standing", "waving", [])}
            },
        )
        people.classify.delay = lambda request: 0.0 if request.skip_vlm else 60.0
        started = time.monotonic()
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.5})
        )
        self.assertFalse(result.success)
        self.assertIn("within", result.message)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(classified(people), [(1, True), (1, False)])

        people.classify.requests.clear()
        started = time.monotonic()
        result = CountPeople(runtime(people)).execute(
            context({"output": "n", "gesture": "waving persons", "timeout": 0.5})
        )
        self.assertFalse(result.success)
        self.assertIn("did not finish", result.message)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_the_look_at_a_person_is_cut_at_the_timeout(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0)),
            attributes={1: {"rules": ("standing", "none")}},
        )
        people.look_settle = 10.0
        started = time.monotonic()
        result = FindDescribedPerson(runtime(people)).execute(
            context({"gesture": "waving person", "timeout": 0.5})
        )
        self.assertFalse(result.success)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_a_name_given_after_the_track_started_is_still_found(self):
        people = FakePeople(people_message(person(5, 1.0, 0.0)))

        def named_later():
            time.sleep(0.3)
            people._on_people(people_message(person(5, 1.0, 0.0, "Jules")))

        thread = threading.Thread(target=named_later)
        thread.start()
        result = FindDescribedPerson(runtime(people)).execute(
            context({"name": "Jules", "timeout": 3.0})
        )
        thread.join()
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 5)


class CountPeopleTests(unittest.TestCase):
    def test_undecided_and_lost_people_do_not_stop_the_count(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0)),
            attributes={
                # The VLM saw no upper body: the clothing stays unknown.
                1: {"vlm": ("standing", "none", [])},
                # 2: the track is gone before it can be judged.
            },
        )
        started = time.monotonic()
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "n",
                    "clothing": "people wearing red t shirts",
                    "pans": [0.0],
                    "dwell": 0.05,
                    "timeout": 10.0,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(result.outputs["n"], 0)
        self.assertIn("2 could not be checked", result.message)
        requests = [r.id for r in people.classify.requests]
        self.assertEqual(requests.count(1), 2)  # rules, then the VLM, once
        self.assertEqual(requests.count(2), MAX_JUDGEMENTS)  # lost: retried once

    def test_counts_matching_people_once_each(self):
        others = (
            person(2, 2.0, 0.0),
            person(3, 0.0, 2.0),
            person(5, 9.0, 0.0),  # outside the area
        )
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), *others),
            attributes={
                1: {"rules": ("standing", ""), "vlm": ("standing", "waving", [])},
                2: {"rules": ("standing", "none")},
                3: {"rules": ("standing", "raising_left_arm")},
                5: {"rules": ("standing", "waving")},
            },
        )
        resight_after_judging(
            people,
            1,
            # Person 1 out of view, then seen again with a new track ID.
            people_message(
                person(1, 1.0, 0.0, visible=False), person(4, 1.2, 0.1), *others
            ),
        )
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "cmd1.count",
                    "gesture": "waving persons",
                    "pans": [0.0, 0.4],
                    "dwell": 0.05,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["cmd1.count"], 1)
        # The last sighting of the person.
        self.assertEqual(result.outputs["cmd1.count.positions"], [[1.2, 0.1, 1.0]])
        # After judging people the head looks the same way again, then moves on
        # and ends straight.
        self.assertEqual(people.panned[:2], [0.0, 0.0])
        order = [
            pan
            for i, pan in enumerate(people.panned)
            if i == 0 or people.panned[i - 1] != pan
        ]
        self.assertEqual(order, [0.0, 0.4, 0.0])
        self.assertNotIn(4, [r.id for r in people.classify.requests])
        self.assertNotIn(5, [r.id for r in people.classify.requests])

    def test_a_person_coming_close_to_a_judged_one_keeps_their_entry(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 1.7, 0.0)),
            attributes={
                1: {"rules": ("standing", "none")},
                2: {"rules": ("standing", ""), "vlm": ("standing", "waving", [])},
            },
            vlm_failures={2: 1},
        )
        answer = people.classify.answer

        def answer_then_come_closer(request):
            reply = answer(request)
            if request.id == 2 and not reply.success:
                # Tracking moves person 2 to 0.4 m of the judged person 1.
                people.message = people_message(
                    person(1, 1.0, 0.0), person(2, 1.4, 0.0)
                )
                people.received = time.monotonic()
            return reply

        people.classify.answer = answer_then_come_closer
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "n",
                    "gesture": "waving persons",
                    "pans": [0.0],
                    "dwell": 0.05,
                    "timeout": 10.0,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["n"], 1)
        vlm = [r.id for r in people.classify.requests if not r.skip_vlm]
        self.assertEqual(vlm, [2, 2])  # failed, then judged again

    def test_people_side_by_side_are_counted_apart(self):
        # 0.45 m apart on a sofa: person_perception keeps them as two tracks.
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 1.0, 0.45)),
            attributes={
                1: {"rules": ("sitting", "none")},
                2: {"rules": ("sitting", "waving")},
            },
        )
        result = CountPeople(runtime(people)).execute(
            context(
                {
                    "output": "n",
                    "gesture": "waving persons",
                    "pans": [0.0],
                    "dwell": 0.05,
                    "timeout": 10.0,
                }
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["n"], 1)
        self.assertEqual(result.outputs["n.positions"], [[1.0, 0.45, 1.0]])

    def test_counts_everyone_without_a_description(self):
        people = FakePeople(
            people_message(person(1, 1.0, 0.0), person(2, 2.0, 0.0), frame="camera")
        )
        result = CountPeople(runtime(people)).execute(
            context({"output": "n", "pans": [0.0], "dwell": 0.05})
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["n"], 2)
        self.assertEqual(people.classify.requests, [])

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        CountPeople.validate(
            {"output": "n", "clothing": "people wearing red coats"}, spec
        )
        for params in ({}, {"output": ""}, {"output": "n", "pose": "jumping person"}):
            with self.assertRaises(ValueError):
                CountPeople.validate(params, spec)


if __name__ == "__main__":
    unittest.main()
