"""answer_question and tell_info with fake speech, people and navigation (no ROS)."""

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
from aha_tasks.talk import AnswerQuestion, TellInfo  # noqa: E402

LOCATIONS = {
    "spawn": [0.0, 0.0, 0.0],
    "instruction_point": [-1.0, 0.0, 3.141593],
    "desk": [3.0, 0.0, 0.0],
}
FACTS = {
    "gpsr.robot_name": "AhaRobot",
    "gpsr.team_name": "TRAIL",
    "gpsr.team_country": "Japan",
    "gpsr.team_affiliation": "the University of Tokyo",
    "gpsr.utc_offset": 9,
}


def point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def person(id, x, y, name=""):
    return SimpleNamespace(
        id=id, name=name, visible=True, position=point(x, y, 1.0), head=point(x, y, 1.6)
    )


def people_message(*people):
    return SimpleNamespace(header=SimpleNamespace(frame_id="map"), people=list(people))


class FakePeople(People):
    now = staticmethod(time.monotonic)

    def __init__(self, message=None):
        self.lock = threading.Lock()
        self.message, self.received = message, time.monotonic()
        self.looked, self.panned, self.followed = [], [], []

    def look_at(self, frame, point):
        self.looked.append((frame, (point.x, point.y, point.z)))

    def look_towards(self, pan, height=1.2):
        self.panned.append(pan)

    def follow(self, track=None, name=""):
        self.followed.append((track, name))


class FakeInteraction:
    def __init__(self):
        self.said = []

    def say(self, text, cancelled):
        self.said.append(text)
        return True, "said"


class FakeNavigation:
    def __init__(self):
        self.goals = []

    def navigate(self, pose, cancelled=None):
        self.goals.append([round(v, 3) for v in pose])
        return True, ""


def runtime(people=None, navigation=None):
    spec = SimpleNamespace(locations=LOCATIONS, initial_state=FACTS)
    return SimpleNamespace(
        node=None,
        spec=spec,
        navigation=navigation,
        people=people,
        interaction=FakeInteraction(),
        gaze=None,
    )


def context(params, memory=None):
    return SkillContext(
        subgoal="exec.step01",
        params=params,
        memory={"robot.location": "spawn", **FACTS, **(memory or {})},
        cancelled=threading.Event(),
        log=lambda text: None,
    )


class AnswerQuestionTests(unittest.TestCase):
    def test_says_the_answer(self):
        skill = AnswerQuestion(runtime())
        result = skill.execute(context({"question": "your teams affiliation"}))
        self.assertTrue(result.success, result.message)
        said = ["My team's affiliation is the University of Tokyo."]
        self.assertEqual(skill.runtime.interaction.said, said)
        result = skill.execute(context({"question": "What time is it?"}))
        self.assertRegex(result.message, r"^It is \d+:\d\d [AP]M\.$")

    def test_params(self):
        spec = SimpleNamespace(locations=LOCATIONS, initial_state=FACTS)
        AnswerQuestion.validate({"question": "the day of the week"}, spec)
        for params in (
            {},
            {"question": ""},
            {"question": "the time", "person": "me"},
        ):
            with self.assertRaises(ValueError, msg=params):
                AnswerQuestion.validate(params, spec)
        # An unknown question mocks only its own step, not the whole plan.
        unknown = {"question": "the capital of France"}
        AnswerQuestion.validate(unknown, spec)
        skill = AnswerQuestion(SimpleNamespace())
        self.assertIn("cannot answer", skill.unsupported(unknown))
        self.assertIsNone(skill.unsupported({"question": "the time"}))
        no_facts = SimpleNamespace(locations=LOCATIONS, initial_state={})
        AnswerQuestion.validate({"question": "the time"}, no_facts)
        with self.assertRaises(ValueError):
            AnswerQuestion.validate({"question": "your teams name"}, no_facts)
        bad_zone = SimpleNamespace(
            locations=LOCATIONS, initial_state={**FACTS, "gpsr.utc_offset": "JST"}
        )
        with self.assertRaises(ValueError):
            AnswerQuestion.validate({"question": "the time"}, bad_zone)


class TellInfoTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("aha_tasks.talk.FACE_TIMEOUT", 0.3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tell(self, params, memory=None, people=None, navigation=None):
        skill = TellInfo(runtime(people, navigation))
        result = skill.execute(context(params, memory))
        return result, skill.runtime.interaction.said

    def test_the_answer_sentence_first_else_the_value(self):
        memory = {"cmd2.count": 3, "cmd2.count.answer": "There are 3 drinks."}
        result, said = self.tell(
            {"key": "cmd2.count", "person": "waving person"}, memory
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(said, ["There are 3 drinks."])
        result, said = self.tell(
            {"key": "person.gesture", "person": "operator"},
            {"person.gesture": "pointing_left"},
        )
        self.assertEqual(said, ["The person is pointing to the left."])

    def test_nothing_to_tell(self):
        for memory in ({}, {"cmd2.count": None}):
            result, said = self.tell({"key": "cmd2.count", "person": "me"}, memory)
            self.assertFalse(result.success)
            self.assertIn("Nothing to tell", result.message)
            self.assertEqual(said, [])

    def test_operator_at_the_instruction_point(self):
        # The operator stands 2 m in front of the instruction point.
        people = FakePeople(people_message(person(4, -3.0, 0.0), person(5, 2.0, 0.0)))
        navigation = FakeNavigation()
        result, said = self.tell(
            {"key": "n", "person": "operator"}, {"n": 2}, people, navigation
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(said, ["The answer is 2."])
        # Went back, then approached the nearest person to 1 m.
        self.assertEqual(navigation.goals, [[-1.0, 0.0, 3.142], [-2.0, 0.0, 3.142]])
        self.assertEqual(result.outputs["robot.location"], [-2.0, 0.0, 3.142])
        self.assertEqual(people.followed, [(4, "")])

    def test_operator_close_by_is_only_turned_to(self):
        people = FakePeople(people_message(person(4, -1.0, 1.0)))
        navigation = FakeNavigation()
        result, _ = self.tell(
            {"key": "n", "person": "me"},
            {"n": 2, "robot.location": "instruction_point"},
            people,
            navigation,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(navigation.goals, [[-1.0, 0.0, 1.571]])

    def test_nobody_seen_still_tells(self):
        navigation = FakeNavigation()
        result, said = self.tell(
            {"key": "n", "person": "operator"},
            {"n": 2},
            FakePeople(people_message()),
            navigation,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(said, ["The answer is 2."])
        self.assertEqual(navigation.goals, [[-1.0, 0.0, 3.142]])
        self.assertEqual(result.outputs["robot.location"], "instruction_point")

    def test_the_person_found_before(self):
        memory = {
            "n": 1,
            "robot.location": "desk",
            "recipient.track": 8,
            "recipient.position": [5.0, 0.0, 1.0],
            "recipient.frame": "map",
        }
        # Tracked again under another ID near where they were found.
        people = FakePeople(people_message(person(9, 5.2, 0.0), person(3, 9.0, 0.0)))
        navigation = FakeNavigation()
        result, said = self.tell(
            {"key": "n", "person": "person at the desk", "role": "recipient"},
            memory,
            people,
            navigation,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(people.looked, [("map", (5.0, 0.0, 1.5))])
        self.assertEqual(people.followed, [(9, "")])
        self.assertEqual(navigation.goals, [[4.2, 0.0, 0.0]])

    def test_a_person_not_seen_is_turned_to_where_they_were(self):
        memory = {
            "n": 1,
            "person.position": [0.0, 1.0, 1.0],
            "person.frame": "map",
        }
        navigation = FakeNavigation()
        result, said = self.tell(
            {"key": "n", "person": "Charlie"}, memory, FakePeople(), navigation
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(navigation.goals, [[0.0, 0.0, 1.571]])
        self.assertEqual(said, ["The answer is 1."])

    def test_without_person_perception(self):
        navigation = FakeNavigation()
        result, said = self.tell(
            {"key": "n", "person": "operator"}, {"n": 2}, None, navigation
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(navigation.goals, [[-1.0, 0.0, 3.142]])

    def test_params_and_expected_location(self):
        spec = SimpleNamespace(locations=LOCATIONS, initial_state=FACTS)
        TellInfo.validate({"key": "n", "person": "operator"}, spec)
        TellInfo.validate({"key": "n", "person": "Jane", "location": "desk"}, spec)
        for params in (
            {"key": "n"},
            {"key": "", "person": "me"},
            {"key": "n", "person": "Jane", "location": "garage"},
            {"key": "n", "person": "Jane", "role": ""},
            {"key": "n", "person": "Jane", "to": "x"},
        ):
            with self.assertRaises(ValueError, msg=params):
                TellInfo.validate(params, spec)
        no_point = SimpleNamespace(locations={"desk": [0.0, 0.0, 0.0]})
        with self.assertRaises(ValueError):
            TellInfo.validate({"key": "n", "person": "operator"}, no_point)
        skill = TellInfo(runtime())
        self.assertEqual(
            skill.expected({"key": "n", "person": "operator"}),
            {"robot.location": "instruction_point"},
        )
        self.assertEqual(skill.expected({"key": "n", "person": "Jane"}), {})


if __name__ == "__main__":
    unittest.main()
