"""Gaze targets against a fake node: path lookahead and person following."""

import math
import os
import sys
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.gaze import (  # noqa: E402
    LOOK_HEIGHT,
    LOOKAHEAD,
    Gaze,
    follow_choice,
    path_lookahead,
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


def plan_message(points, frame="map"):
    poses = [
        SimpleNamespace(pose=SimpleNamespace(position=point(x, y))) for x, y in points
    ]
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame), poses=poses)


class FakeNode:
    def __init__(self):
        self.sent = []
        self.subscriptions = {}

    def create_publisher(self, _type, topic, _qos):
        return SimpleNamespace(publish=self.sent.append)

    def create_subscription(self, _type, topic, callback, _qos):
        self.subscriptions[topic] = callback

    def create_timer(self, _period, _callback):
        pass


class FakePeople:
    def __init__(self, message=None):
        self.message = message

    def latest(self):
        return self.message


def targets(node):
    return [
        (
            m.header.frame_id,
            round(m.point.x, 3),
            round(m.point.y, 3),
            round(m.point.z, 3),
        )
        for m in node.sent
    ]


class PathLookaheadTests(unittest.TestCase):
    def test_point_along_the_path(self):
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]
        self.assertEqual(path_lookahead(points, 0.5, 0.0), (0.5, 0.0))
        self.assertEqual(path_lookahead(points, 1.5, 0.0), (1.0, 0.5))

    def test_continues_in_the_travel_direction_past_the_end(self):
        # The goal faces back and the last grid step goes sideways, but the
        # robot still drives on towards the goal: look ahead.
        points = [(0.0, 0.0), (0.95, 0.0), (1.0, 0.05)]
        ahead = path_lookahead(points, 1.5, math.pi)
        heading = math.atan2(0.05, 1.0)
        self.assertAlmostEqual(math.atan2(ahead[1] - 0.05, ahead[0] - 1.0), heading)

    def test_turns_to_the_goal_heading_when_arriving(self):
        ahead = path_lookahead([(0.0, 0.0), (0.2, 0.0)], 1.5, math.pi / 2)
        self.assertAlmostEqual(ahead[0], 0.2)
        self.assertAlmostEqual(ahead[1], 1.3)
        # At the goal: level and ahead in the final heading, not under the robot.
        self.assertEqual(path_lookahead([(2.0, 1.0)], 1.0, 0.0), (3.0, 1.0))

    def test_empty_plan(self):
        self.assertIsNone(path_lookahead([], 1.0, 0.0))


class FollowChoiceTests(unittest.TestCase):
    def test_track_then_name_then_near_the_last_sighting(self):
        people = [
            person(1, 0.0, 0.0),
            person(2, 1.0, 0.0, "guest1"),
            person(3, 3.0, 0.0),
        ]
        self.assertEqual(follow_choice(people, track=3).id, 3)
        self.assertEqual(follow_choice(people, track=9, name="guest1").id, 2)
        self.assertEqual(follow_choice(people, track=9, last=(0.2, 0.0)).id, 1)

    def test_others_are_not_taken_for_the_lost_person(self):
        people = [
            person(1, 0.1, 0.0, "host"),
            person(2, 2.0, 0.0),
            person(4, 0.0, 0.0, visible=False),
        ]
        self.assertIsNone(follow_choice(people, track=9, last=(0.0, 0.0)))
        self.assertIsNone(follow_choice(people, track=9))


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.node = FakeNode()
        self.gaze = Gaze(self.node)

    def test_follows_the_track_and_takes_over_a_new_track_nearby(self):
        people = FakePeople(people_message(person(5, 1.0, 0.0), person(6, 3.0, 0.0)))
        self.gaze.follow_person(people, track=5)
        people.message = people_message(person(5, 1.1, 0.2), person(6, 3.0, 0.0))
        self.gaze._tick()
        # Lost and seen again under a new track, 0.2 m away.
        people.message = people_message(person(8, 1.2, 0.3), person(6, 3.0, 0.0))
        self.gaze._tick()
        self.assertEqual(
            targets(self.node),
            [("map", 1.0, 0.0, 1.5), ("map", 1.1, 0.2, 1.5), ("map", 1.2, 0.3, 1.5)],
        )
        self.assertEqual(self.gaze.track, 8)

    def test_same_message_and_small_moves_are_not_sent_again(self):
        people = FakePeople(people_message(person(5, 1.0, 0.0)))
        self.gaze.follow_person(people, track=5)
        self.gaze._tick()
        people.message = people_message(person(5, 1.005, 0.0))
        self.gaze._tick()
        self.assertEqual(len(self.node.sent), 1)

    def test_path_looks_at_the_goal_until_a_new_plan_arrives(self):
        self.node.subscriptions["/plan"](plan_message([(9.0, 9.0), (9.0, 8.0)]))
        self.gaze.follow_path([4.0, 0.0, 0.0])
        self.gaze._tick()  # the plan before follow_path is of an older goal
        time.sleep(0.01)
        self.node.subscriptions["/plan"](plan_message([(0.0, 0.0), (4.0, 0.0)]))
        self.gaze._tick()
        self.assertEqual(
            targets(self.node),
            [("map", 4.0, 0.0, LOOK_HEIGHT), ("map", LOOKAHEAD, 0.0, LOOK_HEIGHT)],
        )

    def test_a_point_or_hold_ends_following(self):
        people = FakePeople(people_message(person(5, 1.0, 0.0)))
        self.gaze.follow_person(people, track=5)
        self.gaze.look_towards(0.0)
        people.message = people_message(person(5, 2.0, 0.0))
        self.gaze._tick()
        self.assertEqual(
            targets(self.node)[-1], ("base_footprint", 2.0, 0.0, LOOK_HEIGHT)
        )
        self.gaze.follow_path([4.0, 0.0, 0.0])
        self.gaze.hold()
        self.node.subscriptions["/plan"](plan_message([(0.0, 0.0), (4.0, 0.0)]))
        self.gaze._tick()
        self.assertEqual(targets(self.node)[-1], ("map", 4.0, 0.0, LOOK_HEIGHT))


if __name__ == "__main__":
    unittest.main()
