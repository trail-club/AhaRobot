"""follow_person and guide_person against a scripted world (no ROS graph)."""

import math
import os
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.follow import (  # noqa: E402
    ACTIVE,
    FAILED,
    HEAD_REACH,
    IDLE,
    LETHAL,
    REACQUIRE_MAX,
    SUCCEEDED,
    FollowPerson,
    GuidePerson,
    Stillness,
    Target,
    address,
    approach_poses,
    bearing,
    distance_to_trail,
    follow_done,
    follower,
    free_pose,
    grid_cost,
    needs_goal,
    remembered,
    trail_point,
    turn_to_see,
)
from aha_tasks.people import People  # noqa: E402

DT = 0.1  # s of simulated time per people message
LOCATIONS = {
    "start": [0.0, 0.0, math.pi / 2],
    "kitchen": [0.0, 6.0, math.pi / 2],
    "office": [6.0, 0.0, 0.0],
}


def person(id, x, y, name="", visible=True):
    return SimpleNamespace(
        id=id,
        name=name,
        visible=visible,
        position=SimpleNamespace(x=x, y=y, z=1.0),
        head=SimpleNamespace(x=x, y=y, z=1.75),
    )


def grid(cells, width=10, height=10, resolution=1.0):
    """An OccupancyGrid-like costmap from origin (0, 0); `cells` maps (col, row) to cost."""
    data = [0] * (width * height)
    for (col, row), cost in cells.items():
        data[row * width + col] = cost
    origin = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
    info = SimpleNamespace(
        resolution=resolution, width=width, height=height, origin=origin
    )
    return SimpleNamespace(info=info, data=data)


class GeometryTests(unittest.TestCase):
    def test_bearing_and_turn_to_see(self):
        robot = [0.0, 0.0, 0.0]
        self.assertAlmostEqual(bearing(robot, (0.0, 1.0)), math.pi / 2)
        # Within the head's reach: no turn.
        self.assertEqual(turn_to_see(robot, (1.0, 1.0)), 0.0)
        # Straight behind: turn until the head reaches it on the nearer side.
        yaw = turn_to_see([0.0, 0.0, 0.0], (-1.0, -0.01))
        self.assertAlmostEqual(
            abs(bearing([0.0, 0.0, yaw], (-1.0, -0.01))), HEAD_REACH, places=6
        )
        self.assertLess(yaw, 0.0)  # turned right, the shorter way

    def test_approach_faces_the_person_from_the_robot_side_first(self):
        poses = approach_poses([0.0, 0.0, 0.0], (0.0, 3.0), 1.0)
        x, y, yaw = poses[0]
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(yaw, math.pi / 2)
        for x, y, yaw in poses:
            self.assertAlmostEqual(math.dist((x, y), (0.0, 3.0)), 1.0)
            self.assertAlmostEqual(math.atan2(3.0 - y, -x), yaw)
        # Alternatives turn to both sides, nearer ones first.
        sides = [math.atan2(y - 3.0, x) for x, y, _ in poses]
        self.assertAlmostEqual(abs(sides[1] - sides[0]), abs(sides[2] - sides[0]))

    def test_close_enough_only_turns(self):
        self.assertEqual(
            approach_poses([0.0, 0.0, 0.0], (0.5, 0.0), 1.0), [[0.0, 0.0, 0.0]]
        )

    def test_costmap_picks_the_first_free_pose(self):
        costs = grid({(2, 2): LETHAL, (3, 3): 100, (5, 5): -1, (6, 6): 50})
        self.assertEqual(grid_cost(costs, 2.5, 2.5), LETHAL)
        self.assertIsNone(grid_cost(costs, -0.5, 1.0))
        poses = [[2.5, 2.5, 0.0], [3.5, 3.5, 0.0], [5.5, 5.5, 0.0], [6.5, 6.5, 1.0]]
        self.assertEqual(free_pose(poses, costs), [6.5, 6.5, 1.0])
        self.assertIsNone(free_pose(poses[:3], costs))
        self.assertEqual(free_pose(poses, None), poses[0])  # no costmap yet

    def test_goals_are_replaced_at_most_once_a_period_and_only_when_moved(self):
        active = [0.0, 0.0, 0.0]
        self.assertTrue(needs_goal(active, None, 0.0, IDLE, 0.0))
        near = [0.1, 0.0, 0.1]
        far = [0.5, 0.0, 0.0]
        self.assertFalse(needs_goal(far, active, 0.0, ACTIVE, 0.5))
        self.assertFalse(needs_goal(near, active, 0.0, ACTIVE, 2.0))
        self.assertTrue(needs_goal(far, active, 0.0, ACTIVE, 2.0))
        self.assertTrue(needs_goal([0.0, 0.0, 1.0], active, 0.0, ACTIVE, 2.0))
        # An ended goal is sent again, e.g. after Nav2 aborted it.
        self.assertTrue(needs_goal(near, active, 0.0, FAILED, 2.0))
        self.assertTrue(needs_goal(near, active, 0.0, SUCCEEDED, 2.0))

    def test_trail(self):
        trail = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
        self.assertEqual(trail_point(trail, 1.0), (2.0, 1.0))
        self.assertEqual(trail_point(trail, 3.0), (1.0, 0.0))
        self.assertEqual(trail_point(trail, 9.0), (0.0, 0.0))
        self.assertAlmostEqual(distance_to_trail((1.0, 1.0), trail), 1.0)
        self.assertAlmostEqual(distance_to_trail((3.0, 3.0), trail), math.sqrt(2))
        self.assertAlmostEqual(
            distance_to_trail((1.0, 1.0), [(0.0, 0.0)]), math.sqrt(2)
        )


class TargetTests(unittest.TestCase):
    def test_remembered_target_from_find_described_person(self):
        target = remembered(
            {
                "guest.track": 7,
                "guest.position": [1.0, 2.0, 1.0],
                "guest.frame": "map",
            },
            "guest",
        )
        self.assertEqual((target.track, target.position), (7, (1.0, 2.0)))
        target = remembered({"person.position": [1.0, 2.0, 1.0]}, "person")
        self.assertEqual((target.track, target.position), (None, None))

    def test_first_sighting_near_the_remembered_position_else_the_robot(self):
        people = [person(1, 0.5, 0.0), person(2, 2.5, 0.0)]
        self.assertEqual(Target(position=(2.4, 0.0)).find(people, 0.0).id, 2)
        self.assertEqual(Target().find(people, 0.0, origin=(0.0, 0.0)).id, 1)
        self.assertIsNone(Target().find(people, 0.0, origin=(9.0, 0.0)))
        self.assertIsNone(Target().find(people, 0.0))
        # The remembered track wins wherever it is.
        self.assertEqual(Target(track=1, position=(2.4, 0.0)).find(people, 0.0).id, 1)

    def test_hands_over_to_a_new_track_near_the_last_sighting(self):
        target = Target(track=1)
        target.saw(person(1, 0.0, 0.0), [person(1, 0.0, 0.0), person(9, 3.0, 0.0)], 0.0)
        new = [person(2, 0.4, 0.0), person(9, 3.0, 0.0)]
        self.assertEqual(target.find(new, 0.1).id, 2)
        # Farther than REACQUIRE right after losing them, within reach later.
        later = [person(3, 1.2, 0.0)]
        self.assertIsNone(target.find(later, 0.1))
        self.assertEqual(target.find(later, 1.5).id, 3)
        self.assertEqual(target.reacquire_radius(100.0), REACQUIRE_MAX)

    def test_never_takes_someone_seen_together_or_with_another_name(self):
        target = Target(track=1)
        target.saw(person(1, 0.0, 0.0), [person(1, 0.0, 0.0), person(5, 0.6, 0.0)], 0.0)
        self.assertIsNone(target.find([person(5, 0.3, 0.0)], 5.0))
        self.assertIsNone(target.find([person(6, 0.3, 0.0, name="host")], 5.0))
        self.assertEqual(target.find([person(7, 0.3, 0.0)], 5.0).id, 7)

    def test_a_named_person_is_found_by_name(self):
        target = Target(track=1)
        target.saw(person(1, 0.0, 0.0, name="host"), [], 0.0)
        self.assertEqual(target.find([person(4, 2.9, 0.0, name="host")], 0.1).id, 4)

    def test_ahead_is_the_way_they_walked(self):
        target = Target()
        target.saw(person(1, 0.0, 0.0), [], 0.0)
        self.assertEqual(target.ahead(), (0.0, 0.0))  # not walking
        target.saw(person(1, 0.0, 1.0), [], 1.0)
        self.assertEqual(target.ahead(), (0.0, 2.0))
        target.saw(person(1, 1.0, 1.0), [], 3.0)  # the first sighting is too old
        self.assertEqual(target.ahead(), (1.0, 1.0))

    def test_travel_is_the_farthest_from_the_first_sighting(self):
        target = Target()
        for t, x in enumerate([0.0, 2.0, 1.0]):
            target.saw(person(1, x, 0.0), [], float(t))
        self.assertEqual(target.travelled, 2.0)


class EndTests(unittest.TestCase):
    def test_stillness_counts_only_with_the_robot_next_to_them(self):
        still = Stillness()
        self.assertEqual(still.update((0.0, 0.0), 0.0, close=False), 0.0)
        self.assertEqual(still.update((0.0, 0.0), 1.0, close=True), 0.0)
        self.assertEqual(still.update((0.1, 0.0), 3.0, close=True), 2.0)
        self.assertEqual(still.update((1.0, 0.0), 4.0, close=True), 0.0)  # walked
        self.assertEqual(still.update((1.0, 0.0), 5.0, close=False), 0.0)

    def test_follow_ends_after_walking_and_stopping_in_the_area(self):
        area = (0.0, 5.0, 2.0)
        self.assertTrue(follow_done(5.0, 5.0, 3.0, (0.0, 4.0), area))
        self.assertTrue(follow_done(5.0, 5.0, 3.0, (9.0, 9.0), None))
        self.assertFalse(follow_done(4.9, 5.0, 3.0, (0.0, 4.0), area))
        self.assertFalse(follow_done(5.0, 5.0, 0.5, (0.0, 4.0), area))  # never walked
        self.assertFalse(follow_done(5.0, 5.0, 3.0, (0.0, 2.0), area))  # elsewhere

    def test_follower_is_close_and_on_the_robot_path(self):
        trail = [(0.0, 0.0), (3.0, 0.0)]
        robot = [3.0, 0.0, 0.0]
        target = Target(track=1)
        target.saw(person(1, 0.0, 0.0), [person(1, 0.0, 0.0), person(8, 0.0, 2.0)], 0.0)
        self.assertEqual(
            follower([person(1, 2.0, 0.0)], target, robot, trail, 2.5).id, 1
        )
        self.assertIsNone(follower([person(1, 0.0, 0.0)], target, robot, trail, 2.5))
        self.assertEqual(
            follower([person(2, 1.8, 0.3)], target, robot, trail, 2.5).id, 2
        )
        self.assertIsNone(follower([person(3, 2.0, 2.0)], target, robot, trail, 2.5))
        self.assertIsNone(follower([person(8, 2.0, 0.1)], target, robot, trail, 2.5))

    def test_address_by_name_only(self):
        self.assertEqual(address("Charlie"), "Charlie, ")
        self.assertEqual(address("waving person"), "")
        self.assertEqual(address("operator"), "")


# ---- skills in a scripted world -------------------------------------------------


class World(People):
    """Time, the robot and walking people; time passes as the skill sleeps.

    A people message comes every DT. The robot drives towards the latest goal
    at `speed` and turns at once on arrival. A person is seen while within FOV
    of the robot's heading (the head turns up to its limit, the camera sees a
    bit more).
    """

    FOV = 1.57 + 0.6

    def __init__(self, robot, walkers, speed=0.3):
        self.lock = threading.Lock()
        self.t = 0.0
        self.robot = list(robot)
        self.speed = speed
        self.walkers = walkers  # objects with id, position(t, robot), visible(t)
        self.goal = None
        self.followed, self.looked = [], []
        self.said = []
        self.goals = []
        self.turns = []
        self.pending = 0.0  # s slept towards the next step
        self.message = None
        self.step()

    # People API
    def now(self):
        return self.t

    def latest(self):
        return self.message

    def sleep(self, seconds):
        self.pending += seconds
        while self.pending >= DT - 1e-9:
            self.pending -= DT
            self.step()

    def look_at(self, frame, point):
        self.looked.append((point.x, point.y))

    def look_towards(self, pan, height=1.2):
        self.looked.append(pan)

    def follow(self, track=None, name=""):
        self.followed.append(track)

    def step(self):
        self.t += DT
        self.drive()
        people = []
        for walker in self.walkers:
            x, y = walker.position(self.t, self.robot)
            seen = walker.visible(self.t) and (
                abs(bearing(self.robot, (x, y))) <= self.FOV
            )
            people.append(person(walker.track(self.t), x, y, visible=seen))
        self.message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"), people=people
        )

    def drive(self):
        if self.goal is None:
            return
        gx, gy, gyaw = self.goal
        left = math.dist(self.robot[:2], (gx, gy))
        step = self.speed * DT
        if left <= step:
            self.robot = [gx, gy, gyaw]
            self.goal = None
            self.arrived = True
            return
        heading = math.atan2(gy - self.robot[1], gx - self.robot[0])
        self.robot = [
            self.robot[0] + step * math.cos(heading),
            self.robot[1] + step * math.sin(heading),
            heading,
        ]

    # Base, Pursuit (through PursuitView) and Interaction stand-ins
    grid = None

    def pose(self):
        return list(self.robot)

    def send(self, pose, now):
        self.goals.append(list(pose))
        self.goal, self.arrived = list(pose), False
        self.sent_at = now

    def state(self):
        if self.goal is not None:
            return ACTIVE
        return SUCCEEDED if getattr(self, "arrived", False) else IDLE

    def stop(self):
        self.goal = None

    def halt(self):
        self.goal = None

    def go(self, pose, people, cancelled, timeout):
        self.turns.append(list(pose))
        self.robot = list(pose)
        self.goal = None
        return True, ""

    def say(self, text, cancelled):
        self.said.append(text)
        return True, ""


class PursuitView:
    """The Pursuit interface of a World (goal is the latest sent pose)."""

    def __init__(self, world):
        self.world = world
        self.goal, self.sent_at = None, -math.inf

    def send(self, pose, now):
        self.goal, self.sent_at = list(pose), now
        self.world.send(pose, now)

    def state(self):
        return self.world.state()

    def stop(self):
        self.goal = None
        self.world.stop()

    def halt(self):
        self.stop()

    def go(self, pose, people, cancelled, timeout):
        return self.world.go(pose, people, cancelled, timeout)


class Route:
    """A person walking `points` at `speed` from `start` s; optional hidden spans."""

    def __init__(self, id, points, speed=0.4, start=1.0, hidden=(), new_id_after=None):
        self.id, self.points, self.speed, self.start = id, points, speed, start
        self.hidden = hidden
        self.new_id_after = new_id_after

    def track(self, t):
        if self.new_id_after is not None and t > self.new_id_after:
            return self.id + 100
        return self.id

    def visible(self, t):
        return not any(a <= t < b for a, b in self.hidden)

    def position(self, t, robot):
        left = max(0.0, t - self.start) * self.speed
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            step = math.dist((x0, y0), (x1, y1))
            if left <= step and step > 0.0:
                f = left / step
                return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
            left -= step
        return tuple(self.points[-1])


class Standing(Route):
    def __init__(self, id, point, **kwargs):
        super().__init__(id, [point, point], **kwargs)


class Trailing:
    """Walks the robot's path `gap` behind it at `speed` (as the sim walkers do)."""

    def __init__(self, id, start, gap=1.2, speed=0.4, stop_at=None):
        self.id, self.xy = id, tuple(start)
        self.gap, self.speed = gap, speed
        self.path = []
        self.stop_at = stop_at  # (t0, t1): stands still meanwhile
        self.last = None

    def track(self, t):
        return self.id

    def visible(self, t):
        return True

    def position(self, t, robot):
        if self.last is None or math.dist(self.last, robot[:2]) >= 0.15:
            self.path.append(tuple(robot[:2]))
            self.last = tuple(robot[:2])
        standing = self.stop_at is not None and self.stop_at[0] <= t < self.stop_at[1]
        if not standing and math.dist(self.xy, robot[:2]) > self.gap:
            left = self.speed * DT
            while self.path and left > 0.0:
                target = self.path[0]
                d = math.dist(self.xy, target)
                if d <= left:
                    self.xy = target
                    self.path.pop(0)
                    left -= d
                else:
                    f = left / d
                    self.xy = (
                        self.xy[0] + f * (target[0] - self.xy[0]),
                        self.xy[1] + f * (target[1] - self.xy[1]),
                    )
                    left = 0.0
        return self.xy


def run(skill_class, world, params, memory=None):
    runtime = SimpleNamespace(
        node=None,
        spec=SimpleNamespace(locations=LOCATIONS),
        navigation=object(),
        people=world,
        gaze=SimpleNamespace(follow_path=lambda goal: None, hold=lambda: None),
    )
    pursuit = PursuitView(world)
    context = SkillContext(
        subgoal="test",
        params=params,
        memory=memory or {},
        cancelled=threading.Event(),
        log=lambda text: None,
    )
    with (
        patch("aha_tasks.follow.Base.of", return_value=world),
        patch("aha_tasks.follow.Pursuit", return_value=pursuit),
        patch("aha_tasks.follow.Interaction.of", return_value=world),
        patch("time.sleep", side_effect=world.sleep),
    ):
        return skill_class(runtime).execute(context)


FOUND = {"person.track": 1, "person.position": [0.0, 1.0, 1.0], "person.frame": "map"}


class FollowPersonTests(unittest.TestCase):
    def test_follows_to_the_location_and_stops_next_to_the_person(self):
        walker = Route(1, [(0.0, 1.0), (0.0, 5.5)])
        world = World([0.0, 0.0, math.pi / 2], [walker])
        result = run(
            FollowPerson,
            world,
            {"person": "Charlie", "location": "kitchen", "stop_time": 3.0},
            FOUND,
        )
        self.assertTrue(result.success, result.message)
        x, y, _ = result.outputs["robot.location"]
        gap = math.dist((x, y), (0.0, 5.5))
        self.assertGreater(gap, 0.6)
        self.assertLess(gap, 1.0 + 0.8)
        self.assertEqual(result.outputs["person.track"], 1)
        self.assertEqual(result.outputs["person.position"][:2], [0.0, 5.5])
        self.assertIn("Charlie, I will follow you to the kitchen", world.said[0])
        self.assertEqual(world.followed[0], 1)  # the head is on the person
        # Each goal faces the person from `distance` away.
        self.assertTrue(world.goals)

    def test_turns_to_a_person_who_walks_past_and_follows_them(self):
        # Starts in front, passes 1 m to the right and walks away behind.
        walker = Route(1, [(0.0, 1.2), (1.0, 0.5), (1.0, -1.0), (1.0, -4.0)])
        world = World([0.0, 0.0, math.pi / 2], [walker])
        result = run(FollowPerson, world, {"person": "them", "stop_time": 3.0}, FOUND)
        self.assertTrue(result.success, result.message)
        self.assertLess(world.robot[1], -1.5)
        self.assertLess(math.dist(world.robot[:2], (1.0, -4.0)), 1.8)
        # Kept in view throughout: no search (it would look ahead of them).
        self.assertEqual(world.looked, [(0.0, 1.0)])

    def test_a_stop_outside_the_location_does_not_end_following(self):
        walker = Pausing(1, [(0.0, 1.0), (0.0, 2.5)], [(0.0, 2.5), (0.0, 5.5)], 12.0)
        world = World([0.0, 0.0, math.pi / 2], [walker])
        result = run(
            FollowPerson, world, {"person": "them", "location": "kitchen"}, FOUND
        )
        self.assertTrue(result.success, result.message)
        self.assertGreater(world.robot[1], 3.5)

    def test_without_a_location_a_stop_with_the_robot_next_to_them_ends(self):
        walker = Pausing(1, [(0.0, 1.0), (0.0, 2.5)], [(0.0, 2.5), (0.0, 5.5)], 12.0)
        world = World([0.0, 0.0, math.pi / 2], [walker])
        result = run(FollowPerson, world, {"person": "them"}, FOUND)
        self.assertTrue(result.success, result.message)
        self.assertLess(world.robot[1], 2.5)
        self.assertLess(world.t, 1.0 + 1.5 / 0.4 + 12.0)

    def test_hands_over_a_lost_track_and_ignores_a_bystander(self):
        walker = Route(
            1, [(0.0, 1.0), (0.0, 5.5)], hidden=[(5.0, 6.0)], new_id_after=5.5
        )
        bystander = Standing(2, (0.8, 3.0))
        world = World([0.0, 0.0, math.pi / 2], [walker, bystander])
        result = run(
            FollowPerson, world, {"person": "them", "location": "kitchen"}, FOUND
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["person.track"], 101)
        self.assertNotIn(2, world.followed)

    def test_a_person_who_never_walks_is_not_followed_forever(self):
        world = World([0.0, 0.0, math.pi / 2], [Standing(1, (0.0, 1.0))])
        result = run(FollowPerson, world, {"person": "them", "timeout": 20.0}, FOUND)
        self.assertFalse(result.success)
        self.assertIn("Timed out", result.message)

    def test_searches_and_gives_up_on_a_vanished_person(self):
        walker = Route(1, [(0.0, 1.0), (0.0, 5.5)], hidden=[(4.0, 999.0)])
        world = World([0.0, 0.0, math.pi / 2], [walker])
        result = run(
            FollowPerson, world, {"person": "them", "lost_timeout": 15.0}, FOUND
        )
        self.assertFalse(result.success)
        self.assertIn("Lost", result.message)
        self.assertTrue(any(isinstance(look, float) for look in world.looked))

    def test_nobody_in_view_fails_at_the_start(self):
        world = World([0.0, 0.0, math.pi / 2], [Standing(1, (0.0, -2.0))])
        result = run(FollowPerson, world, {"person": "them"})
        self.assertFalse(result.success)
        self.assertIn("No them in view", result.message)

    def test_validation(self):
        spec = SimpleNamespace(locations=LOCATIONS)
        FollowPerson.validate({"person": "them"}, spec)
        FollowPerson.validate({"person": "Charlie", "location": "kitchen"}, spec)
        for params in (
            {},
            {"person": ""},
            {"person": "them", "location": "attic"},
            {"person": "them", "distance": 0},
            {"person": "them", "speed": 1.0},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                FollowPerson.validate(params, spec)
        self.assertEqual(
            FollowPerson(None).expected({"person": "a", "location": "kitchen"}),
            {"robot.location": "kitchen"},
        )
        self.assertEqual(FollowPerson(None).expected({"person": "a"}), {})


class Pausing(Route):
    """Walks `first`, stands `pause` s, then walks `second`."""

    def __init__(self, id, first, second, pause, speed=0.4, start=1.0):
        super().__init__(id, first, speed, start)
        self.second = Route(
            id, second, speed, start + self._length(first) / speed + pause
        )

    @staticmethod
    def _length(points):
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    def position(self, t, robot):
        if t < self.second.start:
            return super().position(t, robot)
        return self.second.position(t, robot)


class GuidePersonTests(unittest.TestCase):
    def test_guides_a_follower_and_checks_on_the_way(self):
        world = World([0.0, 0.0, math.pi / 2], [Trailing(1, (0.0, 1.0))])
        result = run(
            GuidePerson,
            world,
            {"person": "Alex", "location": "office", "check_every": 3.0},
            FOUND,
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("checks on the way: 1", result.message)
        # The follower right behind is out of the head's reach: the base turned
        # at the check and on arrival.
        self.assertEqual(len(world.turns), 2)
        self.assertEqual(
            world.said[0], "Alex, please follow me. I will take you to the office."
        )
        self.assertEqual(world.said[-1], "Alex, we have arrived at the office.")
        self.assertEqual(result.outputs["person.track"], 1)
        follower_xy = result.outputs["person.position"][:2]
        self.assertLess(math.dist(follower_xy, (6.0, 0.0)), 2.5)

    def test_calls_a_person_who_stopped_and_continues_when_they_come(self):
        world = World(
            [0.0, 0.0, math.pi / 2], [Trailing(1, (0.0, 1.0), stop_at=(5.0, 40.0))]
        )
        result = run(
            GuidePerson,
            world,
            {"person": "them", "location": "office", "check_every": 3.0, "wait": 20.0},
            FOUND,
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("I am here. Please follow me.", world.said)

    def test_fails_when_nobody_follows(self):
        world = World([0.0, 0.0, math.pi / 2], [Standing(1, (0.0, 1.0))])
        result = run(
            GuidePerson,
            world,
            {"person": "them", "location": "office", "wait": 3.0, "calls": 2},
            FOUND,
        )
        self.assertFalse(result.success)
        self.assertIn("did not follow", result.message)
        self.assertEqual(world.said.count("I am here. Please follow me."), 2)

    def test_validation(self):
        spec = SimpleNamespace(locations=LOCATIONS)
        GuidePerson.validate({"person": "them", "location": "office"}, spec)
        for params in (
            {"person": "them"},
            {"person": "them", "location": "office", "calls": -1},
            {"person": "them", "location": "office", "calls": 1.5},
            {"person": "them", "location": "office", "gap": -1.0},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                GuidePerson.validate(params, spec)
        self.assertEqual(
            GuidePerson(None).expected({"person": "a", "location": "office"}),
            {"robot.location": "office"},
        )


if __name__ == "__main__":
    unittest.main()
