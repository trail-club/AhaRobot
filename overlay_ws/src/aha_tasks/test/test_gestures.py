"""OfferSeat against a fake arm, gaze and navigation (no ROS graph)."""

import math
import os
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_manipulation.pointing import lift_for_height, point_at  # noqa: E402
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.gestures import OfferSeat  # noqa: E402

START = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint_r1, joint_r2..joint_r6
SEAT = [1.75, 3.5, 0.5]


class FakeArm:
    """Robot at `pose` in map; records moves as (lift, arm)."""

    def __init__(self, pose=(1.75, 2.5, math.pi / 2), lift=True, ok=True):
        self.pose = list(pose)
        self.lift = lift
        self.ok = ok
        self.moves = []

    def to_base(self, frame, xyz):
        x, y, yaw = self.pose
        dx, dy = xyz[0] - x, xyz[1] - y
        return (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
            xyz[2],
        )

    def robot_pose(self):
        return list(self.pose)

    def positions(self, names):
        return list(START)

    def has_lift(self):
        return self.lift

    def move(self, lift, arm, cancelled):
        self.moves.append((lift, tuple(arm)))
        return (True, "") if self.ok or len(self.moves) > 1 else (False, "aborted")


class FakeGaze:
    def __init__(self):
        self.calls = []

    def look_at(self, frame, point):
        self.calls.append(("look_at", frame, (point.x, point.y, point.z)))

    def follow_person(self, people, track=None, name=""):
        self.calls.append(("follow", track, name))


class FakeNavigation:
    def __init__(self, arm):
        self.arm = arm
        self.goals = []

    def navigate(self, pose, cancelled=None):
        self.goals.append(pose)
        self.arm.pose = list(pose)
        return True, ""


def runtime(arm, navigation=None):
    clock = SimpleNamespace(ros_time_is_active=False)
    node = SimpleNamespace(get_clock=lambda: clock)
    return SimpleNamespace(
        node=node,
        spec=None,
        navigation=navigation,
        gaze=FakeGaze(),
        arm=arm,
        people=SimpleNamespace(follow=lambda **kw: runtime_follow.append(kw)),
    )


runtime_follow = []


def context(params=None, memory=None):
    return SkillContext(
        subgoal="guest1.seat",
        params={
            "guest": "guest1",
            "seat": "chair_left",
            "position": SEAT,
            "hold": 0.01,
            **(params or {}),
        },
        memory=dict(memory or {}),
        cancelled=threading.Event(),
        log=lambda text: None,
    )


class OfferSeatTests(unittest.TestCase):
    def setUp(self):
        runtime_follow.clear()

    def test_points_at_the_seat_and_returns(self):
        arm = FakeArm()
        rt = runtime(arm)
        result = OfferSeat(rt).execute(context(memory={"guest1.track": 4}))
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs, {"guest1.seat": "chair_left"})
        expected = point_at(arm.to_base("map", SEAT), lift_for_height(0.8))
        self.assertEqual(arm.moves[0], (expected.lift, expected.arm))
        self.assertEqual(len(arm.moves), 2)  # no correction: the base held still
        self.assertEqual(arm.moves[1], (START[0], tuple(START[1:])))
        self.assertEqual(rt.gaze.calls, [("look_at", "map", tuple(SEAT))])
        self.assertEqual(runtime_follow, [{"track": 4, "name": "guest1"}])

    def test_a_base_that_settles_meanwhile_is_corrected(self):
        arm = FakeArm()
        moves = arm.move

        def settle(lift, joints, cancelled):
            if not arm.moves:
                arm.pose[2] += 0.05  # turned 3 deg while the arm moved
            return moves(lift, joints, cancelled)

        arm.move = settle
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(arm.moves), 3)
        corrected = point_at(arm.to_base("map", SEAT), lift_for_height(0.8))
        self.assertEqual(arm.moves[1][1], corrected.arm)

    def test_without_lift_points_from_the_current_height(self):
        arm = FakeArm(lift=False)
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertIsNone(arm.moves[0][0])
        self.assertEqual(arm.moves[0][1], point_at(arm.to_base("map", SEAT), 0.0).arm)

    def test_holding_something_shows_the_seat_with_the_head_only(self):
        arm = FakeArm()
        rt = runtime(arm)
        result = OfferSeat(rt).execute(context(memory={"robot.holding": "bag"}))
        self.assertTrue(result.success, result.message)
        self.assertEqual(arm.moves, [])
        self.assertEqual(rt.gaze.calls[0][0], "look_at")

    def test_seat_behind_turns_the_base_first(self):
        arm = FakeArm(pose=(1.75, 2.5, -math.pi / 2))
        navigation = FakeNavigation(arm)
        result = OfferSeat(runtime(arm, navigation)).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(navigation.goals), 1)
        self.assertAlmostEqual(navigation.goals[0][2], math.pi / 2)
        self.assertEqual(result.outputs["robot.location"], [1.75, 2.5, 1.571])
        self.assertEqual(len(arm.moves), 2)

    def test_seat_behind_without_navigation_fails(self):
        arm = FakeArm(pose=(1.75, 2.5, -math.pi / 2))
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertFalse(result.success)
        self.assertEqual(arm.moves, [])

    def test_failed_motion_still_returns_the_arm(self):
        arm = FakeArm(ok=False)
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertFalse(result.success)
        self.assertIn("aborted", result.message)
        self.assertEqual(arm.moves[-1], (START[0], tuple(START[1:])))

    def test_an_error_while_pointing_still_returns_the_arm(self):
        arm = FakeArm()
        moves = arm.move

        def time_out(lift, joints, cancelled):
            result = moves(lift, joints, cancelled)
            if len(arm.moves) == 1:
                raise TimeoutError("the controller did not finish in time")
            return result

        arm.move = time_out
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertFalse(result.success)
        self.assertIn("TimeoutError", result.message)
        self.assertEqual(len(arm.moves), 2)
        self.assertEqual(arm.moves[-1], (START[0], tuple(START[1:])))

    def test_a_lost_transform_after_pointing_still_returns_the_arm(self):
        class TransformException(Exception):  # tf2_ros's is not a ValueError
            pass

        arm = FakeArm()
        to_base = arm.to_base

        def lose_tf(frame, xyz):
            if arm.moves:
                raise TransformException("map does not exist")
            return to_base(frame, xyz)

        arm.to_base = lose_tf
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertFalse(result.success)
        self.assertIn("TransformException", result.message)
        self.assertEqual(len(arm.moves), 2)
        self.assertEqual(arm.moves[-1], (START[0], tuple(START[1:])))

    def test_an_error_on_the_way_back_is_reported_with_the_failure(self):
        arm = FakeArm(ok=False)
        moves = arm.move

        def fail_back(lift, joints, cancelled):
            result = moves(lift, joints, cancelled)
            if len(arm.moves) > 1:
                raise TimeoutError("the clock stopped while the arm moved")
            return result

        arm.move = fail_back
        result = OfferSeat(runtime(arm)).execute(context())
        self.assertFalse(result.success)
        self.assertIn("aborted", result.message)
        self.assertIn("not back: TimeoutError", result.message)

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        OfferSeat.validate(
            {"guest": "g", "seat": "s", "position": [1, 2, 0.5], "height": 0.7}, spec
        )
        for params in (
            {"guest": "g", "seat": "s"},
            {"guest": "g", "seat": "s", "position": [1, 2]},
            {"guest": "g", "seat": "", "position": [1, 2, 0.5]},
            {"guest": "g", "seat": "s", "position": [1, 2, 0.5], "hold": 0},
            {"guest": "g", "seat": "s", "position": [1, 2, 0.5], "x": 1},
        ):
            with self.assertRaises(ValueError):
                OfferSeat.validate(params, spec)

    def test_expected_seat_for_mocks(self):
        params = {"guest": "guest2", "seat": "chair_right", "position": [0, 0, 0]}
        self.assertEqual(
            OfferSeat(None).expected(params), {"guest2.seat": "chair_right"}
        )


if __name__ == "__main__":
    unittest.main()
