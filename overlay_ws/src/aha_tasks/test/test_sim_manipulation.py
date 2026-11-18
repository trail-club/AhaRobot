"""Geometry and models of the sim_manipulation stand-in (no ROS or Gazebo)."""

import math
import os
from pathlib import Path
import importlib.util
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.sim_manipulation import (  # noqa: E402
    AIM_MIN,
    PLACE_LIFT,
    aim_point,
    find_model,
    held_pose,
    horizontal_distance,
    pose_matrix,
    resting_centre,
    static_copy,
    yaw_of,
)
from aha_tasks import sim_manipulation_node  # noqa: E402
from aha_tasks.sim_manipulation_node import (  # noqa: E402
    Failed,
    Held,
    Stand,
    failed_status,
)
from aha_tasks.sim_objects import PrintedModel, printed_sdf  # noqa: E402
from aha_perception.object_list import load_object_list  # noqa: E402

OBJECT_LIST = load_object_list(
    Path(__file__).resolve().parents[2]
    / "aha_perception"
    / "config"
    / "objects"
    / "robocup_objects.yaml"
)


class ModelTests(unittest.TestCase):
    POSES = {
        "cola": pose_matrix(-1.8, -1.55, 0.86),
        "cola_2": pose_matrix(3.14, 3.42, 0.97),
        "apple": pose_matrix(-1.75, -1.55, 0.84),
        "aha_robot": pose_matrix(-1.8, -1.5, 0.0),
        "dining_table": pose_matrix(-2.0, -2.25, 0.0),
    }

    def test_the_model_of_the_label_nearest_the_target(self):
        self.assertEqual(
            find_model(self.POSES, OBJECT_LIST, "cola", (-1.78, -1.55, 0.86))[0],
            "cola",
        )
        self.assertEqual(
            find_model(self.POSES, OBJECT_LIST, "cola", (3.1, 3.4, 0.97))[0], "cola_2"
        )
        # The apple is nearer, but the cola has the label.
        self.assertEqual(
            find_model(self.POSES, OBJECT_LIST, "cola", (-1.76, -1.55, 0.85))[0],
            "cola",
        )

    def test_a_mislabelled_target_takes_the_object_there(self):
        name, _ = find_model(self.POSES, OBJECT_LIST, "orange", (-1.75, -1.55, 0.84))
        self.assertEqual(name, "apple")

    def test_nothing_near_or_not_an_object(self):
        self.assertIsNone(find_model(self.POSES, OBJECT_LIST, "cola", (0.0, 0.0, 0.9)))
        self.assertIsNone(
            find_model(self.POSES, OBJECT_LIST, "cola", (-2.0, -2.25, 0.0))
        )

    def test_the_static_copy_has_no_physics(self):
        model = PrintedModel("cola", "cylinder", (0.066, 0.066, 0.12), {"mass": 0.3})
        sdf = printed_sdf(model, "cola", (1.0, 2.0, 0.8, 0.5), Path("/tmp/cola.obj"))
        copy = ET.fromstring(static_copy(sdf).replace('<?xml version="1.0"?>', ""))
        model = copy.find("model")
        self.assertEqual(model.get("name"), "cola")
        self.assertEqual(model.findtext("static"), "true")
        self.assertEqual(len(model.findall("static")), 1)
        link = model.find("link")
        self.assertIsNone(link.find("collision"))
        self.assertIsNone(link.find("inertial"))
        self.assertIsNotNone(link.find("visual"))
        self.assertEqual(model.findtext("pose").split()[:2], ["1.000000", "2.000000"])


class PoseTests(unittest.TestCase):
    def test_the_held_object_is_upright_at_the_tcp(self):
        tcp = pose_matrix(1.0, 2.0, 0.9, 0.3)
        pitch = np.eye(4)
        pitch[:3, :3] = [[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]]  # x tilted down
        x, y, z, yaw = held_pose(tcp @ pitch, 0.2)
        self.assertEqual((x, y, z), (1.0, 2.0, 0.9))
        self.assertAlmostEqual(yaw, 0.5)  # the heading of the TCP + the offset

    def test_placed_objects_stand_on_the_spot_or_drop_from_it(self):
        self.assertEqual(
            resting_centre((1.0, 2.0, 0.4), 0.12, False), (1.0, 2.0, 0.46 + PLACE_LIFT)
        )
        self.assertEqual(
            resting_centre((1.0, 2.0, 0.375), 0.12, True), (1.0, 2.0, 0.375)
        )

    def test_near_targets_are_pointed_past(self):
        shoulder = (0.0, 0.0)
        far = (1.0, 0.0, 0.8)
        self.assertEqual(aim_point(far, shoulder), far)
        near = aim_point((0.3, 0.4, 0.8), shoulder)
        self.assertAlmostEqual(math.hypot(*near[:2]), AIM_MIN)
        self.assertAlmostEqual(near[1] / near[0], 0.4 / 0.3)
        self.assertEqual(near[2], 0.8)

    def test_frames(self):
        base = pose_matrix(1.0, 1.0, 0.0, math.pi / 2)
        self.assertAlmostEqual(yaw_of(base), math.pi / 2)
        self.assertAlmostEqual(horizontal_distance(base, (1.0, 2.0, 0.8)), 1.0)


class FakeStand(Stand):
    """Stand without Gazebo, TF or the arm; `carry` and the removal of a model
    in Gazebo may fail."""

    def __init__(self, carry_fails=False, gazebo_fails=False, tf_fails=False):
        self.held, self.object_list, self.carry_fails = None, OBJECT_LIST, carry_fails
        self.tf_fails, self.backs = tf_fails, 0
        self.gazebo_fails, self.created = gazebo_fails, []
        self.lock, self.placed, self.arm = threading.Lock(), None, None
        self.furniture = SimpleNamespace(get=lambda name: None)
        self.log = SimpleNamespace(info=lambda text: None, warning=lambda text: None)

    def set_pose(self, name, x, y, z, yaw):
        return True

    def remove(self, name):
        if self.gazebo_fails:
            raise Failed(f"{name} stays in Gazebo after its removal")

    def create(self, sdf, name):
        self.created.append(name)

    def sdf(self, name, label, x, y, z, yaw):
        return name

    def param(self, name):
        return {"reach": 1.0}[name]

    def world_point(self, frame, xyz):
        return xyz

    def poses(self):
        return {"cola": pose_matrix(0.6, -0.2, 0.86)}, pose_matrix(0.0, 0.0, 0.0)

    def base_point(self, xyz):
        return xyz

    def point(self, base_xyz, height, cancelled, what):
        return True

    def attach(self, name, label, world_object):
        self.held = Held(name, label, 0.0, (0.066, 0.066, 0.12))

    def move(self, lift, joints, cancelled, what):
        if self.carry_fails:
            raise Failed(f"Arm ({what}): Cancelled")

    def ros_pose(self, world_xyz, yaw=0.0):
        if self.tf_fails:
            raise Failed("No TF base_footprint <- map")
        return world_xyz

    def back(self, cancelled):
        self.backs += 1
        return True


class StandPickTests(unittest.TestCase):
    GOAL = SimpleNamespace(
        arm="",
        target=SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            class_name="cola",
            pose=SimpleNamespace(
                pose=SimpleNamespace(position=SimpleNamespace(x=0.6, y=-0.2, z=0.86))
            ),
        ),
    )

    def pick(self, stand):
        handle = SimpleNamespace(is_cancel_requested=False)
        return stand.pick(self.GOAL, handle, lambda *args: None)

    def test_the_object_in_the_hand_is_a_picked_object(self):
        name, back, _, note = self.pick(FakeStand(carry_fails=False))
        self.assertEqual((name, back, note), ("cola", True, ""))

    def test_a_failed_carry_still_picks(self):
        # Failing here would keep the object in the hand while the task
        # thinks the hand is empty, and every later pick is refused.
        stand = FakeStand(carry_fails=True)
        name, back, _, note = self.pick(stand)
        self.assertEqual((name, back), ("cola", False))  # `at_home` false
        self.assertIn("not at carry", note)
        self.assertEqual(stand.held.name, "cola")
        with self.assertRaises(Failed):
            self.pick(stand)  # already holding


class StandUnreachableTests(unittest.TestCase):
    def test_an_arm_that_cannot_point_fails_before_any_motion(self):
        stand = FakeStand(carry_fails=False)
        stand.point = lambda *args: False  # point_at: no pose for the point
        with self.assertRaises(Failed) as caught:
            StandPickTests().pick(stand)
        self.assertIsNone(caught.exception.at_home)  # the arm did not move
        self.assertIsNone(stand.held)
        stand.held = Held("cola", "cola", 0.0, (0.066, 0.066, 0.12))
        with self.assertRaises(Failed) as caught:
            StandReleaseTests().place(stand)
        self.assertEqual(caught.exception.status, "STATUS_UNREACHABLE")
        self.assertEqual(stand.held.name, "cola")


class GazeboModelTests(unittest.TestCase):
    """remove / create judged by pose/info: a request whose reply is lost
    may still be carried out."""

    def stand(self, models):
        stand = Stand.__new__(Stand)
        stand.gazebo = SimpleNamespace(latest=lambda: ({m: None for m in models}, 0.1))
        stand.request = lambda service, message, reply: False  # no reply in time
        return stand

    def test_a_removal_without_a_reply_that_happened(self):
        if importlib.util.find_spec("gz") is None:
            self.skipTest("no gz-transport (outside the container)")
        self.stand([]).remove("cola")  # gone: no Failed
        with mock.patch.object(sim_manipulation_node, "REMOVE_TIMEOUT", 0.2):
            with self.assertRaises(Failed):
                self.stand(["cola"]).remove("cola")

    def test_presence(self):
        self.assertFalse(self.stand([]).present("cola", False, 0.0))
        self.assertTrue(self.stand(["cola"]).present("cola", True, 0.0))
        self.assertTrue(self.stand(["cola"]).present("cola", False, 0.0))


class StatusTests(unittest.TestCase):
    def test_a_failure_after_the_arm_moved_is_not_a_before_motion_status(self):
        self.assertEqual(failed_status(Failed("only right")), "STATUS_ERROR")
        self.assertEqual(
            failed_status(Failed("Arm: stuck", at_home=False)),
            "STATUS_EXECUTION_FAILED",
        )
        self.assertEqual(
            failed_status(Failed("Arm: stuck", at_home=True)),
            "STATUS_EXECUTION_FAILED",
        )
        self.assertEqual(
            failed_status(Failed("Cancelled", "STATUS_CANCELED", False)),
            "STATUS_CANCELED",
        )

    def test_a_handover_motion_that_fails_after_moving(self):
        stand = FakeStand()
        stand.held = Held("cola", "cola", 0.0, (0.066, 0.066, 0.12))

        def point(base_xyz, height, cancelled, what):
            raise Failed("Arm (person): stuck", at_home=False)

        stand.point = point
        stand.back = lambda cancelled: False  # and not at home either
        goal = SimpleNamespace(
            arm="",
            target=StandReleaseTests.point(None, 1.0, 0.0, 0.9),
            release="take",
            timeout=1.0,
        )
        handle = SimpleNamespace(is_cancel_requested=False)
        with self.assertRaises(Failed) as caught:
            stand.hand_over(goal, handle, lambda *a: None)
        self.assertEqual(caught.exception.at_home, False)
        self.assertEqual(failed_status(caught.exception), "STATUS_EXECUTION_FAILED")


class StandAttachTests(unittest.TestCase):
    def test_a_failed_static_copy_puts_the_object_back(self):
        class NoCopy(FakeStand):
            def sdf(self, name, label, x, y, z, yaw):
                return f'<sdf version="1.9"><model name="{name}"><link name="l"/></model></sdf>'

            def world_tcp(self):
                return pose_matrix(0.6, -0.2, 1.0)

            def create(self, sdf, name):
                if "<static>" in sdf:
                    raise Failed(f"No reply to the creation of {name}")
                self.created.append(name)

        stand = NoCopy()
        with self.assertRaises(Failed):
            Stand.attach(stand, "cola", "cola", pose_matrix(0.6, -0.2, 0.86))
        self.assertEqual(stand.created, ["cola"])  # the object as it was
        self.assertIsNone(stand.held)

    def test_no_tcp_pose_leaves_the_object_as_it_is(self):
        class NoTcp(FakeStand):
            removed = ()

            def sdf(self, name, label, x, y, z, yaw):
                return f'<sdf version="1.9"><model name="{name}"><link name="l"/></model></sdf>'

            def remove(self, name):
                self.removed += (name,)

            def world_tcp(self):
                raise Failed("No TF base_footprint <- link_ree")

        stand = NoTcp()
        with self.assertRaises(Failed):
            Stand.attach(stand, "cola", "cola", pose_matrix(0.6, -0.2, 0.86))
        self.assertEqual((stand.removed, stand.created), ((), []))
        self.assertIsNone(stand.held)

    def test_a_failed_attach_brings_the_arm_back(self):
        stand = FakeStand()

        def attach(name, label, world_object):
            raise Failed("Could not create cola")

        stand.attach = attach
        handle = SimpleNamespace(is_cancel_requested=False)
        with self.assertRaises(Failed) as caught:
            stand.pick(StandPickTests.GOAL, handle, lambda *a: None)
        self.assertEqual((stand.backs, caught.exception.at_home), (1, True))
        self.assertIsNone(stand.held)
        # And not at home either: the result says so (`at_home` false).
        stand.back = lambda cancelled: False
        with self.assertRaises(Failed) as caught:
            stand.pick(StandPickTests.GOAL, handle, lambda *a: None)
        self.assertEqual(caught.exception.at_home, False)


class StandPoseTests(unittest.TestCase):
    def test_no_grasp_pose_still_picks(self):
        stand = FakeStand(tf_fails=True)
        handle = SimpleNamespace(is_cancel_requested=False)
        name, back, pose, note = stand.pick(
            StandPickTests.GOAL, handle, lambda *a: None
        )
        self.assertEqual((name, back, pose), ("cola", True, None))
        self.assertIn("no grasp_pose (No TF", note)
        self.assertEqual(stand.held.name, "cola")


class StandReachTests(unittest.TestCase):
    """A goal that moved the arm ends with it at a safe pose, but a cancel."""

    def failing(self, status):
        stand = FakeStand()

        def point(base_xyz, height, cancelled, what):
            raise Failed("Arm (spot): stuck", status, at_home=False)

        stand.point = point
        with self.assertRaises(Failed) as caught:
            stand.reach_out((0.6, 0.0, 0.8), 0.9, None, "spot")
        return stand, caught.exception

    def test_a_failed_motion_brings_the_arm_back(self):
        stand, error = self.failing(None)
        self.assertEqual((stand.backs, error.at_home), (1, True))

    def test_a_cancel_leaves_the_arm_where_it_is(self):
        stand, error = self.failing("STATUS_CANCELED")
        self.assertEqual((stand.backs, error.at_home), (0, False))


class StandMoveTests(unittest.TestCase):
    def test_an_arm_timeout_is_a_failure(self):
        def no_joint_states(joints):
            raise TimeoutError("no /joint_states")

        stand = FakeStand(carry_fails=False)
        stand.arm = SimpleNamespace(positions=no_joint_states)
        with self.assertRaises(Failed):
            Stand.move(stand, 0.5, (0.0,) * 5, None, "carry")

    def test_a_cancel_during_a_motion_is_a_cancel(self):
        stand = FakeStand()
        stand.arm = SimpleNamespace(
            positions=lambda joints: [0.0] * 6,
            move=lambda lift, joints, cancelled: (False, "Cancelled"),
        )
        for requested, status in ((True, "STATUS_CANCELED"), (False, None)):
            with self.subTest(requested=requested):
                cancelled = SimpleNamespace(is_set=lambda: requested)
                with self.assertRaises(Failed) as caught:
                    Stand.move(stand, 0.5, (0.0,) * 5, cancelled, "person")
                self.assertEqual(caught.exception.status, status)


class StandReleaseTests(unittest.TestCase):
    """Once the object left the hand, place and handover succeed (with a
    note on a Gazebo failure), so that the task's memory matches."""

    def holding(self, gazebo_fails):
        stand = FakeStand(gazebo_fails=gazebo_fails)
        stand.held = Held("cola", "cola", 0.0, (0.066, 0.066, 0.12))
        return stand

    def point(self, x, y, z):
        return SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            point=SimpleNamespace(x=x, y=y, z=z),
        )

    def place(self, stand):
        goal = SimpleNamespace(
            arm="", near=self.point(0.6, -0.2, 0.8), furniture="", drop=False
        )
        handle = SimpleNamespace(is_cancel_requested=False)
        return stand.place(goal, handle, lambda *args: None)

    def hand_over(self, stand):
        goal = SimpleNamespace(
            arm="", target=self.point(1.0, 0.0, 0.9), release="now", timeout=0.0
        )
        handle = SimpleNamespace(is_cancel_requested=False)
        return stand.hand_over(goal, handle, lambda *args: None)

    def test_place_and_handover(self):
        stand = self.holding(gazebo_fails=False)
        name, back, pose, note = self.place(stand)
        self.assertEqual(
            (name, back, note, stand.created), ("cola", True, "", ["cola"])
        )
        self.assertAlmostEqual(pose[2], 0.8 + PLACE_LIFT)  # the bottom on the spot
        stand = self.holding(gazebo_fails=False)
        self.assertEqual(self.hand_over(stand), ("cola", True, ""))
        self.assertIsNone(stand.held)

    def test_gazebo_failures_after_letting_go_are_notes(self):
        stand = self.holding(gazebo_fails=True)
        name, _, _, note = self.place(stand)
        self.assertIn("let go, but cola stays in Gazebo", note)
        self.assertIsNone(stand.held)
        stand = self.holding(gazebo_fails=True)
        name, _, note = self.hand_over(stand)
        self.assertIn("let go, but cola stays in Gazebo", note)
        self.assertIsNone(stand.held)
        with self.assertRaises(Failed):  # nothing held any more
            self.hand_over(stand)


if __name__ == "__main__":
    unittest.main()
