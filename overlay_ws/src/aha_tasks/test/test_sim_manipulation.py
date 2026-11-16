"""Geometry and models of the sim_manipulation stand-in (no ROS or Gazebo)."""

import math
import os
from pathlib import Path
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
