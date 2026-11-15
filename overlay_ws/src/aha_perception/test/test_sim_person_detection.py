"""Ground-truth people for debug runs (aha_perception.sim_person_detection)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_perception.sim_person_detection import (  # noqa: E402
    Sighting,
    TruthTracker,
    body_bbox,
    is_person,
    person_visible,
)

W, H = 640, 480
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
HEIGHT, RADIUS, TOLERANCE = 1.75, 0.25, 0.1
CAMERA_Z = 1.3  # m above the floor


def camera_from_model(distance, lateral=0.0):
    """Optical frame (z forward, y down) to a person standing `distance` ahead."""
    T = np.eye(4)
    # Model z (up) is optical -y; the model stands on the floor below the camera.
    T[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    T[:3, 3] = [lateral, CAMERA_Z, distance]
    return T


def depth_image(value):
    return np.full((H, W), value, dtype=np.float32)


def sighting(model, x=1.0):
    return Sighting(
        model, np.array([x, 0.0, 0.9]), np.array([x, 0.0, 1.6]), [0, 0, 1, 1]
    )


class VisibilityTests(unittest.TestCase):
    def test_person_in_front_of_a_far_wall_is_seen(self):
        self.assertTrue(
            person_visible(
                depth_image(6.0), K, camera_from_model(3.0), HEIGHT, RADIUS, TOLERANCE
            )
        )

    def test_the_person_surface_is_in_front_of_the_axis(self):
        depth = depth_image(3.0 - RADIUS)  # the depth camera sees the body's front
        self.assertTrue(
            person_visible(depth, K, camera_from_model(3.0), HEIGHT, RADIUS, TOLERANCE)
        )

    def test_a_low_wall_hides_the_legs_but_not_the_head(self):
        depth = depth_image(6.0)
        depth[260:, :] = 2.0  # rows below the horizon: a wall 2 m away
        self.assertTrue(
            person_visible(depth, K, camera_from_model(3.0), HEIGHT, RADIUS, TOLERANCE)
        )

    def test_hidden_or_out_of_view_people_are_not_seen(self):
        cases = {
            "behind a closer wall": (depth_image(1.5), camera_from_model(3.0)),
            "behind the camera": (depth_image(6.0), camera_from_model(-3.0)),
            "far to the side": (depth_image(6.0), camera_from_model(3.0, lateral=8.0)),
            "under the floor": (
                depth_image(2.0),
                camera_from_model(3.0) @ shifted(-5.0),
            ),
        }
        for name, (depth, pose) in cases.items():
            with self.subTest(name):
                self.assertFalse(
                    person_visible(depth, K, pose, HEIGHT, RADIUS, TOLERANCE)
                )

    def test_bbox_covers_the_body_and_stays_in_the_image(self):
        x0, y0, x1, y1 = body_bbox(K, camera_from_model(3.0), HEIGHT, RADIUS, (H, W))
        self.assertLess(x0, 320)
        self.assertGreater(x1, 320)
        self.assertEqual(y1, H - 1)  # the feet are below the view at 3 m
        self.assertLess(y0, 240)
        self.assertEqual(
            body_bbox(K, camera_from_model(-3.0), HEIGHT, RADIUS, (H, W)), [0, 0, 0, 0]
        )

    def test_models_by_pattern(self):
        self.assertTrue(is_person("guest1", ["guest*"]))
        self.assertFalse(is_person("arena_door", ["guest*", "person_*"]))


def shifted(z):
    T = np.eye(4)
    T[2, 3] = z
    return T


class TrackerTests(unittest.TestCase):
    def test_ids_are_kept_while_seen_and_renewed_after_forgetting(self):
        tracker = TruthTracker(forget_after=3.0)
        first = tracker.update([sighting("guest1")], 10.0)
        self.assertEqual(tracker.update([], 12.0), set())
        self.assertEqual(tracker.update([sighting("guest1")], 12.5), first)
        tracker.update([], 15.6)  # 3.1 s unseen: forgotten
        self.assertNotIn("guest1", tracker.tracks)
        again = tracker.update([sighting("guest1")], 16.0)
        self.assertNotEqual(again, first)

    def test_a_remembered_name_follows_the_model_to_its_next_track(self):
        tracker = TruthTracker(forget_after=3.0)
        (track_id,) = tracker.update([sighting("guest1")], 1.0)
        self.assertEqual(tracker.remember(track_id, "Charlie"), "")
        tracker.update([], 5.0)
        tracker.update([sighting("guest1"), sighting("guest2", 3.0)], 6.0)
        names = {t.model: t.name for t in tracker.tracks.values()}
        self.assertEqual(names, {"guest1": "Charlie", "guest2": ""})

    def test_a_name_moves_to_the_last_remembered_model(self):
        tracker = TruthTracker()
        ids = tracker.update([sighting("guest1"), sighting("guest2", 3.0)], 1.0)
        one, two = sorted(ids)
        tracker.remember(one, "Emma")
        tracker.remember(two, "Emma")
        names = {t.model: t.name for t in tracker.tracks.values()}
        self.assertEqual(names, {"guest1": "", "guest2": "Emma"})
        self.assertEqual(tracker.names, {"guest2": "Emma"})

    def test_remember_needs_a_tracked_person_and_a_name(self):
        tracker = TruthTracker()
        (track_id,) = tracker.update([sighting("guest1")], 1.0)
        self.assertIn("not tracked", tracker.remember(track_id + 1, "Emma"))
        self.assertIn("empty", tracker.remember(track_id, ""))


if __name__ == "__main__":
    unittest.main()
