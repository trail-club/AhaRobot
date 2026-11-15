"""Projection and drawing of perceived objects for the screen (no ROS)."""

import io
import math
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.object_view import (  # noqa: E402
    EDGE_COLOR,
    FRAME_COLOR,
    box_corners,
    draw_object,
    encode_jpeg,
    image_array,
    image_box,
    object_sentence,
    place_words,
    project,
    rotation_matrix,
)

K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
WIDTH, HEIGHT = 640, 480
IDENTITY = (0.0, 0.0, 0.0, 1.0)
# From a frame with x forward, y left, z up into the optical frame (z forward,
# x right, y down): the camera at the origin looks along +x.
FORWARD = (0.5, -0.5, 0.5, 0.5)


class RotationTests(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_allclose(rotation_matrix(*IDENTITY), np.eye(3))

    def test_forward_camera(self):
        expected = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        np.testing.assert_allclose(rotation_matrix(*FORWARD), expected, atol=1e-12)

    def test_unnormalized_quaternion(self):
        np.testing.assert_allclose(rotation_matrix(0, 0, 0, 2.0), np.eye(3))

    def test_zero_quaternion(self):
        with self.assertRaises(ValueError):
            rotation_matrix(0, 0, 0, 0)


class ProjectionTests(unittest.TestCase):
    def test_corners(self):
        corners = box_corners([1.0, 2.0, 3.0], [0.2, 0.4, 0.6], yaw=0.0)
        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_allclose(corners.min(axis=0), [0.9, 1.8, 2.7])
        np.testing.assert_allclose(corners.max(axis=0), [1.1, 2.2, 3.3])

    def test_corners_turned_by_yaw(self):
        corners = box_corners([0.0, 0.0, 0.0], [0.2, 0.04, 0.1], yaw=math.pi / 2)
        np.testing.assert_allclose(corners.max(axis=0), [0.02, 0.1, 0.05], atol=1e-12)

    def test_corners_without_yaw_hold_any_yaw(self):
        size = [0.2, 0.04, 0.1]
        free = box_corners([0.0, 0.0, 0.0], size).max(axis=0)
        np.testing.assert_allclose(free, [math.hypot(0.1, 0.02)] * 2 + [0.05])
        for yaw in np.linspace(0.0, math.pi, 13):
            turned = box_corners([0.0, 0.0, 0.0], size, yaw=float(yaw)).max(axis=0)
            self.assertTrue((turned <= free + 1e-12).all(), yaw)

    def test_project_drops_points_behind(self):
        pixels = project([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], K)
        np.testing.assert_allclose(pixels, [[320.0, 240.0]])
        self.assertIsNone(project([[0.0, 0.0, -1.0]], K))

    def test_box_ahead_is_centred(self):
        box, cut = image_box(
            [0, 0, 2.0],
            [0.2, 0.2, 0.2],
            IDENTITY,
            [0, 0, 0],
            K,
            WIDTH,
            HEIGHT,
            yaw=0.0,
            pad=0,
        )
        # The nearest face (z = 1.9 m) is the widest: 500 * 0.1 / 1.9 = 26.3 px.
        self.assertEqual(box, (294, 214, 346, 266))
        self.assertFalse(cut)

    def test_frame_axes(self):
        """An object on the robot's left appears left of the centre, higher is up."""
        box, _ = image_box(
            [2.0, 0.5, 0.3],
            [0.1, 0.1, 0.1],
            FORWARD,
            [0, 0, 0],
            K,
            WIDTH,
            HEIGHT,
            yaw=0.0,
        )
        left, top, right, bottom = box
        self.assertLess(right, 320)
        self.assertLess(bottom, 240)

    def test_translation(self):
        # The camera 1 m behind the object's frame origin: same as the object 1 m further.
        near, _ = image_box(
            [0, 0, 1.0], [0.2] * 3, IDENTITY, [0, 0, 1.0], K, WIDTH, HEIGHT, yaw=0.0
        )
        far, _ = image_box(
            [0, 0, 2.0], [0.2] * 3, IDENTITY, [0, 0, 0], K, WIDTH, HEIGHT, yaw=0.0
        )
        self.assertEqual(near, far)

    def test_partly_out_is_clipped(self):
        box, cut = image_box(
            [0.6, 0, 1.0], [0.2] * 3, IDENTITY, [0, 0, 0], K, WIDTH, HEIGHT, yaw=0.0
        )
        self.assertTrue(cut)
        self.assertEqual(box[2], WIDTH)
        self.assertLess(box[0], WIDTH)

    def test_out_of_view(self):
        self.assertEqual(
            image_box(
                [0, 0, -2.0], [0.2] * 3, IDENTITY, [0, 0, 0], K, WIDTH, HEIGHT, yaw=0.0
            ),
            (None, None),
        )
        self.assertEqual(
            image_box(
                [3.0, 0, 1.0], [0.2] * 3, IDENTITY, [0, 0, 0], K, WIDTH, HEIGHT, yaw=0.0
            ),
            (None, None),
        )


class ImageArrayTests(unittest.TestCase):
    def test_rgb_with_row_padding(self):
        width, height, step = 2, 2, 8  # 6 bytes of pixels + 2 of padding per row
        data = bytes([1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0])
        rgb = image_array(data, width, height, step, "rgb8")
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb[1, 1].tolist(), [10, 11, 12])

    def test_bgra_is_reordered(self):
        rgb = image_array(bytes([3, 2, 1, 255]), 1, 1, 4, "bgra8")
        self.assertEqual(rgb[0, 0].tolist(), [1, 2, 3])

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            image_array(bytes(4), 2, 2, 2, "mono8")
        with self.assertRaises(ValueError):
            image_array(bytes(5), 1, 2, 3, "rgb8")


class DrawingTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)

    def test_frame_outside_the_box_and_rest_kept(self):
        box = (200, 200, 300, 300)
        image = np.asarray(draw_object(self.rgb, box, "mug"))
        self.assertEqual(image.shape, self.rgb.shape)
        # The object inside the box is not covered.
        self.assertTrue((image[205:295, 205:295] == 128).all())
        # Black edges on both sides of the green frame (4 px at 480 px height).
        self.assertEqual(tuple(image[250, 200]), EDGE_COLOR)
        self.assertEqual(tuple(image[250, 197]), FRAME_COLOR)
        self.assertEqual(tuple(image[250, 195]), EDGE_COLOR)
        # The surroundings stay visible.
        self.assertTrue((image[400:, :] == 128).all())
        self.assertTrue((image[:, 400:] == 128).all())

    def test_label_tag_above_the_box(self):
        image = np.asarray(draw_object(self.rgb, (200, 200, 300, 300), "mug"))
        above = image[150:195, 190:320]
        green = (above == FRAME_COLOR).all(axis=2)
        dark = (above < 60).all(axis=2)
        self.assertGreater(green.sum(), 200)
        self.assertGreater(dark.sum(), 20)  # the text

    def test_label_tag_below_a_box_at_the_top(self):
        image = np.asarray(draw_object(self.rgb, (200, 0, 300, 100), "plate"))
        below = image[105:160, 190:320]
        self.assertGreater((below == FRAME_COLOR).all(axis=2).sum(), 200)

    def test_tag_stays_in_the_image(self):
        image = draw_object(self.rgb, (600, 0, WIDTH, HEIGHT), "a very long label")
        self.assertEqual(image.size, (WIDTH, HEIGHT))

    def test_jpeg(self):
        data = encode_jpeg(draw_object(self.rgb, (10, 10, 50, 50), "fork"))
        self.assertEqual(data[:2], b"\xff\xd8")
        self.assertEqual(Image.open(io.BytesIO(data)).size, (WIDTH, HEIGHT))


class SentenceTests(unittest.TestCase):
    def test_place(self):
        self.assertEqual(place_words("dining_table"), "on the dining table")
        self.assertEqual(place_words("floor"), "on the floor")
        self.assertEqual(place_words(""), "")

    def test_sentences(self):
        self.assertEqual(
            object_sentence("mug", "dining_table"),
            "I found the mug on the dining table.",
        )
        self.assertEqual(
            object_sentence("fork", "dining_table", shown=False),
            "I recognized the fork on the dining table.",
        )
        self.assertEqual(
            object_sentence("cookies", "floor", trash=True),
            "I found the cookies on the floor. That goes in the trash.",
        )
        self.assertEqual(object_sentence("milk", ""), "I found the milk.")


if __name__ == "__main__":
    unittest.main()
