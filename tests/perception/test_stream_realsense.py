import base64
import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np


SCRIPT = pathlib.Path(__file__).parents[2] / "tools/perception/macos/stream_realsense.py"
SPEC = importlib.util.spec_from_file_location("stream_realsense", SCRIPT)
stream_realsense = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = stream_realsense
SPEC.loader.exec_module(stream_realsense)


class StreamRealsenseMessageTests(unittest.TestCase):
    def setUp(self):
        self.stamp = stream_realsense.Stamp(sec=12, nanosec=345)

    def test_depth_zero_is_nan_and_valid_samples_are_metres(self):
        raw = np.array([[0, 1000], [2500, 0]], dtype=np.uint16)

        depth = stream_realsense.depth_to_meters(raw, 0.001)

        self.assertTrue(np.isnan(depth[0, 0]))
        self.assertTrue(np.isnan(depth[1, 1]))
        np.testing.assert_allclose(depth[0, 1], 1.0)
        np.testing.assert_allclose(depth[1, 0], 2.5)

    def test_image_message_has_ros_shape_and_base64_data(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        message = stream_realsense.make_image_message(
            image,
            encoding="rgb8",
            stamp=self.stamp,
            frame_id=stream_realsense.COLOR_FRAME,
        )

        self.assertEqual(message["header"]["stamp"], {"sec": 12, "nanosec": 345})
        self.assertEqual(message["header"]["frame_id"], "camera_color_optical_frame")
        self.assertEqual(message["height"], 2)
        self.assertEqual(message["width"], 2)
        self.assertEqual(message["step"], 6)
        self.assertEqual(message["is_bigendian"], 0)
        self.assertEqual(base64.b64decode(message["data"]), image.tobytes())

    def test_camera_info_message_contains_color_intrinsics(self):
        intrinsics = SimpleNamespace(
            width=640,
            height=480,
            fx=600.0,
            fy=601.0,
            ppx=320.0,
            ppy=240.0,
            model="brown_conrady",
            coeffs=(0.1, -0.2, 0.003, 0.004, 0.0),
        )

        message = stream_realsense.make_camera_info_message(
            intrinsics,
            stamp=self.stamp,
            frame_id=stream_realsense.COLOR_FRAME,
        )

        self.assertEqual(message["header"]["frame_id"], "camera_color_optical_frame")
        self.assertEqual(message["width"], 640)
        self.assertEqual(message["height"], 480)
        self.assertEqual(message["distortion_model"], "plumb_bob")
        self.assertEqual(message["d"], [0.1, -0.2, 0.003, 0.004, 0.0])
        self.assertEqual(message["k"], [600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0])
        self.assertEqual(len(message["p"]), 12)

    def test_aligned_frame_bundle_uses_one_stamp_and_color_frame(self):
        intrinsics = SimpleNamespace(
            width=2, height=1, fx=1, fy=1, ppx=0, ppy=0,
            model="none", coeffs=(0, 0, 0, 0, 0),
        )
        color = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        depth = np.array([[1000, 0]], dtype=np.uint16)

        messages = stream_realsense.make_frame_bundle(
            color, depth, intrinsics, 0.001, self.stamp
        )

        self.assertEqual(len(messages), 3)
        for message in messages:
            self.assertEqual(message["header"]["stamp"], self.stamp.as_message())
            self.assertEqual(message["header"]["frame_id"], stream_realsense.COLOR_FRAME)
        self.assertEqual(messages[0]["encoding"], "rgb8")
        self.assertEqual(messages[2]["encoding"], "32FC1")
        decoded_depth = np.frombuffer(base64.b64decode(messages[2]["data"]), dtype=np.float32)
        np.testing.assert_allclose(decoded_depth[0], 1.0)
        self.assertTrue(np.isnan(decoded_depth[1]))

    def test_aligned_frame_bundle_rejects_mismatched_sizes(self):
        intrinsics = SimpleNamespace(
            width=2, height=1, fx=1, fy=1, ppx=0, ppy=0,
            model="none", coeffs=(0, 0, 0, 0, 0),
        )
        color = np.zeros((1, 2, 3), dtype=np.uint8)
        depth = np.zeros((2, 2), dtype=np.uint16)

        with self.assertRaisesRegex(stream_realsense.StreamError, "same size"):
            stream_realsense.make_frame_bundle(
                color, depth, intrinsics, 0.001, self.stamp
            )

    def test_rosbridge_advertisement_requests_sensor_qos(self):
        operation = stream_realsense.advertise_operation(
            stream_realsense.DEPTH_TOPIC, stream_realsense.ROS_IMAGE
        )

        self.assertEqual(operation["op"], "advertise")
        self.assertEqual(operation["topic"], "/camera/depth_registered/image_rect")
        self.assertEqual(operation["type"], "sensor_msgs/msg/Image")
        self.assertEqual(operation["qos"]["reliability"], "reliable")
        self.assertEqual(operation["qos"]["durability"], "volatile")
        self.assertEqual(operation["qos"]["depth"], 5)

    def test_mac_topic_contract_has_no_pointcloud_publisher(self):
        topics = {
            stream_realsense.COLOR_TOPIC,
            stream_realsense.CAMERA_INFO_TOPIC,
            stream_realsense.DEPTH_TOPIC,
        }

        self.assertEqual(
            topics,
            {
                "/camera/color/image_raw",
                "/camera/color/camera_info",
                "/camera/depth_registered/image_rect",
            },
        )
        self.assertFalse(any(topic.endswith("/points") for topic in topics))


if __name__ == "__main__":
    unittest.main()
