"""End to end: object_perception.py against the fake worker (no DGX, no CUDA).

Runs the node from the source tree on ROS_DOMAIN_ID 94 (AHA_TEST_DOMAIN_ID to
change), so it does not see a running sim.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

from aha_msgs.msg import DetectedObjectArray
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster
import yaml

from aha_perception.plane_detection_io import quaternion_from_matrix

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(PKG, "scripts", "object_perception.py")
FAKE = os.path.join(PKG, "test", "fake_object_worker.py")
OBJECTS = os.path.join(PKG, "config", "objects", "robocup_objects.yaml")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "94"))
W, H, FX, CX, CY, DEPTH = 320, 240, 200.0, 160.0, 120.0, 0.5
CAMERA = "camera_color_optical_frame"
HEIGHT = 1.0  # camera above the map origin, looking along map x
# Optical axes (x right, y down, z forward) in map.
OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
HALF = 20  # px; each blob is 40 x 40 pixels (0.1 m) on a wall at DEPTH


def expected_centre(label, u, v):
    """Map centre of a blob centred on (u, v) straight ahead: the wall point
    plus half the label's smaller footprint side behind it (the hidden depth)."""
    with open(OBJECTS, encoding="utf-8") as f:
        size = next(
            o["size"] for o in yaml.safe_load(f)["objects"] if o["label"] == label
        )
    return np.array(
        [
            DEPTH + min(size[:2]) / 2,
            -(u - CX) * DEPTH / FX,
            HEIGHT - (v - CY) * DEPTH / FX,
        ]
    )


class ObjectPerceptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        params = os.path.join(cls.tmpdir.name, "params.yaml")
        with open(params, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "/**": {
                        "ros__parameters": {
                            "worker_command": f"{sys.executable} {FAKE}",
                            "objects_config": OBJECTS,
                            "target_frame": "map",
                            "prompts": ["can", "fruit"],
                            "max_rate": 20.0,
                            "startup_timeout": 30.0,
                        }
                    }
                },
                f,
            )
        env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [PKG, env.get("PYTHONPATH")]))
        cls.process = subprocess.Popen(
            [sys.executable, NODE, "--ros-args", "--params-file", params], env=env
        )
        os.environ["ROS_DOMAIN_ID"] = str(DOMAIN_ID)
        rclpy.init()
        cls.node = rclpy.create_node("object_perception_test")
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.color = cls.node.create_publisher(Image, "/camera/color/image_raw", 10)
        cls.depth = cls.node.create_publisher(
            Image, "/camera/depth_registered/image_rect", 10
        )
        cls.info = cls.node.create_publisher(
            CameraInfo, "/camera/color/camera_info", 10
        )
        cls.objects = []
        cls.node.create_subscription(
            DetectedObjectArray, "/aha/perception/objects", cls.objects.append, 10
        )
        cls.images = []
        cls.node.create_subscription(
            Image, "/aha/perception/objects/image", cls.images.append, 2
        )
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = CAMERA
        transform.transform.translation.z = HEIGHT
        q = transform.transform.rotation
        q.x, q.y, q.z, q.w = quaternion_from_matrix(OPTICAL)
        cls.tf = StaticTransformBroadcaster(cls.node)
        cls.tf.sendTransform(transform)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.process.kill()
        cls.executor.shutdown()
        cls.node.destroy_node()
        rclpy.shutdown()
        cls.tmpdir.cleanup()

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def publish(self, blobs):
        rgb = np.full((H, W, 3), 128, np.uint8)
        for colour, (u, v) in blobs.items():
            rgb[v - HALF : v + HALF, u - HALF : u + HALF] = COLOURS[colour]
        depth = np.full((H, W), DEPTH, np.float32)
        stamp = self.node.get_clock().now().to_msg()
        color = Image(
            height=H, width=W, encoding="rgb8", step=W * 3, data=rgb.tobytes()
        )
        aligned = Image(
            height=H, width=W, encoding="32FC1", step=W * 4, data=depth.tobytes()
        )
        info = CameraInfo(height=H, width=W)
        info.k = [FX, 0.0, CX, 0.0, FX, CY, 0.0, 0.0, 1.0]
        for message in (color, aligned, info):
            message.header.stamp = stamp
            message.header.frame_id = CAMERA
        self.color.publish(color)
        self.depth.publish(aligned)
        self.info.publish(info)

    def stream(self, blobs, seconds, until):
        start = self.node.get_clock().now()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.publish(blobs)
            self.spin(0.1)
            if self.objects and Time.from_msg(self.objects[-1].header.stamp) >= start:
                if until(self.objects[-1]):
                    return self.objects[-1]
        return None

    def test_labels_positions_and_track_ids(self):
        # Straight ahead, so that the wall's extent along the viewing direction
        # is zero and the hidden depth is the label's footprint.
        blobs = {"red": (160, 60), "green": (160, 180), "blue": (60, 120)}
        message = self.stream(blobs, 30.0, lambda m: len(m.objects) == 2)
        self.assertIsNotNone(message, "two objects were never published")
        self.assertEqual(message.header.frame_id, "map")
        found = {o.class_name: o for o in message.objects}
        # Blue is none of the list.
        self.assertEqual(set(found), {"apple", "cola"})
        for label, colour in (("apple", "red"), ("cola", "green")):
            position = found[label].pose.pose.position
            np.testing.assert_allclose(
                [position.x, position.y, position.z],
                expected_centre(label, *blobs[colour]),
                atol=0.05,
            )
            size = found[label].size
            self.assertAlmostEqual(size.y, 2 * HALF * DEPTH / FX, delta=0.05)
            self.assertAlmostEqual(found[label].pose.pose.orientation.w, 1.0)
        ids = {label: o.track_id for label, o in found.items()}
        self.assertEqual(len(set(ids.values())), 2)

        # Same positions keep the same track IDs.
        later = self.stream(blobs, 5.0, lambda m: len(m.objects) == 2)
        self.assertEqual({o.class_name: o.track_id for o in later.objects}, ids)

        # An object out of view is not published; the other keeps its ID.
        alone = self.stream(
            {"green": blobs["green"]}, 5.0, lambda m: len(m.objects) == 1
        )
        self.assertEqual(
            [(o.class_name, o.track_id) for o in alone.objects],
            [("cola", ids["cola"])],
        )
        self.assertTrue(self.images, "no annotated image was published")


if __name__ == "__main__":
    unittest.main()
