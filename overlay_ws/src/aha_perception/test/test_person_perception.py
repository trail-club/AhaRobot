"""End to end: person_perception.py against the fake worker (no DGX, no CUDA).

Runs the node from the source tree on ROS_DOMAIN_ID 96 (AHA_TEST_DOMAIN_ID to
change), so it does not see a running sim.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

from aha_msgs.msg import PersonArray
from aha_msgs.srv import DescribePerson, RememberPerson
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(PKG, "scripts", "person_perception.py")
FAKE = os.path.join(PKG, "test", "fake_person_worker.py")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "96"))
W, H, FX, CX, CY, DEPTH = 320, 240, 200.0, 160.0, 120.0, 2.0
CAMERA = "camera_color_optical_frame"
OFFSET = np.array([1.0, 0.5, 0.0])  # map -> camera translation


def expected_centre(u):
    """Map position of a rectangle centred on column u at the image centre row."""
    return OFFSET + np.array([(u - CX) * DEPTH / FX, 0.0, DEPTH])


class PersonPerceptionTest(unittest.TestCase):
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
                            "target_frame": "map",
                            "max_rate": 20.0,
                            "forget_after": 1.0,
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
        cls.node = rclpy.create_node("person_perception_test")
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.color = cls.node.create_publisher(Image, "/camera/color/image_raw", 10)
        cls.depth = cls.node.create_publisher(
            Image, "/camera/depth_registered/image_rect", 10
        )
        cls.info = cls.node.create_publisher(
            CameraInfo, "/camera/color/camera_info", 10
        )
        cls.people = []
        cls.node.create_subscription(
            PersonArray, "/aha/perception/people", cls.people.append, 10
        )
        cls.remember = cls.node.create_client(
            RememberPerson, "/aha/perception/remember_person"
        )
        cls.describe = cls.node.create_client(
            DescribePerson, "/aha/perception/describe_person"
        )
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = CAMERA
        transform.transform.translation.x, transform.transform.translation.y = OFFSET[
            :2
        ]
        transform.transform.rotation.w = 1.0
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

    def publish(self, columns):
        """One frame: a 40x120 pixel rectangle per colour, centred on its column."""
        rgb = np.full((H, W, 3), 128, np.uint8)
        for colour, u in columns.items():
            rgb[CY_SLICE, u - 20 : u + 20] = {"red": (255, 0, 0), "blue": (0, 0, 255)}[
                colour
            ]
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

    def stream(self, columns, seconds, until=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.publish(columns)
            self.spin(0.1)
            if until is not None and self.people and until(self.people[-1]):
                return self.people[-1]
        # Without a condition: the last message of the stream.
        return self.people[-1] if until is None and self.people else None

    def call(self, client, request, timeout=20.0):
        self.assertTrue(client.wait_for_service(timeout_sec=timeout))
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            self.spin(0.05)
        self.assertTrue(future.done(), "service did not answer")
        return future.result()

    @staticmethod
    def visible(message):
        return {p.id: p for p in message.people if p.visible}

    def test_track_remember_describe_and_rename_after_swap(self):
        message = self.stream(
            {"red": 100, "blue": 220}, 30.0, lambda m: len(self.visible(m)) == 2
        )
        self.assertIsNotNone(message, "two people were never published")
        self.assertEqual(message.header.frame_id, "map")
        people = sorted(self.visible(message).values(), key=lambda p: p.position.x)
        red, blue = people
        for person, u in ((red, 100), (blue, 220)):
            centre = np.array([person.position.x, person.position.y, person.position.z])
            np.testing.assert_allclose(centre, expected_centre(u), atol=0.05)
            # The head is above the centre: smaller y in the optical frame.
            self.assertLess(person.head.y, person.position.y - 0.3)

        # Same positions keep the same IDs.
        later = self.stream({"red": 100, "blue": 220}, 2.0)
        self.assertEqual(set(self.visible(later)), {red.id, blue.id})

        for person, name in ((red, "guest1"), (blue, "guest2")):
            reply = self.call(
                self.remember, RememberPerson.Request(id=person.id, name=name)
            )
            self.assertTrue(reply.success, reply.message)
        reply = self.call(self.describe, DescribePerson.Request(id=red.id))
        self.assertTrue(reply.success, reply.message)
        self.assertEqual(list(reply.features), ["red shirt"])
        missing = self.call(self.describe, DescribePerson.Request(id=999))
        self.assertFalse(missing.success)

        # Out of view until the tracks are dropped, then back with seats swapped:
        # the VLM names the new tracks.
        self.spin(2.0)
        message = self.stream(
            {"red": 220, "blue": 100},
            10.0,
            lambda m: all(p.name for p in self.visible(m).values())
            and len(self.visible(m)) == 2,
        )
        self.assertIsNotNone(message, "the remembered people were not renamed")
        named = {p.name: p for p in self.visible(message).values()}
        self.assertEqual(set(named), {"guest1", "guest2"})
        self.assertFalse({p.id for p in named.values()} & {red.id, blue.id})
        for name, u in (("guest1", 220), ("guest2", 100)):
            np.testing.assert_allclose(
                named[name].position.x, expected_centre(u)[0], atol=0.05
            )


CY_SLICE = slice(int(CY) - 60, int(CY) + 60)


if __name__ == "__main__":
    unittest.main()
