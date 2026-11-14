"""Gazebo model poses over gz-transport, for the simulation-only detectors.

Used by scripts/sim_object_detector.py and scripts/sim_person_detector.py.
"""

import threading
import time

import numpy as np

from aha_perception.plane_detection_io import transform_to_matrix


def tf_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class GazeboPoses:
    """Latest top-level model poses of a world from gz-transport.

    The subscription is throttled to ``rate`` messages per second and a
    message is only parsed when asked for (pose/info is large and frequent)."""

    def __init__(self, world: str, timeout: float, rate: float):
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node as GzNode
        from gz.transport13 import SubscribeOptions

        self.node = GzNode()
        self.world = world or self._find_world(timeout)
        self.lock = threading.Lock()
        self.msg = None
        self.stamp = 0.0
        options = SubscribeOptions()
        options.msgs_per_sec = max(int(rate), 1)
        topic = f"/world/{self.world}/pose/info"
        if not self.node.subscribe(Pose_V, topic, self._on_poses, options):
            raise RuntimeError(f"cannot subscribe to {topic}")

    def _find_world(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for topic in self.node.topic_list():
                parts = topic.split("/")
                if len(parts) == 4 and parts[1] == "world" and parts[3] == "clock":
                    return parts[2]
            time.sleep(0.5)
        raise RuntimeError("no Gazebo world found (check GZ_PARTITION)")

    def _on_poses(self, msg):
        with self.lock:
            self.msg = msg
            self.stamp = time.monotonic()

    def latest(self) -> tuple[dict, float]:
        """(world pose 4x4 by name, age in s of the message)."""
        with self.lock:
            msg, stamp = self.msg, self.stamp
        poses = {}
        for pose in msg.pose if msg is not None else ():
            p, q = pose.position, pose.orientation
            poses[pose.name] = transform_to_matrix(
                (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
            )
        return poses, time.monotonic() - stamp
