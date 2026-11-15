#!/usr/bin/env python3
"""Add D435i-like depth noise to the simulated head depth image.

Subscribes input_topic (the Gazebo depth from ros_gz_bridge, 32FC1 m or
16UC1 mm) and publishes the corrupted image on output_topic with the same
header and encoding (32FC1: NaN for invalid; 16UC1 / mono16 in, 16UC1 out:
0 for invalid). Invalid input pixels stay invalid and pixels beyond
NoiseParams.max_range (6 m) become invalid. The noise is
aha_perception.depth_noise with the preset named by the parameter preset and
f = fx of camera_info_topic. The simulated camera uses the fx of the real
color camera (about 603, 55.9 deg horizontal FOV), so sigma(Z) in meters
matches the real D435i. The Gazebo sensor itself has no depth noise
(aha_description/urdf/sensors.xacro sets only the RGB noise).

The frozen components (static error field, holes) are drawn with the first
frame from seed and stay fixed in image coordinates like the systematic error
of the real camera; they are redrawn when the image size changes. The
temporal noise, flicker next to invalid pixels, dropouts, flying pixels and
the stereo shadow (left of foreground edges) are computed per frame.

Only the latest frame is kept and processed in a worker thread; frames
arriving while it is busy are dropped. The processing time and the number of
dropped frames are logged every SUMMARY_PERIOD_S. Runs on the wall clock:
stamps are copied from the input.
"""

import sys
import threading
import time
from array import array

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image

from aha_perception.depth_noise import NOISE_PRESETS, DepthNoiseSimulator
from aha_perception.plane_detection_io import depth_from_buffer, depth_to_buffer

NODE_NAME = "aha_perception_sim_depth_noise"
WARN_PERIOD_S = 5.0
SUMMARY_PERIOD_S = 10.0
WORKER_JOIN_S = 3.0


class SimDepthNoise(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        preset = self.declare_parameter("preset", "d435i_filtered").value
        if preset not in NOISE_PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}, choose from {sorted(NOISE_PRESETS)}"
            )
        input_topic = self.declare_parameter(
            "input_topic", "/camera/depth_registered/image_rect_clean"
        ).value
        output_topic = self.declare_parameter(
            "output_topic", "/camera/depth_registered/image_rect"
        ).value
        info_topic = self.declare_parameter(
            "camera_info_topic", "/camera/color/camera_info"
        ).value
        seed = int(self.declare_parameter("seed", 0).value)
        self.noise = DepthNoiseSimulator(preset, seed)
        self.shape = None
        self.camera_info = None

        self.cond = threading.Condition()
        self.pending = None
        self.stopping = False
        self.dropped = 0
        self.times_ms = []
        self.last_summary = time.monotonic()

        self.create_subscription(
            CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, input_topic, self.on_depth, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(
            Image,
            output_topic,
            QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"{preset} noise (seed {seed}): {input_topic} -> {output_topic}"
        )

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_depth(self, msg):
        with self.cond:
            if self.pending is not None:
                self.dropped += 1
            self.pending = msg
            self.cond.notify()

    def stop(self):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        self.worker.join(timeout=WORKER_JOIN_S)

    def run_worker(self):
        while True:
            with self.cond:
                while self.pending is None and not self.stopping:
                    self.cond.wait()
                if self.stopping:
                    return
                msg, self.pending = self.pending, None
            try:
                self.process(msg)
            except Exception as e:  # keep the worker alive
                if self.stopping or not rclpy.ok():
                    return
                self.get_logger().error(
                    f"depth noise failed: {e!r}", throttle_duration_sec=WARN_PERIOD_S
                )

    def process(self, msg):
        info = self.camera_info
        if info is None or info.k[0] <= 0.0:
            self.get_logger().warn(
                "waiting for camera_info with intrinsics",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        if (info.width, info.height) != (msg.width, msg.height):
            self.get_logger().warn(
                f"depth {msg.width}x{msg.height} does not match camera_info "
                f"{info.width}x{info.height}, skipped",
                throttle_duration_sec=WARN_PERIOD_S,
            )
            return
        start = time.perf_counter()
        depth = depth_from_buffer(
            msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian
        )
        if depth.shape != self.shape:
            self.get_logger().info(
                f"drawing the frozen noise for {msg.width}x{msg.height}"
            )
            self.shape = depth.shape
        noisy = self.noise.apply(depth, info.k[0])
        encoding = "32FC1" if msg.encoding == "32FC1" else "16UC1"
        data, step = depth_to_buffer(noisy, encoding)
        out = Image(
            header=msg.header,
            height=msg.height,
            width=msg.width,
            encoding=encoding,
            is_bigendian=False,
            step=step,
            data=array("B", data),
        )
        if self.stopping:
            return
        self.pub.publish(out)
        self.times_ms.append((time.perf_counter() - start) * 1e3)
        self.log_summary()

    def log_summary(self):
        now = time.monotonic()
        if now - self.last_summary < SUMMARY_PERIOD_S:
            return
        with self.cond:
            dropped, self.dropped = self.dropped, 0
        times = np.array(self.times_ms)
        self.times_ms = []
        self.get_logger().info(
            f"{len(times)} frames in {now - self.last_summary:.0f} s, "
            f"{times.mean():.0f} ms mean / {times.max():.0f} ms max, "
            f"{dropped} dropped"
        )
        self.last_summary = now


def main():
    rclpy.init()
    try:
        node = SimDepthNoise()
    except ValueError as e:
        rclpy.logging.get_logger(NODE_NAME).fatal(str(e))
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
