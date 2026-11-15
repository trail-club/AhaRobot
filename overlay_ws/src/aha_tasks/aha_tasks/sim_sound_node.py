"""Simulation only: the microphone stream heard at the robot (aha_tasks.sim_sounds).

Publishes 16 kHz mono audio on AUDIO_TOPIC in real time (wall clock, like a
microphone) and mixes in a doorbell or knocks for each request on BELL_TOPIC,
which sim_guests sends when a guest reaches the door. aha_audio_doorbell reads
AUDIO_TOPIC with `input: topic`.
"""

import time

from aha_tasks.sim_sounds import RATE, Mixer

AUDIO_TOPIC = "/aha/sim/audio"  # std_msgs/Float32MultiArray, RATE Hz mono samples
BELL_TOPIC = "/aha/sim/doorbell"  # std_msgs/String: "chime" | "knock"
PERIOD = 0.1  # s between audio messages


def main(argv=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    rclpy.init(args=argv)
    node = Node("sim_sounds")
    mixer = Mixer()
    pub = node.create_publisher(Float32MultiArray, AUDIO_TOPIC, 10)

    def on_bell(message):
        try:
            kind = mixer.add(message.data)
        except ValueError as error:
            node.get_logger().warn(str(error))
            return
        node.get_logger().info(f"Playing {message.data} ({kind})")

    node.create_subscription(String, BELL_TOPIC, on_bell, 10)
    start = time.monotonic()
    sent = 0

    def tick():
        nonlocal sent
        due = int((time.monotonic() - start) * RATE) - sent
        if due > 0:
            pub.publish(Float32MultiArray(data=mixer.read(due).tolist()))
            sent += due

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
