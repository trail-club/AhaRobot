"""Simulation only: serves the ASR's Listen action with simulated voices.

The answer is chosen by aha_tasks.sim_speech from the robot's pose (TF map ->
base_footprint) and the robot's last caption (/aha/display/caption). It is
also published as a transcript on /aha/perception/audio/transcript, as the
ASR does. Without an answer the goal ends after its timeout like silence.
A new run in the task status (/aha/tasks/status) starts the commands over.
"""

import argparse
import json
import math
from pathlib import Path
import threading
import time

from aha_tasks.core import load_spec
from aha_tasks.display import CAPTION_TOPIC
from aha_tasks.runner import STATUS_TOPIC
from aha_tasks.sim_people import load_scene
from aha_tasks.sim_speech import Voices, asks_name, load_config, speaker_for

LISTEN_ACTION = "/aha/perception/audio/listen"
TRANSCRIPT_TOPIC = "/aha/perception/audio/transcript"
DEFAULT_TIMEOUT = 10.0  # s, as the ASR's listen_timeout
MODEL = "sim_speech"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True, help="Task definition JSON")
    parser.add_argument("--people", type=Path, help="People scene JSON (named people)")
    args, ros_args = parser.parse_known_args(argv)
    spec = load_spec(args.task)
    people = load_scene(args.people) if args.people else None
    settings, speakers = load_config(args.config, spec.locations, people)
    voices = Voices(settings)

    import rclpy
    from aha_msgs.action import Listen
    from aha_msgs.msg import SpeechTranscript
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init(args=ros_args)
    node = Node("sim_speech")
    log = node.get_logger()
    group = ReentrantCallbackGroup()
    tf = Buffer()
    TransformListener(tf, node)
    caption = {"text": ""}
    busy = threading.Lock()

    def on_caption(message):
        caption["text"] = message.data

    def on_status(message):
        try:
            run = (json.loads(message.data).get("run") or {}).get("number")
        except ValueError:
            return
        voices.new_run(run)

    node.create_subscription(
        String, CAPTION_TOPIC, on_caption, 10, callback_group=group
    )
    latched = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    node.create_subscription(
        String, STATUS_TOPIC, on_status, latched, callback_group=group
    )
    transcripts = node.create_publisher(SpeechTranscript, TRANSCRIPT_TOPIC, 10)

    def robot_pose():
        try:
            t = tf.lookup_transform("map", "base_footprint", Time()).transform
        except TransformException as error:
            log.warning(f"No robot pose: {error}")
            return None
        q = t.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        return (t.translation.x, t.translation.y, yaw)

    def on_goal(goal):
        # One listener at a time, as the ASR.
        return (
            GoalResponse.ACCEPT if busy.acquire(blocking=False) else GoalResponse.REJECT
        )

    def execute(handle):
        try:
            return listen(handle)
        finally:
            busy.release()

    def cancelled(handle, until):
        """Wait until the monotonic time `until`; True if the goal was canceled."""
        while time.monotonic() < until:
            if handle.is_cancel_requested:
                handle.canceled()
                return True
            time.sleep(0.05)
        return False

    def listen(handle):
        timeout = handle.request.timeout
        timeout = timeout if timeout > 0 else DEFAULT_TIMEOUT
        started = time.monotonic()
        if cancelled(handle, started + settings["delay"]):
            return Listen.Result(message="canceled")
        question = caption["text"]
        pose = robot_pose()
        speaker = (
            speaker_for(
                pose,
                speakers,
                settings["radius"],
                settings["fov"],
                person=asks_name(question),
            )
            if pose is not None
            else None
        )
        text = voices.answer(question, speaker)
        who = "nobody" if speaker is None else speaker.name or "the operator"
        log.info(f"{question!r} -> {who}: {text!r}")
        if text is not None:
            handle.publish_feedback(Listen.Feedback(state="hearing"))
            message = SpeechTranscript(
                text=text,
                language="en",
                model=MODEL,
                audio_duration=0.3 * len(text.split()),
            )
            message.header.stamp = node.get_clock().now().to_msg()
            transcripts.publish(message)
            handle.succeed()
            return Listen.Result(success=True, transcript=message)
        if cancelled(handle, started + timeout):
            return Listen.Result(message="canceled")
        handle.abort()
        return Listen.Result(message=f"no transcript within {timeout:.1f}s")

    ActionServer(
        node,
        Listen,
        LISTEN_ACTION,
        execute_callback=execute,
        goal_callback=on_goal,
        cancel_callback=lambda _: CancelResponse.ACCEPT,
        callback_group=group,
    )
    log.info(
        f"Answering as the operator and {sum(not s.operator for s in speakers)} "
        f"named people ({len(settings['commands'])} commands)"
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
