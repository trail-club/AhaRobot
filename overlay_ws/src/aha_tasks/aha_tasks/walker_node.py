"""Simulation only: moves the GPSR walkers in Gazebo as the plan steps run.

Uses the true robot pose, which the skills never see: they only get the
walkers through the head camera and person_perception. Motion runs on the
simulation clock, so the walking speed holds at any real time factor.
"""

import argparse
import json
from pathlib import Path
import threading

from aha_tasks.guest_node import MAX_DT, MOVED, PERIOD, REQUEST_TIMEOUT, set_pose
from aha_tasks.guests import guest_sdf
from aha_tasks.runner import STATUS_TOPIC
from aha_tasks.walkers import SimWalker, load_config, step_attempts, step_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    scene = load_config(args.config)

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode
    from gz.transport13 import SubscribeOptions
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    world = f"/world/{scene['world']}"
    gz = GzNode()
    lock = threading.Lock()
    state = {"robot": None, "status": None}

    def on_poses(message):
        for pose in message.pose:
            if pose.name == scene["robot_model"]:
                with lock:
                    state["robot"] = (pose.position.x, pose.position.y)
                return

    options = SubscribeOptions()
    options.msgs_per_sec = int(1.0 / PERIOD)
    if not gz.subscribe(Pose_V, f"{world}/pose/info", on_poses, options):
        raise RuntimeError(f"cannot subscribe to {world}/pose/info")

    rclpy.init(args=ros_args)
    node = Node("sim_walkers")
    log = node.get_logger()

    def on_status(message):
        with lock:
            state["status"] = json.loads(message.data)

    node.create_subscription(
        String,
        STATUS_TOPIC,
        on_status,
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ),
    )
    walkers = [SimWalker(name, c, scene) for name, c in scene["walkers"].items()]
    missing = list(walkers)
    sent = {}
    last = None

    def spawn():
        for walker in list(missing):
            sdf = guest_sdf(
                walker.name, walker.config["start"], walker.config["colors"]
            )
            ok, reply = gz.request(
                f"{world}/create",
                EntityFactory(sdf=sdf),
                EntityFactory,
                Boolean,
                REQUEST_TIMEOUT,
            )
            # A model of that name left from an earlier run is moved instead.
            if not (ok and reply.data) and not set_pose(
                gz, scene["world"], walker.name, walker.x, walker.y, 0.0, 0.0
            ):
                return
            missing.remove(walker)
            log.info(
                f"Spawned {walker.name} ({walker.action}) at {walker.config['start']}"
            )

    def tick():
        nonlocal last
        now = node.get_clock().now().nanoseconds * 1e-9
        dt, last = (0.0 if last is None else min(max(now - last, 0.0), MAX_DT)), now
        if missing:
            spawn()
            return
        with lock:
            robot, status = state["robot"], state["status"]
        for walker in walkers:
            stage = walker.stage
            if status is None:
                walker.set_status(None)
            else:
                walker.set_status(
                    step_status(status, walker.action),
                    step_attempts(status, walker.action),
                )
            if walker.stage != stage:
                log.info(f"{walker.name}: {stage} -> {walker.stage}")
            walker.step(dt, robot)
            pose = (walker.x, walker.y, 0.0, walker.model_yaw)
            previous = sent.get(walker.name)
            if (
                previous is None
                or max(abs(a - b) for a, b in zip(pose, previous)) > MOVED
            ):
                if set_pose(gz, scene["world"], walker.name, *pose):
                    sent[walker.name] = pose

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
