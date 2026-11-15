"""Simulation only: moves the guests in Gazebo as the task runner's subgoals advance.

Uses the true robot pose, which the skills never see: they only get the guests
through the head camera and person_perception. Motion runs on the simulation
clock, so the walking speed holds at any real time factor.
"""

import argparse
import json
import math
from pathlib import Path
import threading

from aha_tasks.guests import (
    Guest,
    door_open,
    guest_phase,
    guest_sdf,
    load_config,
    statuses_of,
)
from aha_tasks.runner import STATUS_TOPIC

PERIOD = 0.1  # s (simulation clock)
REQUEST_TIMEOUT = 2000  # ms
MAX_DT = 0.5  # s; the clock may jump when Gazebo restarts
MOVED = 1e-3  # m or rad; below this a guest keeps its pose in Gazebo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    scene = load_config(args.config)

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.pose_pb2 import Pose
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
    state = {"robot": None, "statuses": {}}

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

    def set_pose(name, x, y, z, yaw):
        request = Pose(name=name)
        request.position.x, request.position.y, request.position.z = x, y, z
        request.orientation.z = math.sin(yaw / 2.0)
        request.orientation.w = math.cos(yaw / 2.0)
        ok, reply = gz.request(
            f"{world}/set_pose", request, Pose, Boolean, REQUEST_TIMEOUT
        )
        return ok and reply.data

    rclpy.init(args=ros_args)
    node = Node("sim_guests")
    log = node.get_logger()

    def on_status(message):
        with lock:
            state["statuses"] = statuses_of(json.loads(message.data))

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
    guests = [Guest(name, c, scene) for name, c in scene["guests"].items()]
    door = scene["door"]
    missing = list(guests)
    sent = {}
    opened = None
    last = None

    def spawn():
        for guest in list(missing):
            sdf = guest_sdf(guest.name, guest.config["away"], guest.config["colors"])
            ok, reply = gz.request(
                f"{world}/create",
                EntityFactory(sdf=sdf),
                EntityFactory,
                Boolean,
                REQUEST_TIMEOUT,
            )
            if not (ok and reply.data):
                return
            missing.remove(guest)
            log.info(f"Spawned {guest.name} at {guest.config['away']}")

    def tick():
        nonlocal last, opened
        now = node.get_clock().now().nanoseconds * 1e-9
        dt, last = (0.0 if last is None else min(max(now - last, 0.0), MAX_DT)), now
        if missing:
            spawn()
            return
        with lock:
            robot, statuses = state["robot"], state["statuses"]
        is_open = door_open([g.name for g in guests], statuses)
        if is_open != opened:
            x, y, z, yaw = door["open" if is_open else "closed"]
            if set_pose(door["model"], x, y, z, yaw):
                opened = is_open
                log.info(f"Door {'open' if is_open else 'closed'}")
        for guest in guests:
            phase = guest_phase(guest.name, statuses)
            if phase != guest.phase:
                log.info(f"{guest.name}: phase {guest.phase} -> {phase}")
            guest.set_phase(phase)
            guest.step(dt, robot)
            pose = (guest.x, guest.y, guest.z, guest.model_yaw)
            previous = sent.get(guest.name)
            if (
                previous is None
                or max(abs(a - b) for a, b in zip(pose, previous)) > MOVED
            ):
                if set_pose(guest.name, *pose):
                    sent[guest.name] = pose

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
