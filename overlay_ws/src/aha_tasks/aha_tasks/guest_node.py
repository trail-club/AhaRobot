"""Simulation only: moves the guests in Gazebo as the task runner's subgoals advance.

Uses the true robot pose and the task memory (the offered seats), which the
skills never see: they only get the guests through the head camera and
person_perception. Motion runs on the simulation clock, so the walking speed
holds at any real time factor. Seated people (the host, the sitting mesh of
aha_tasks.sim_people) and added chairs are created with the guests.
"""

import argparse
import json
import math
import os
from pathlib import Path
import threading

from aha_tasks.guests import (
    CHAIR_MODEL,
    Guest,
    chair_sdf,
    door_open,
    guest_phase,
    guest_sdf,
    load_config,
    offered_seat,
    seated_pose,
    statuses_of,
)
from aha_tasks.people_scene_node import DEFAULT_CACHE
from aha_tasks.runner import STATUS_TOPIC
from aha_tasks.sim_people import HIDDEN_Z, build_meshes, person_sdf

SITTING = "sitting"  # the sim_people label of seated people

PERIOD = 0.1  # s (simulation clock)
REQUEST_TIMEOUT = 2000  # ms
MAX_DT = 0.5  # s; the clock may jump when Gazebo restarts
MOVED = 1e-3  # m or rad; below this a guest keeps its pose in Gazebo


def set_pose(gz, world, name, x, y, z, yaw, timeout=REQUEST_TIMEOUT):
    """Move a Gazebo model (gz-transport Node, world name) to a yawed pose."""
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose

    request = Pose(name=name)
    request.position.x, request.position.y, request.position.z = x, y, z
    request.orientation.z = math.sin(yaw / 2.0)
    request.orientation.w = math.cos(yaw / 2.0)
    ok, reply = gz.request(f"/world/{world}/set_pose", request, Pose, Boolean, timeout)
    return ok and reply.data


def find_model_sdf(name):
    """model.sdf of model://<name> in GZ_SIM_RESOURCE_PATH or sobits_gazebo_worlds."""
    paths = os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep)
    roots = [Path(p) for p in paths if p]
    try:
        from ament_index_python.packages import get_package_share_directory

        roots.append(
            Path(get_package_share_directory("sobits_gazebo_worlds")) / "models"
        )
    except LookupError:
        pass
    for root in roots:
        path = root / name / "model.sdf"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"model://{name} not found")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
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
    state = {"robot": None, "statuses": {}, "memory": {}}

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

    def move(name, x, y, z, yaw):
        return set_pose(gz, scene["world"], name, x, y, z, yaw)

    rclpy.init(args=ros_args)
    node = Node("sim_guests")
    log = node.get_logger()

    def on_status(message):
        status = json.loads(message.data)
        with lock:
            state["statuses"] = statuses_of(status)
            state["memory"] = status.get("memory", {})

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
    # People are created hidden under the floor and then moved.
    missing = [
        (name, guest_sdf(name, person["away"], person["colors"]))
        for name, person in scene["guests"].items()
    ]
    seated = scene["seated"]
    if seated:
        try:
            meshes = build_meshes(
                args.cache, [(SITTING, p["shirt"]) for p in seated.values()]
            )
            for name, person in seated.items():
                x, y, _, yaw = seated_pose(scene, name)
                mesh = meshes[(SITTING, person["shirt"])]
                missing.append((name, person_sdf(name, mesh, x, y, HIDDEN_Z, yaw)))
        except (OSError, RuntimeError) as error:
            log.error(f"No seated people: {error}")
            seated = {}
    chairs = {name: s["chair"] for name, s in scene["seats"].items() if "chair" in s}
    if chairs:
        try:
            model = find_model_sdf(CHAIR_MODEL)
            missing += [(n, chair_sdf(model, n, p)) for n, p in chairs.items()]
        except FileNotFoundError as error:
            log.error(f"No chairs added: {error}")
    sent = {}
    opened = None
    last = None

    def spawn():
        for name, sdf in list(missing):
            ok, reply = gz.request(
                f"{world}/create",
                EntityFactory(sdf=sdf),
                EntityFactory,
                Boolean,
                REQUEST_TIMEOUT,
            )
            if not (ok and reply.data):
                return
            missing.remove((name, sdf))
            log.info(f"Spawned {name}")

    def tick():
        nonlocal last, opened
        now = node.get_clock().now().nanoseconds * 1e-9
        dt, last = (0.0 if last is None else min(max(now - last, 0.0), MAX_DT)), now
        if missing:
            spawn()
            return
        with lock:
            robot, statuses = state["robot"], state["statuses"]
            memory = state["memory"]
        is_open = door_open([g.name for g in guests], statuses)
        if is_open != opened:
            x, y, z, yaw = door["open" if is_open else "closed"]
            if move(door["model"], x, y, z, yaw):
                opened = is_open
                log.info(f"Door {'open' if is_open else 'closed'}")
        poses = {name: seated_pose(scene, name) for name in seated}
        for guest in guests:
            phase = guest_phase(guest.name, statuses)
            seat = offered_seat(guest.name, memory, scene["seats"])
            if phase != guest.phase:
                log.info(f"{guest.name}: phase {guest.phase} -> {phase}")
            guest.set_phase(phase, seat)
            guest.step(dt, robot)
            poses[guest.name] = (guest.x, guest.y, guest.z, guest.model_yaw)
        for name, pose in poses.items():
            previous = sent.get(name)
            if (
                previous is None
                or max(abs(a - b) for a, b in zip(pose, previous)) > MOVED
            ):
                if move(name, *pose):
                    sent[name] = pose

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
