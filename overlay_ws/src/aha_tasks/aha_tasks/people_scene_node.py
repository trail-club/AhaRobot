"""Simulation only: puts the posed people of a scene file into Gazebo.

The people stand still in their posture or gesture for the whole run; waving
people switch between two meshes on the simulation clock. Skills never see
the scene file, only the head camera through person_perception.
"""

import argparse
import math
from pathlib import Path

from aha_tasks.sim_people import (
    HIDDEN_Z,
    build_meshes,
    load_scene,
    models,
    person_sdf,
    shown_frame,
)

PERIOD = 0.05  # s (simulation clock)
REQUEST_TIMEOUT = 2000  # ms
DEFAULT_CACHE = Path.home() / ".cache" / "aharobot" / "sim_people"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args, ros_args = parser.parse_known_args(argv)
    scene = load_scene(args.config)
    paths = build_meshes(
        args.cache, [(p["label"], p["shirt"]) for p in scene["people"].values()]
    )
    entries = models(scene, paths)

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.pose_pb2 import Pose
    from gz.transport13 import Node as GzNode
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node

    world = f"/world/{scene['world']}"
    gz = GzNode()

    def place(person, hidden):
        x, y, z = person["position"]
        return x, y, HIDDEN_Z if hidden else z, person["yaw"]

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
    node = Node("sim_people")
    log = node.get_logger()
    missing = list(entries)
    shown = {}  # person name -> frame on show

    def spawn():
        for entry in list(missing):
            model, name, frame, mesh = entry
            person = scene["people"][name]
            # Every frame but the first of a waving person starts hidden.
            first = frame == shown_frame(person["label"], 0.0, scene["wave_period"])
            sdf = person_sdf(model, mesh, *place(person, not first))
            ok, reply = gz.request(
                f"{world}/create",
                EntityFactory(sdf=sdf),
                EntityFactory,
                Boolean,
                REQUEST_TIMEOUT,
            )
            if not (ok and reply.data):
                return
            missing.remove(entry)
            if first:
                shown[name] = frame
            log.info(f"Spawned {model} ({person['label']}, {person['shirt']} shirt)")

    def tick():
        if missing:
            spawn()
            return
        now = node.get_clock().now().nanoseconds * 1e-9
        for model, name, frame, _ in entries:
            person = scene["people"][name]
            if model == name:
                continue  # one mesh: never moves
            wanted = shown_frame(person["label"], now, scene["wave_period"])
            if shown.get(name) == wanted or frame != wanted:
                continue
            for other, other_name, other_frame, _ in entries:
                if other_name == name:
                    set_pose(other, *place(person, other_frame != wanted))
            shown[name] = wanted

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
