"""Simulation only: puts the objects of a scene file onto the furniture in Gazebo.

Objects missing from the world are spawned; objects of the same name left
from an earlier run (e.g. moved by the robot) are put back to their scene
pose, so running it again resets the scene. Models that the object list
would detect as an object but that are not in the scene (e.g. of another
scene placed earlier in the same Gazebo) are removed first; other models
(furniture, the robot, guests, people) are never touched. After ``--settle`` seconds the
objects are checked against their resting poses, and the program exits:
0 when every object is in the world, 1 otherwise. A moved or tipped object
only gives a warning. ``--check`` only does the check (nothing is moved or
removed; objects not in the scene are reported).
Skills never see the scene file.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import threading
import time

from aha_tasks.sim_objects import (
    Catalog,
    SceneError,
    displacement,
    furniture_config,
    load_scene,
    stale_models,
)

REQUEST_TIMEOUT = 2000  # ms
# AHA_CACHE_DIR: a directory that outlives the container (docker-compose.yml)
DEFAULT_CACHE = (
    Path(os.environ.get("AHA_CACHE_DIR") or Path.home() / ".cache" / "aharobot")
    / "sim_objects"
)
FURNITURE = "config/furniture"  # aha_perception share, <world>.yaml
OBJECT_LIST = "config/objects/robocup_objects.yaml"  # aha_perception share
MOVED = 0.02  # m, warn when an object rests this far from its scene pose
TIPPED = math.radians(10.0)


def scene_path(value: str) -> Path:
    """A scene name of config/sim/objects/, or a JSON path."""
    if value.endswith(".json"):
        return Path(value)
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("aha_tasks"))
    return share / "config" / "sim" / "objects" / f"{value}.json"


def load(config: Path):
    from ament_index_python.packages import get_package_share_directory

    from aha_perception.furniture import load_config
    from aha_perception.object_list import load_object_list

    tasks = Path(get_package_share_directory("aha_tasks"))
    perception = Path(get_package_share_directory("aha_perception"))
    tmc = Path(get_package_share_directory("tmc_wrs_gz_worlds"))
    catalog = Catalog.load(
        tasks / "config" / "sim" / "object_models.json",
        load_object_list(perception / OBJECT_LIST),
        tmc / "models",
    )
    furniture = load_config(furniture_config(perception / FURNITURE, config))
    places = {f.name: f for f in furniture.furniture}
    return catalog, load_scene(config, catalog, places)


def log(message: str) -> None:
    print(f"[sim_objects] {message}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="scene name in config/sim/objects/ or JSON path"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="s to wait for the Gazebo world"
    )
    parser.add_argument(
        "--settle", type=float, default=3.0, help="s (wall) before the check"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report missing, moved, tipped and stale objects",
    )
    args, _ = parser.parse_known_args(argv)
    try:
        catalog, scene = load(scene_path(args.config))
    except (SceneError, OSError, ValueError) as error:
        log(f"invalid scene: {error}")
        return 2
    sdfs = {}
    for obj in [] if args.check else scene.objects:  # --check builds nothing
        try:
            sdfs[obj.name] = catalog.sdf(obj, args.cache)
        except Exception as error:  # e.g. offline, a bad archive: place the others
            log(f"cannot build {obj.name}: {type(error).__name__}: {error}")

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode
    from gz.transport13 import SubscribeOptions

    from aha_perception.plane_detection_io import transform_to_matrix

    world = f"/world/{scene.world}"
    gz = GzNode()
    lock = threading.Lock()
    latest = {}

    def on_poses(message):
        poses = {}
        for pose in message.pose:
            p, q = pose.position, pose.orientation
            poses[pose.name] = transform_to_matrix(
                (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
            )
        with lock:
            latest.clear()
            latest.update(poses)
            latest[None] = time.monotonic()

    options = SubscribeOptions()
    options.msgs_per_sec = 5
    if not gz.subscribe(Pose_V, f"{world}/pose/info", on_poses, options):
        log(f"cannot subscribe to {world}/pose/info")
        return 1

    def poses(newer_than=0.0):
        with lock:
            if latest.get(None, -1.0) <= newer_than:
                return None
            return {k: v for k, v in latest.items() if k is not None}

    deadline = time.monotonic() + args.timeout
    while poses() is None:
        if time.monotonic() > deadline:
            log(f"no Gazebo world {scene.world} within {args.timeout:.0f} s")
            return 1
        time.sleep(0.5)

    def request(service, message, reply_type):
        ok, reply = gz.request(
            f"{world}/{service}", message, type(message), reply_type, REQUEST_TIMEOUT
        )
        return ok and reply.data

    present = poses()
    stale = stale_models(present, scene, catalog.object_list)
    if stale and args.check:
        log(f"warning: objects not in the scene: {', '.join(stale)}")
    elif stale:
        for name in stale:
            removed = request("remove", Entity(name=name, type=Entity.MODEL), Boolean)
            log(
                f"removed {name} (not in the scene)"
                if removed
                else f"could not remove {name}"
            )
        asked = time.monotonic()
        while time.monotonic() < asked + 10.0:
            current = poses(newer_than=asked)
            if current is not None and not set(stale) & set(current):
                break
            time.sleep(0.2)
        else:
            log("warning: removed objects are still in Gazebo")
    for obj in [] if args.check else scene.objects:
        x, y, z, yaw = obj.spawn_pose()
        if obj.name in present:
            pose = Pose(name=obj.name)
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.z = math.sin(yaw / 2.0)
            pose.orientation.w = math.cos(yaw / 2.0)
            ok = request("set_pose", pose, Boolean)
            action = "reset"
        elif obj.name in sdfs:
            ok = request("create", EntityFactory(sdf=sdfs[obj.name]), Boolean)
            action = "spawned"
        else:
            continue  # no model: reported missing below
        where = obj.on if obj.on == "floor" else f"{obj.on} level {obj.level}"
        role = f", {obj.role}" if obj.role else ""
        if ok:
            log(f"{action} {obj.name} ({obj.label}) on {where}{role}")
        else:
            log(f"could not place {obj.name}")

    start = time.monotonic()
    time.sleep(0.0 if args.check else max(args.settle, 0.0))
    current = None
    settle = 0.0 if args.check else args.settle
    while current is None and time.monotonic() < start + settle + 10.0:
        current = poses(newer_than=start + settle)
        time.sleep(0.2)
    current = current or {}
    missing = [o.name for o in scene.objects if o.name not in current]
    moved = 0
    for obj in scene.objects:
        if obj.name in current:
            distance, tilt = displacement(obj, current[obj.name])
            if distance > MOVED or tilt > TIPPED:
                moved += 1
                log(
                    f"warning: {obj.name} rests {distance * 100:.1f} cm,"
                    f" {math.degrees(tilt):.0f} deg from its scene pose"
                )
    if missing:
        log(f"missing in Gazebo: {', '.join(missing)}")
        return 1
    log(f"{len(scene.objects)} objects in {scene.world}, {moved} off their pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
