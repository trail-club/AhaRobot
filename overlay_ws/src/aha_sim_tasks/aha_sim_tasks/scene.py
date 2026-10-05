"""Static benchmark geometry shared by the world, criteria, and scripted baseline."""

import json
import math
from pathlib import Path


def scene_path():
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("aha_sim_tasks")) / "config/scene.json"


def load_scene(path=None):
    return json.loads((Path(path) if path is not None else scene_path()).read_text())


def approach_target(scene, destination=False):
    """Robot world pose that aligns the configured gripper with a table's object."""
    position = scene["destination_position" if destination else "source_position"]
    heading = scene["robot_spawn"]["yaw"]
    ox, oy = scene["scripted"]["gripper_offset"]
    c, s = math.cos(heading), math.sin(heading)
    return [position[0] - (c * ox - s * oy), position[1] - (s * ox + c * oy), heading]


def scripted_distances(scene):
    """Distances along the initial heading; this baseline only drives straight."""
    spawn = scene["robot_spawn"]
    c, s = math.cos(spawn["yaw"]), math.sin(spawn["yaw"])
    distances = []
    for destination in (False, True):
        tx, ty, _ = approach_target(scene, destination)
        dx, dy = tx - spawn["x"], ty - spawn["y"]
        if abs(-s * dx + c * dy) > 1e-6 or c * dx + s * dy < 0:
            raise ValueError(
                "Scripted scene targets must be forward along the spawn heading"
            )
        distances.append(c * dx + s * dy)
    return tuple(distances)


def expand_world(template, scene_config):
    import xacro

    return xacro.process_file(
        str(template), mappings={"scene_config": str(scene_config)}
    ).toxml()
