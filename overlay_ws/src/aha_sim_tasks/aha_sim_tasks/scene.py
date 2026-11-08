"""YCB assets, repeatable episode geometry, and world expansion."""

from copy import deepcopy
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import random
import struct
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


def package_share(package="aha_sim_tasks"):
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory(package))


def scene_path():
    return Path(
        os.environ.get("AHA_SCENE_CONFIG", package_share() / "config/scene.json")
    )


def load_catalog():
    return json.loads((package_share() / "config/objects.json").read_text())


def model_path(model):
    return package_share("tmc_wrs_gz_worlds") / "models" / model / "model-1_4.sdf"


@lru_cache(maxsize=16)
def object_geometry(model):
    """Measure installed collision meshes; never scale an object to fit the hand."""
    path = model_path(model)
    link = ET.parse(path).getroot().find("model/link")
    vertices = []
    for collision in link.findall("collision"):
        pose = [float(v) for v in collision.findtext("pose", "0 0 0 0 0 0").split()]
        if any(abs(v) > 1e-12 for v in pose[3:]):
            raise ValueError(f"Unsupported rotated collision mesh: {model}")
        uri = collision.findtext("geometry/mesh/uri")
        mesh = path.parent / uri.split(f"model://{model}/", 1)[1]
        data = mesh.read_bytes()
        triangles = struct.unpack_from("<I", data, 80)[0]
        if len(data) != 84 + 50 * triangles:
            raise ValueError(f"Expected a binary STL collision mesh: {mesh}")
        for triangle in range(triangles):
            for vertex in range(3):
                xyz = struct.unpack_from(
                    "<3f", data, 84 + 50 * triangle + 12 + 12 * vertex
                )
                vertices.append(tuple(xyz[i] + pose[i] for i in range(3)))
    lower = [min(p[i] for p in vertices) for i in range(3)]
    upper = [max(p[i] for p in vertices) for i in range(3)]
    center = [(a + b) / 2 for a, b in zip(lower, upper)]
    size = [b - a for a, b in zip(lower, upper)]
    radius = max(math.hypot(p[0] - center[0], p[1] - center[1]) for p in vertices)
    return {
        "center": center,
        "size": size,
        "radius": radius,
        "mass": float(link.findtext("inertial/mass")),
        "vertices": [
            list(p)
            for p in sorted(
                {tuple(p[i] - center[i] for i in range(3)) for p in vertices}
            )
        ],
    }


def select_object(scene, name):
    catalog = load_catalog()
    if name not in catalog:
        raise ValueError(f"Unknown object: {name}; choose from {', '.join(catalog)}")
    obj = {"id": name, **catalog[name]}
    obj.update(deepcopy(object_geometry(obj["model"])))
    scene["object"] = obj
    scene["object_id"] = name
    # Policy and evaluator address the collision bounding-box centre.
    for prefix in ("source", "destination"):
        height = (
            scene[prefix + "_table_position"][2]
            + scene["table_size"][2] / 2
            + obj["size"][2] / 2
            + scene["spawn_clearance"]
        )
        scene[prefix + "_position"] = [*scene[prefix + "_position"][:2], height]
    settings = scene["scripted"]
    settings["pick_lift"] = scene["source_position"][2] - settings["grasp_zero_height"]
    settings["transport_lift"] = settings["pick_lift"] + 0.18
    settings["grasp_position"] = min(0.055, max(0.003, obj["size"][1] / 2 - 0.008))
    return scene


def load_scene(path=None):
    scene = json.loads((Path(path) if path is not None else scene_path()).read_text())
    if scene.get("scene_type") == "laundry":
        from .laundry_scene import load_laundry_scene

        return load_laundry_scene(scene)
    validate_scene(scene)
    if scene.get("resolved"):
        if "vertices" not in scene["object"]:
            scene["object"].update(deepcopy(object_geometry(scene["object"]["model"])))
        return scene
    for prefix in ("source", "destination"):
        scene[prefix + "_table_position"] = [
            scene[prefix + "_position"][0] + scene["table_offset"][0],
            scene[prefix + "_position"][1] + scene["table_offset"][1],
            scene["table_height"] - scene["table_size"][2] / 2,
        ]
    scene.setdefault("distractors", [])
    scene.setdefault("object_yaw", 0.0)
    scene["resolved"] = True
    return select_object(scene, scene.get("object_id", "apple"))


def validate_scene(scene):
    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError("Scene values must be finite")

    check_finite(scene)
    if len(scene["table_size"]) != 3 or min(scene["table_size"]) <= 0:
        raise ValueError("table_size must contain three positive lengths")
    if scene["table_height"] <= scene["table_size"][2]:
        raise ValueError("Table height must exceed the tabletop thickness")
    sensors = scene["sensors"]
    if (
        not 0 <= sensors["depth_dropout"] <= 1
        or sensors["depth_noise_stddev"] < 0
        or sensors["rgb_noise_stddev"] < 0
    ):
        raise ValueError(
            "Sensor noise must be nonnegative and dropout must be in [0, 1]"
        )
    if min(scene["physics"].values()) < 0:
        raise ValueError("Friction must be nonnegative")


def sample_scene(base, object_id="apple", *, seed=0, randomize=False, distractors=0):
    """Sample feasible, separated placements, independent of the policy."""
    if seed < 0 or not 0 <= distractors <= 3:
        raise ValueError("seed must be nonnegative and distractors must be in 0..3")
    scene = select_object(deepcopy(base), object_id)
    scene.update(seed=seed, randomized=randomize, distractors=[])
    rng = random.Random(seed)
    variation = scene["randomization"]
    if randomize:
        for i in range(2):
            scene["source_position"][i] += rng.uniform(
                -variation["object_xy"], variation["object_xy"]
            )
        scene["object_yaw"] = rng.uniform(-math.pi, math.pi)
        for key in ("x", "y"):
            scene["robot_spawn"][key] += rng.uniform(
                -variation["robot_xy"], variation["robot_xy"]
            )
        scene["robot_spawn"]["yaw"] += rng.uniform(
            -variation["robot_yaw"], variation["robot_yaw"]
        )
        scene["lighting"]["intensity"] = rng.uniform(*variation["light_intensity"])
        scene["lighting"]["direction"] = [
            rng.uniform(-0.7, -0.2),
            rng.uniform(-0.4, 0.4),
            -0.9,
        ]
    center = scene["source_table_position"]
    placed = [(scene["source_position"], scene["object"]["radius"])]
    if any(
        abs(scene["source_position"][i] - center[i]) + scene["object"]["radius"]
        > scene["table_size"][i] / 2
        for i in range(2)
    ):
        raise ValueError("Target does not fit on the source table")
    catalog = load_catalog()
    choices = [key for key in catalog if key != object_id]
    for index in range(distractors):
        name = rng.choice(choices)
        obj = {
            "id": name,
            **catalog[name],
            **deepcopy(object_geometry(catalog[name]["model"])),
        }
        radius = obj["radius"]
        for _ in range(200):
            position = [
                center[i]
                + rng.uniform(
                    -scene["table_size"][i] / 2 + radius + 0.015,
                    scene["table_size"][i] / 2 - radius - 0.015,
                )
                for i in range(2)
            ]
            # Keep clutter behind the target to leave a feasible approach.
            if (
                position[0]
                < scene["source_position"][0]
                + radius
                + scene["object"]["radius"]
                + 0.025
            ):
                continue
            if all(
                math.dist(position, p[:2]) > radius + other + 0.025
                for p, other in placed
            ):
                break
        else:
            raise ValueError("Cannot place the requested distractors without overlap")
        position.append(
            scene["table_height"] + obj["size"][2] / 2 + scene["spawn_clearance"]
        )
        scene["distractors"].append(
            {
                "name": f"distractor_{index}",
                "object": obj,
                "position": position,
                "yaw": rng.uniform(-math.pi, math.pi),
            }
        )
        placed.append((position, radius))
    return scene


def approach_target(scene, destination=False):
    position = scene["destination_position" if destination else "source_position"]
    heading = scene.get("approach_yaw", scene["robot_spawn"]["yaw"])
    ox, oy = scene["scripted"]["gripper_offset"]
    c, s = math.cos(heading), math.sin(heading)
    return [position[0] - (c * ox - s * oy), position[1] - (s * ox + c * oy), heading]


def scripted_distances(scene):
    spawn = scene["robot_spawn"]
    heading = spawn["yaw"]
    distances = []
    for destination in (False, True):
        tx, ty, _ = approach_target(scene, destination)
        dx, dy = tx - spawn["x"], ty - spawn["y"]
        if (
            abs(-math.sin(heading) * dx + math.cos(heading) * dy) > 1e-6
            or math.cos(heading) * dx + math.sin(heading) * dy < 0
        ):
            raise ValueError(
                "Scripted scene targets must be forward along the spawn heading"
            )
        distances.append(math.cos(heading) * dx + math.sin(heading) * dy)
    return tuple(distances)


def bounds_corners(size):
    return list(itertools.product(*[(-v / 2, v / 2) for v in size]))


def object_model(obj, name, position, yaw, entities, friction):
    model = deepcopy(ET.parse(model_path(obj["model"])).getroot().find("model"))
    model.set("name", name)
    ET.SubElement(model, "pose").text = " ".join(map(str, (*position, 0, 0, yaw)))
    link = model.find("link")
    link.set("name", entities["object_link"])
    for element in [
        link.find("inertial"),
        *link.findall("visual"),
        *link.findall("collision"),
    ]:
        pose = element.find("pose")
        if pose is None:
            pose = ET.SubElement(element, "pose")
        values = [float(v) for v in (pose.text or "0 0 0 0 0 0").split()]
        pose.text = " ".join(
            map(str, (*(values[i] - obj["center"][i] for i in range(3)), *values[3:]))
        )
    for index, collision in enumerate(link.findall("collision")):
        collision.set(
            "name", entities["object_collision"] + (f"_{index}" if index else "")
        )
        old_surface = collision.find("surface")
        if old_surface is not None:
            collision.remove(old_surface)
        surface = ET.SubElement(collision, "surface")
        ode = ET.SubElement(ET.SubElement(surface, "friction"), "ode")
        for key in ("mu", "mu2"):
            ET.SubElement(ode, key).text = str(friction)
        ET.SubElement(
            ET.SubElement(surface, "bounce"), "restitution_coefficient"
        ).text = "0.0"
    return model


def expand_world(template, scene_config, cameras=True, ground_truth=False):
    import xacro

    scene = load_scene(scene_config)
    with TemporaryDirectory(prefix="aha-resolved-scene-") as directory:
        resolved = Path(directory) / "scene.json"
        resolved.write_text(json.dumps(scene))
        root = ET.fromstring(
            xacro.process_file(
                str(template),
                mappings={
                    "scene_config": str(resolved),
                    "cameras": str(cameras).lower(),
                    "ground_truth": str(ground_truth).lower(),
                },
            ).toxml()
        )
    world = root.find("world")
    world.append(
        object_model(
            scene["object"],
            scene["entities"]["object_model"],
            scene["source_position"],
            scene["object_yaw"],
            scene["entities"],
            scene["physics"]["object_friction"],
        )
    )
    for item in scene["distractors"]:
        world.append(
            object_model(
                item["object"],
                item["name"],
                item["position"],
                item["yaw"],
                scene["entities"],
                scene["physics"]["object_friction"],
            )
        )
    return ET.tostring(root, encoding="unicode")
