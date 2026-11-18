"""Simulated objects on the furniture of the task world, without ROS or Gazebo.

A scene file (config/sim/objects/<name>.json) lists the objects of a task::

    {"world": "rcjo2026_arena",
     "objects": [{"name": "mug", "label": "mug", "on": "dining_table",
                  "position": [-1.75, -1.55], "yaw": 0.0, "role": "table.cup"}]}

``name`` is the Gazebo model name and must match the label's model patterns
of the object list (aha_perception.object_list), so that
sim_object_detector.py reports it under that label. ``on`` is a furniture of
the furniture config (aha_perception.furniture, frame ``world``) or
``floor``; ``level`` (1 = lowest) is needed for furniture with several
levels. ``position`` is the x, y of the object center and ``yaw`` its turn
(rad); the height follows from the level and the object's size, so every
object rests on its surface. ``role`` (optional) is the ground truth of the
task, e.g. the Pick and Place memory key the object belongs to.

The model catalog (config/sim/object_models.json) gives each label a model:

- ``ycb``: a YCB model of tmc_wrs_gz_worlds (mesh, mass, inertia), turned by
  ``yaw`` degrees inside the model so that its sides line up with x and y,
  with the origin moved to the center of its collision bounding box.
- ``scanned``: a dataset's scan of the real object (aha_tasks.object_models).
- ``measured``: a measured shape painted with images, for objects that no
  dataset has (aha_tasks.object_models).

Every model's origin is its bounding box center, as sim_object_detector.py
assumes. Models must be within ``SIZE_TOLERANCE`` of the object list's size.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

from aha_tasks.object_models import (
    FACE_AXES,
    MeasuredModel,
    ScannedModel,
    build_measured,
    build_scanned,
    yaw_rotation,
)

FLOOR = "floor"
SCENE_KEYS = {"world", "objects"}
OBJECT_KEYS = {"name", "label", "on", "position", "yaw"}
OPTIONAL_OBJECT_KEYS = {"level", "role"}
SCANNED_KEYS = {
    "model",
    "url",
    "sha256",
    "mesh",
    "rpy",
    "scale",
    "mass",
    "size",
    "source",
}
MEASURED_KEYS = {"model", "shape", "paint", "mass", "size", "source"}
MEASURED_SHAPES = {"revolve", "box", "scan", "profile"}
MEASURED_PAINTS = {"color", "side", "top", "bottom", *FACE_AXES}
DROP = 0.002  # m, spawn height above the resting pose
SIZE_TOLERANCE = 0.015  # m, model bounding box against the object list per axis
EDGE_MARGIN = 0.01  # m, footprint corners inside the furniture top
GAP = 0.01  # m, between the footprints of two objects on one surface
HEAD_ROOM = 0.01  # m, object top under the next board
FRICTION = 1.0


class SceneError(ValueError):
    """Invalid scene or model catalog."""


@dataclass(frozen=True)
class YcbModel:
    label: str
    model: str  # tmc_wrs_gz_worlds model directory name
    yaw: float  # rad, turn of the mesh inside the model
    directory: Path


@dataclass(frozen=True)
class SceneObject:
    name: str
    label: str
    category: str
    on: str  # furniture name or "floor"
    level: int  # furniture level (1 = lowest), 0 on the floor
    role: str
    position: tuple[float, float, float]  # world, the resting bounding box center
    yaw: float
    size: tuple[float, float, float]  # bounding box of the model

    def spawn_pose(self) -> tuple[float, float, float, float]:
        x, y, z = self.position
        return x, y, z + DROP, self.yaw


@dataclass(frozen=True)
class Scene:
    world: str
    objects: tuple[SceneObject, ...]

    def get(self, name: str) -> SceneObject | None:
        return next((o for o in self.objects if o.name == name), None)

    def by_role(self) -> dict[str, SceneObject]:
        return {o.role: o for o in self.objects if o.role}


# ---------------------------------------------------------------- YCB meshes


def _pose_values(element) -> list[float]:
    pose = element.find("pose")
    text = pose.text if pose is not None and pose.text else "0 0 0 0 0 0"
    values = [float(v) for v in text.split()]
    if len(values) != 6:
        raise SceneError(f"unsupported pose {text!r}")
    return values


def _stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise SceneError(f"expected a binary STL: {path}")
    rows = np.frombuffer(data, dtype=np.uint8, offset=84, count=50 * count)
    xyz = rows.reshape(count, 50)[:, 12:48].copy().view("<f4").reshape(-1, 3)
    return xyz.astype(np.float64)


def ycb_bounds(directory: Path, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """(center, size) of the collision meshes of a YCB model turned by yaw."""
    sdf = directory / "model-1_4.sdf"
    link = ET.parse(sdf).getroot().find("model/link")
    points = []
    for collision in link.findall("collision"):
        pose = _pose_values(collision)
        if any(abs(v) > 1e-9 for v in pose[3:]):
            raise SceneError(f"{sdf}: rotated collision mesh")
        uri = collision.findtext("geometry/mesh/uri", "").strip()
        prefix = f"model://{directory.name}/"
        if not uri.startswith(prefix):
            raise SceneError(f"{sdf}: unexpected mesh {uri!r}")
        points.append(_stl_vertices(directory / uri[len(prefix) :]) + pose[:3])
    if not points:
        raise SceneError(f"{sdf}: no collision mesh")
    turned = np.concatenate(points) @ yaw_rotation(yaw).T
    lower, upper = turned.min(axis=0), turned.max(axis=0)
    return (lower + upper) / 2, upper - lower


def ycb_sdf(model: YcbModel, name: str, pose) -> str:
    """The YCB model as ``name``: mesh turned by model.yaw, origin at the box center."""
    out = ET.parse(model.directory / "model-1_4.sdf").getroot().find("model")
    out.set("name", name)
    for old in out.findall("pose"):
        out.remove(old)
    pose_element = ET.Element("pose")
    pose_element.text = " ".join(f"{v:.6f}" for v in (*pose[:3], 0, 0, pose[3]))
    out.insert(0, pose_element)
    center, _ = ycb_bounds(model.directory, model.yaw)
    turn = yaw_rotation(model.yaw)
    link = out.find("link")
    parts = ("inertial", "visual", "collision")
    for element in (e for tag in parts for e in link.findall(tag)):
        values = _pose_values(element)
        xyz = turn @ np.array(values[:3]) - center
        old = element.find("pose")
        if old is None:
            old = ET.SubElement(element, "pose")
        old.text = " ".join(
            f"{v:.6f}" for v in (*xyz, values[3], values[4], values[5] + model.yaw)
        )
    for collision in link.findall("collision"):
        old_surface = collision.find("surface")
        if old_surface is not None:
            collision.remove(old_surface)
        collision.append(_surface())
    _file_uris(out, model.directory)
    return (
        '<?xml version="1.0"?><sdf version="1.8">'
        + ET.tostring(out, encoding="unicode")
        + "</sdf>"
    )


def _file_uris(sdf_model, directory: Path) -> None:
    """Point the model://<directory name>/ mesh URIs at the files themselves,
    so a model need not be on Gazebo's resource path (the scans are built
    into a cache at run time)."""
    prefix = f"model://{directory.name}/"
    for uri in sdf_model.iter("uri"):
        text = (uri.text or "").strip()
        if text.startswith(prefix):
            path = directory / text[len(prefix) :]
            uri.text = "file://" + urllib.parse.quote(str(path))


def _surface():
    return ET.fromstring(
        "<surface><friction><ode>"
        f"<mu>{FRICTION}</mu><mu2>{FRICTION}</mu2>"
        "</ode></friction><bounce><restitution_coefficient>0"
        "</restitution_coefficient></bounce></surface>"
    )


# ------------------------------------------------------------------- catalog


def _crop(value):
    if value is None:
        return None
    box = tuple(tuple(float(v) for v in corner) for corner in value)
    if len(box) != 2 or any(len(c) != 3 for c in box):
        raise SceneError(f"crop must be [[x, y, z], [x, y, z]]: {value!r}")
    return box


class Catalog:
    """Model of every simulated label; see the module docstring."""

    def __init__(self, models: dict, object_list):
        self.models = models  # label -> YcbModel | ScannedModel | MeasuredModel
        self.object_list = object_list

    @classmethod
    def load(cls, path, object_list, ycb_root) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != {"ycb", "scanned", "measured"}:
            raise SceneError(f"{path}: needs exactly 'ycb', 'scanned' and 'measured'")
        models = {}

        def add(label, model):
            if label in models:
                raise SceneError(f"{path}: {label!r} has two models")
            if object_list.get(label) is None:
                raise SceneError(f"{path}: {label!r} is not in the object list")
            models[label] = model

        for label, entry in data["ycb"].items():
            if set(entry) - {"model", "yaw"} or "model" not in entry:
                raise SceneError(f"{path}: ycb {label!r} needs 'model' (and 'yaw')")
            yaw = math.radians(float(entry.get("yaw", 0.0)))
            directory = Path(ycb_root) / entry["model"]
            add(label, YcbModel(label, entry["model"], yaw, directory))
        for label, entry in data["scanned"].items():
            if not SCANNED_KEYS <= set(entry) <= SCANNED_KEYS | {"crop"}:
                raise SceneError(
                    f"{path}: scanned {label!r} needs {sorted(SCANNED_KEYS)} (and 'crop')"
                )
            rpy = tuple(math.radians(float(v)) for v in entry["rpy"])
            size = tuple(float(v) for v in entry["size"])
            if len(rpy) != 3 or len(size) != 3 or not float(entry["mass"]) > 0:
                raise SceneError(f"{path}: scanned {label!r}: rpy, size or mass")
            model = ScannedModel(
                label,
                str(entry["model"]),
                str(entry["url"]),
                str(entry["sha256"]),
                str(entry["mesh"]),
                rpy,
                float(entry["scale"]),
                _crop(entry.get("crop")),
                float(entry["mass"]),
                size,
                str(entry["source"]),
            )
            add(label, model)
        for label, entry in data["measured"].items():
            if set(entry) != MEASURED_KEYS or len(entry["shape"]) != 1:
                raise SceneError(
                    f"{path}: measured {label!r} needs {sorted(MEASURED_KEYS)}"
                )
            if not set(entry["shape"]) <= MEASURED_SHAPES:
                raise SceneError(f"{path}: measured {label!r}: unknown shape")
            if not set(entry["paint"]) <= MEASURED_PAINTS:
                raise SceneError(f"{path}: measured {label!r}: unknown paint")
            model = MeasuredModel(
                label,
                str(entry["model"]),
                entry["shape"],
                entry["paint"],
                float(entry["mass"]),
                tuple(float(v) for v in entry["size"]),
                str(entry["source"]),
            )
            add(label, model)
        return cls(models, object_list)

    def size(self, label: str) -> tuple[float, float, float]:
        """Bounding box x, y, z (m) of the label's model."""
        model = self.models.get(label)
        if model is None:
            raise SceneError(f"no simulated model for {label!r}")
        if isinstance(model, YcbModel):
            return tuple(float(v) for v in ycb_bounds(model.directory, model.yaw)[1])
        return model.size

    def sdf(self, obj: SceneObject, cache: Path) -> str:
        """The SDF of the object at its spawn pose; scans and measured models
        are built into ``cache`` first."""
        model = self.models[obj.label]
        if not isinstance(model, YcbModel):
            build = build_scanned if isinstance(model, ScannedModel) else build_measured
            model = YcbModel(model.label, model.model, 0.0, build(model, cache))
        return ycb_sdf(model, obj.name, obj.spawn_pose())


# --------------------------------------------------------------------- scene


def footprint(obj: SceneObject) -> np.ndarray:
    """(4, 2) world corners of the object's bounding box seen from above."""
    hx, hy = obj.size[0] / 2, obj.size[1] / 2
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    c, s = math.cos(obj.yaw), math.sin(obj.yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array(obj.position[:2])


def footprints_overlap(a: np.ndarray, b: np.ndarray, gap: float = 0.0) -> bool:
    """Whether two convex polygons come closer than ``gap`` (separating axes)."""
    for polygon in (a, b):
        for k in range(len(polygon)):
            edge = polygon[(k + 1) % len(polygon)] - polygon[k]
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            pa, pb = a @ axis, b @ axis
            if pa.max() + gap <= pb.min() or pb.max() + gap <= pa.min():
                return False
    return True


def _resolve(item, k, path, catalog, furniture) -> SceneObject:
    what = f"{path}: objects[{k}]"
    if not isinstance(item, dict):
        raise SceneError(f"{what}: expected a mapping")
    unknown = set(item) - OBJECT_KEYS - OPTIONAL_OBJECT_KEYS
    missing = OBJECT_KEYS - set(item)
    if unknown or missing:
        raise SceneError(
            f"{what}: unknown {sorted(unknown)}, missing {sorted(missing)}"
        )
    name, label, on = str(item["name"]), str(item["label"]), str(item["on"])
    what = f"{path}: {name}"
    listed = catalog.object_list.get(label)
    if listed is None:
        raise SceneError(f"{what}: label {label!r} is not in the object list")
    detected = catalog.object_list.for_model(name)
    if detected is None or detected.label != label:
        raise SceneError(
            f"{what}: the detector would report {detected.label if detected else None!r}"
            f" for this model name, not {label!r}"
        )
    size = catalog.size(label)
    position = item["position"]
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
    ):
        raise SceneError(f"{what}: position must be [x, y]")
    yaw = float(item["yaw"])
    if not math.isfinite(yaw):
        raise SceneError(f"{what}: yaw must be finite")
    if on == FLOOR:
        if "level" in item:
            raise SceneError(f"{what}: no level on the floor")
        level, height = 0, 0.0
    else:
        place = furniture.get(on)
        if place is None:
            raise SceneError(f"{what}: unknown furniture {on!r}")
        level = int(item.get("level", 1 if len(place.heights) == 1 else 0))
        surface = place.level(level)
        if surface is None:
            raise SceneError(f"{what}: {on} needs a level 1-{len(place.heights)}")
        height = surface.height
    return SceneObject(
        name,
        label,
        listed.category,
        on,
        level,
        str(item.get("role", "")),
        (float(position[0]), float(position[1]), height + size[2] / 2),
        yaw,
        size,
    )


def check_placement(scene: Scene, furniture: dict) -> None:
    """SceneError if an object hangs over an edge, hits the next board or
    another object, or a floor object stands on a furniture footprint."""
    for obj in scene.objects:
        corners = footprint(obj)
        if obj.on == FLOOR:
            for place in furniture.values():
                if footprints_overlap(corners, place.corners(), GAP):
                    raise SceneError(f"{obj.name}: on the floor under {place.name}")
            continue
        place = furniture[obj.on]
        if not np.all(place.contains(corners, -EDGE_MARGIN)):
            raise SceneError(f"{obj.name}: over the edge of {obj.on}")
        room = place.level(obj.level).max_object_height
        if room is not None and obj.size[2] > room - HEAD_ROOM:
            raise SceneError(
                f"{obj.name}: {obj.size[2]:.3f} m high, {obj.on} level {obj.level}"
                f" has {room:.3f} m"
            )
    for k, a in enumerate(scene.objects):
        for b in scene.objects[k + 1 :]:
            if (a.on, a.level) != (b.on, b.level):
                continue
            if footprints_overlap(footprint(a), footprint(b), GAP):
                raise SceneError(f"{a.name} and {b.name} overlap")


def furniture_config(directory, scene_path) -> Path:
    """The furniture config of the scene's world: ``<directory>/<world>.yaml``
    (aha_perception config/furniture); SceneError if the world has none."""
    data = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    world = data.get("world") if isinstance(data, dict) else None
    path = Path(directory) / f"{world}.yaml"
    if not isinstance(world, str) or not world or not path.is_file():
        raise SceneError(f"{scene_path}: no furniture config for the world {world!r}")
    return path


def load_scene(path, catalog: Catalog, furniture: dict) -> Scene:
    """The resolved objects of a scene file; SceneError if it is invalid.

    furniture: name -> aha_perception.furniture.Furniture, in the world frame."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != SCENE_KEYS:
        raise SceneError(f"{path}: needs exactly {sorted(SCENE_KEYS)}")
    if not isinstance(data["objects"], list) or not data["objects"]:
        raise SceneError(f"{path}: objects must be a non-empty list")
    objects = tuple(
        _resolve(item, k, path, catalog, furniture)
        for k, item in enumerate(data["objects"])
    )
    for kind, values in (
        ("names", [o.name for o in objects]),
        ("roles", [o.role for o in objects if o.role]),
    ):
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            raise SceneError(f"{path}: duplicate {kind} {duplicates}")
    scene = Scene(str(data["world"]), objects)
    check_placement(scene, furniture)
    return scene


def stale_models(names, scene: Scene, object_list) -> list[str]:
    """Gazebo model names that look like objects but are not in the scene.

    A name counts as an object when it matches a pattern of the object list
    (sim_object_detector.py would report it), e.g. the objects of another
    scene placed earlier in the same Gazebo. Other models (furniture, the
    robot, guests, people) never match."""
    wanted = {o.name for o in scene.objects}
    return sorted(
        name
        for name in names
        if name and name not in wanted and object_list.for_model(name) is not None
    )


def displacement(obj: SceneObject, pose: np.ndarray) -> tuple[float, float]:
    """(distance m, tilt rad) of a Gazebo world pose (4x4) from the resting pose."""
    distance = float(np.linalg.norm(pose[:3, 3] - np.array(obj.position)))
    tilt = math.acos(max(-1.0, min(1.0, float(pose[2, 2]))))
    return distance, tilt
