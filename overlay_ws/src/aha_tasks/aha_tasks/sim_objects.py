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
- ``printed``: a box or upright cylinder of the object list's shape and size
  with a generated texture (colours and the ``text`` printed on its sides),
  for objects without a YCB model.

Every model's origin is its bounding box center, as sim_object_detector.py
assumes. YCB models must be within ``SIZE_TOLERANCE`` of the object list's
size, printed ones have it exactly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

FLOOR = "floor"
SCENE_KEYS = {"world", "objects"}
OBJECT_KEYS = {"name", "label", "on", "position", "yaw"}
OPTIONAL_OBJECT_KEYS = {"level", "role"}
STYLE_KEYS = {"text", "body", "band", "ink", "cap", "mass"}
DROP = 0.002  # m, spawn height above the resting pose
SIZE_TOLERANCE = 0.015  # m, YCB bounding box against the object list per axis
EDGE_MARGIN = 0.01  # m, footprint corners inside the furniture top
GAP = 0.01  # m, between the footprints of two objects on one surface
HEAD_ROOM = 0.01  # m, object top under the next board
FRICTION = 1.0
CYLINDER_SEGMENTS = 48
GENERATOR_VERSION = 2  # raise when generated printed models change
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


class SceneError(ValueError):
    """Invalid scene or model catalog."""


@dataclass(frozen=True)
class Geometry:
    shape: str  # box | cylinder (printed); the YCB models are meshes
    size: tuple[float, float, float]  # bounding box x, y, z (m)
    mass: float  # kg


@dataclass(frozen=True)
class YcbModel:
    label: str
    model: str  # tmc_wrs_gz_worlds model directory name
    yaw: float  # rad, turn of the mesh inside the model
    directory: Path


@dataclass(frozen=True)
class PrintedModel:
    label: str
    shape: str
    size: tuple[float, float, float]
    style: dict


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

    def truth(self) -> dict:
        """Ground truth as written to logs and read by evaluations."""
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "on": self.on,
            "level": self.level,
            "role": self.role,
            "position": [round(v, 4) for v in self.position],
            "yaw": self.yaw,
            "size": [round(v, 4) for v in self.size],
        }


@dataclass(frozen=True)
class Scene:
    world: str
    objects: tuple[SceneObject, ...]

    def get(self, name: str) -> SceneObject | None:
        return next((o for o in self.objects if o.name == name), None)

    def by_role(self) -> dict[str, SceneObject]:
        return {o.role: o for o in self.objects if o.role}


# ---------------------------------------------------------------- YCB meshes


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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
    turned = np.concatenate(points) @ _rotation(yaw).T
    lower, upper = turned.min(axis=0), turned.max(axis=0)
    return (lower + upper) / 2, upper - lower


def ycb_sdf(model: YcbModel, name: str, pose) -> str:
    """The YCB model as ``name``: mesh turned by model.yaw, origin at the box center."""
    source = ET.parse(model.directory / "model-1_4.sdf").getroot().find("model")
    out = deepcopy(source)
    out.set("name", name)
    for old in out.findall("pose"):
        out.remove(old)
    pose_element = ET.Element("pose")
    pose_element.text = " ".join(f"{v:.6f}" for v in (*pose[:3], 0, 0, pose[3]))
    out.insert(0, pose_element)
    center, _ = ycb_bounds(model.directory, model.yaw)
    turn = _rotation(model.yaw)
    link = out.find("link")
    elements = [link.find("inertial"), *link.findall("visual")]
    elements += link.findall("collision")
    for element in elements:
        if element is None:
            continue
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
    return (
        '<?xml version="1.0"?><sdf version="1.8">'
        + ET.tostring(out, encoding="unicode")
        + "</sdf>"
    )


def _surface():
    surface = ET.fromstring(
        "<surface><friction><ode>"
        f"<mu>{FRICTION}</mu><mu2>{FRICTION}</mu2>"
        "</ode></friction><bounce><restitution_coefficient>0"
        "</restitution_coefficient></bounce></surface>"
    )
    return surface


# ------------------------------------------------------------ printed models


def _colour(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise SceneError(f"colour must be #rrggbb: {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _font(size: int):
    from PIL import ImageFont

    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit_text(draw, text: str, width: float, height: float):
    """(font, lines) of the largest font that fits ``text`` into a box."""
    words = text.split()
    candidates = [[text]]
    for split in range(1, len(words)):
        candidates.append([" ".join(words[:split]), " ".join(words[split:])])
    best = None
    for lines in candidates:
        low, high = 4, max(int(height), 5)
        while low < high:
            size = (low + high + 1) // 2
            font = _font(size)
            boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
            w = max(b[2] - b[0] for b in boxes)
            h = sum(b[3] - b[1] for b in boxes) * 1.25
            if w <= width and h <= height:
                low = size
            else:
                high = size - 1
        if best is None or low > best[0]:
            best = (low, lines)
    return _font(best[0]), best[1]


def _draw_text(draw, text, colour, box):
    if not text:
        return
    x0, y0, x1, y1 = box
    font, lines = _fit_text(draw, text, (x1 - x0) * 0.86, (y1 - y0) * 0.9)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    step = max(b[3] - b[1] for b in boxes) * 1.25
    y = (y0 + y1) / 2 - step * len(lines) / 2
    for line, b in zip(lines, boxes):
        draw.text(
            (
                (x0 + x1) / 2 - (b[2] - b[0]) / 2 - b[0],
                y + (step - (b[3] - b[1])) / 2 - b[1],
            ),
            line,
            fill=colour,
            font=font,
        )
        y += step


def printed_layout(shape: str, size) -> dict:
    """Texture layout (pixels) of a printed model: the side wrap, then the cap."""
    sx, sy, sz = size
    if shape == "cylinder":
        diameter = max(sx, sy)
        perimeter = math.pi * diameter
        cap = (diameter, diameter)
        faces = []
    else:
        perimeter = 2 * (sx + sy)
        cap = (sx, sy)
        faces = [sy, sx, sy, sx]  # +x, +y, -x, -y faces, counterclockwise
    ppm = min(2500.0, 1024.0 / max(perimeter, sz + cap[1], cap[0]))
    side_w, side_h = round(perimeter * ppm), max(round(sz * ppm), 8)
    cap_w, cap_h = max(round(cap[0] * ppm), 8), max(round(cap[1] * ppm), 8)
    return {
        "ppm": ppm,
        "width": max(side_w, cap_w),
        "height": side_h + cap_h,
        "side": (side_w, side_h),
        "cap": (cap_w, cap_h),
        "faces": [round(f * ppm) for f in faces],
    }


def printed_texture(model: PrintedModel):
    """PIL image of the printed model's texture (see printed_layout)."""
    from PIL import Image, ImageDraw

    style = model.style
    layout = printed_layout(model.shape, model.size)
    width, height = layout["width"], layout["height"]
    side_w, side_h = layout["side"]
    cap_w, cap_h = layout["cap"]
    image = Image.new("RGB", (width, height), _colour(style["cap"]))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, side_h), fill=_colour(style["body"]))
    band = _colour(style["band"])
    draw.rectangle((0, 0, width, round(side_h * 0.14)), fill=band)
    draw.rectangle((0, side_h - round(side_h * 0.1), width, side_h), fill=band)
    ink = _colour(style["ink"])
    text = style["text"]
    middle = (round(side_h * 0.14), side_h - round(side_h * 0.1))
    if model.shape == "cylinder":
        for k in range(2):
            x0 = side_w * k / 2
            _draw_text(draw, text, ink, (x0, middle[0], x0 + side_w / 2, middle[1]))
    else:
        x0 = 0
        for face in layout["faces"]:
            _draw_text(draw, text, ink, (x0, middle[0], x0 + face, middle[1]))
            x0 += face
    flat = model.size[2] < min(model.size[:2])
    if flat and model.shape == "box":
        top = (0, side_h, cap_w, side_h + cap_h)
        draw.rectangle(top, fill=_colour(style["body"]))
        draw.rectangle(top, outline=band, width=max(cap_h // 8, 1))
        _draw_text(draw, text, ink, top)
    return image


def printed_obj(model: PrintedModel, name: str) -> str:
    """Wavefront OBJ of the printed model, centered, UVs into printed_texture;
    its material and texture are ``<name>.mtl`` / ``<name>.png`` next to it."""
    layout = printed_layout(model.shape, model.size)
    width, height = layout["width"], layout["height"]
    side_w, side_h = layout["side"]
    cap_w, cap_h = layout["cap"]
    sx, sy, sz = model.size
    vertices, uvs, normals, faces = [], [], [], []

    def uv(px, py):  # pixel -> OBJ texture coordinate (v up)
        return px / width, 1.0 - py / height

    def add(points, coords, normal, triangles):
        base_v, base_t, base_n = len(vertices), len(uvs), len(normals)
        vertices.extend(points)
        uvs.extend(coords)
        normals.append(normal)
        for tri in triangles:
            faces.append([(base_v + i + 1, base_t + i + 1, base_n + 1) for i in tri])

    def cap_uv(x, y, extent_x, extent_y):
        return uv((x / extent_x + 0.5) * cap_w, side_h + (0.5 - y / extent_y) * cap_h)

    if model.shape == "cylinder":
        r = max(sx, sy) / 2
        n = CYLINDER_SEGMENTS
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            mid = (a0 + a1) / 2
            p = [
                (r * math.cos(a0), r * math.sin(a0), -sz / 2),
                (r * math.cos(a1), r * math.sin(a1), -sz / 2),
                (r * math.cos(a1), r * math.sin(a1), sz / 2),
                (r * math.cos(a0), r * math.sin(a0), sz / 2),
            ]
            u0, u1 = side_w * i / n, side_w * (i + 1) / n
            t = [uv(u0, side_h), uv(u1, side_h), uv(u1, 0), uv(u0, 0)]
            add(p, t, (math.cos(mid), math.sin(mid), 0.0), [(0, 1, 2), (0, 2, 3)])
        for z, normal in ((sz / 2, 1.0), (-sz / 2, -1.0)):
            ring = [
                (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
                for i in range(n)
            ]
            points = [(0.0, 0.0, z)] + [(x, y, z) for x, y in ring]
            coords = [cap_uv(0, 0, 2 * r, 2 * r)] + [
                cap_uv(x, y, 2 * r, 2 * r) for x, y in ring
            ]
            tris = [
                (0, 1 + i, 1 + (i + 1) % n)
                if normal > 0
                else (0, 1 + (i + 1) % n, 1 + i)
                for i in range(n)
            ]
            add(points, coords, (0.0, 0.0, normal), tris)
    else:
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        # Counterclockwise from above: start and end corner of each side face.
        sides = [
            ((hx, -hy), (hx, hy), (1.0, 0.0)),
            ((hx, hy), (-hx, hy), (0.0, 1.0)),
            ((-hx, hy), (-hx, -hy), (-1.0, 0.0)),
            ((-hx, -hy), (hx, -hy), (0.0, -1.0)),
        ]
        u0 = 0
        for (start, end, normal), face in zip(sides, layout["faces"]):
            p = [(*start, -hz), (*end, -hz), (*end, hz), (*start, hz)]
            t = [uv(u0, side_h), uv(u0 + face, side_h), uv(u0 + face, 0), uv(u0, 0)]
            add(p, t, (*normal, 0.0), [(0, 1, 2), (0, 2, 3)])
            u0 += face
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        for z, normal in ((hz, 1.0), (-hz, -1.0)):
            points = [(x, y, z) for x, y in corners]
            coords = [cap_uv(x, y, sx, sy) for x, y in corners]
            tris = [(0, 1, 2), (0, 2, 3)] if normal > 0 else [(0, 2, 1), (0, 3, 2)]
            add(points, coords, (0.0, 0.0, normal), tris)
    lines = [
        "# generated by aha_tasks.sim_objects",
        f"mtllib {name}.mtl",
        f"usemtl {name}",
    ]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    lines += ["f " + " ".join(f"{a}/{b}/{c}" for a, b, c in face) for face in faces]
    return "\n".join(lines) + "\n"


# Gazebo keeps a mesh material by its name across models: every printed
# model has its own material and texture names.
MTL = """newmtl {name}
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.050 0.050 0.050
Ns 10.0
d 1.0
illum 2
map_Kd {name}.png
"""


def printed_name(model: PrintedModel) -> str:
    """Name of the generated files and the material: label and a style hash."""
    data = json.dumps(
        [GENERATOR_VERSION, model.shape, model.size, model.style], sort_keys=True
    )
    key = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"{model.label.replace(' ', '_')}_{key}"


def build_printed(model: PrintedModel, cache: Path) -> Path:
    """Write (once) the mesh, material and texture of a printed model; the OBJ path."""
    name = printed_name(model)
    directory = cache / name
    mesh = directory / f"{name}.obj"
    if mesh.is_file() and (directory / f"{name}.png").is_file():
        return mesh
    directory.mkdir(parents=True, exist_ok=True)
    printed_texture(model).save(directory / f"{name}.png")
    (directory / f"{name}.mtl").write_text(MTL.format(name=name))
    temporary = directory / f"{name}.obj.tmp"
    temporary.write_text(printed_obj(model, name))
    temporary.replace(mesh)
    return mesh


def _inertia(shape: str, size, mass: float) -> tuple[float, float, float]:
    x, y, z = size
    if shape == "cylinder":
        r = max(x, y) / 2
        side = mass * (3 * r * r + z * z) / 12
        return side, side, mass * r * r / 2
    return (
        mass * (y * y + z * z) / 12,
        mass * (x * x + z * z) / 12,
        mass * (x * x + y * y) / 12,
    )


def printed_sdf(model: PrintedModel, name: str, pose, mesh: Path) -> str:
    """A box or cylinder collision with the generated mesh as its visual."""
    sx, sy, sz = model.size
    if model.shape == "cylinder":
        geometry = (
            f"<cylinder><radius>{max(sx, sy) / 2}</radius>"
            f"<length>{sz}</length></cylinder>"
        )
    else:
        geometry = f"<box><size>{sx} {sy} {sz}</size></box>"
    mass = float(model.style["mass"])
    ixx, iyy, izz = _inertia(model.shape, model.size, mass)
    uri = "file://" + urllib.parse.quote(str(mesh))
    x, y, z, yaw = pose
    surface = ET.tostring(_surface(), encoding="unicode")
    return (
        '<?xml version="1.0"?><sdf version="1.8">'
        f'<model name="{name}"><pose>{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.6f}</pose>'
        '<link name="link">'
        f"<inertial><mass>{mass}</mass><inertia><ixx>{ixx}</ixx><ixy>0</ixy>"
        f"<ixz>0</ixz><iyy>{iyy}</iyy><iyz>0</iyz><izz>{izz}</izz></inertia>"
        "</inertial>"
        f'<collision name="collision"><geometry>{geometry}</geometry>{surface}'
        "</collision>"
        f'<visual name="visual"><geometry><mesh><uri>{uri}</uri></mesh>'
        "</geometry></visual></link></model></sdf>"
    )


# ------------------------------------------------------------------- catalog


class Catalog:
    """Model of every simulated label; see the module docstring."""

    def __init__(self, models: dict, object_list):
        self.models = models  # label -> YcbModel | PrintedModel
        self.object_list = object_list

    @classmethod
    def load(cls, path, object_list, ycb_root) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != {"ycb", "printed"}:
            raise SceneError(f"{path}: needs exactly 'ycb' and 'printed'")
        models = {}
        for label, entry in data["ycb"].items():
            if set(entry) - {"model", "yaw"} or "model" not in entry:
                raise SceneError(f"{path}: ycb {label!r} needs 'model' (and 'yaw')")
            directory = Path(ycb_root) / entry["model"]
            models[label] = YcbModel(
                label,
                entry["model"],
                math.radians(float(entry.get("yaw", 0.0))),
                directory,
            )
        for label, style in data["printed"].items():
            if set(style) != STYLE_KEYS:
                raise SceneError(
                    f"{path}: printed {label!r} needs exactly {sorted(STYLE_KEYS)}"
                )
            for key in ("body", "band", "ink", "cap"):
                _colour(style[key])
            if not float(style["mass"]) > 0:
                raise SceneError(f"{path}: printed {label!r} mass must be positive")
            if label in models:
                raise SceneError(f"{path}: {label!r} is both ycb and printed")
            entry = object_list.get(label)
            if entry is None:
                raise SceneError(f"{path}: {label!r} is not in the object list")
            models[label] = PrintedModel(label, entry.shape, entry.size, dict(style))
        for label in models:
            if object_list.get(label) is None:
                raise SceneError(f"{path}: {label!r} is not in the object list")
        return cls(models, object_list)

    def geometry(self, label: str) -> Geometry:
        model = self.models.get(label)
        if model is None:
            raise SceneError(f"no simulated model for {label!r}")
        if isinstance(model, PrintedModel):
            return Geometry(model.shape, model.size, float(model.style["mass"]))
        _, size = ycb_bounds(model.directory, model.yaw)
        mass = float(
            ET.parse(model.directory / "model-1_4.sdf")
            .getroot()
            .findtext("model/link/inertial/mass", "0")
        )
        return Geometry("mesh", tuple(float(v) for v in size), mass)

    def size_errors(self) -> dict[str, float]:
        """Largest per-axis difference (m) of each model to the object list's size."""
        out = {}
        for label in self.models:
            size = np.array(self.geometry(label).size)
            listed = np.array(self.object_list.get(label).size)
            out[label] = float(np.max(np.abs(size - listed)))
        return out

    def sdf(self, obj: SceneObject, cache: Path) -> str:
        model = self.models[obj.label]
        pose = obj.spawn_pose()
        if isinstance(model, YcbModel):
            return ycb_sdf(model, obj.name, pose)
        return printed_sdf(model, obj.name, pose, build_printed(model, cache))


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
    geometry = catalog.geometry(label)
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
    z = height + geometry.size[2] / 2
    return SceneObject(
        name,
        label,
        listed.category,
        on,
        level,
        str(item.get("role", "")),
        (float(position[0]), float(position[1]), z),
        yaw,
        geometry.size,
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
    names = [o.name for o in objects]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SceneError(f"{path}: duplicate names {duplicates}")
    roles = [o.role for o in objects if o.role]
    duplicates = sorted({r for r in roles if roles.count(r) > 1})
    if duplicates:
        raise SceneError(f"{path}: duplicate roles {duplicates}")
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
