"""Synthetic organized depth scenes with per-pixel ground-truth planes.

Analytic boxes, cylinders and rectangles are ray-cast from a head camera pose
into a D435i-like depth image, then corrupted with the stereo noise model of
``aha_perception.depth_noise``.
Coordinates are in the robot base frame (x forward, y left, z up, floor z=0).
"""

from __future__ import annotations

import pathlib
import sys
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import label as connected_label

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PERCEPTION = _REPO / "overlay_ws" / "src" / "aha_perception"
if str(_PERCEPTION) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION))

# The noise model and presets are re-exported for the bench scripts.
from aha_perception.depth_noise import NOISE_PRESETS as NOISE_PRESETS  # noqa: E402
from aha_perception.depth_noise import FrozenNoise as FrozenNoise  # noqa: E402
from aha_perception.depth_noise import NoiseParams, corrupt_depth  # noqa: E402
from aha_perception.depth_noise import noise_preset as noise_preset  # noqa: E402

INVALID = -3
DONT_CARE = -2
NON_PLANAR = -1
_EIGHT = np.ones((3, 3), dtype=bool)
_FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


@dataclass(frozen=True)
class Intrinsics:
    width: int = 640
    height: int = 480
    fx: float = 465.7
    fy: float = 465.7
    cx: float = 319.5
    cy: float = 239.5

    def rays(self) -> np.ndarray:
        """Optical-frame rays with unit z, so ray parameter equals depth."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return np.stack(
            [(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones(u.shape)], -1
        )


@dataclass(frozen=True)
class CameraPose:
    """Head camera pose; positive tilt looks down, positive pan looks left."""

    x: float = 0.0
    y: float = 0.0
    height: float = 1.43
    pan_deg: float = 0.0
    tilt_deg: float = 30.0

    def camera_to_base(self) -> np.ndarray:
        optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        t, p = np.radians(self.tilt_deg), np.radians(self.pan_deg)
        pitch = np.array(
            [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
        )
        yaw = np.array(
            [[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]]
        )
        transform = np.eye(4)
        transform[:3, :3] = yaw @ pitch @ optical
        transform[:3, 3] = (self.x, self.y, self.height)
        return transform


@dataclass
class Quad:
    """Rectangle ``origin + a*edge_u + b*edge_v`` with a, b in [0, 1]."""

    origin: tuple
    edge_u: tuple
    edge_v: tuple
    tag: object = DONT_CARE


@dataclass
class Box:
    """Box with optional yaw; ``faces`` maps '+z', '-x', ... to tags."""

    center: tuple
    size: tuple
    yaw_deg: float = 0.0
    faces: dict = field(default_factory=dict)


@dataclass
class Cylinder:
    """Vertical cylinder; the side is non-planar, the top cap uses ``top``."""

    center_xy: tuple
    radius: float
    z0: float
    z1: float
    top: object = DONT_CARE


@dataclass
class Scene:
    name: str
    tags: tuple
    pose: CameraPose
    surfaces: dict
    primitives: list


@dataclass
class GtPlane:
    surface: str
    cls: str
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    pixels: int
    area: float


@dataclass
class SceneFrame:
    name: str
    tags: tuple
    noise_scale: float
    cloud: np.ndarray
    labels: np.ndarray
    planes: list
    camera_to_base: np.ndarray
    intrinsics: Intrinsics


def _hit_quad(quad: Quad, origin, rays):
    p0, u, v = (
        np.asarray(a, dtype=np.float64) for a in (quad.origin, quad.edge_u, quad.edge_v)
    )
    normal = np.cross(u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0 - origin) @ normal) / (rays @ normal)
    rel = origin + t[..., None] * rays - p0
    a = rel @ u / (u @ u)
    b = rel @ v / (v @ v)
    ok = (t > 1e-6) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    return np.where(ok, t, np.inf), np.zeros(t.shape, dtype=np.int64)


def _hit_box(box: Box, origin, rays):
    yaw = np.radians(box.yaw_deg)
    inv = np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    o = inv @ (origin - np.asarray(box.center, dtype=np.float64))
    d = rays @ inv.T
    half = np.asarray(box.size, dtype=np.float64) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    parallel = d == 0
    inside = np.abs(o) <= half
    t1 = np.where(parallel, np.where(inside, -np.inf, np.inf), t1)
    t2 = np.where(parallel, np.where(inside, np.inf, np.inf), t2)
    near = np.minimum(t1, t2)
    far = np.maximum(t1, t2)
    t_near = near.max(axis=-1)
    t_far = far.min(axis=-1)
    ok = (t_far >= t_near) & (t_near > 1e-6)
    axis = near.argmax(axis=-1)
    positive = np.take_along_axis(d, axis[..., None], -1)[..., 0] < 0
    face = axis * 2 + positive.astype(np.int64)
    return np.where(ok, t_near, np.inf), face


def _hit_cylinder(cyl: Cylinder, origin, rays):
    center = np.asarray(cyl.center_xy, dtype=np.float64)
    dx = origin[:2] - center
    dxy = rays[..., :2]
    a = (dxy**2).sum(-1)
    b = 2 * (dxy @ dx)
    c = dx @ dx - cyl.radius**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_side = (-b - np.sqrt(disc)) / (2 * a)
        z_side = origin[2] + t_side * rays[..., 2]
        side_ok = (
            (disc >= 0) & (t_side > 1e-6) & (z_side >= cyl.z0) & (z_side <= cyl.z1)
        )
        t_top = (cyl.z1 - origin[2]) / rays[..., 2]
    top_xy = origin[:2] + t_top[..., None] * dxy - center
    top_ok = (t_top > 1e-6) & ((top_xy**2).sum(-1) <= cyl.radius**2)
    t_side = np.where(side_ok, t_side, np.inf)
    t_top = np.where(top_ok, t_top, np.inf)
    return np.minimum(t_side, t_top), (t_top < t_side).astype(np.int64)


def _codes(scene: Scene, primitive) -> np.ndarray:
    """Per-face integer codes: surface index, DONT_CARE or NON_PLANAR."""
    names = list(scene.surfaces)

    def code(tag):
        return names.index(tag) if isinstance(tag, str) else int(tag)

    if isinstance(primitive, Quad):
        return np.array([code(primitive.tag)])
    if isinstance(primitive, Box):
        return np.array([code(primitive.faces.get(f, DONT_CARE)) for f in _FACES])
    return np.array([NON_PLANAR, code(primitive.top)])


def raycast_faces(scene: Scene, intrinsics: Intrinsics):
    """Noise-free depth and, per pixel, the hit primitive index and face (-1 = none).

    Faces index ``_FACES`` for a ``Box``, 0 for a ``Quad``, and 0 (side) / 1 (top)
    for a ``Cylinder``.
    """
    transform = scene.pose.camera_to_base()
    rays = intrinsics.rays() @ transform[:3, :3].T
    origin = transform[:3, 3]
    depth = np.full(rays.shape[:2], np.inf)
    index = np.full(rays.shape[:2], -1, dtype=np.int64)
    faces = np.full(rays.shape[:2], -1, dtype=np.int64)
    hitters = {Quad: _hit_quad, Box: _hit_box, Cylinder: _hit_cylinder}
    for k, primitive in enumerate(scene.primitives):
        t, face = hitters[type(primitive)](primitive, origin, rays)
        closer = t < depth
        depth = np.where(closer, t, depth)
        index = np.where(closer, k, index)
        faces = np.where(closer, face, faces)
    return depth, index, faces


def raycast(scene: Scene, intrinsics: Intrinsics):
    """Noise-free depth (optical z, inf = no hit) and per-pixel surface codes."""
    depth, index, faces = raycast_faces(scene, intrinsics)
    codes = np.full(depth.shape, NON_PLANAR, dtype=np.int64)
    for k, primitive in enumerate(scene.primitives):
        hit = index == k
        codes[hit] = _codes(scene, primitive)[faces[hit]]
    return depth, codes


def corrupt(depth, codes, noise: NoiseParams, rng: np.random.Generator):
    """Apply the stereo noise model; corrupted pixels are relabeled."""
    z, codes, _ = corrupt_frame(depth, codes, noise, rng)
    return z, codes


def corrupt_frame(
    depth,
    codes,
    noise: NoiseParams,
    rng: np.random.Generator,
    frozen: FrozenNoise | None = None,
):
    """``corrupt`` that also returns, or reuses, the frozen components.

    With ``frozen`` None the frozen components are drawn from ``rng`` in the
    same order as the single-frame model.
    """
    result = corrupt_depth(depth, noise, rng, frozen)
    codes = codes.copy()
    codes[result.flying | result.outlier] = NON_PLANAR
    codes[np.isnan(result.depth)] = INVALID
    return result.depth, codes, result.frozen


def _instances(
    codes, depth_true, rays_cam, transform, intrinsics, scene, min_pixels, min_area
):
    """Split each surface into image-connected GT plane instances."""
    labels = np.where(codes >= 0, DONT_CARE, codes).astype(np.int32)
    points = (
        rays_cam * np.where(np.isfinite(depth_true), depth_true, 0)[..., None]
    ) @ (transform[:3, :3].T) + transform[:3, 3]
    origin = transform[:3, 3]
    classes = list(scene.surfaces.values())
    names = list(scene.surfaces)
    planes = []
    for index in np.unique(codes[codes >= 0]):
        comp, count = connected_label(codes == index, structure=_EIGHT)
        for c in range(1, count + 1):
            mask = comp == c
            if mask.sum() < min_pixels:
                continue
            pts = points[mask]
            centroid = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
            normal = vt[2]
            offset = -float(normal @ centroid)
            if normal @ origin + offset < 0:
                normal, offset = -normal, -offset
            # Pixel footprint on the plane: z^2 / (fx fy |r . n|) for rays with r_z = 1.
            z = depth_true[mask]
            r_dot_n = np.abs((pts - origin) @ normal) / z
            footprint = z**2 / (
                intrinsics.fx * intrinsics.fy * np.maximum(r_dot_n, 0.05)
            )
            area = float(footprint.sum())
            if area < min_area:
                continue
            labels[mask] = len(planes)
            planes.append(
                GtPlane(
                    surface=names[index],
                    cls=classes[index],
                    normal=normal,
                    offset=offset,
                    centroid=centroid,
                    pixels=int(mask.sum()),
                    area=area,
                )
            )
    return labels, planes


def generate(
    scene: Scene,
    noise_scale: float = 1.0,
    seed: int = 0,
    intrinsics: Intrinsics | None = None,
    noise: NoiseParams | None = None,
    eval_range: tuple[float, float] = (0.3, 4.0),
    min_pixels: int = 400,
    min_area: float = 0.02,
) -> SceneFrame:
    return generate_sequence(
        scene,
        1,
        noise,
        seed,
        intrinsics,
        noise_scale,
        eval_range=eval_range,
        min_pixels=min_pixels,
        min_area=min_area,
    )[0]


def generate_sequence(
    scene: Scene,
    n_frames: int,
    noise: NoiseParams | None = None,
    seed: int = 0,
    intrinsics: Intrinsics | None = None,
    noise_scale: float = 1.0,
    eval_range: tuple[float, float] = (0.3, 4.0),
    min_pixels: int = 400,
    min_area: float = 0.02,
) -> list[SceneFrame]:
    """Frames of a static camera; the first equals ``generate`` with ``seed``.

    Frozen noise components are drawn with the first frame and reused; GT
    labels follow each frame's validity as in ``generate``.
    """
    intrinsics = intrinsics or Intrinsics()
    noise = replace(noise or NoiseParams(focal=intrinsics.fx), scale=noise_scale)
    rng = np.random.default_rng(seed)
    transform = scene.pose.camera_to_base()
    rays_cam = intrinsics.rays()
    depth_true, codes = raycast(scene, intrinsics)
    in_range = (depth_true >= eval_range[0]) & (depth_true <= eval_range[1])
    codes = np.where(in_range, codes, INVALID)
    frames, frozen = [], None
    for _ in range(n_frames):
        depth, noisy_codes, frozen = corrupt_frame(
            depth_true, codes, noise, rng, frozen
        )
        clean = np.where(noisy_codes == codes, codes, INVALID)
        labels, planes = _instances(
            clean,
            depth_true,
            rays_cam,
            transform,
            intrinsics,
            scene,
            min_pixels,
            min_area,
        )
        labels = np.where(noisy_codes == NON_PLANAR, NON_PLANAR, labels)
        cloud = (rays_cam * depth[..., None]).astype(np.float32)
        frames.append(
            SceneFrame(
                name=scene.name,
                tags=scene.tags,
                noise_scale=noise_scale,
                cloud=cloud,
                labels=labels.astype(np.int32),
                planes=planes,
                camera_to_base=transform,
                intrinsics=intrinsics,
            )
        )
    return frames


def floor(extent: float = 10.0) -> Quad:
    return Quad((-1.0, -extent / 2, 0.0), (extent, 0, 0), (0, extent, 0), "floor")


def wall_x(x: float, y0: float, y1: float, tag: str, height: float = 2.5) -> Quad:
    """Wall in the plane x = const."""
    return Quad((x, y0, 0.0), (0, y1 - y0, 0), (0, 0, height), tag)


def wall_y(y: float, x0: float, x1: float, tag: str, height: float = 2.5) -> Quad:
    """Wall in the plane y = const."""
    return Quad((x0, y, 0.0), (x1 - x0, 0, 0), (0, 0, height), tag)


def _place(center, offset, yaw_deg):
    yaw = np.radians(yaw_deg)
    dx, dy = offset
    return (
        center[0] + np.cos(yaw) * dx - np.sin(yaw) * dy,
        center[1] + np.sin(yaw) * dx + np.cos(yaw) * dy,
    )


def table(tag, center, size=(1.2, 0.7), height=0.72, yaw_deg=0.0) -> list:
    """Table top (GT surface ``tag``) on four legs."""
    thickness, leg = 0.03, 0.04
    parts = [
        Box(
            (center[0], center[1], height - thickness / 2),
            (size[0], size[1], thickness),
            yaw_deg,
            {"+z": tag},
        )
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = _place(
                center, (sx * (size[0] / 2 - leg), sy * (size[1] / 2 - leg)), yaw_deg
            )
            leg_height = height - thickness
            parts.append(Box((x, y, leg_height / 2), (leg, leg, leg_height), yaw_deg))
    return parts


def shelf(
    prefix, front_x, center_y, boards, width=0.8, depth=0.35, height=1.8, board_t=0.02
):
    """Open shelf facing the robot (-x). Returns (primitives, surfaces).

    Boards (``board_t`` thick) are named ``<prefix>_board<i>`` (top face, class
    shelf); their other faces and the side panels are don't care, the back panel
    front is ``<prefix>_back`` (class other).
    """
    t = 0.02
    cx = front_x + depth / 2
    parts, surfaces = [], {}
    for i, h in enumerate(boards):
        name = f"{prefix}_board{i}"
        surfaces[name] = "shelf"
        parts.append(
            Box(
                (cx, center_y, h - board_t / 2),
                (depth, width, board_t),
                0,
                {"+z": name},
            )
        )
    for sy in (-1, 1):
        side_y = center_y + sy * (width / 2 + t / 2)
        parts.append(Box((cx, side_y, height / 2), (depth, t, height)))
    back = f"{prefix}_back"
    surfaces[back] = "other"
    parts.append(
        Box(
            (front_x + depth + t / 2, center_y, height / 2),
            (t, width + 2 * t, height),
            0,
            {"-x": back},
        )
    )
    return parts, surfaces


def box_object(x, y, base, size, yaw_deg=0.0) -> Box:
    return Box((x, y, base + size[2] / 2), size, yaw_deg)


def can_object(x, y, base, radius, height) -> Cylinder:
    return Cylinder((x, y), radius, base, base + height)


def _objects_on(center, top, layout, yaw_deg=0.0) -> list:
    """Boxes ("box", (dx, dy), (sx, sy, sz), yaw) and cans ("can", (dx, dy),
    (radius, height)) at offsets from ``center`` resting on height ``top``."""
    out = []
    for item in layout:
        x, y = _place(center, item[1], yaw_deg)
        if item[0] == "box":
            out.append(box_object(x, y, top, item[2], item[3] + yaw_deg))
        else:
            out.append(can_object(x, y, top, *item[2]))
    return out


def _table_objects(center, top, yaw_deg=0.0) -> list:
    layout = [
        ("box", (-0.35, -0.15), (0.12, 0.08, 0.20), 15),
        ("box", (0.10, 0.18), (0.18, 0.12, 0.06), -10),
        ("box", (0.35, -0.05), (0.10, 0.10, 0.10), 30),
        ("can", (-0.05, 0.02), (0.035, 0.12)),
        ("can", (0.22, -0.20), (0.040, 0.22)),
    ]
    return _objects_on(center, top, layout, yaw_deg)


def _shelf_objects(front_x, center_y, boards, depth=0.35) -> list:
    out = []
    for i, h in enumerate(boards):
        x = front_x + depth * 0.4
        out.append(box_object(x, center_y - 0.25 + 0.05 * i, h, (0.10, 0.07, 0.15), 10))
        out.append(can_object(x - 0.03, center_y + 0.05, h, 0.035, 0.12 + 0.02 * i))
        if i % 2 == 0:
            out.append(box_object(x + 0.05, center_y + 0.25, h, (0.12, 0.12, 0.08)))
    return out


def _table_scene(name, distance, tilt, pan=0.0, yaw=0.0, tags=("table",)) -> Scene:
    center = (distance + 0.35, 0.0)
    prims = [floor(), *table("table", center, yaw_deg=yaw)]
    prims += _table_objects(center, 0.72, yaw)
    return Scene(
        name,
        tags,
        CameraPose(pan_deg=pan, tilt_deg=tilt),
        {"floor": "floor", "table": "table"},
        prims,
    )


def _shelf_scene(name, front_x, center_y, boards, tilt, pan=0.0) -> Scene:
    parts, surfaces = shelf("shelf", front_x, center_y, boards)
    prims = [floor(), *parts, *_shelf_objects(front_x, center_y, boards)]
    return Scene(
        name,
        ("shelf",),
        CameraPose(pan_deg=pan, tilt_deg=tilt),
        {"floor": "floor", **surfaces},
        prims,
    )


def _clutter_objects(rng, count, x_range, y_range, keep_out) -> list:
    out = []
    while len(out) < count:
        x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in keep_out):
            continue
        if rng.random() < 0.6:
            size = (
                rng.uniform(0.1, 0.35),
                rng.uniform(0.1, 0.35),
                rng.uniform(0.05, 0.4),
            )
            out.append(box_object(x, y, 0.0, size, rng.uniform(0, 90)))
        else:
            out.append(
                can_object(x, y, 0.0, rng.uniform(0.03, 0.12), rng.uniform(0.1, 0.5))
            )
    return out


def scene_set() -> list[Scene]:
    """The fixed benchmark scenes."""
    scenes = [
        _table_scene("table_0.8m", 0.8, 45),
        _table_scene("table_1.5m", 1.5, 30),
        _table_scene("table_1.5m_pan", 1.5, 40, pan=15, yaw=20),
        _table_scene("table_2.5m", 2.5, 20, tags=("table", "far")),
        _shelf_scene("shelf_3board", 1.1, 0.0, (0.4, 0.8, 1.2), tilt=25),
        _shelf_scene("shelf_4board", 0.9, 0.1, (0.3, 0.6, 0.9, 1.2), tilt=30, pan=10),
        Scene(
            "floor_only",
            ("floor",),
            CameraPose(tilt_deg=30),
            {"floor": "floor"},
            [floor()],
        ),
    ]

    prims = [floor()]
    for side, y in (("left", 0.65), ("right", -0.65)):
        prims += table(f"table_{side}", (1.9, y), size=(1.0, 0.6))
        prims += _table_objects((1.9, y), 0.72)[:3]
    scenes.append(
        Scene(
            "two_tables",
            ("coplanar",),
            CameraPose(tilt_deg=30),
            {"floor": "floor", "table_left": "table", "table_right": "table"},
            prims,
        )
    )

    scenes.append(
        Scene(
            "corner_walls",
            ("walls", "far"),
            CameraPose(pan_deg=25, tilt_deg=20),
            {"floor": "floor", "wall_front": "wall", "wall_left": "wall"},
            [
                floor(),
                wall_x(2.6, -3.0, 1.2, "wall_front"),
                wall_y(1.2, -1.0, 2.6, "wall_left"),
            ],
        )
    )

    prims = [floor(), wall_x(2.2, -3.0, 3.0, "wall"), *table("table", (1.85, 0.0))]
    prims += _table_objects((1.85, 0.0), 0.72)
    scenes.append(
        Scene(
            "table_wall",
            ("table", "walls"),
            CameraPose(tilt_deg=30),
            {"floor": "floor", "wall": "wall", "table": "table"},
            prims,
        )
    )

    rng = np.random.default_rng(7)
    prims = [floor(), wall_x(3.2, -3.0, 3.0, "wall")]
    prims += table("table", (1.7, 0.35), yaw_deg=10)
    prims += _table_objects((1.7, 0.35), 0.72, 10)
    prims.append(Box((1.3, -0.95, 0.225), (0.5, 0.4, 0.45), 0, {"+z": "cabinet"}))
    prims.append(box_object(1.0, 0.9, 0.0, (0.45, 0.45, 0.45), 20))
    prims += _clutter_objects(
        rng,
        10,
        (0.6, 2.9),
        (-1.6, 1.6),
        [(0.95, 2.45, -0.15, 0.85), (1.0, 1.6, -1.2, -0.7)],
    )
    scenes.append(
        Scene(
            "clutter_room",
            ("clutter", "walls"),
            CameraPose(tilt_deg=30, pan_deg=-5),
            {"floor": "floor", "wall": "wall", "table": "table", "cabinet": "table"},
            prims,
        )
    )

    rng = np.random.default_rng(11)
    parts, surfaces = shelf("shelf", 1.6, -0.9, (0.35, 0.7, 1.05))
    prims = [floor(), *parts, *_shelf_objects(1.6, -0.9, (0.35, 0.7, 1.05))]
    prims += table("table", (1.5, 0.5), size=(0.8, 0.8))
    prims += _table_objects((1.5, 0.5), 0.72)[:4]
    prims += _clutter_objects(
        rng, 8, (0.6, 2.6), (-1.4, 1.4), [(1.0, 2.0, 0.0, 1.0), (1.5, 2.1, -1.4, -0.4)]
    )
    scenes.append(
        Scene(
            "clutter_shelf_table",
            ("clutter", "shelf"),
            CameraPose(tilt_deg=28),
            {"floor": "floor", "table": "table", **surfaces},
            prims,
        )
    )
    return scenes


# Board heights (m) by spacing; the top boards of 0.25 / 0.40 are above the camera.
SHELF_BOARDS = {
    0.25: (0.30, 0.55, 0.80, 1.05, 1.30, 1.55),
    0.35: (0.30, 0.65, 1.00, 1.35),
    0.40: (0.40, 0.80, 1.20, 1.60),
}
SHELF_DEPTH = 0.35


def _shelf_study_scene(
    name,
    front_x,
    spacing,
    tilt,
    pan=0.0,
    objects=True,
    board_t=0.02,
    table_height=None,
) -> Scene:
    """Shelf straight ahead of the panned camera, ``front_x`` m in front of it.

    Tags carry ``distance=<front_x>`` and ``spacing=<spacing>`` for the report.
    With ``table_height`` a table stands to the left of the shelf.
    """
    boards = SHELF_BOARDS[spacing]
    center_y = (front_x + SHELF_DEPTH / 2) * np.tan(np.radians(pan))
    if table_height is not None:
        center_y -= 0.45
    parts, surfaces = shelf(
        "shelf",
        front_x,
        center_y,
        boards,
        depth=SHELF_DEPTH,
        height=boards[-1] + 0.2,
        board_t=board_t,
    )
    prims = [floor(), *parts]
    surfaces = {"floor": "floor", **surfaces}
    if objects:
        prims += _shelf_objects(front_x, center_y, boards, SHELF_DEPTH)
    tags = ["shelf", f"distance={front_x:g}", f"spacing={spacing:g}"]
    tags.append("objects" if objects else "empty")
    if table_height is not None:
        center = (front_x + 0.3, center_y + 1.0)
        prims += table("table", center, size=(0.8, 0.6), height=table_height)
        prims += _table_objects(center, table_height)[1:3]
        surfaces["table"] = "table"
        tags.append("table")
    return Scene(
        name,
        tuple(tags),
        CameraPose(pan_deg=pan, tilt_deg=tilt),
        surfaces,
        prims,
    )


def shelf_scene_set() -> list[Scene]:
    """Shelf scenes for the place study (run_shelf.py); not part of ``scene_set``."""
    return [
        _shelf_study_scene("shelf_s25_d0.7", 0.7, 0.25, tilt=40),
        _shelf_study_scene("shelf_s25_d1.0", 1.0, 0.25, tilt=30),
        _shelf_study_scene("shelf_s25_d1.5", 1.5, 0.25, tilt=20),
        _shelf_study_scene("shelf_s25_d1.0_empty", 1.0, 0.25, tilt=30, objects=False),
        _shelf_study_scene("shelf_s25_d1.0_pan20", 1.0, 0.25, tilt=30, pan=20),
        _shelf_study_scene("shelf_s40_d0.7", 0.7, 0.40, tilt=35),
        _shelf_study_scene("shelf_s40_d1.0", 1.0, 0.40, tilt=25),
        _shelf_study_scene("shelf_s40_d1.5", 1.5, 0.40, tilt=15),
        _shelf_study_scene(
            "shelf_s40_d1.0_pan-20_thick", 1.0, 0.40, tilt=25, pan=-20, board_t=0.04
        ),
        _shelf_study_scene("shelf_s35_d1.0_empty", 1.0, 0.35, tilt=30, objects=False),
        _shelf_study_scene("shelf_s35_d1.0", 1.0, 0.35, tilt=30),
        _shelf_study_scene(
            "shelf_s40_d1.0_table", 1.0, 0.40, tilt=30, table_height=0.75
        ),
    ]


def _place_table_scene(name, front_x, size, height, tilt, layout, pan=0.0) -> Scene:
    center = (front_x + size[0] / 2, 0.0)
    prims = [floor(), *table("table", center, size=size, height=height)]
    prims += _objects_on(center, height, layout)
    return Scene(
        name,
        ("place", "table"),
        CameraPose(pan_deg=pan, tilt_deg=tilt),
        {"floor": "floor", "table": "table"},
        prims,
    )


def place_scene_set() -> list[Scene]:
    """Extra table and shelf scenes for the place study (run_place.py)."""
    mixed = [
        ("box", (-0.30, -0.15), (0.10, 0.07, 0.18), 20),
        ("can", (-0.05, 0.12), (0.04, 0.12)),
        ("box", (0.20, -0.12), (0.15, 0.10, 0.05), -15),
        ("can", (0.32, 0.15), (0.03, 0.20)),
        ("box", (0.02, -0.17), (0.06, 0.06, 0.06), 0),
    ]
    crowded = [
        ("box", (dx, dy), (0.11, 0.11, h), 0)
        for (dx, dy), h in zip(
            [
                (-0.24, -0.19),
                (-0.24, 0.0),
                (-0.24, 0.19),
                (0.04, -0.21),
                (0.04, 0.21),
                (0.27, -0.19),
                (0.27, 0.0),
                (0.27, 0.19),
            ],
            (0.08, 0.06, 0.10, 0.15, 0.12, 0.25, 0.20, 0.18),
        )
    ]
    sparse = [
        ("box", (-0.35, 0.20), (0.12, 0.08, 0.20), 30),
        ("can", (0.10, -0.15), (0.035, 0.12)),
        ("box", (0.40, 0.10), (0.10, 0.10, 0.10), 0),
    ]
    scenes = [
        _place_table_scene("place_table_d1.0", 1.0, (1.0, 0.6), 0.75, 40, mixed),
        _place_table_scene("place_crowded_d1.0", 1.0, (0.8, 0.65), 0.75, 40, crowded),
        _place_table_scene("place_table_d2.0", 2.0, (1.2, 0.7), 0.75, 20, sparse),
        _place_table_scene("place_low_table_d0.8", 0.8, (0.9, 0.5), 0.45, 45, mixed),
    ]

    # Low shelf: boards 0.20 m apart (0.18 m gap under the next board).
    boards = (0.30, 0.50, 0.70, 0.90, 1.10)
    parts, surfaces = shelf("shelf", 1.0, 0.0, boards, depth=0.35, height=1.3)
    prims = [floor(), *parts]
    for i, h in enumerate(boards):
        x = 1.0 + 0.35 * 0.45
        prims += _objects_on(
            (x, 0.0),
            h,
            [
                ("box", (0.0, -0.25 + 0.04 * i), (0.10, 0.07, 0.12), 10),
                ("can", (-0.03, 0.08), (0.035, 0.10)),
            ],
        )
    scenes.append(
        Scene(
            "place_shelf_low_d1.0",
            ("place", "shelf"),
            CameraPose(tilt_deg=30),
            {"floor": "floor", **surfaces},
            prims,
        )
    )
    return scenes


NOISE_LEVELS = (0.5, 1.0, 2.0)


def frame_seed(name: str, noise_scale: float) -> int:
    return zlib.crc32(f"{name}:{noise_scale}".encode())


def iter_frames(
    noise_levels=NOISE_LEVELS, names=None, intrinsics: Intrinsics | None = None
) -> Iterator[SceneFrame]:
    """Yield every (scene, noise level) frame with a fixed per-frame seed."""
    for scene in scene_set():
        if names and scene.name not in names:
            continue
        for scale in noise_levels:
            yield generate(scene, scale, frame_seed(scene.name, scale), intrinsics)
