"""Arm mount and joint offsets from the head camera's point cloud (ROS free).

The arm model (URDF meshes posed by the joint readings) and the arm in the
point cloud disagree when the mount yaw or a joint zero is off. With the arm
held still in a few poses, the cloud points on the arm should lie on the
model's surface; the offsets that best put them there are the calibration.

Parameters, all deltas to the model the cloud was compared with (radians and
metres; ``PARAMETERS``):

- ``shoulder``: added to the mount yaw of link_<l|r>1 (frame config ``yaw``),
  turning the whole arm about the joint_<l|r>2 axis, as frame.py does. It is
  a joint_<l|r>2 reading offset that also turns link_<l|r>1.
- ``elbow`` / ``wrist``: joint_<l|r>3 / joint_<l|r>4 reading offsets (the
  true angle is reading + offset; frame config ``joint_offsets``).
- ``height``: the whole arm up (frame config ``shoulder_xyz`` z, measured
  with a tape; the camera height comes from the plane calibration).
- ``camera_x`` / ``camera_y`` / ``camera_yaw``: a horizontal correction of
  the camera (camera_arm_calibration.HorizontalCorrection about the vertical
  through the camera), applied to the cloud points. It moves the arm in the
  cloud like a shoulder offset does, so with one arm both are not separable:
  free the camera only with the shoulder fixed.

``ArmModel`` samples the visual meshes of the arm's links (area-weighted,
fixed seed) with their facet normals, and poses them by the URDF chain from
the root (base_footprint). ``prepare`` reduces a snapshot's cloud to the
points that can be the arm: a voxel grid, above the support surface (given,
or the densest height under the arm, ``support_height``) and within ``crop``
of the nominal model, not nearer to the fixed robot geometry (frame members,
head base) than to the arm, with a normal from its neighbours. ``fit`` then
alternates, ICP-like, between associating each point with the nearest posed
model sample facing the camera (within a gate shrinking to
``max_distance``; dropped when the sample's normal is off the point's, as
at edges, where depth noise would bias the pairing) and a robust
least-squares solve over all
snapshots of the point-to-plane distances (Huber, ``loss_scale``), until the
parameters settle. Points nearest to a gripper finger (its opening may be
off) or to nothing within the gate (a held object, other objects) do not
count.

The result has the per-snapshot RMS distance and inlier counts before and
after, and an observability check: parameter correlations and the
condition number from the Jacobian at the solution, and, with 3 or more
snapshots, how far each parameter moves when a snapshot is left out. One
pose rarely separates the shoulder from the elbow; ``warnings`` say so.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from aha_description.frame import ARMS, rpy_matrix

PARAMETERS = (
    "shoulder",
    "elbow",
    "wrist",
    "height",
    "camera_x",
    "camera_y",
    "camera_yaw",
)
GROUPS = {
    "shoulder": ("shoulder",),
    "elbow": ("elbow",),
    "wrist": ("wrist",),
    # The whole arm up (shoulder_xyz z); the camera height is the plane
    # calibration's.
    "height": ("height",),
    "camera": ("camera_x", "camera_y", "camera_yaw"),
}
DEFAULT_FREE = ("shoulder", "elbow")
ROOT = "base_footprint"
SPACING = 0.004  # m between arm surface samples
STATIC_SPACING = 0.01  # m between samples of the fixed robot geometry
VOXEL = 0.005  # m, cloud grid
CROP = 0.08  # m from the nominal model
SUPPORT_MARGIN = 0.015  # m above the support surface
MAX_DISTANCE = 0.02  # m, final association gate
LOSS_SCALE = 0.004  # m, Huber
CORRELATION_LIMIT = 0.99
SINGULAR = 1e-8  # eigenvalue ratio of a null direction of the normalised J^T J
NORMAL_RADIUS = 0.012  # m, neighbourhood of a cloud normal
NORMAL_AGREEMENT = math.cos(math.radians(30.0))  # cloud vs model normal
NEIGHBOURS = 32  # nearest model samples tried per point for an agreeing normal
STEP_TOLERANCE = 1e-4  # rad or m, parameter change that ends the iteration
SPREAD_LIMIT = 0.01  # rad or m a parameter may move with one snapshot left out
MIN_POSES = 3
MIN_INLIERS = 100


# --- meshes ---------------------------------------------------------------


def read_stl(path) -> np.ndarray:
    """Triangles (N, 3, 3) of a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if 84 + count * 50 == len(data):
            record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, record, count, 84)["v"].astype(float)
    text = data.decode("ascii", errors="replace")
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not vertices or len(vertices) % 3:
        raise ValueError(f"{path} is not an STL file")
    return np.asarray(vertices, float).reshape(-1, 3, 3)


def box_triangles(size) -> np.ndarray:
    """Triangles (12, 3, 3) of a box of ``size`` centred at the origin."""
    h = np.asarray(size, float) / 2.0
    corners = (
        np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        * h
    )
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    triangles = []
    for a, b, c, d in faces:
        triangles += [corners[[a, b, c]], corners[[a, c, d]]]
    return np.asarray(triangles)


def sample_surface(
    triangles: np.ndarray, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(points, unit normals) on the triangles, one per ``spacing``^2 of area
    (at least one), area-weighted."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(b - a, c - a)
    doubled = np.linalg.norm(cross, axis=1)
    keep = doubled > 0.0
    a, b, c, cross, doubled = a[keep], b[keep], c[keep], cross[keep], doubled[keep]
    if len(a) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    area = doubled / 2.0
    count = max(1, int(math.ceil(area.sum() / spacing**2)))
    index = rng.choice(len(a), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    points = a[index] + u[:, None] * (b - a)[index] + v[:, None] * (c - a)[index]
    normals = cross[index] / doubled[index, None]
    return points, normals


def package_path(uri: str) -> Path:
    """File of a package:// or file:// mesh URI (needs ament_index)."""
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    if not uri.startswith("package://"):
        return Path(uri)
    from ament_index_python.packages import get_package_share_directory

    package, _, rest = uri[len("package://") :].partition("/")
    return Path(get_package_share_directory(package)) / rest


# --- kinematics -----------------------------------------------------------


def _floats(text, default) -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def transform(rotation=None, translation=None) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


def axis_rotation(axis, angle: float) -> np.ndarray:
    """3x3 rotation by ``angle`` about the unit ``axis`` (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _origin_matrix(element) -> np.ndarray:
    if element is None:
        return np.eye(4)
    return transform(
        rpy_matrix(_floats(element.get("rpy"), "0 0 0")),
        _floats(element.get("xyz"), "0 0 0"),
    )


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4 parent -> joint frame at zero
    axis: np.ndarray  # unit, in the joint frame
    mimic: tuple[str, float, float] | None  # (joint, multiplier, offset)

    def motion(self, value: float) -> np.ndarray:
        if self.type in ("revolute", "continuous"):
            return transform(axis_rotation(self.axis, value))
        if self.type == "prismatic":
            return transform(translation=self.axis * value)
        return np.eye(4)


@dataclass(frozen=True)
class Visual:
    origin: np.ndarray  # 4x4 link -> geometry
    kind: str  # "mesh" or "box"
    source: str  # mesh URI
    size: np.ndarray  # box size, or mesh scale


class UrdfTree:
    """Links, joints and visuals of a URDF string."""

    def __init__(self, urdf: str):
        root = ET.fromstring(urdf)
        self.joints: dict[str, Joint] = {}  # by child link
        self.by_name: dict[str, Joint] = {}
        for element in root.findall("joint"):
            axis = _floats(
                None
                if element.find("axis") is None
                else element.find("axis").get("xyz"),
                "1 0 0",
            )
            mimic = element.find("mimic")
            joint = Joint(
                name=element.get("name"),
                type=element.get("type"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                origin=_origin_matrix(element.find("origin")),
                axis=axis / (np.linalg.norm(axis) or 1.0),
                mimic=None
                if mimic is None
                else (
                    mimic.get("joint"),
                    float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0")),
                ),
            )
            self.joints[joint.child] = joint
            self.by_name[joint.name] = joint
        self.visuals: dict[str, list[Visual]] = {}
        for link in root.findall("link"):
            visuals = []
            for element in link.findall("visual"):
                geometry = element.find("geometry")
                shape = None if geometry is None or len(geometry) == 0 else geometry[0]
                if shape is None:
                    continue
                origin = _origin_matrix(element.find("origin"))
                if shape.tag == "mesh":
                    scale = _floats(shape.get("scale"), "1 1 1")
                    visuals.append(Visual(origin, "mesh", shape.get("filename"), scale))
                elif shape.tag == "box":
                    visuals.append(
                        Visual(origin, "box", "", _floats(shape.get("size"), ""))
                    )
            self.visuals[link.get("name")] = visuals
        self.links = list(self.visuals)

    def joint(self, name: str) -> Joint:
        if name not in self.by_name:
            raise ValueError(f"URDF has no joint {name}")
        return self.by_name[name]

    def path(self, link: str, root: str = ROOT) -> list[Joint]:
        """Joints from ``root`` down to ``link``."""
        joints = []
        while link != root:
            if link not in self.joints:
                raise ValueError(f"{link} is not below {root}")
            joints.append(self.joints[link])
            link = joints[-1].parent
        return joints[::-1]

    def subtree(self, link: str) -> list[str]:
        """``link`` and every link below it."""
        out, stack = [], [link]
        while stack:
            current = stack.pop()
            out.append(current)
            stack += [j.child for j in self.joints.values() if j.parent == current]
        return out

    def value(self, joint: Joint, values: dict, offsets: dict) -> float:
        if joint.mimic is not None:
            master, multiplier, offset = joint.mimic
            base = values.get(master, 0.0) + offsets.get(master, 0.0)
            return multiplier * base + offset
        return values.get(joint.name, 0.0) + offsets.get(joint.name, 0.0)

    def poses(
        self,
        links,
        values: dict,
        offsets: dict | None = None,
        motions: dict | None = None,
        root: str = ROOT,
    ) -> dict[str, np.ndarray]:
        """Root -> link transforms for joint ``values`` plus ``offsets``;
        ``motions`` (link -> 4x4) are applied right after a link's joint."""
        offsets, motions = offsets or {}, motions or {}
        cache = {root: np.eye(4)}

        def pose(link):
            if link not in cache:
                joint = self.joints.get(link)
                if joint is None:
                    raise ValueError(f"{link} is not below {root}")
                T = pose(joint.parent) @ joint.origin
                T = T @ joint.motion(self.value(joint, values, offsets))
                if link in motions:
                    T = T @ motions[link]
                cache[link] = T
            return cache[link]

        return {link: pose(link) for link in links}


# --- arm model ------------------------------------------------------------


def arm_joints(side: str) -> dict[str, str]:
    """Joint names of the calibrated parameters for ``side``."""
    letter = ARMS[side]
    return {
        "mount": f"joint_{letter}1",
        "shoulder": f"joint_{letter}2",
        "elbow": f"joint_{letter}3",
        "wrist": f"joint_{letter}4",
    }


@dataclass
class ArmModel:
    tree: UrdfTree
    side: str
    order: list[str]  # link_<x>1 and the links below it, parents first
    links: list[str]  # of order, the links with samples
    fingers: list[str]  # of links, gripper fingers
    samples: dict  # link -> (points, normals) in the link frame
    static: np.ndarray  # (M, 3) fixed robot geometry in ROOT
    shoulder_axis: tuple  # (point, direction) in link_<x>1
    bounds: dict = field(default_factory=dict, repr=False)  # link -> (center, radius)

    def __post_init__(self):
        for link in self.links:
            points = self.samples[link][0]
            center = (points.min(axis=0) + points.max(axis=0)) / 2.0
            radius = float(np.max(np.linalg.norm(points - center, axis=1)))
            self.bounds[link] = (center, radius)

    @classmethod
    def from_urdf(
        cls,
        urdf: str,
        side: str,
        resolve=package_path,
        spacing: float = SPACING,
        static_spacing: float = STATIC_SPACING,
        seed: int = 0,
    ) -> ArmModel:
        if side not in ARMS:
            raise ValueError(f"unknown arm {side!r}, use {sorted(ARMS)}")
        tree = UrdfTree(urdf)
        names = arm_joints(side)
        mount = tree.joint(names["mount"])
        shoulder = tree.joint(names["shoulder"])
        if shoulder.parent != mount.child:
            raise ValueError(f"{shoulder.name} does not follow {mount.name}")
        rng = np.random.default_rng(seed)
        meshes: dict[str, np.ndarray] = {}

        def triangles(link):
            parts = []
            for visual in tree.visuals.get(link, []):
                if visual.kind == "mesh":
                    if visual.source not in meshes:
                        meshes[visual.source] = read_stl(resolve(visual.source))
                    local = meshes[visual.source] * visual.size
                else:
                    local = box_triangles(visual.size)
                parts.append(local @ visual.origin[:3, :3].T + visual.origin[:3, 3])
            return np.concatenate(parts) if parts else np.zeros((0, 3, 3))

        order = tree.subtree(mount.child)
        samples = {}
        for link in order:
            found = triangles(link)
            if len(found):
                samples[link] = sample_surface(found, spacing, rng)
        fingers = [
            link
            for link in samples
            if tree.joints[link].type == "prismatic"
            and tree.joints[link].parent != mount.child
        ]
        static = []
        for link in tree.links:
            if link in order or not tree.visuals.get(link):
                continue
            try:
                path = tree.path(link)
            except ValueError:
                continue
            if any(j.type != "fixed" for j in path):
                continue
            points, _ = sample_surface(triangles(link), static_spacing, rng)
            T = tree.poses([link], {})[link]
            static.append(points @ T[:3, :3].T + T[:3, 3])
        return cls(
            tree=tree,
            side=side,
            order=order,
            links=list(samples),
            fingers=fingers,
            samples=samples,
            static=np.concatenate(static) if static else np.zeros((0, 3)),
            shoulder_axis=(
                shoulder.origin[:3, 3],
                shoulder.origin[:3, :3] @ shoulder.axis,
            ),
        )

    def poser(self, joints: dict):
        """Function of the parameters (``PARAMETERS``, missing ones 0) giving
        the root -> link transforms of the arm for the joint readings."""
        names = arm_joints(self.side)
        tree = self.tree
        mount = tree.joint(names["mount"])
        base = tree.poses([mount.child], joints)[mount.child]
        point, direction = self.shoulder_axis
        tuned = {names["elbow"]: "elbow", names["wrist"]: "wrist"}
        steps = []  # (link, parent, fixed transform or (joint, parameter))
        for link in self.order[1:]:
            joint = tree.joints[link]
            value = tree.value(joint, joints, {})
            if joint.name in tuned:
                steps.append((link, joint.parent, (joint, value, tuned[joint.name])))
            else:
                steps.append((link, joint.parent, joint.origin @ joint.motion(value)))

        def poses(params: dict) -> dict[str, np.ndarray]:
            turn = transform(axis_rotation(direction, params.get("shoulder", 0.0)))
            shoulder = transform(translation=point) @ turn
            lift = transform(translation=(0.0, 0.0, params.get("height", 0.0)))
            out = {
                self.order[0]: lift @ base @ shoulder @ transform(translation=-point)
            }
            for link, parent, step in steps:
                if isinstance(step, tuple):
                    joint, value, name = step
                    step = joint.origin @ joint.motion(value + params.get(name, 0.0))
                out[link] = out[parent] @ step
            return out

        return poses

    def link_poses(self, joints: dict, params: dict) -> dict[str, np.ndarray]:
        """Root -> link transforms of the arm (see ``poser``)."""
        return self.poser(joints)(params)

    def surface(self, joints: dict, params: dict):
        """(points, normals, link index into ``links``) of the posed samples."""
        poses = self.link_poses(joints, params)
        points, normals, index = [], [], []
        for i, link in enumerate(self.links):
            p, n = self.samples[link]
            T = poses[link]
            points.append(p @ T[:3, :3].T + T[:3, 3])
            normals.append(n @ T[:3, :3].T)
            index.append(np.full(len(p), i))
        return np.concatenate(points), np.concatenate(normals), np.concatenate(index)


# --- snapshots ------------------------------------------------------------


@dataclass
class Snapshot:
    points: np.ndarray  # (N, 3) cloud in ROOT, through the modelled camera
    joints: dict  # joint name -> reading
    camera: np.ndarray  # (3,) camera position in ROOT
    support: float | None = None  # m, support surface height (None: estimate)


def voxel_filter(points: np.ndarray, size: float) -> np.ndarray:
    """Mean point of each occupied ``size`` cube (finite points only)."""
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0 or size <= 0.0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse.reshape(-1), points)
    return sums / counts[:, None]


def support_height(
    points: np.ndarray,
    arm: np.ndarray,
    radius: float = 0.15,
    bin_size: float = 0.005,
    min_points: int = 150,
) -> float | None:
    """Height of the surface under the arm: the ``bin_size`` height bin with
    the most cloud points within ``radius`` (horizontally) of the model
    ``arm`` points and below its lowest point + 5 cm. None if that bin has
    fewer than ``min_points``."""
    if len(points) == 0 or len(arm) == 0:
        return None
    near = cKDTree(arm[:, :2]).query(points[:, :2], distance_upper_bound=radius)[0]
    z = points[np.isfinite(near), 2]
    z = z[z < arm[:, 2].min() + 0.05]
    if len(z) < min_points:
        return None
    bins = np.floor(z / bin_size).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    if counts.max() < min_points:
        return None
    top = values[np.argmax(counts)]
    return float(np.median(z[np.abs(bins - top) <= 1]))


@dataclass
class Prepared:
    points: np.ndarray  # (N, 3) candidate arm points in ROOT
    normals: np.ndarray  # (N, 3) unit, from the neighbouring points
    joints: dict
    camera: np.ndarray
    support: float | None  # m, the height used


def cloud_normals(points: np.ndarray, radius: float = NORMAL_RADIUS, k: int = 16):
    """(normals, valid) of ``points`` (N, 3): the direction of least spread of
    the up to ``k`` neighbours within ``radius``; valid with 5 or more
    neighbours spread over a plane rather than a line."""
    normals = np.zeros_like(points)
    valid = np.zeros(len(points), bool)
    if len(points) < 5:
        return normals, valid
    k = min(k, len(points))
    distance, index = cKDTree(points).query(points, k=k, distance_upper_bound=radius)
    found = np.isfinite(distance)
    index = np.where(found, index, 0)
    count = found.sum(axis=1)
    neighbours = np.where(found[..., None], points[index], 0.0)
    mean = neighbours.sum(axis=1) / count[:, None]
    centred = np.where(found[..., None], points[index] - mean[:, None], 0.0)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / count[:, None, None]
    eigen, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    valid = (count >= 5) & (eigen[:, 1] > 4.0 * eigen[:, 0])
    return normals, valid


def prepare(
    model: ArmModel,
    snapshot: Snapshot,
    voxel: float = VOXEL,
    crop: float = CROP,
    support_margin: float = SUPPORT_MARGIN,
) -> Prepared:
    """The snapshot's points that can be the arm (see the module doc)."""
    points = voxel_filter(np.asarray(snapshot.points, float), voxel)
    arm, _, _ = model.surface(snapshot.joints, {})
    support = snapshot.support
    if support is None:
        support = support_height(points, arm)
    if support is not None:
        points = points[points[:, 2] > support + support_margin]
    to_arm = cKDTree(arm).query(points, distance_upper_bound=crop)[0]
    keep = np.isfinite(to_arm)
    if len(model.static):
        to_static = cKDTree(model.static).query(points, distance_upper_bound=crop)[0]
        keep &= ~(to_static < to_arm)
    points = points[keep]
    normals, valid = cloud_normals(points)
    camera = np.asarray(snapshot.camera, float)
    return Prepared(
        points[valid], normals[valid], dict(snapshot.joints), camera, support
    )


# --- fit ------------------------------------------------------------------


def free_parameters(groups) -> tuple[str, ...]:
    """Parameter names of the free ``groups`` (``GROUPS`` keys)."""
    names = []
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"unknown parameter group {group!r}, use {sorted(GROUPS)}")
        names += GROUPS[group]
    if not names:
        raise ValueError("no free parameters")
    return tuple(n for n in PARAMETERS if n in names)


def camera_motion(params: dict, pivot) -> np.ndarray:
    """4x4 of the camera correction (rotation about the vertical through
    ``pivot``, then the shift), as HorizontalCorrection.motion."""
    yaw = params.get("camera_yaw", 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    p = np.asarray(pivot, float)[:2]
    shift = np.array([params.get("camera_x", 0.0), params.get("camera_y", 0.0)])
    m = np.eye(4)
    m[:2, :2] = rotation
    m[:2, 3] = p + shift - rotation @ p
    return m


def unit(name: str) -> str:
    return "m" if name in ("height", "camera_x", "camera_y") else "rad"


@dataclass(frozen=True)
class SnapshotStats:
    rms_before: float  # m, point-to-surface distance of the inliers, nominal
    rms_after: float  # m, fitted
    inliers_before: int  # points nearest to the arm within max_distance
    inliers_after: int
    points: int  # prepared points
    support: float | None


@dataclass
class FitResult:
    params: dict  # all PARAMETERS, fixed ones 0
    free: tuple
    std: dict  # free parameter -> 1-sigma for independent point noise
    correlation: np.ndarray  # free x free
    condition: float  # of J^T J with normalised columns
    unobservable: tuple  # free parameters in a (near) null direction of J
    snapshots: list
    pivot: np.ndarray  # camera correction pivot (x, y)
    leave_one_out: dict | None  # free parameter -> max change with one left out
    warnings: list = field(default_factory=list)
    reliable: bool = True  # no observability warning
    iterations: int = 0
    seconds: float = 0.0


class _Problem:
    """Point-to-model associations and residuals over the snapshots."""

    def __init__(self, model, prepared, free, pivot, exclude_fingers):
        self.model, self.prepared, self.free, self.pivot = model, prepared, free, pivot
        self.posers = [model.poser(s.joints) for s in prepared]
        self.active = np.array(
            [not (exclude_fingers and link in model.fingers) for link in model.links]
        )
        # Only the surface facing the camera can be seen: a noisy point behind
        # a thin part must not pair with its back. Which samples face the
        # camera hardly changes with the parameters; take the nominal pose.
        self.facing = []
        for snap, poser in zip(prepared, self.posers):
            poses = poser({})
            facing = {}
            for name in model.links:
                T = poses[name]
                p, n = model.samples[name]
                eye = (np.asarray(snap.camera, float) - T[:3, 3]) @ T[:3, :3]
                rows = np.flatnonzero(np.einsum("ij,ij->i", eye - p, n) > 0.0)
                if len(rows):
                    facing[name] = (rows, cKDTree(p[rows]))
            self.facing.append(facing)
        self.pairs = []

    def params(self, x) -> dict:
        params = {n: 0.0 for n in PARAMETERS}
        params.update(zip(self.free, (float(v) for v in x)))
        return params

    def nearest(self, params: dict, gate: float):
        """Per snapshot: the distance, link index and sample row of each
        point's nearest model sample facing the camera (inf / -1 beyond
        ``gate``)."""
        C = camera_motion(params, self.pivot)
        out = []
        for snap, poser, facing in zip(self.prepared, self.posers, self.facing):
            points = snap.points @ C[:3, :3].T + C[:3, 3]
            normals = snap.normals @ C[:3, :3].T
            poses = poser(params)
            best = np.full(len(points), np.inf)
            link = np.full(len(points), -1)
            sample = np.full(len(points), -1)
            for i, name in enumerate(self.model.links):
                if name not in facing:
                    continue
                T = poses[name]
                center, radius = self.model.bounds[name]
                local = (points - T[:3, 3]) @ T[:3, :3]
                rows = np.flatnonzero(
                    np.linalg.norm(local - center, axis=1) < radius + gate
                )
                if len(rows) == 0:
                    continue
                index, tree = facing[name]
                d, k = tree.query(local[rows], k=NEIGHBOURS, distance_upper_bound=gate)
                d, k = d.reshape(len(rows), -1), k.reshape(len(rows), -1)
                # A point near an edge of the arm, or paired across one, has a
                # normal off the sample's: depth noise would bias it. The
                # nearest sample whose normal agrees is taken (inside a model
                # that sits too high, a side face is nearer than the top).
                found = np.isfinite(d)
                n = self.model.samples[name][1][index[np.where(found, k, 0)]]
                agree = np.abs(np.einsum("ij,ikj->ik", normals[rows] @ T[:3, :3], n))
                d[found & (agree < NORMAL_AGREEMENT)] = np.inf
                pick = np.argmin(d, axis=1)
                d = d[np.arange(len(rows)), pick]
                k = k[np.arange(len(rows)), pick]
                better = d < best[rows]
                rows = rows[better]
                best[rows], link[rows] = d[better], i
                sample[rows] = index[k[better]]
            out.append((best, link, sample))
        return out

    def associate(self, params: dict, gate: float) -> int:
        """Pairs each point with its nearest sample if that is on an active
        link within ``gate``; returns the number of pairs."""
        self.pairs = []
        for snap, (best, link, sample) in zip(
            self.prepared, self.nearest(params, gate)
        ):
            ok = np.isfinite(best)
            ok[ok] = self.active[link[ok]]
            q = np.zeros((int(ok.sum()), 3))
            n = np.zeros_like(q)
            for i in np.unique(link[ok]):
                rows = link[ok] == i
                p, normals = self.model.samples[self.model.links[i]]
                q[rows], n[rows] = p[sample[ok][rows]], normals[sample[ok][rows]]
            self.pairs.append((snap.points[ok], link[ok], q, n))
        return sum(len(p[0]) for p in self.pairs)

    def parts(self, x) -> list[np.ndarray]:
        """Per snapshot, the point-to-plane distances of the pairs."""
        params = self.params(x)
        C = camera_motion(params, self.pivot)
        out = []
        for poser, (points, link, q, n) in zip(self.posers, self.pairs):
            poses = poser(params)
            T = np.stack([poses[name] for name in self.model.links])[link]
            world = points @ C[:3, :3].T + C[:3, 3]
            local = np.einsum("kji,kj->ki", T[:, :3, :3], world - T[:, :3, 3])
            out.append(np.einsum("ij,ij->i", local - q, n))
        return out

    def residuals(self, x) -> np.ndarray:
        return np.concatenate([np.zeros(0)] + self.parts(x))

    def stats(self, x, gate: float) -> list[tuple[float, int]]:
        """Per snapshot (RMS point-to-plane distance, count) of the points
        nearest to an active link within ``gate``."""
        self.associate(self.params(x), gate)
        return [
            (float(np.sqrt(np.mean(r**2))) if len(r) else float("nan"), len(r))
            for r in self.parts(x)
        ]


def _solve(problem, x0, gate, max_distance, loss_scale, iterations):
    x = np.array(x0, float)
    done = 0
    for done in range(1, iterations + 1):
        if problem.associate(problem.params(x), gate) <= 10 * len(x):
            raise ValueError("too few cloud points near the arm model")
        result = least_squares(
            problem.residuals, x, loss="huber", f_scale=loss_scale, x_scale=0.01
        )
        step = float(np.max(np.abs(result.x - x)))
        x = result.x
        at_gate = gate <= max_distance
        gate = max(max_distance, gate * 0.6)
        if at_gate and step < STEP_TOLERANCE:
            break
    return x, done


def _jacobian(problem, x, step: float = 1e-5) -> np.ndarray:
    columns = []
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = step
        columns.append(
            (problem.residuals(x + dx) - problem.residuals(x - dx)) / (2 * step)
        )
    return np.column_stack(columns)


def observability(J: np.ndarray, r: np.ndarray, names):
    """(std, correlation, condition, unobservable names) of a least-squares
    solution with Jacobian ``J`` and residuals ``r``."""
    JTJ = J.T @ J
    scale = np.sqrt(np.diag(JTJ))
    scale[scale == 0.0] = 1.0
    eigen, vectors = np.linalg.eigh(JTJ / np.outer(scale, scale))
    condition = float(eigen[-1] / max(eigen[0], eigen[-1] * 1e-16, 1e-300))
    null = eigen < SINGULAR * eigen[-1]
    unobservable = tuple(
        n
        for n, weight in zip(names, np.abs(vectors[:, null]).max(axis=1, initial=0.0))
        if weight > 0.1
    )
    floored = np.maximum(eigen, SINGULAR * eigen[-1])
    inverse = (vectors / floored) @ vectors.T / np.outer(scale, scale)
    diagonal = np.diag(inverse)
    noise = float(np.sqrt(np.mean(r**2))) if len(r) else float("nan")
    std = np.sqrt(diagonal) * noise
    correlation = inverse / np.sqrt(np.outer(diagonal, diagonal))
    return std, correlation, condition, unobservable


def fit(
    model: ArmModel,
    snapshots,
    free=DEFAULT_FREE,
    *,
    exclude_fingers: bool = True,
    crop: float = CROP,
    max_distance: float = MAX_DISTANCE,
    loss_scale: float = LOSS_SCALE,
    iterations: int = 40,
    leave_one_out: bool = True,
) -> FitResult:
    """Fit the free parameter groups (``GROUPS``) to ``Prepared`` snapshots."""
    started = time.monotonic()
    names = free_parameters(free)
    prepared = list(snapshots)
    if not prepared:
        raise ValueError("no snapshots")
    pivot = np.mean([s.camera for s in prepared], axis=0)[:2]
    problem = _Problem(model, prepared, names, pivot, exclude_fingers)
    before = problem.stats(np.zeros(len(names)), max_distance)
    x, done = _solve(
        problem, np.zeros(len(names)), crop, max_distance, loss_scale, iterations
    )
    params = problem.params(x)
    after = problem.stats(x, max_distance)

    # Observability on the inliers of the final association.
    r = problem.residuals(x)
    inlier = np.abs(r) < 3.0 * loss_scale
    J = _jacobian(problem, x)[inlier]
    std, correlation, condition, unobservable = observability(J, r[inlier], names)

    spread = None
    if leave_one_out and len(prepared) >= MIN_POSES:
        changes = []
        for i in range(len(prepared)):
            rest = prepared[:i] + prepared[i + 1 :]
            sub = _Problem(model, rest, names, pivot, exclude_fingers)
            xi, _ = _solve(sub, x, max_distance, max_distance, loss_scale, 10)
            changes.append(np.abs(xi - x))
        spread = {n: float(v) for n, v in zip(names, np.max(changes, axis=0))}

    warnings = []
    if len(prepared) < MIN_POSES:
        warnings.append(
            f"{len(prepared)} snapshot(s): one pose cannot tell a shoulder from an "
            f"elbow offset reliably; use {MIN_POSES} or more distinct poses"
        )
    if unobservable:
        warnings.append(
            f"{', '.join(unobservable)} not observable together"
            + (
                " (the camera correction moves the arm like the shoulder offset: "
                "free the camera only with the shoulder fixed)"
                if "shoulder" in unobservable and "camera_yaw" in unobservable
                else ""
            )
        )
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            if abs(correlation[i, j]) > CORRELATION_LIMIT:
                warnings.append(
                    f"{a} and {names[j]} are hardly separable in these poses "
                    f"(correlation {correlation[i, j]:+.3f}): vary the poses"
                )
    if spread:
        for n, value in spread.items():
            if value > SPREAD_LIMIT:
                warnings.append(
                    f"{n} changes by {value:.4f} {unit(n)} with a snapshot left "
                    "out: add poses, check that the arm is in view and still"
                )
    reliable = not warnings
    for k, (_, count) in enumerate(after, 1):
        if count < MIN_INLIERS:
            warnings.append(f"snapshot {k}: only {count} cloud points on the arm")
    stats = [
        SnapshotStats(b[0], a[0], b[1], a[1], len(s.points), s.support)
        for s, b, a in zip(prepared, before, after)
    ]
    return FitResult(
        params=params,
        free=names,
        std={n: float(v) for n, v in zip(names, std)},
        correlation=correlation,
        condition=condition,
        unobservable=unobservable,
        snapshots=stats,
        pivot=pivot,
        leave_one_out=spread,
        warnings=warnings,
        reliable=reliable,
        iterations=done,
        seconds=time.monotonic() - started,
    )


def camera_moved(reference: np.ndarray, camera: np.ndarray, tolerance) -> bool:
    """Whether the 4x4 ``camera`` is off ``reference`` by more than
    ``tolerance`` (m, rad)."""
    shift = float(np.linalg.norm(camera[:3, 3] - reference[:3, 3]))
    relative = reference[:3, :3].T @ camera[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0)))
    return shift > tolerance[0] or angle > tolerance[1]
