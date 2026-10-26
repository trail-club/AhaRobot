"""Planning scene objects for placing: perceived planes, obstacles and the held object.

Horizontal planes (tables, shelf boards) become thin boxes under the surface,
fitted to the minimum-area rectangle of the convex hull and grown by
``support_padding`` on the sides; walls become thin
boxes centered on the plane. Perceived obstacles (aha_msgs/PlaceObstacle)
become padded boxes standing on their surface. The held object is attached
to the TCP link at its pose in the hand (the goal's ``object_in_hand``, or
the grasp's nominal ``place_geometry.object_in_tcp``).
Also box tests (overlap, clipping the top) and allowed collision matrix
edits for the place server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from aha_msgs.msg import PlaceObstacle
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    AttachedCollisionObject,
    CollisionObject,
)
from shape_msgs.msg import SolidPrimitive

from aha_test_manipulation.place_geometry import (
    matrix_from_quaternion,
    quaternion_from_matrix,
    rot_z,
    transform,
)

SURFACE_PREFIX = "aha_place_surface_"
WALL_PREFIX = "aha_place_wall_"
OBSTACLE_PREFIX = "aha_place_obstacle_"
PERCEIVED_PREFIXES = (SURFACE_PREFIX, WALL_PREFIX, OBSTACLE_PREFIX)

BOX = "box"
CYLINDER = "cylinder"


@dataclass
class SceneParams:
    support_thickness: float = 0.03  # box below a horizontal surface
    surface_clearance: float = 0.005  # box top below the detected surface
    # Horizontal growth of a support box on every side: plans that graze the
    # detected edge hit the real board (arm tracking, held object offset).
    support_padding: float = 0.0
    wall_thickness: float = 0.02
    min_height: float = 0.05  # horizontal planes below this are floor
    horizontal_min_nz: float = 0.9
    vertical_max_nz: float = 0.2


SURFACE = "surface"
WALL = "wall"
OBSTACLE = "obstacle"


@dataclass
class Box:
    kind: str  # SURFACE | WALL | OBSTACLE
    pose: np.ndarray  # 4x4 center pose in the plane frame
    size: tuple[float, float, float]


def min_area_rectangle(
    points: np.ndarray,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Minimum-area rectangle of 2D points: (center, angle of the first side, (length, width))."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("no points")
    if len(pts) < 3:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (lo + hi) / 2, 0.0, (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    best = None
    for i in range(len(pts)):
        edge = pts[(i + 1) % len(pts)] - pts[i]
        if np.hypot(*edge) < 1e-9:
            continue
        angle = math.atan2(edge[1], edge[0])
        R = rot_z(-angle)[:2, :2]
        local = pts @ R.T
        lo, hi = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            center = rot_z(angle)[:2, :2] @ ((lo + hi) / 2)
            best = (area, center, angle, (float(hi[0] - lo[0]), float(hi[1] - lo[1])))
    if best is None:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (lo + hi) / 2, 0.0, (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    return best[1], best[2], best[3]


def _hull_points(plane) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in plane.hull.points], dtype=float).reshape(
        -1, 3
    )


def _normal(plane) -> np.ndarray:
    n = np.array([plane.normal.x, plane.normal.y, plane.normal.z], dtype=float)
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n


def plane_box(plane, params: SceneParams | None = None) -> Box | None:
    """Collision box for an aha_msgs/Plane, or None (floor, slanted, empty hull)."""
    params = params or SceneParams()
    points = _hull_points(plane)
    if len(points) < 3 or plane.label == "floor":
        return None
    n = _normal(plane)
    if abs(n[2]) >= params.horizontal_min_nz:
        center, angle, (length, width) = min_area_rectangle(points[:, :2])
        # Height on the plane at the rectangle center: n . p + offset = 0.
        z = -(n[0] * center[0] + n[1] * center[1] + plane.offset) / n[2]
        if z < params.min_height:
            return None
        top = z - params.surface_clearance
        T = transform(
            rot_z(angle), [center[0], center[1], top - params.support_thickness / 2]
        )
        grow = 2 * params.support_padding
        return Box(SURFACE, T, (length + grow, width + grow, params.support_thickness))
    if abs(n[2]) <= params.vertical_max_nz:
        horizontal = np.array([n[0], n[1], 0.0])
        horizontal /= np.linalg.norm(horizontal)
        along = np.array([-horizontal[1], horizontal[0], 0.0])
        c = np.array([plane.centroid.x, plane.centroid.y, plane.centroid.z])
        u = (points - c) @ along
        z = points[:, 2]
        center = c + along * (u.min() + u.max()) / 2
        center[2] = (z.min() + z.max()) / 2
        yaw = math.atan2(horizontal[1], horizontal[0])
        size = (
            params.wall_thickness,
            float(u.max() - u.min()),
            float(z.max() - z.min()),
        )
        return Box(WALL, transform(rot_z(yaw), center), size)
    return None


def obstacle_box(obstacle, padding: float = 0.0) -> Box:
    """Box for an aha_msgs/PlaceObstacle, grown by ``padding`` on the sides and
    the top; the bottom stays on the surface."""
    p, q = obstacle.pose.position, obstacle.pose.orientation
    T = transform(
        matrix_from_quaternion(q.x, q.y, q.z, q.w),
        [p.x, p.y, p.z],
    )
    T[:3, 3] += T[:3, 2] * padding / 2
    s = obstacle.size
    size = (s.x + 2 * padding, s.y + 2 * padding, s.z + padding)
    return Box(OBSTACLE, T, tuple(float(v) for v in size))


def box_distance(box: Box, point) -> float:
    """Distance from a point to the box (0 inside)."""
    local = box.pose[:3, :3].T @ (np.asarray(point, dtype=float) - box.pose[:3, 3])
    outside = np.maximum(np.abs(local) - np.asarray(box.size) / 2, 0.0)
    return float(np.linalg.norm(outside))


def object_at(
    obstacles, point, surface_height: float, margin: float, height_tolerance: float
) -> bool:
    """Whether a perceived obstacle (aha_msgs/PlaceObstacle) stands on the
    surface at ``surface_height`` within ``margin`` (horizontal) of ``point``."""
    for obstacle in obstacles:
        box = obstacle_box(obstacle)
        center = box.pose[:3, 3]
        if abs(center[2] - box.size[2] / 2 - surface_height) > height_tolerance:
            continue
        if box_distance(box, (point[0], point[1], center[2])) <= margin:
            return True
    return False


def boxes_overlap(a: Box, b: Box) -> bool:
    """Whether two oriented boxes intersect (separating axis test)."""
    Ra, Rb = a.pose[:3, :3], b.pose[:3, :3]
    ha, hb = np.asarray(a.size, dtype=float) / 2, np.asarray(b.size, dtype=float) / 2
    t = b.pose[:3, 3] - a.pose[:3, 3]
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        ra = float(np.sum(ha * np.abs(Ra.T @ axis)))
        rb = float(np.sum(hb * np.abs(Rb.T @ axis)))
        if abs(float(t @ axis)) > ra + rb:
            return False
    return True


def box_corners(box: Box) -> np.ndarray:
    """(8, 3) corners of a box (base frame)."""
    half = np.asarray(box.size, dtype=float) / 2
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return (signs * half) @ box.pose[:3, :3].T + box.pose[:3, 3]


def clear_box(box: Box, points, margin: float, min_size: float) -> Box | None:
    """An upright box cut so that it keeps ``margin`` from the convex hull of
    ``points`` (base frame; e.g. the corners of the touching link), losing
    the least volume: its top lowered below the lowest point, or one side
    face moved back past the points. The bottom stays on the surface. None
    when every cut would leave less than ``min_size``; the box itself when it
    already clears the points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    R, center = box.pose[:3, :3], box.pose[:3, 3]
    half = np.asarray(box.size, dtype=float) / 2
    local = (pts - center) @ R
    lo, hi = local.min(axis=0) - margin, local.max(axis=0) + margin
    if np.any(hi < -half) or np.any(lo > half):
        return box
    options = []  # (removed volume, new lower corner, new upper corner)
    for axis in range(2):  # cut a side: keep the part away from the points
        for keep_low in (True, False):
            low, high = -half.copy(), half.copy()
            if keep_low:
                high[axis] = lo[axis]
            else:
                low[axis] = hi[axis]
            options.append((low, high))
    low, high = -half.copy(), half.copy()
    high[2] = lo[2]  # lower the top
    options.append((low, high))
    best = None
    for low, high in options:
        size = high - low
        if np.any(size < min_size):
            continue
        removed = float(np.prod(2 * half) - np.prod(size))
        if best is None or removed < best[0]:
            best = (removed, low, high)
    if best is None:
        return None
    _, low, high = best
    T = box.pose.copy()
    T[:3, 3] = center + R @ ((low + high) / 2)
    return Box(box.kind, T, tuple(float(v) for v in high - low))


def obstacle_msg(box: Box) -> PlaceObstacle:
    """aha_msgs/PlaceObstacle (center pose and size) of a box."""
    msg = PlaceObstacle()
    msg.pose = pose_msg(box.pose)
    msg.size.x, msg.size.y, msg.size.z = (float(v) for v in box.size)
    return msg


def acm_forget(acm: AllowedCollisionMatrix, forget) -> AllowedCollisionMatrix:
    """A copy of ``acm`` without the entries whose name satisfies ``forget``."""
    keep = [i for i, name in enumerate(acm.entry_names) if not forget(name)]
    out = AllowedCollisionMatrix()
    out.entry_names = [acm.entry_names[i] for i in keep]
    out.entry_values = [
        AllowedCollisionEntry(
            enabled=[bool(acm.entry_values[i].enabled[j]) for j in keep]
        )
        for i in keep
    ]
    pairs = [
        (name, bool(value))
        for name, value in zip(acm.default_entry_names, acm.default_entry_values)
        if not forget(name)
    ]
    out.default_entry_names = [name for name, _ in pairs]
    out.default_entry_values = [value for _, value in pairs]
    return out


def acm_allow(acm: AllowedCollisionMatrix, name: str, others) -> AllowedCollisionMatrix:
    """A copy of ``acm`` that allows collisions between ``name`` and ``others``
    (entries are added as needed)."""
    out = acm_forget(acm, lambda _: False)
    rows = [list(entry.enabled) for entry in out.entry_values]
    for entry in [name, *others]:
        if entry not in out.entry_names:
            out.entry_names.append(entry)
            for row in rows:
                row.append(False)
            rows.append([False] * len(out.entry_names))
    i = out.entry_names.index(name)
    for other in others:
        j = out.entry_names.index(other)
        rows[i][j] = rows[j][i] = True
    out.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
    return out


def pose_msg(T: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    q = quaternion_from_matrix(T[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def primitive(shape: str, size) -> SolidPrimitive:
    """Box (x, y, z) or upright cylinder (diameter max(x, y), height z)."""
    prim = SolidPrimitive()
    if shape == CYLINDER:
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [float(size[2]), float(max(size[0], size[1])) / 2]
    elif shape == BOX:
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(v) for v in size]
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return prim


def collision_object(
    object_id: str, frame: str, shape: str, size, T: np.ndarray, stamp=None
) -> CollisionObject:
    obj = CollisionObject()
    obj.id = object_id
    obj.header.frame_id = frame
    if stamp is not None:
        obj.header.stamp = stamp
    obj.pose = pose_msg(T)
    obj.primitives = [primitive(shape, size)]
    obj.primitive_poses = [Pose()]
    obj.primitive_poses[0].orientation.w = 1.0
    obj.operation = CollisionObject.ADD
    return obj


def remove_object(object_id: str, frame: str = "") -> CollisionObject:
    obj = CollisionObject()
    obj.id = object_id
    obj.header.frame_id = frame
    obj.operation = CollisionObject.REMOVE
    return obj


def plane_objects(
    planes, frame: str, params: SceneParams | None = None
) -> list[CollisionObject]:
    """Collision objects for the planes (supports first, then walls); ids by index."""
    objects = []
    counts = {SURFACE: 0, WALL: 0}
    prefixes = {SURFACE: SURFACE_PREFIX, WALL: WALL_PREFIX}
    for plane in planes:
        box = plane_box(plane, params)
        if box is None:
            continue
        object_id = f"{prefixes[box.kind]}{counts[box.kind]}"
        counts[box.kind] += 1
        objects.append(collision_object(object_id, frame, BOX, box.size, box.pose))
    return objects


def obstacle_objects(boxes, frame: str, start: int = 0) -> list[CollisionObject]:
    """Collision objects ``aha_place_obstacle_<index>`` for obstacle boxes,
    numbered from ``start``."""
    return [
        collision_object(f"{OBSTACLE_PREFIX}{k}", frame, BOX, box.size, box.pose)
        for k, box in enumerate(boxes, start)
    ]


def held_object(
    object_id: str,
    link: str,
    touch_links,
    shape: str,
    size,
    T_tcp_object: np.ndarray,
) -> AttachedCollisionObject:
    """AttachedCollisionObject on the TCP link with the object at T_tcp_object."""
    attached = AttachedCollisionObject()
    attached.link_name = link
    attached.touch_links = list(touch_links)
    attached.object = collision_object(object_id, link, shape, size, T_tcp_object)
    return attached


def detach_object(object_id: str, link: str) -> AttachedCollisionObject:
    attached = AttachedCollisionObject()
    attached.link_name = link
    attached.object.id = object_id
    attached.object.operation = CollisionObject.REMOVE
    return attached
