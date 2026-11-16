"""Robot self filter for cloud_accumulator.py: one box per link.

Kept free of rclpy so it can be tested on the host.

``link_boxes`` reads a URDF and returns, per link with collision geometry,
the axis-aligned bounding box in the link frame of all its collision
geometries (binary STL meshes with their scale, boxes, cylinders and
spheres, placed by the collision origin), grown by ``padding``. A point is
on the robot if it is inside the box of any link at the link's current pose
(``points_in_boxes``). The arm and head links fit their boxes closely; the
base_link box also covers the space in front of the mast, but not beyond the
base's front.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

STL_HEADER = 80
STL_TRIANGLE = 50  # normal, 3 vertices (12 float32) and the attribute count


@dataclass(frozen=True)
class LinkBox:
    """Bounding box of a link's collision geometry in the link frame."""

    link: str
    lower: np.ndarray
    upper: np.ndarray


@lru_cache(maxsize=64)
def read_stl_vertices(path: str) -> np.ndarray:
    """(3 * triangles, 3) float64 vertices of a binary STL (ValueError otherwise)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read()
    if size < STL_HEADER + 4:
        raise ValueError(f"{path}: too short for a binary STL")
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=STL_HEADER)[0])
    if size != STL_HEADER + 4 + count * STL_TRIANGLE:
        raise ValueError(f"{path}: not a binary STL (ASCII STL is not supported)")
    triangles = np.frombuffer(
        data,
        dtype=np.dtype([("normal", "<f4", 3), ("v", "<f4", 9), ("attr", "<u2")]),
        count=count,
        offset=STL_HEADER + 4,
    )
    return triangles["v"].reshape(-1, 3).astype(np.float64)


def package_path(uri: str) -> str:
    """Local path of a package:// or file:// URI (other strings are paths)."""
    if uri.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory

        package, _, relative = uri[len("package://") :].partition("/")
        return os.path.join(get_package_share_directory(package), relative)
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _floats(text: str | None, count: int, default: float) -> np.ndarray:
    if text is None:
        return np.full(count, default)
    values = np.array([float(v) for v in text.split()])
    return values if len(values) == count else np.full(count, values[0])


def _corners(half: np.ndarray) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return signs * half


def geometry_vertices(
    geometry: ET.Element, resolve: Callable[[str], str]
) -> np.ndarray:
    """(N, 3) points whose bounding box is that of a URDF <geometry>."""
    shape = geometry[0]
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise ValueError("mesh without filename")
        scale = _floats(shape.get("scale"), 3, 1.0)
        return read_stl_vertices(resolve(filename)) * scale
    if shape.tag == "box":
        return _corners(_floats(shape.get("size"), 3, 0.0) / 2.0)
    if shape.tag == "cylinder":
        r = float(shape.get("radius"))
        return _corners(np.array([r, r, float(shape.get("length")) / 2.0]))
    if shape.tag == "sphere":
        return _corners(np.full(3, float(shape.get("radius"))))
    raise ValueError(f"unsupported geometry {shape.tag!r}")


def link_boxes(
    urdf: str,
    padding: float,
    resolve: Callable[[str], str] = package_path,
) -> tuple[list[LinkBox], list[str]]:
    """(boxes, warnings) of the links with collision geometry of a URDF string.

    A collision that cannot be read (missing or ASCII mesh, unknown shape)
    is skipped with a warning; the link keeps its other collisions.
    """
    root = ET.fromstring(urdf)
    boxes, warnings = [], []
    for link in root.findall("link"):
        name = link.get("name")
        vertices = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            try:
                local = geometry_vertices(geometry, resolve)
            except (OSError, ValueError, LookupError, TypeError) as e:
                warnings.append(f"{name}: collision skipped: {e}")
                continue
            origin = collision.find("origin")
            xyz = _floats(origin.get("xyz") if origin is not None else None, 3, 0.0)
            rpy = _floats(origin.get("rpy") if origin is not None else None, 3, 0.0)
            vertices.append(local @ rpy_matrix(*rpy).T + xyz)
        if vertices:
            stacked = np.concatenate(vertices)
            boxes.append(
                LinkBox(
                    name,
                    stacked.min(axis=0) - padding,
                    stacked.max(axis=0) + padding,
                )
            )
    return boxes, warnings


def points_in_boxes(
    points: np.ndarray, boxes: Sequence[LinkBox], poses: Sequence[np.ndarray]
) -> np.ndarray:
    """(N,) bool: points inside any box, each at its 4x4 link -> points pose."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    for box, pose in zip(boxes, poses, strict=True):
        pose = np.asarray(pose, dtype=np.float64)
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        inside |= np.all((local >= box.lower) & (local <= box.upper), axis=1)
    return inside
