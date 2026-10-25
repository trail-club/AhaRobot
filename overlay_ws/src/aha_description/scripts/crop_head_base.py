#!/usr/bin/env python3
"""Cut the head base (the printed HeadYaw part with the pan servo) out of the
upstream base_link mesh, in coordinates relative to the joint_head_pan
origin (m, base_link axes), for the frame model (aha_description/frame.py).

The Astra body mesh carries the head base on the column top: the triangles
above HEAD_BASE_MIN_Z with their center in front of HEAD_BASE_MIN_X.

Usage:  crop_head_base.py <astra_description.urdf> <base_link.STL> <output.stl>
"""

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HEAD_BASE_MIN_Z = 1.46  # m in base_link: below the head base, above the column
HEAD_BASE_MIN_X = 0.0  # m: the column top plate behind it stays out


def read_stl(path: Path) -> list[tuple[tuple[float, ...], ...]]:
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if 84 + count * 50 != len(data):
        raise ValueError(f"{path} is not a binary STL")
    triangles = []
    for i in range(count):
        values = struct.unpack("<12f", data[84 + i * 50 : 84 + i * 50 + 48])
        triangles.append((values[3:6], values[6:9], values[9:12]))
    return triangles


def normal(triangle) -> tuple[float, float, float]:
    """Unit facet normal (right-hand winding); zero for a degenerate facet."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    length = sum(c * c for c in n) ** 0.5
    return (0.0, 0.0, 0.0) if length == 0.0 else tuple(c / length for c in n)


def write_stl(path: Path, triangles) -> None:
    out = bytearray(b"aha_description head base".ljust(80, b"\0"))
    out += struct.pack("<I", len(triangles))
    for triangle in triangles:
        out += struct.pack("<3f", *normal(triangle))
        for vertex in triangle:
            out += struct.pack("<3f", *vertex)
        out += b"\0\0"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def pan_origin(urdf: Path) -> tuple[float, float, float]:
    origin = ET.parse(urdf).getroot().find("joint[@name='joint_head_pan']/origin")
    if origin is None:
        raise ValueError(f"{urdf} has no joint_head_pan origin")
    return tuple(float(v) for v in origin.get("xyz").split())


def crop(triangles, origin):
    out = []
    for triangle in triangles:
        center_x = sum(v[0] for v in triangle) / 3
        if min(v[2] for v in triangle) > HEAD_BASE_MIN_Z and center_x > HEAD_BASE_MIN_X:
            out.append(
                tuple(tuple(v[i] - origin[i] for i in range(3)) for v in triangle)
            )
    return out


def main(urdf, mesh, output) -> int:
    triangles = crop(read_stl(Path(mesh)), pan_origin(Path(urdf)))
    if not triangles:
        print(f"no head base triangles in {mesh}", file=sys.stderr)
        return 1
    write_stl(Path(output), triangles)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
