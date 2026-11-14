#!/usr/bin/env python3
"""Merge the printed display mount parts into one mesh for display_link.

Usage: display_mount_mesh.py <hardware/display_mount> <output.stl>

The inputs are the print STLs of hardware/display_mount (binary, mm). Each is
in its part's frame: the minimum corner at the part origin of
fusion/build_display_mount.py, and the frame adapter turned onto its print
face. The output is a binary STL in display_link coordinates (mm): origin at
the centre of the screen's front face, x out of the screen, y left, z up.
"""

import struct
import sys
from pathlib import Path

# Part origins in the Astra STEP frame (X left, Y up, Z forward; mm), as placed
# by fusion/build_display_mount.py.
PARTS = {
    "01_frame_adapter": (-28.0, 1180.0, 425.26820827052),
    "02_lower_left": (-133.3, 992.0, 510.0),
    "03_lower_right": (-18.0, 992.0, 510.0),
    "04_top_keeper": (-45.0, 1132.0, 510.0),
}
# The print STL of the frame adapter is (x, z, height - y) of its part frame.
ADAPTER_HEIGHT = 161.0
# display_link origin in the STEP frame: centre of the case front face.
SCREEN_CENTER = (0.0, 1093.5, 586.1)


def read_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    return [
        [struct.unpack_from("<3f", data, 84 + 50 * i + 12 * (k + 1)) for k in range(3)]
        for i in range(count)
    ]


def to_display_link(name, vertex):
    x, y, z = vertex
    if name == "01_frame_adapter":
        x, y, z = x, ADAPTER_HEIGHT - z, y
    step = [v + o for v, o in zip((x, y, z), PARTS[name])]
    # STEP (X left, Y up, Z forward) -> (x forward, y left, z up); a rotation.
    return (
        step[2] - SCREEN_CENTER[2],
        step[0] - SCREEN_CENTER[0],
        step[1] - SCREEN_CENTER[1],
    )


def normal(a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = sum(x * x for x in n) ** 0.5
    return tuple(x / length for x in n) if length else (0.0, 0.0, 0.0)


def main(argv):
    source, output = Path(argv[1]), Path(argv[2])
    triangles = [
        [to_display_link(name, vertex) for vertex in triangle]
        for name in PARTS
        for triangle in read_stl(source / f"{name}.stl")
    ]
    header = b"AhaRobot display mount in display_link (mm)".ljust(80, b" ")
    body = b"".join(
        struct.pack("<12fH", *normal(*t), *t[0], *t[1], *t[2], 0) for t in triangles
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + struct.pack("<I", len(triangles)) + body)
    print(f"Wrote {len(triangles)} triangles to {output}")


if __name__ == "__main__":
    main(sys.argv)
