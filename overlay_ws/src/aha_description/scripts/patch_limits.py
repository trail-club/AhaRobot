#!/usr/bin/env python3
"""Patch upstream astra_description URDF joint limits.

Upstream ships every joint with `<limit effort="0" velocity="0">`, which
gz_ros2_control reads as "zero max torque" so joints never move. This script
rewrites those two attributes on every joint with sensible defaults, keyed by
joint name prefix. Preserves lower/upper.

Usage:  patch_limits.py <input.urdf> <output.urdf>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


DEFAULTS = {
    # prismatic lift (0..1.2 m, needs to hold ~20 kg of upper body -> generous)
    ("prismatic", "joint_r1"): (400.0, 0.3),
    ("prismatic", "joint_l1"): (400.0, 0.3),
    # gripper prismatic (small stroke)
    ("prismatic", "joint_r7"): (30.0, 0.2),
    ("prismatic", "joint_l7"): (30.0, 0.2),
    # arm revolute
    ("revolute",  "joint_r"):  (30.0, 2.0),
    ("revolute",  "joint_l"):  (30.0, 2.0),
    # head revolute (light)
    ("revolute",  "joint_head"): (10.0, 3.0),
}
FALLBACK = {"prismatic": (50.0, 0.5), "revolute": (20.0, 2.0)}


def pick(joint_type: str, name: str) -> tuple[float, float]:
    for (jt, prefix), val in DEFAULTS.items():
        if jt == joint_type and name.startswith(prefix):
            return val
    return FALLBACK.get(joint_type, (10.0, 1.0))


def main(src: str, dst: str) -> int:
    tree = ET.parse(src)
    root = tree.getroot()
    patched = 0
    for j in root.findall("joint"):
        jt = j.get("type")
        if jt in ("fixed", "floating", "planar"):
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        eff, vel = pick(jt, j.get("name", ""))
        lim.set("effort", str(eff))
        lim.set("velocity", str(vel))
        patched += 1
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    print(f"[patch_limits] wrote {dst} ({patched} joints patched)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
