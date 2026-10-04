#!/usr/bin/env python3
"""Patch the upstream astra_description URDF at build time.

1. Joint limits: upstream ships every joint with `<limit effort="0"
   velocity="0">`, which gz_ros2_control reads as "zero max torque" so joints
   never move. Rewrite those two attributes with sensible defaults, keyed by
   joint name prefix. Preserves lower/upper.
2. Head tilt link: replace the upstream link_head_tilt mesh (old HeadCamMount +
   old camera) with the D435i bracket from
   aha_perception/hardware/head_cam_mount_d435i, for both visual and
   collision, and replace its inertial with bracket + D435i.

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
    ("revolute", "joint_r"): (30.0, 2.0),
    ("revolute", "joint_l"): (30.0, 2.0),
    # head revolute (light)
    ("revolute", "joint_head"): (10.0, 3.0),
}
FALLBACK = {"prismatic": (50.0, 0.5), "revolute": (20.0, 2.0)}

# D435i head bracket (aha_perception/hardware/head_cam_mount_d435i). The mesh is
# exported from Fusion directly in link_head_tilt coordinates (x forward,
# y down, z left, origin on the tilt/pan axis intersection) in millimetres.
HEAD_TILT_LINK = "link_head_tilt"
HEAD_TILT_MESH = "package://aha_description/meshes/head_cam_mount_d435i.stl"
HEAD_TILT_MESH_SCALE = "0.001 0.001 0.001"
# Inertial = PLA bracket (solid, 1.24 g/cm3, 28.5 g) + D435i as a 72 g box at
# camera_link (52.525 mm forward), about the combined COM in link axes.
HEAD_TILT_INERTIAL = {
    "xyz": "0.04297 0 -0.00285",
    "mass": "0.10050",
    "inertia": {
        "ixx": "6.2585e-05",
        "ixy": "0",
        "ixz": "-1.1244e-05",
        "iyy": "9.3762e-05",
        "iyz": "0",
        "izz": "4.2398e-05",
    },
}


def pick(joint_type: str, name: str) -> tuple[float, float]:
    for (jt, prefix), val in DEFAULTS.items():
        if jt == joint_type and name.startswith(prefix):
            return val
    return FALLBACK.get(joint_type, (10.0, 1.0))


def patch_head_tilt(root: ET.Element) -> bool:
    link = root.find(f"link[@name='{HEAD_TILT_LINK}']")
    if link is None:
        return False
    for tag in ("visual", "collision"):
        for elem in link.findall(tag):
            origin = elem.find("origin")
            if origin is None:
                origin = ET.SubElement(elem, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")
            mesh = elem.find("geometry/mesh")
            if mesh is None:
                raise RuntimeError(f"{HEAD_TILT_LINK} {tag} has no mesh geometry")
            mesh.set("filename", HEAD_TILT_MESH)
            mesh.set("scale", HEAD_TILT_MESH_SCALE)
    inertial = link.find("inertial")
    if inertial is not None:
        inertial.find("origin").set("xyz", HEAD_TILT_INERTIAL["xyz"])
        inertial.find("origin").set("rpy", "0 0 0")
        inertial.find("mass").set("value", HEAD_TILT_INERTIAL["mass"])
        for k, v in HEAD_TILT_INERTIAL["inertia"].items():
            inertial.find("inertia").set(k, v)
    return True


def main(src: str, dst: str) -> int:
    tree = ET.parse(src)
    root = tree.getroot()
    head = patch_head_tilt(root)
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
    print(
        f"[patch_limits] wrote {dst} ({patched} joints patched, "
        f"{HEAD_TILT_LINK} mesh {'replaced' if head else 'NOT FOUND'})"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
