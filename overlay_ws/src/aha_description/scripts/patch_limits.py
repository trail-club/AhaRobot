#!/usr/bin/env python3
"""Patch the upstream astra_description URDF at build time.

1. Joint limits: upstream ships every joint with `<limit effort="0"
   velocity="0">`, which gz_ros2_control reads as "zero max torque" so joints
   never move. Rewrite those two attributes with sensible defaults, keyed by
   joint name prefix. Preserves lower/upper.
2. Camera mount links: replace the visual and collision meshes and the
   inertial of
   - link_head_tilt (old HeadCamMount + old camera) with the D435i bracket from
     aha_perception/hardware/head_cam_mount_d435i, inertial bracket + D435i;
   - link_l6 / link_r6 (gripper base with the old camera) with the same
     assembly using the wrist part from
     aha_perception/hardware/wrist_cam_mount_d435i, inertial upstream + added
     plastic + D435i.

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

MESH_SCALE = "0.001 0.001 0.001"

# D435i head bracket (aha_perception/hardware/head_cam_mount_d435i). The mesh is
# exported from Fusion directly in link_head_tilt coordinates (x forward,
# y down, z left, origin on the tilt/pan axis intersection) in millimetres.
# Inertial = PLA bracket (solid, 1.24 g/cm3, 28.5 g) + D435i as a 72 g box at
# camera_link (52.525 mm forward), about the combined COM in link axes.
HEAD_TILT_PATCH = {
    "mesh": "package://aha_description/meshes/head_cam_mount_d435i.stl",
    "inertial": {
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
    },
}

# Gripper base with the D435i wrist part; mesh in link_6 coordinates in
# millimetres (including the D435i box, as the upstream mesh included its
# camera), inertial in SI (m, kg, kg m2). Both come from
# aha_perception/hardware/wrist_cam_mount_d435i/export_urdf_mesh.py.
# Inertial: upstream link_6 + added PLA (solid, 13.3 g) + D435i as a 72 g box
# at the wrist camera link, about the combined COM in link axes.
WRIST_PATCH = {
    "mesh": "package://aha_description/meshes/wrist_cam_mount_d435i_link6.stl",
    "inertial": {
        "xyz": "0.02580 0.00003 0.07162",
        "mass": "0.22465",
        "inertia": {
            "ixx": "1.8250e-04",
            "ixy": "-8.2150e-08",
            "ixz": "2.1709e-05",
            "iyy": "1.6825e-04",
            "iyz": "5.3912e-08",
            "izz": "1.9965e-04",
        },
    },
}

LINK_PATCHES = {
    "link_head_tilt": HEAD_TILT_PATCH,
    "link_l6": WRIST_PATCH,
    "link_r6": WRIST_PATCH,
}


def pick(joint_type: str, name: str) -> tuple[float, float]:
    for (jt, prefix), val in DEFAULTS.items():
        if jt == joint_type and name.startswith(prefix):
            return val
    return FALLBACK.get(joint_type, (10.0, 1.0))


def child(parent: ET.Element, tag: str) -> ET.Element:
    """The first `tag` child of parent, created when missing."""
    elem = parent.find(tag)
    return elem if elem is not None else ET.SubElement(parent, tag)


def patch_link(root: ET.Element, name: str, patch: dict) -> None:
    """Raises RuntimeError when the upstream link no longer has the expected shape."""
    link = root.find(f"link[@name='{name}']")
    if link is None:
        raise RuntimeError(f"link {name} not found")
    if link.find("visual") is None:
        raise RuntimeError(f"{name} has no visual")
    for tag in ("visual", "collision"):
        for elem in link.findall(tag):
            origin = child(elem, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")
            mesh = elem.find("geometry/mesh")
            if mesh is None:
                raise RuntimeError(f"{name} {tag} has no mesh geometry")
            mesh.set("filename", patch["mesh"])
            mesh.set("scale", MESH_SCALE)
    values = patch["inertial"]
    inertial = child(link, "inertial")
    origin = child(inertial, "origin")
    origin.set("xyz", values["xyz"])
    origin.set("rpy", "0 0 0")
    child(inertial, "mass").set("value", values["mass"])
    inertia = child(inertial, "inertia")
    for k, v in values["inertia"].items():
        inertia.set(k, v)


def main(src: str, dst: str) -> int:
    tree = ET.parse(src)
    root = tree.getroot()
    try:
        for name, patch in LINK_PATCHES.items():
            patch_link(root, name, patch)
    except RuntimeError as e:
        # Fail the build rather than ship the old mesh and inertia.
        print(f"[patch_limits] {src}: {e}", file=sys.stderr)
        return 1
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
        f"{', '.join(LINK_PATCHES)} meshes replaced)"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
