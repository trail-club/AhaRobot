# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "networkx", "scipy"]
# ///
"""Build the link_l6 / link_r6 mesh and inertial for this wrist part.

The upstream link_r6.STL (geometrically identical to link_l6.STL) is one
assembly of 33 bodies. Drop the original Effector and the old upstream camera,
add wrist_cam_mount_d435i.stl and the D435i as a box, and write the result in
link_6 coordinates (mm) to aha_description/meshes/wrist_cam_mount_d435i_link6.stl.
Like the old upstream camera, the D435i is part of link_6 (visual and
collision), so it is drawn, collides and shows up in arm point cloud models.

Also prints the inertial for aha_description/scripts/patch_limits.py: the
upstream link_6 inertial, plus the plastic added to the Effector (solid PLA),
plus the D435i as a 72 g box at camera_link. The upstream inertial is read
from astra_description.urdf and the camera pose and size from
aha_description/urdf/sensors.xacro.

Usage (from anywhere):  uv run export_urdf_mesh.py
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[5]
UPSTREAM_MESH = ROOT / "upstream/astra_description/meshes/link_r6.STL"
UPSTREAM_URDF = ROOT / "upstream/astra_description/urdf/astra_description.urdf"
SENSORS_XACRO = ROOT / "overlay_ws/src/aha_description/urdf/sensors.xacro"
PART_MESH = Path(__file__).with_name("wrist_cam_mount_d435i.stl")
OUT_MESH = (
    ROOT / "overlay_ws/src/aha_description/meshes/wrist_cam_mount_d435i_link6.stl"
)

# Part frame (mm) -> link_6 frame (mm): p_link = R @ p_part + t.
PART_TO_LINK_R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
PART_TO_LINK_T = np.array([56.0, 0.0, 75.0])

# Bodies of the upstream mesh replaced by this part, as part-frame bounds (mm).
REPLACED_BODIES = {
    "Effector": ((-18.0, -51.8, -11.0), (18.0, 40.0, 74.0)),
    "old camera": ((-19.5, 10.9, 4.0), (19.5, 42.9, 49.1)),
}
BOUNDS_TOL_MM = 0.5
UPSTREAM_BODY_COUNT = 33

PLA_DENSITY = 1240.0  # kg/m3, solid
CAMERA_MASS = 0.072


def upstream_inertial() -> tuple[float, np.ndarray, np.ndarray]:
    """link_r6 mass, COM and inertia about the COM (SI, link frame).

    Raises RuntimeError when link_l6 differs, since both use this mesh.
    """
    root = ET.parse(UPSTREAM_URDF).getroot()

    def read(name: str) -> tuple[float, np.ndarray, np.ndarray]:
        inertial = root.find(f"link[@name='{name}']/inertial")
        if inertial is None:
            raise RuntimeError(f"{UPSTREAM_URDF.name}: no inertial for {name}")
        origin = inertial.find("origin")
        if origin is None or any(map(float, origin.get("rpy", "0 0 0").split())):
            raise RuntimeError(
                f"{UPSTREAM_URDF.name}: {name} inertial origin unexpected"
            )
        com = np.array([float(v) for v in origin.get("xyz").split()])
        mass = float(inertial.find("mass").get("value"))
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        inertia = np.array(
            [
                [i["ixx"], i["ixy"], i["ixz"]],
                [i["ixy"], i["iyy"], i["iyz"]],
                [i["ixz"], i["iyz"], i["izz"]],
            ]
        )
        return mass, com, inertia

    right, left = read("link_r6"), read("link_l6")
    if not all(np.array_equal(a, b) for a, b in zip(right, left)):
        raise RuntimeError(
            f"{UPSTREAM_URDF.name}: link_l6 and link_r6 inertials differ"
        )
    return right


def xacro_property(text: str, name: str, value_pattern: str) -> float:
    match = re.search(
        rf'<xacro:property\s+name="{name}"\s+value="{value_pattern}"\s*/>', text
    )
    if match is None:
        raise RuntimeError(
            f"{SENSORS_XACRO.name}: property {name} not found as {value_pattern!r}"
        )
    return float(match.group(1))


def camera_design() -> tuple[np.ndarray, float, np.ndarray]:
    """D435i on the back plate from sensors.xacro: camera_link size (x, y, z),
    tilt (rad) and back edge in link_6 (SI)."""
    text = SENSORS_XACRO.read_text(encoding="utf-8")
    number = r"([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)"

    def prop(name: str) -> float:
        return xacro_property(text, name, number)

    size = np.array([prop(f"camera_body_{axis}") for axis in "xyz"])
    tilt = np.radians(
        xacro_property(text, "wrist_camera_tilt", rf"\$\{{radians\({number}\)\}}")
    )
    back_edge = np.array(
        [prop("wrist_camera_back_edge_x"), 0.0, prop("wrist_camera_back_edge_z")]
    )
    return size, tilt, back_edge


def to_link(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = PART_TO_LINK_R
    transform[:3, 3] = PART_TO_LINK_T
    out.apply_transform(transform)
    return out


def to_part(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = PART_TO_LINK_R.T
    transform[:3, 3] = -PART_TO_LINK_R.T @ PART_TO_LINK_T
    out.apply_transform(transform)
    return out


def split_upstream(
    link_mm: trimesh.Trimesh,
) -> tuple[list[trimesh.Trimesh], dict[str, trimesh.Trimesh]]:
    """Kept bodies, and the replaced ones by name."""
    bodies = link_mm.split(only_watertight=False)
    if len(bodies) != UPSTREAM_BODY_COUNT:
        raise RuntimeError(
            f"{UPSTREAM_MESH.name}: {len(bodies)} bodies, expected {UPSTREAM_BODY_COUNT}"
        )
    kept = []
    replaced: dict[str, list[trimesh.Trimesh]] = {name: [] for name in REPLACED_BODIES}
    for body in bodies:
        bounds = to_part(body).bounds
        for name, expected in REPLACED_BODIES.items():
            if np.allclose(bounds, expected, atol=BOUNDS_TOL_MM):
                replaced[name].append(body)
                break
        else:
            kept.append(body)
    counts = {name: len(found) for name, found in replaced.items()}
    if any(n != 1 for n in counts.values()):
        raise RuntimeError(f"replaced bodies not matched exactly once: {counts}")
    return kept, {name: found[0] for name, found in replaced.items()}


def second_moment(mass: float, com: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Inertia about the link origin (parallel axis theorem)."""
    return inertia + mass * (com @ com * np.eye(3) - np.outer(com, com))


def solid(mesh_mm: trimesh.Trimesh) -> tuple[float, np.ndarray, np.ndarray]:
    mesh = mesh_mm.copy()
    mesh.apply_scale(0.001)
    mesh.density = PLA_DENSITY
    if not mesh.is_watertight:
        raise RuntimeError("mass properties need a watertight mesh")
    return mesh.mass, mesh.center_mass, mesh.moment_inertia


def camera_pose(
    size: np.ndarray, tilt: float, back_edge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """camera_link position and rotation (columns = camera x, y, z) in link_6."""
    s, c = np.sin(tilt), np.cos(tilt)
    x_axis = np.array([s, 0.0, c])
    z_axis = np.array([-c, 0.0, s])
    rot = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    pos = back_edge + size[0] / 2 * x_axis + size[2] / 2 * z_axis
    return pos, rot


def inertial(part_link_mm: trimesh.Trimesh, effector_link_mm: trimesh.Trimesh):
    cam_size, cam_tilt, cam_back_edge = camera_design()
    cam_pos, cam_rot = camera_pose(cam_size, cam_tilt, cam_back_edge)
    dx, dy, dz = cam_size
    cam_local = (
        CAMERA_MASS / 12 * np.diag([dy**2 + dz**2, dx**2 + dz**2, dx**2 + dy**2])
    )
    contributions = [
        (*upstream_inertial(), 1.0),
        (*solid(part_link_mm), 1.0),
        (*solid(effector_link_mm), -1.0),
        (CAMERA_MASS, cam_pos, cam_rot @ cam_local @ cam_rot.T, 1.0),
    ]
    mass = sum(sign * m for m, _, _, sign in contributions)
    com = sum(sign * m * p for m, p, _, sign in contributions) / mass
    about_origin = sum(sign * second_moment(m, p, i) for m, p, i, sign in contributions)
    about_com = about_origin - mass * (com @ com * np.eye(3) - np.outer(com, com))
    added_plastic = contributions[1][0] - contributions[2][0]
    return mass, com, about_com, added_plastic, cam_pos


def main() -> None:
    link_mm = trimesh.load(UPSTREAM_MESH)
    link_mm.apply_scale(1000.0)
    part_link = to_link(trimesh.load(PART_MESH))
    kept, replaced = split_upstream(link_mm)

    cam_size, cam_tilt, cam_back_edge = camera_design()
    cam_pos, cam_rot = camera_pose(cam_size, cam_tilt, cam_back_edge)
    cam_transform = np.eye(4)
    cam_transform[:3, :3] = cam_rot
    cam_transform[:3, 3] = cam_pos * 1000.0
    camera_box = trimesh.creation.box(cam_size * 1000.0, transform=cam_transform)

    combined = trimesh.util.concatenate([*kept, part_link, camera_box])
    OUT_MESH.parent.mkdir(parents=True, exist_ok=True)
    combined.export(OUT_MESH)
    print(f"wrote {OUT_MESH.relative_to(ROOT)} ({len(combined.faces)} faces)")

    mass, com, inertia, added, cam_pos = inertial(part_link, replaced["Effector"])
    print(f"added plastic {added * 1000:.2f} g, camera_link at {np.round(cam_pos, 6)}")
    print(f'    "xyz": "{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}",')
    print(f'    "mass": "{mass:.5f}",')
    names = (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2))
    names += (("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))
    for name, r, c in names:
        print(f'        "{name}": "{inertia[r, c]:.4e}",')


if __name__ == "__main__":
    main()
