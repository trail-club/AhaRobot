"""Real robot on a measured frame instead of the Astra body (ROS free).

``frame_robot`` rewrites the URDF of aha_robot.urdf.xacro (sim:=false) for a
robot whose arms and head stand on a frame of aluminum extrusions, without
the mobile base and the lifts. The YAML config (``load_frame``) gives, in
base_link (x forward, y left, z up):

- ``base_height``: base_footprint (floor) -> base_link (e.g. a table top)
- ``head``: ``pan_xyz``, the pan axis at the height of the tilt axis, and
  ``tilt_offset``, the tilt axis from that point (base_link axes, zero pan).
  The head base (printed HeadYaw part, ``meshes/head_base.stl`` cut from the
  Astra body by scripts/crop_head_base.py) is fixed at the pan joint origin
  unless ``base: false``
- ``camera`` (optional): ``xyz`` / ``rpy`` of camera_joint (link_head_tilt ->
  camera_link), else the bracket design values of sensors.xacro
- ``arms``: per side ``shoulder_xyz``, the joint_<l|r>2 axis at the height
  of link_<l|r>2's frame (the middle of the upper arm bar), ``yaw`` of
  link_<l|r>1 (the URDF's is +-pi/2), turning the whole arm about the
  shoulder axis, and optional ``joint_offsets`` ({joint_<l|r>N: rad}): the
  true angle of a joint is its reading plus the offset. The offset turns the
  joint's origin about the joint axis, which must be vertical in the parent
  link (a revolute joint about z with an origin roll of 0 or pi and no pitch,
  as joint_<l|r>2..4); the model then shows the true pose for the readings
- ``extrusions``: the frame members (``name``, centerline ``from`` / ``to``,
  optional ``section`` [width, depth], default EXTRUSION_SECTION square),
  each a link fixed to base_link. ``width`` is across the member
  horizontally (base_link y for an upright), ``depth`` the other side
  (base_link x for an upright, z for a horizontal member). The visual is a
  slotted profile (a groove in the middle of every 30 mm module of each
  face, ``profile_boxes``); the collision is the outer box

The lift joints become fixed joints that put link_<l|r>1 where the shoulder
axis lands at ``shoulder_xyz``; the wheel joints become fixed, and the Astra
body, wheels and caster lose their visuals and collisions.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ARMS = {"left": "l", "right": "r"}
REMOVED_GEOMETRY = ("base_link", "left_wheel_link", "right_wheel_link", "caster_link")
FIXED_JOINTS = ("left_wheel_joint", "right_wheel_joint")
FRAME_COLOR = (0.78, 0.79, 0.81, 1.0)
GROOVE_COLOR = (0.45, 0.46, 0.48, 1.0)
EXTRUSION_SECTION = 0.03  # m, 30 mm aluminum profile
MODULE = 0.03  # m, profile module (one slot per module and face)
HEAD_BASE_MESH = "package://aha_description/meshes/head_base.stl"
# Extents of the head base mesh from the joint_head_pan origin (base_link
# axes), measured on the crop: x -0.06 .. 0.018, y +-0.03, z 0.059 .. 0.0976.
HEAD_BASE_BOX = ((-0.021, 0.0, 0.0783), (0.078, 0.06, 0.0386))  # center, size
HEAD_BASE_COLOR = (0.95, 0.95, 0.95, 1.0)
GROOVE_WIDTH = 0.008  # m, slot opening of 30-series profiles
GROOVE_DEPTH = 0.004  # m, drawn depth of the slots


def load_frame(path) -> dict:
    """The frame config at ``path``; ValueError if a required key is missing."""
    config = yaml.safe_load(Path(path).read_text())
    for key in ("base_height", "head", "arms"):
        if key not in config:
            raise ValueError(f"{path}: missing {key}")
    if "pan_xyz" not in config["head"]:
        raise ValueError(f"{path}: head needs pan_xyz")
    for side, arm in config["arms"].items():
        if side not in ARMS:
            raise ValueError(f"{path}: unknown arm {side!r}, use {sorted(ARMS)}")
        missing = [k for k in ("shoulder_xyz", "yaw") if k not in (arm or {})]
        if missing:
            raise ValueError(f"{path}: arm {side} needs {missing}")
        offsets = arm.get("joint_offsets") or {}
        if not isinstance(offsets, dict):
            raise ValueError(f"{path}: arm {side} joint_offsets must be a mapping")
        pattern = re.compile(rf"joint_{ARMS[side]}[0-9]+")
        for name, value in offsets.items():
            if not pattern.fullmatch(str(name)):
                raise ValueError(f"{path}: {name!r} is not a joint of the {side} arm")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}: joint offset {name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{path}: joint offset {name} is not finite")
    camera = config.get("camera")
    if camera is not None and "xyz" not in (camera or {}):
        raise ValueError(f"{path}: camera needs xyz")
    for i, extrusion in enumerate(config.get("extrusions") or []):
        missing = [k for k in ("name", "from", "to") if k not in (extrusion or {})]
        if missing:
            raise ValueError(f"{path}: extrusion {i} needs {missing}")
    return config


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _values(text, default="0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def _text(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _joint(root, name):
    joint = root.find(f"joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"URDF has no joint {name}")
    return joint


def _origin(joint):
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    return origin


def _offset_joint(joint, offset: float):
    """Turn a joint's origin by ``offset`` about its axis (vertical in the
    parent link): the joint reads ``offset`` less than its true angle."""
    name = joint.get("name")
    if joint.get("type") not in ("revolute", "continuous"):
        raise ValueError(f"joint offset on {name}: not a revolute joint")
    axis_element = joint.find("axis")
    axis = _values(None if axis_element is None else axis_element.get("xyz"), "1 0 0")
    origin = _origin(joint)
    rpy = _values(origin.get("rpy"))
    vertical = rpy_matrix(rpy) @ (axis / np.linalg.norm(axis))
    if math.hypot(vertical[0], vertical[1]) > 1e-4:
        raise ValueError(
            f"joint offset on {name}: the axis is not vertical in the parent link"
        )
    # origin * Rz(offset) about the axis = Rz(+-offset) * origin rotation.
    rpy[2] += offset if vertical[2] > 0.0 else -offset
    origin.set("rpy", _text(rpy))


def _make_fixed(joint):
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
        for element in joint.findall(tag):
            joint.remove(element)


def _head_base_link(root, pan_xyz):
    pan = np.asarray(pan_xyz, float)
    link = ET.SubElement(root, "link", name="frame_head_base")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz=_text(pan), rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=HEAD_BASE_MESH)
    material = ET.SubElement(visual, "material", name="aha_frame_printed")
    ET.SubElement(material, "color", rgba=_text(HEAD_BASE_COLOR))
    center, size = HEAD_BASE_BOX
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz=_text(pan + center), rpy="0 0 0")
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=_text(size))
    joint = ET.SubElement(root, "joint", name="frame_head_base_joint", type="fixed")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="frame_head_base")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")


def _extrusion_rpy(direction) -> tuple[float, float, float]:
    """rpy turning the x axis onto ``direction`` (no roll about it)."""
    x, y, z = direction / np.linalg.norm(direction)
    return 0.0, -math.asin(max(-1.0, min(1.0, z))), math.atan2(y, x)


def _land_intervals(span: float) -> list[tuple[float, float]]:
    """(center, length) of the face strips between the slots of a face
    ``span`` wide: one slot in the middle of every MODULE, none if the face
    is narrower than a module."""
    modules = int(round(span / MODULE))
    if modules < 1 or abs(modules * MODULE - span) > 1e-6:
        return [(0.0, span)]
    edges = [-span / 2]
    for i in range(modules):
        center = -span / 2 + MODULE * (i + 0.5)
        edges += [center - GROOVE_WIDTH / 2, center + GROOVE_WIDTH / 2]
    edges.append(span / 2)
    pairs = zip(edges[0::2], edges[1::2])
    return [((a + b) / 2, b - a) for a, b in pairs]


def profile_boxes(length: float, width: float, depth: float):
    """((center y, center z), (size y, size z), color) boxes along x drawing
    a slotted profile: a core inside the slot bottoms and, on each face, the
    strips between the slots."""
    s = min(GROOVE_DEPTH, width / 4, depth / 4)
    boxes = [((0.0, 0.0), (width - 2 * s, depth - 2 * s), GROOVE_COLOR)]
    for sign in (-1.0, 1.0):
        for center, size in _land_intervals(width):
            boxes.append(((center, sign * (depth - s) / 2), (size, s), FRAME_COLOR))
        for center, size in _land_intervals(depth):
            boxes.append(((sign * (width - s) / 2, center), (s, size), FRAME_COLOR))
    return [((0.0, y, z), (length, sy, sz), color) for (y, z), (sy, sz), color in boxes]


def _extrusion_link(root, extrusion):
    name = f"frame_{extrusion['name']}"
    start = np.asarray(extrusion["from"], float)
    end = np.asarray(extrusion["to"], float)
    length = float(np.linalg.norm(end - start))
    if length <= 0.0:
        raise ValueError(f"extrusion {extrusion['name']} has no length")
    section = extrusion.get("section", (EXTRUSION_SECTION, EXTRUSION_SECTION))
    if not isinstance(section, (list, tuple)):
        section = (section, section)
    width, depth = float(section[0]), float(section[1])
    center = (start + end) / 2.0
    rotation = rpy_matrix(_extrusion_rpy(end - start))
    rpy = _text(_extrusion_rpy(end - start))
    link = ET.SubElement(root, "link", name=name)
    for offset, size, color in profile_boxes(length, width, depth):
        visual = ET.SubElement(link, "visual")
        xyz = center + rotation @ np.asarray(offset)
        ET.SubElement(visual, "origin", xyz=_text(xyz), rpy=rpy)
        ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=_text(size))
        shade = "aha_frame_aluminum" if color == FRAME_COLOR else "aha_frame_slot"
        material = ET.SubElement(visual, "material", name=shade)
        ET.SubElement(material, "color", rgba=_text(color))
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz=_text(center), rpy=rpy)
    ET.SubElement(
        ET.SubElement(collision, "geometry"), "box", size=_text((length, width, depth))
    )
    joint = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")


def replace_yaw(text: str, side: str, yaw: float) -> str:
    """Config text with the ``yaw`` of ``side``'s arm entry set to ``yaw``."""
    pattern = re.compile(rf"^(\s*{side}:\s*\{{.*?yaw:\s*)(-?[0-9.eE+-]+)(.*)$", re.M)
    text, count = pattern.subn(lambda m: f"{m.group(1)}{yaw:.4f}{m.group(3)}", text)
    if count != 1:
        raise ValueError(f"no single '{side}: {{... yaw: ...}}' line in the config")
    return text


def replace_shoulder_xyz(text: str, side: str, xyz) -> str:
    """Config text with the ``shoulder_xyz`` of ``side``'s arm entry set."""
    pattern = re.compile(
        rf"^(\s*{side}:\s*\{{.*?shoulder_xyz:\s*)\[[^\]]*\](.*)$", re.M
    )
    values = ", ".join(f"{float(v):.4f}" for v in xyz)
    text, count = pattern.subn(lambda m: f"{m.group(1)}[{values}]{m.group(2)}", text)
    if count != 1:
        raise ValueError(
            f"no single '{side}: {{shoulder_xyz: ...}}' line in the config"
        )
    return text


def replace_joint_offsets(text: str, side: str, offsets: dict) -> str:
    """Config text with ``joint_offsets`` of ``side``'s one-line arm entry set
    to ``offsets`` (replaced, or added at the end of the entry)."""
    pattern = re.compile(rf"^\s*{side}:\s*\{{.*$", re.M)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"no single '{side}: {{...}}' line in the config")
    match = matches[0]
    line = match.group(0)
    entries = ", ".join(
        f"{name}: {value:.4f}" for name, value in sorted(offsets.items())
    )
    value = f"joint_offsets: {{{entries}}}"
    existing = re.compile(r"joint_offsets:\s*\{[^}]*\}")
    if existing.search(line):
        line = existing.sub(value, line, count=1)
    else:
        body, hash_, comment = line.partition("#")
        end = body.rstrip()
        if not end.endswith("}"):
            raise ValueError(f"the '{side}:' entry is not a one-line mapping")
        line = f"{end[:-1]}, {value}}}" + (f"  {hash_}{comment}" if hash_ else "")
    return text[: match.start()] + line + text[match.end() :]


def frame_description(frame_config) -> str:
    """URDF of the installed aha_robot.urdf.xacro (sim:=false) on the frame
    at ``frame_config``: the real robot's robot_description (needs xacro and
    the aha_description share directory, unlike the rest of this module)."""
    import xacro
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("aha_description"))
    urdf = xacro.process_file(
        str(share / "urdf" / "aha_robot.urdf.xacro"), mappings={"sim": "false"}
    ).toxml()
    return frame_robot(urdf, load_frame(frame_config))


def frame_robot(urdf: str, config: dict) -> str:
    """URDF string of the robot on the frame described by ``config``."""
    root = ET.fromstring(urdf)

    _origin(_joint(root, "base_footprint_to_base_link")).set(
        "xyz", _text((0.0, 0.0, config["base_height"]))
    )
    for name in FIXED_JOINTS:
        joint = root.find(f"joint[@name='{name}']")
        if joint is not None:
            _make_fixed(joint)
    for name in REMOVED_GEOMETRY:
        link = root.find(f"link[@name='{name}']")
        if link is None:
            continue
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)

    head = config["head"]
    pan = _joint(root, "joint_head_pan")
    pan_origin = _origin(pan)
    pan_origin.set("xyz", _text(head["pan_xyz"]))
    pan_rotation = rpy_matrix(_values(pan_origin.get("rpy")))
    tilt_origin = _origin(_joint(root, "joint_head_tilt"))
    # The offset is given in base_link axes; the tilt joint is in the pan frame.
    offset = pan_rotation.T @ np.asarray(head.get("tilt_offset", (0, 0, 0)), float)
    tilt_origin.set("xyz", _text(offset))

    if head.get("base", True):
        _head_base_link(root, head["pan_xyz"])

    camera = config.get("camera")
    if camera:
        camera_origin = _origin(_joint(root, "camera_joint"))
        camera_origin.set("xyz", _text(camera["xyz"]))
        if "rpy" in camera:
            camera_origin.set("rpy", _text(camera["rpy"]))

    for side, arm in config["arms"].items():
        letter = ARMS[side]
        lift = _joint(root, f"joint_{letter}1")
        shoulder = _origin(_joint(root, f"joint_{letter}2"))
        yaw = float(arm["yaw"])
        # link_<x>1 -> joint_<x>2 offset of the URDF, turned by the mount yaw.
        step = rpy_matrix((0.0, 0.0, yaw)) @ _values(shoulder.get("xyz"))
        mount = np.asarray(arm["shoulder_xyz"], float) - step
        _make_fixed(lift)
        lift_origin = _origin(lift)
        lift_origin.set("xyz", _text(mount))
        lift_origin.set("rpy", _text((0.0, 0.0, yaw)))
        for name, offset in (arm.get("joint_offsets") or {}).items():
            _offset_joint(_joint(root, name), float(offset))

    for extrusion in config.get("extrusions", ()):
        _extrusion_link(root, extrusion)
    return ET.tostring(root, encoding="unicode")
