"""MoveIt config of the real robot on its frame (ROS free).

The frame model (aha_test_frame.frame) fixes the lifts and wheels and adds
the frame members (``frame_*``) and the head base (``frame_head_base``) as
links fixed to base_link. moveit.launch.py (``frame_config``) builds its
config from the simulation's with these functions:

- ``frame_srdf``: the SRDF without named-state values of joints that are not
  movable in the URDF (the lifts), and with the collisions disabled between
  links that never move relative to each other or that no planning group moves
- ``frame_joint_limits``: joint limits narrowed to the arm trajectory
  bridge's (aha_manipulation config/arm.yaml), which rejects a trajectory
  outside them as a whole
- ``frame_controllers``: trajectory execution through the bridges of the
  given arms only (no lift_controller)

``side_grasp_tcp_height`` gives the one TCP height of the side grasp on
this model for place_server's ``tcp_height`` (manip_real.launch.py).
"""

from __future__ import annotations

import copy
import itertools
import math
import xml.etree.ElementTree as ET

import numpy as np
from aha_test_frame.frame import rpy_matrix

# MoveIt waits this long (s) beyond the bridge's goal_time for the result.
GOAL_MARGIN = 0.5
# MoveIt's time parameterization exceeded the bridge's acceleration check by
# up to 1.4x (the bridge differentiates between the points): MoveIt plans
# with this share of the bridge's acceleration limit.
ACCELERATION_SHARE = 0.5
# rad, largest tilt of the TCP z axis from vertical accepted as the side grasp
SIDE_GRASP_TILT = 1e-3


def movable_joints(urdf: ET.Element) -> set[str]:
    return {j.get("name") for j in urdf.findall("joint") if j.get("type") != "fixed"}


def _parents(urdf: ET.Element) -> dict:
    """child link -> (joint name, joint type, parent link)."""
    return {
        j.find("child").get("link"): (
            j.get("name"),
            j.get("type"),
            j.find("parent").get("link"),
        )
        for j in urdf.findall("joint")
    }


def _ancestors(parents: dict, link: str):
    """(joint name, joint type) from ``link`` up to the root."""
    while link in parents:
        name, kind, link = parents[link]
        yield name, kind


def _rigid_body(parents: dict, link: str) -> str:
    """The link nearest to the root that ``link`` is fixed to."""
    while link in parents and parents[link][1] == "fixed":
        link = parents[link][2]
    return link


def group_joints(srdf: ET.Element, urdf: ET.Element) -> set[str]:
    """Movable joints of the SRDF's chain and joint groups."""
    parents = _parents(urdf)
    joints = set()
    for group in srdf.findall("group"):
        joints |= {j.get("name") for j in group.findall("joint")}
        for chain in group.findall("chain"):
            link = chain.get("tip_link")
            while link != chain.get("base_link"):
                name, kind, link = parents[link]
                joints.add(name)
    return joints & movable_joints(urdf)


def _disabled_pairs(srdf: ET.Element) -> set[frozenset]:
    pairs = {
        frozenset((e.get("link1"), e.get("link2")))
        for e in srdf.findall("disable_collisions")
    }
    return pairs | {
        frozenset((e.get("link"),)) for e in srdf.findall("disable_default_collisions")
    }


def frame_srdf(srdf_xml: str, urdf_xml: str) -> str:
    """The SRDF ``srdf_xml`` made valid for the URDF ``urdf_xml``.

    Named-state values of joints that are fixed or missing in the URDF are
    dropped (a state left empty is removed). Collisions are disabled between
    links with collision geometry that are fixed to the same link (the frame
    members, the arm mounts link_<l|r>1 and the head base on base_link) or
    that no planning group joint moves (also the head links against those):
    such contacts are the same in every planned state and would otherwise
    make every state invalid. Pairs already disabled are kept as they are.
    """
    srdf = ET.fromstring(srdf_xml)
    urdf = ET.fromstring(urdf_xml)
    movable = movable_joints(urdf)
    for state in srdf.findall("group_state"):
        for joint in state.findall("joint"):
            if joint.get("name") not in movable:
                state.remove(joint)
        if not state.findall("joint"):
            srdf.remove(state)

    parents = _parents(urdf)
    planned = group_joints(srdf, urdf)
    links = sorted(
        link.get("name")
        for link in urdf.findall("link")
        if link.find("collision") is not None
    )

    def static(link):
        return not any(name in planned for name, _ in _ancestors(parents, link))

    disabled = _disabled_pairs(srdf)
    added = []
    for a, b in itertools.combinations(links, 2):
        if (
            frozenset((a, b)) in disabled
            or {frozenset((a,)), frozenset((b,))} & disabled
        ):
            continue
        if _rigid_body(parents, a) == _rigid_body(parents, b):
            added.append((a, b, "Adjacent"))
        elif static(a) and static(b):
            added.append((a, b, "Never"))
    if added:
        srdf.append(
            ET.Comment(
                " Frame model (aha_test_manipulation.frame_moveit): links fixed "
                "together (Adjacent) or not moved by any group (Never). "
            )
        )
    for a, b, reason in added:
        ET.SubElement(srdf, "disable_collisions", link1=a, link2=b, reason=reason)
    ET.indent(srdf, space="  ")
    return ET.tostring(srdf, encoding="unicode")


def bridge_joints(bridge_config: dict, arms) -> dict:
    """{controller name: [joint names]} of the bridges of ``arms`` in the
    bridge parameter file (aha_manipulation config/arm.yaml)."""
    out = {}
    for arm in arms:
        params = bridge_config[f"aha_{arm}_arm_bridge"]["ros__parameters"]
        out[params["controller_name"]] = list(params["joints"])
    return out


def frame_joint_limits(limits: dict, bridge_config: dict, urdf_xml: str) -> dict:
    """``limits`` (joint_limits.yaml) with the arm joints' position,
    velocity and acceleration bounds narrowed to the bridges' (both arms;
    acceleration to ACCELERATION_SHARE of it) and the URDF's."""
    out = copy.deepcopy(limits)
    urdf = {j.get("name"): j for j in ET.fromstring(urdf_xml).findall("joint")}
    common = bridge_config.get("/**", {}).get("ros__parameters", {})
    for key, node in bridge_config.items():
        if key == "/**":
            continue
        params = {**common, **node["ros__parameters"]}
        for joint in params["joints"]:
            bridge = params[joint]
            entry = out["joint_limits"].setdefault(joint, {})
            limit = urdf[joint].find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            if entry.get("has_position_limits"):
                lower = max(lower, entry["min_position"])
                upper = min(upper, entry["max_position"])
            entry["has_position_limits"] = True
            entry["min_position"] = max(lower, float(bridge["min"]))
            entry["max_position"] = min(upper, float(bridge["max"]))
            for name, value in (
                ("velocity", bridge["max_velocity"]),
                ("acceleration", bridge["max_acceleration"] * ACCELERATION_SHARE),
            ):
                key_max = f"max_{name}"
                current = (
                    entry.get(key_max) if entry.get(f"has_{name}_limits") else None
                )
                entry[f"has_{name}_limits"] = True
                entry[key_max] = float(
                    value if current is None else min(current, value)
                )
    return out


def frame_controllers(bridge_config: dict, arms) -> dict:
    """MoveIt simple controller manager parameters for the bridges of
    ``arms``. MoveIt waits for a result the bridge's ``goal_time`` (time to
    reach the last point) plus GOAL_MARGIN beyond the trajectory, and
    accepts the bridge's ``start_tolerance`` between the plan's start and the
    current state."""
    common = bridge_config.get("/**", {}).get("ros__parameters", {})
    controllers = bridge_joints(bridge_config, arms)
    manager = {"controller_names": list(controllers)}
    for name, joints in controllers.items():
        manager[name] = {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": joints,
        }
    return {
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": manager,
        "trajectory_execution": {
            "allowed_execution_duration_scaling": 1.2,
            "allowed_goal_duration_margin": float(common["goal_time"]) + GOAL_MARGIN,
            "allowed_start_tolerance": float(common["start_tolerance"]),
        },
    }


def _axis_rotation(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K


def _floats(text, default="0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def link_pose(urdf_xml: str, link: str, joints: dict | None = None) -> np.ndarray:
    """4x4 pose of ``link`` in the URDF's root link with the movable joints
    at ``joints`` (rad / m, others at 0)."""
    joints = joints or {}
    by_child = {
        j.find("child").get("link"): j for j in ET.fromstring(urdf_xml).findall("joint")
    }
    chain = []
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link].find("parent").get("link")
    T = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        step = np.eye(4)
        if origin is not None:
            step[:3, :3] = rpy_matrix(_floats(origin.get("rpy")))
            step[:3, 3] = _floats(origin.get("xyz"))
        T = T @ step
        kind, value = joint.get("type"), float(joints.get(joint.get("name"), 0.0))
        axis = joint.find("axis")
        axis = _floats(axis.get("xyz") if axis is not None else None, "1 0 0")
        motion = np.eye(4)
        if kind in ("revolute", "continuous"):
            motion[:3, :3] = _axis_rotation(axis, value)
        elif kind == "prismatic":
            motion[:3, 3] = axis / np.linalg.norm(axis) * value
        T = T @ motion
    return T


def side_grasp_tcp_height(urdf_xml: str, arm: str) -> float:
    """Height (z in the URDF root, base_footprint) of the arm's TCP
    link_<l|r>ee in the side grasp on a model without the lift: every arm
    joint at 0, where joint_<l|r>5 holds the TCP z axis up (checked within
    SIDE_GRASP_TILT, else ValueError). joint_<l|r>2..4 turn about vertical
    axes and keep the height; only joint_<l|r>5 tilts the hand."""
    T = link_pose(urdf_xml, f"link_{arm[0]}ee")
    tilt = math.acos(max(-1.0, min(1.0, float(T[2, 2]))))
    if tilt > SIDE_GRASP_TILT:
        raise ValueError(
            f"{arm} TCP z axis is {math.degrees(tilt):.2f} deg from vertical "
            "with the arm joints at 0"
        )
    return float(T[2, 3])
