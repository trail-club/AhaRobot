"""Simulation only, without ROS: geometry and models of sim_manipulation.

sim_manipulation (aha_tasks.sim_manipulation_node) stands in for the pick,
place and handover servers in e2e:=debug. A picked object leaves the
physics: its model is re-created under the same name as a static model
without collision and follows the gripper (TCP) upright; a placed object is
re-created as the dynamic model, standing at the spot (dropped from it into
a container); a handed-over object is removed. Keeping the name keeps the
label and the track of sim_object_detector.py.

Poses are 4x4 matrices in Gazebo's world frame unless named otherwise.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

# aha_manipulation srdf group_state "carry" of right_arm: lift, then
# joint_r2..joint_r6. The hand is at about (0.22, -0.30, 0.92) in base_footprint.
CARRY = (0.5, (-0.6, 1.2, 0.6, 0.0, 0.0))
MATCH_DISTANCE = 0.2  # m from the goal's target to the object model
REACH = 0.9  # m (horizontal) from the base centre the stand-in reaches
PLACE_LIFT = 0.005  # m above the support where a placed object is created
# m from the shoulder: a nearer target is pointed at through a point this far
# along the same direction (the stretched arm's hand ends about 0.75 m out,
# and aha_manipulation.pointing needs the target beyond the hand).
AIM_MIN = 1.0


def pose_matrix(x, y, z, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = (x, y, z)
    return out


def yaw_of(matrix):
    """Heading of the x axis about world z."""
    return math.atan2(matrix[1, 0], matrix[0, 0])


def apply(matrix, xyz):
    return tuple(float(v) for v in (matrix @ np.array([*xyz, 1.0]))[:3])


def norm(label):
    return " ".join(str(label).lower().replace("_", " ").split())


def find_model(poses, object_list, label, point, max_distance=MATCH_DISTANCE):
    """(name, distance) of the model of `label` nearest `point` (world xyz)
    within `max_distance`, else of any object model that close; None if none.

    A detector may give another label to the model it saw (mislabel_rate)."""
    same, other = [], []
    for name, pose in poses.items():
        entry = object_list.for_model(name) if name else None
        if entry is None:
            continue
        distance = float(np.linalg.norm(pose[:3, 3] - np.asarray(point)))
        if distance <= max_distance:
            (same if norm(entry.label) == norm(label) else other).append(
                (distance, name)
            )
    for found in (same, other):
        if found:
            distance, name = min(found)
            return name, distance
    return None


def horizontal_distance(world_base, point):
    return math.dist(world_base[:2, 3], point[:2])


def held_pose(world_tcp, yaw_offset):
    """(x, y, z, yaw) of a held object: its centre at the TCP, upright, turned
    with the TCP's heading."""
    x, y, z = world_tcp[:3, 3]
    return float(x), float(y), float(z), yaw_of(world_tcp) + yaw_offset


def resting_centre(near, size_z, drop):
    """Centre of a placed object: standing on the support at `near` (bottom
    centre), or, dropped, at `near` itself."""
    x, y, z = near
    return (x, y, z) if drop else (x, y, z + size_z / 2 + PLACE_LIFT)


def static_copy(sdf):
    """The model of an SDF string as a static model without collision and
    inertia: Gazebo moves it only by set_pose, and it pushes nothing."""
    root = ET.fromstring(sdf.replace('<?xml version="1.0"?>', "", 1))
    model = root if root.tag == "model" else root.find("model")
    if model is None:
        raise ValueError("no model in the SDF")
    for old in model.findall("static"):
        model.remove(old)
    static = ET.Element("static")
    static.text = "true"
    model.insert(0, static)
    for link in model.findall("link"):
        for tag in ("collision", "inertial"):
            for element in link.findall(tag):
                link.remove(element)
    return '<?xml version="1.0"?>' + ET.tostring(root, encoding="unicode")


def aim_point(target, shoulder):
    """A point to point the arm at for `target` (base_footprint xyz): AIM_MIN
    from `shoulder` (x, y) in the target's horizontal direction when it is
    nearer, at the target's height."""
    dx, dy = target[0] - shoulder[0], target[1] - shoulder[1]
    distance = math.hypot(dx, dy)
    if distance >= AIM_MIN or distance < 1e-6:
        return tuple(target)
    scale = AIM_MIN / distance
    return (shoulder[0] + dx * scale, shoulder[1] + dy * scale, target[2])
