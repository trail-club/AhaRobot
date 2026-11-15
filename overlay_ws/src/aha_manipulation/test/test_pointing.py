"""Check the pointing model against forward kinematics of the installed URDF."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import xacro

from aha_manipulation.pointing import (
    ARM_JOINTS,
    EE_DROP,
    LIFT_JOINT,
    LIMITS,
    lift_for_height,
    point_at,
    wrist_position,
)


@pytest.fixture(scope="module")
def joints():
    path = Path(get_package_share_directory("aha_description"))
    urdf = ET.fromstring(
        xacro.process_file(str(path / "urdf/aha_robot.urdf.xacro")).toxml()
    )
    return {j.find("child").attrib["link"]: j for j in urdf.findall("joint")}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (f(a) for a in (r, p, y) for f in (math.cos, math.sin))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_matrix(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def fk(joints, link, values, root="base_footprint"):
    """4x4 pose of `link` in `root` for joint values (missing ones are 0)."""
    chain = []
    while link != root:
        joint = joints[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    pose = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = rpy_matrix(*rpy), xyz
        pose = pose @ step
        value = values.get(joint.attrib["name"], 0.0)
        axis_tag = joint.find("axis")
        axis = np.array(
            [
                float(v)
                for v in (
                    axis_tag.attrib["xyz"] if axis_tag is not None else "1 0 0"
                ).split()
            ]
        )
        motion = np.eye(4)
        if joint.attrib["type"] in ("revolute", "continuous"):
            motion[:3, :3] = axis_matrix(axis, value)
        elif joint.attrib["type"] == "prismatic":
            motion[:3, 3] = axis * value
        pose = pose @ motion
    return pose


def ray_miss(pose, target):
    """Distance from `target` to the ray along the pose's x axis, and the ray's t."""
    origin, direction = pose[:3, 3], pose[:3, 0]
    t = float(np.dot(np.array(target) - origin, direction))
    return float(np.linalg.norm(origin + t * direction - np.array(target))), t


SAMPLES = [
    {"joint_r1": 0.3, "joint_r2": 0.4, "joint_r3": -0.7, "joint_r4": 0.2},
    {"joint_r1": 0.0, "joint_r2": -1.0, "joint_r3": 1.5, "joint_r4": 0.5},
    {"joint_r1": 0.8, "joint_r2": 1.2, "joint_r3": 0.3, "joint_r5": 0.6},
]


@pytest.mark.parametrize("values", SAMPLES)
def test_model_matches_the_urdf(joints, values):
    wrist = fk(joints, "link_r4", values)[:3, 3]
    expected = wrist_position(
        values["joint_r1"], values.get("joint_r2", 0.0), values.get("joint_r3", 0.0)
    )
    assert np.allclose(wrist, expected, atol=1e-3)
    ee = fk(joints, "link_ree", values)
    yaw = math.atan2(ee[1, 0], ee[0, 0])
    pitch = math.atan2(ee[2, 0], math.hypot(ee[0, 0], ee[1, 0]))
    model_yaw = values.get("joint_r2", 0.0) + values.get("joint_r3", 0.0)
    model_yaw -= values.get("joint_r4", 0.0)
    assert math.remainder(yaw - model_yaw, math.tau) == pytest.approx(0.0, abs=2e-3)
    assert pitch == pytest.approx(-values.get("joint_r5", 0.0), abs=2e-3)
    miss, _ = ray_miss(ee, wrist)
    assert miss == pytest.approx(EE_DROP, abs=1e-3)


@pytest.mark.parametrize(
    "target",
    [
        (1.0, 0.0, 0.5),  # a seat ahead
        (1.2, -0.8, 0.45),  # front right
        (0.0, -1.5, 0.9),  # right
        (1.0, 0.6, 0.5),  # front left: forearm turns
        (2.5, 0.3, 1.6),  # far and above
    ],
)
def test_ray_passes_through_the_target(joints, target):
    lift = lift_for_height(0.8)
    pointing = point_at(target, lift)
    values = dict(zip(ARM_JOINTS, pointing.arm), **{LIFT_JOINT: pointing.lift})
    for name, value in values.items():
        low, high = LIMITS[name]
        assert low <= value <= high
    miss, t = ray_miss(fk(joints, "link_ree", values), target)
    assert miss < 1e-3
    assert t > 0.0  # in front of the gripper, not behind it


def test_targets_behind_or_too_close_are_refused():
    with pytest.raises(ValueError, match="bearing"):
        point_at((-1.0, 0.5, 0.5), 0.5)
    with pytest.raises(ValueError, match="too close"):
        wrist = wrist_position(0.5, 1.38525, -1.3852)
        point_at((wrist[0] + 0.1, wrist[1], wrist[2]), 0.5)


def test_lift_for_height_is_clamped():
    assert lift_for_height(0.8) == pytest.approx(0.618)
    assert lift_for_height(0.0) == 0.0
    assert lift_for_height(5.0) == LIMITS[LIFT_JOINT][1]
