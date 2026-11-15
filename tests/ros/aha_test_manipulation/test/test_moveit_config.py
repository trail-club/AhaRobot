"""Consistency checks between the MoveIt config, the URDF and controllers.yaml."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory

PACKAGE = Path(__file__).resolve().parents[1]
DESCRIPTION = Path(get_package_share_directory("aha_test_description"))


@pytest.fixture(scope="module")
def urdf():
    doc = xacro.process_file(
        str(DESCRIPTION / "urdf" / "aha_robot_test.urdf.xacro"),
        mappings={"sim": "true"},
    )
    return ET.fromstring(doc.toxml())


@pytest.fixture(scope="module")
def srdf():
    return ET.parse(PACKAGE / "srdf" / "aha_robot.srdf").getroot()


def load_yaml(path):
    return yaml.safe_load(path.read_text())


def links(urdf):
    return {link.get("name") for link in urdf.findall("link")}


def joints(urdf):
    return {joint.get("name"): joint for joint in urdf.findall("joint")}


def chain_joints(urdf, base, tip):
    by_child = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    result = []
    link = tip
    while link != base:
        joint = by_child[link]
        if joint.get("type") != "fixed":
            result.append(joint.get("name"))
        link = joint.find("parent").get("link")
    return list(reversed(result))


def group_joints(urdf, srdf, name):
    group = srdf.find(f"group[@name='{name}']")
    chain = group.find("chain")
    if chain is not None:
        return chain_joints(urdf, chain.get("base_link"), chain.get("tip_link"))
    return [joint.get("name") for joint in group.findall("joint")]


def position_bounds(urdf_joint, override):
    if override.get("has_position_limits"):
        return override["min_position"], override["max_position"]
    limit = urdf_joint.find("limit")
    return float(limit.get("lower")), float(limit.get("upper"))


def test_groups_and_states_match_urdf(urdf, srdf):
    urdf_joints = joints(urdf)
    limits = load_yaml(PACKAGE / "config" / "joint_limits.yaml")["joint_limits"]
    assert {g.get("name") for g in srdf.findall("group")} == {
        "right_arm",
        "left_arm",
        "right_hand",
        "left_hand",
    }
    assert group_joints(urdf, srdf, "right_arm") == [f"joint_r{i}" for i in range(1, 7)]
    assert group_joints(urdf, srdf, "left_arm") == [f"joint_l{i}" for i in range(1, 7)]
    for state in srdf.findall("group_state"):
        expected = set(group_joints(urdf, srdf, state.get("group")))
        values = {j.get("name"): float(j.get("value")) for j in state.findall("joint")}
        assert set(values) == expected, state.get("name")
        for name, value in values.items():
            lower, upper = position_bounds(urdf_joints[name], limits.get(name, {}))
            assert lower <= value <= upper, name


def test_collision_links_exist(urdf, srdf):
    names = links(urdf)
    pairs = srdf.findall("disable_collisions")
    assert pairs
    for pair in pairs:
        assert {pair.get("link1"), pair.get("link2")} <= names
    for entry in srdf.findall("disable_default_collisions"):
        assert entry.get("link") in names
    for effector in srdf.findall("end_effector"):
        assert effector.get("parent_link") in names


def test_joint_limits_cover_groups(urdf, srdf):
    limits = load_yaml(PACKAGE / "config" / "joint_limits.yaml")["joint_limits"]
    urdf_joints = joints(urdf)
    for group in ("right_arm", "left_arm", "right_hand", "left_hand"):
        for name in group_joints(urdf, srdf, group):
            entry = limits[name]
            assert entry["has_acceleration_limits"] and entry["max_acceleration"] > 0
            limit = urdf_joints[name].find("limit")
            assert 0 < entry["max_velocity"] <= float(limit.get("velocity")), name
            lower, upper = position_bounds(urdf_joints[name], entry)
            assert float(limit.get("lower")) <= lower, name
            assert lower < upper <= float(limit.get("upper")), name


def test_moveit_controllers_match_ros2_controllers():
    moveit = load_yaml(PACKAGE / "config" / "moveit_controllers.yaml")
    managed = moveit["moveit_simple_controller_manager"]
    ros2 = load_yaml(DESCRIPTION / "config" / "controllers.yaml")
    for name in managed["controller_names"]:
        assert managed[name]["joints"] == ros2[name]["ros__parameters"]["joints"], name
    lift = ros2["lift_controller"]["ros__parameters"]
    assert lift["allow_partial_joints_goal"] is True
