"""frame_moveit: the MoveIt config of the real robot on its frame, on a
minimal URDF and on the installed frame model with this package's SRDF and
aha_manipulation's bridge config."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from aha_test_frame.frame import frame_description
from ament_index_python.packages import get_package_share_directory

from aha_test_manipulation.frame_moveit import (
    GOAL_MARGIN,
    frame_controllers,
    frame_joint_limits,
    frame_srdf,
    group_joints,
    link_pose,
    side_grasp_tcp_height,
)

PACKAGE = Path(__file__).resolve().parents[1]
FRAME_CONFIG = (
    Path(get_package_share_directory("aha_test_frame"))
    / "config"
    / "frame_table_202610.yaml"
)
BRIDGE_CONFIG = (
    Path(get_package_share_directory("aha_manipulation")) / "config" / "arm.yaml"
)
BOX = "<collision><geometry><box size='0.1 0.1 0.1'/></geometry></collision>"

# base_link with a frame member and a fixed lift (link_r1); the arm joint
# joint_r2 and the head pan move.
MINI_URDF = f"""
<robot name="mini">
  <link name="base_footprint"/>
  <link name="base_link"/>
  <link name="frame_tower">{BOX}</link>
  <link name="link_r1">{BOX}</link>
  <link name="link_r2">{BOX}</link>
  <link name="link_ree"/>
  <link name="link_head_pan">{BOX}</link>
  <joint name="root" type="fixed"><parent link="base_footprint"/><child link="base_link"/></joint>
  <joint name="frame_tower_joint" type="fixed"><parent link="base_link"/><child link="frame_tower"/></joint>
  <joint name="joint_r1" type="fixed"><parent link="base_link"/><child link="link_r1"/></joint>
  <joint name="joint_r2" type="revolute"><parent link="link_r1"/><child link="link_r2"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="1"/></joint>
  <joint name="joint_ree" type="fixed"><parent link="link_r2"/><child link="link_ree"/></joint>
  <joint name="joint_head_pan" type="revolute"><parent link="base_link"/><child link="link_head_pan"/>
    <limit lower="-1.57" upper="1.57" velocity="3.0" effort="1"/></joint>
</robot>
"""
MINI_SRDF = """
<robot name="mini">
  <group name="right_arm"><chain base_link="base_link" tip_link="link_ree"/></group>
  <group_state name="home" group="right_arm">
    <joint name="joint_r1" value="0.01"/>
    <joint name="joint_r2" value="0"/>
  </group_state>
  <group_state name="lift_only" group="right_arm">
    <joint name="joint_r1" value="0.5"/>
  </group_state>
  <disable_collisions link1="link_r1" link2="link_r2" reason="Adjacent"/>
</robot>
"""


def disabled(srdf):
    return {
        frozenset((e.get("link1"), e.get("link2"))): e.get("reason")
        for e in srdf.findall("disable_collisions")
    }


def states(srdf):
    return {
        (s.get("group"), s.get("name")): {
            j.get("name"): float(j.get("value")) for j in s.findall("joint")
        }
        for s in srdf.findall("group_state")
    }


def test_mini_states_drop_fixed_joints():
    srdf = ET.fromstring(frame_srdf(MINI_SRDF, MINI_URDF))
    assert states(srdf) == {("right_arm", "home"): {"joint_r2": 0.0}}


def test_mini_collisions():
    pairs = disabled(ET.fromstring(frame_srdf(MINI_SRDF, MINI_URDF)))
    # Fixed to base_link together.
    assert pairs[frozenset(("frame_tower", "link_r1"))] == "Adjacent"
    # Moved by no group: the head against the frame.
    assert pairs[frozenset(("frame_tower", "link_head_pan"))] == "Never"
    assert pairs[frozenset(("link_head_pan", "link_r1"))] == "Never"
    # Kept as given.
    assert pairs[frozenset(("link_r1", "link_r2"))] == "Adjacent"
    # The arm stays checked against the frame and the head.
    assert frozenset(("frame_tower", "link_r2")) not in pairs
    assert frozenset(("link_head_pan", "link_r2")) not in pairs


@pytest.fixture(scope="module")
def urdf_xml():
    return frame_description(FRAME_CONFIG)


@pytest.fixture(scope="module")
def srdf(urdf_xml):
    original = (PACKAGE / "srdf" / "aha_robot.srdf").read_text()
    return ET.fromstring(frame_srdf(original, urdf_xml))


@pytest.fixture(scope="module")
def bridges():
    return yaml.safe_load(BRIDGE_CONFIG.read_text())


@pytest.fixture(scope="module")
def limits(urdf_xml, bridges):
    original = yaml.safe_load((PACKAGE / "config" / "joint_limits.yaml").read_text())
    return frame_joint_limits(original, bridges, urdf_xml)["joint_limits"]


def test_frame_srdf_is_valid_for_the_frame_model(urdf_xml, srdf):
    urdf = ET.fromstring(urdf_xml)
    links = {link.get("name") for link in urdf.findall("link")}
    for pair in srdf.findall("disable_collisions"):
        assert {pair.get("link1"), pair.get("link2")} <= links
    joints = group_joints(srdf, urdf)
    assert {f"joint_r{i}" for i in range(2, 7)} <= joints
    assert not {"joint_r1", "joint_l1"} & joints
    for state in srdf.findall("group_state"):
        names = {j.get("name") for j in state.findall("joint")}
        assert not {"joint_r1", "joint_l1"} & names, state.get("name")
    assert set(states(srdf)[("right_arm", "carry")]) == {
        f"joint_r{i}" for i in range(2, 7)
    }


def test_frame_srdf_disables_contacts_by_construction(srdf):
    pairs = disabled(srdf)
    for pair in (
        ("frame_column", "frame_head_bar"),
        ("frame_bottom", "frame_right_tower"),
        ("frame_right_tower", "link_r1"),
        ("frame_left_tower", "link_l1"),
        ("frame_head_base", "link_head_pan"),
        ("frame_head_bar", "link_head_tilt"),
    ):
        assert frozenset(pair) in pairs, pair
    for pair in (
        ("frame_right_tower", "link_r2"),
        ("frame_column", "link_r6"),
        ("link_head_tilt", "link_r6"),
    ):
        assert frozenset(pair) not in pairs, pair


def test_frame_limits_within_bridge_and_urdf(urdf_xml, bridges, limits):
    urdf = {j.get("name"): j for j in ET.fromstring(urdf_xml).findall("joint")}
    for node in ("aha_right_arm_bridge", "aha_left_arm_bridge"):
        params = bridges[node]["ros__parameters"]
        for joint in params["joints"]:
            entry, bridge = limits[joint], params[joint]
            limit = urdf[joint].find("limit")
            assert entry["has_position_limits"]
            assert entry["min_position"] >= max(
                bridge["min"], float(limit.get("lower"))
            )
            assert entry["max_position"] <= min(
                bridge["max"], float(limit.get("upper"))
            )
            assert entry["min_position"] < entry["max_position"]
            assert entry["max_velocity"] <= bridge["max_velocity"]
            assert entry["max_acceleration"] <= bridge["max_acceleration"]


def test_right_named_states_within_frame_limits(srdf, limits):
    for (group, name), values in states(srdf).items():
        if group != "right_arm":
            continue
        for joint, value in values.items():
            entry = limits[joint]
            assert entry["min_position"] <= value <= entry["max_position"], (
                name,
                joint,
            )


def test_frame_controllers_are_the_bridges(bridges):
    params = frame_controllers(bridges, ["right"])
    manager = params["moveit_simple_controller_manager"]
    assert manager["controller_names"] == ["right_arm_controller"]
    right = bridges["aha_right_arm_bridge"]["ros__parameters"]
    assert manager["right_arm_controller"]["joints"] == right["joints"]
    common = bridges["/**"]["ros__parameters"]
    execution = params["trajectory_execution"]
    assert execution["allowed_goal_duration_margin"] == pytest.approx(
        common["goal_time"] + GOAL_MARGIN
    )
    assert execution["allowed_start_tolerance"] == common["start_tolerance"]
    both = frame_controllers(bridges, ["right", "left"])
    assert both["moveit_simple_controller_manager"]["controller_names"] == [
        "right_arm_controller",
        "left_arm_controller",
    ]


# A lift (prismatic), a vertical shoulder joint and a wrist tilting about y.
TCP_URDF = """
<robot name="tcp">
  <link name="base_footprint"/>
  <link name="link_r1"/>
  <link name="link_r2"/>
  <link name="link_r5"/>
  <link name="link_ree"/>
  <joint name="joint_r1" type="prismatic"><parent link="base_footprint"/><child link="link_r1"/>
    <origin xyz="0 0 0.5" rpy="0 0 -1.7"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" velocity="1" effort="1"/></joint>
  <joint name="joint_r2" type="revolute"><parent link="link_r1"/><child link="link_r2"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="1" effort="1"/></joint>
  <joint name="joint_r5" type="revolute"><parent link="link_r2"/><child link="link_r5"/>
    <origin xyz="0.2 0 -0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" velocity="1" effort="1"/></joint>
  <joint name="joint_ree" type="fixed"><parent link="link_r5"/><child link="link_ree"/>
    <origin xyz="0.1 0 0" rpy="TILT"/></joint>
</robot>
"""


def test_link_pose_of_a_tiny_chain():
    urdf = TCP_URDF.replace("TILT", "0 0 0")
    T = link_pose(urdf, "link_ree", {"joint_r1": 0.1, "joint_r2": 0.4})
    yaw = -1.7 + 0.4
    expected = np.array([0.0, 0.0, 0.6]) + [
        0.3 * math.cos(-1.7) + 0.3 * math.cos(yaw),
        0.3 * math.sin(-1.7) + 0.3 * math.sin(yaw),
        -0.1,
    ]
    np.testing.assert_allclose(T[:3, 3], expected, atol=1e-12)
    np.testing.assert_allclose(T[:3, 2], [0, 0, 1], atol=1e-12)
    # The wrist tilt lowers the TCP.
    T = link_pose(urdf, "link_ree", {"joint_r5": 0.3})
    assert T[2, 3] == pytest.approx(0.4 - 0.1 * math.sin(0.3))


def test_tiny_side_grasp_tcp_height():
    assert side_grasp_tcp_height(TCP_URDF.replace("TILT", "0 0 0"), "right") == (
        pytest.approx(0.4)
    )
    with pytest.raises(ValueError):
        side_grasp_tcp_height(TCP_URDF.replace("TILT", "0 0.1 0"), "right")


@pytest.mark.parametrize("arm", ["right", "left"])
def test_frame_side_grasp_tcp_height(urdf_xml, arm):
    """The TCP with z up stays at one height whatever joint_?2..4 do; only
    joint_?5 moves it."""
    height = side_grasp_tcp_height(urdf_xml, arm)
    letter = arm[0]
    tcp = f"link_{letter}ee"
    rng = np.random.default_rng(0)
    for _ in range(20):
        joints = {f"joint_{letter}{i}": rng.uniform(-1.5, 1.5) for i in (2, 3, 4)}
        T = link_pose(urdf_xml, tcp, joints)
        assert abs(T[2, 3] - height) < 1e-5, joints
        assert T[2, 2] > math.cos(1e-3)
    tilted = link_pose(urdf_xml, tcp, {f"joint_{letter}5": 0.1})
    assert abs(tilted[2, 3] - height) > 0.01
