"""Verify MoveIt references and controller coverage against the installed URDF."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro
import yaml

SHARE = Path(__file__).resolve().parents[1]


def model():
    description = Path(get_package_share_directory("aha_description"))
    return ET.fromstring(
        xacro.process_file(str(description / "urdf/aha_robot.urdf.xacro")).toxml()
    )


def test_srdf_references_existing_model_links_and_serial_chain():
    urdf = model()
    links = {item.attrib["name"] for item in urdf.findall("link")}
    parents = {
        joint.find("child").attrib["link"]: joint for joint in urdf.findall("joint")
    }
    srdf = ET.parse(SHARE / "srdf/aha_robot.srdf").getroot()
    assert srdf.attrib["name"] == urdf.attrib["name"]
    for pair in srdf.findall("disable_collisions"):
        assert pair.attrib["link1"] in links and pair.attrib["link2"] in links
    chain = srdf.find("group[@name='right_arm']/chain")
    tip = chain.attrib["tip_link"]
    active = set()
    while tip != chain.attrib["base_link"]:
        joint = parents[tip]
        if joint.attrib["type"] != "fixed":
            active.add(joint.attrib["name"])
        tip = joint.find("parent").attrib["link"]
    assert active == {f"joint_r{i}" for i in range(1, 7)}


def test_moveit_controller_joints_match_ros2_control_and_cover_arm():
    controller = yaml.safe_load(
        (
            Path(get_package_share_directory("aha_description"))
            / "config/controllers.yaml"
        ).read_text()
    )
    moveit = yaml.safe_load((SHARE / "config/controllers.yaml").read_text())[
        "moveit_simple_controller_manager"
    ]
    covered = set()
    for name in moveit["controller_names"]:
        joints = moveit[name]["joints"]
        assert joints == controller[name]["ros__parameters"]["joints"]
        covered.update(joints)
    assert {f"joint_r{i}" for i in range(1, 7)} <= covered
    assert controller["lift_controller"]["ros__parameters"]["allow_partial_joints_goal"]


def test_time_parameterization_limits_fit_urdf():
    urdf = model()
    limits = yaml.safe_load((SHARE / "config/joint_limits.yaml").read_text())[
        "joint_limits"
    ]
    for name, settings in limits.items():
        joint = urdf.find(f"joint[@name='{name}']")
        assert (
            0
            < settings["max_velocity"]
            <= float(joint.find("limit").attrib["velocity"])
        )
        assert settings["has_acceleration_limits"] and settings["max_acceleration"] > 0


def test_named_states_set_every_arm_joint_within_limits():
    urdf = model()
    srdf = ET.parse(SHARE / "srdf/aha_robot.srdf").getroot()
    states = {
        state.attrib["name"]: {
            joint.attrib["name"]: float(joint.attrib["value"])
            for joint in state.findall("joint")
        }
        for state in srdf.findall("group_state[@group='right_arm']")
    }
    assert {"ready", "standby", "carry"} <= set(states)
    for values in states.values():
        assert set(values) == {f"joint_r{i}" for i in range(1, 7)}
        for name, value in values.items():
            limit = urdf.find(f"joint[@name='{name}']/limit")
            assert float(limit.attrib["lower"]) <= value <= float(limit.attrib["upper"])


def test_rviz_view_matches_planning_group_and_topics():
    rviz = yaml.safe_load((SHARE / "rviz/manip.rviz").read_text())
    displays = {
        display["Name"]: display
        for display in rviz["Visualization Manager"]["Displays"]
    }
    planning = displays["MotionPlanning"]["Planning Request"]["Planning Group"]
    assert planning in yaml.safe_load((SHARE / "config/kinematics.yaml").read_text())
    # The dock layout (QMainWindow State) restores docks by these names.
    state = bytes.fromhex(rviz["Window Geometry"]["QMainWindow State"])
    for name in ("MotionPlanning", "Displays", "HeadCamera", "RightWristCamera"):
        assert name.encode("utf-16-be") in state
    bridged = {
        entry["ros_topic_name"]
        for entry in yaml.safe_load(
            (
                Path(get_package_share_directory("aha_perception"))
                / "config/sim_camera_bridge.yaml"
            ).read_text()
        )
    }
    for name in ("HeadCamera", "RightWristCamera"):
        assert displays[name]["Topic"]["Value"] in bridged
