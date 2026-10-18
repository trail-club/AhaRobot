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
