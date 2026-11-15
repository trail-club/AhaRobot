"""The test bed xacro expands with the held object welds and lift fixes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory

xacro = pytest.importorskip("xacro")


@pytest.fixture(scope="module")
def robot():
    share = Path(get_package_share_directory("aha_test_description"))
    document = xacro.process_file(
        str(share / "urdf" / "aha_robot_test.urdf.xacro"),
        mappings={"sim": "true", "cameras": "true"},
    )
    return ET.fromstring(document.toxml())


def test_held_object_welds_for_both_arms(robot):
    plugins = [
        p
        for p in robot.iter("plugin")
        if p.get("name") == "gz::sim::systems::DetachableJoint"
    ]
    assert sorted(p.findtext("child_model") for p in plugins) == [
        "aha_held_object_left",
        "aha_held_object_right",
    ]
    assert {p.findtext("parent_link") for p in plugins} == {"link_l6", "link_r6"}


def test_lift_starts_above_its_stop(robot):
    control = robot.find("ros2_control")
    lift = control.find("joint[@name='joint_r1']")
    assert lift.find("state_interface/param").text.strip() == "0.01"
    params = next(
        p.findtext("parameters")
        for p in robot.iter("plugin")
        if p.get("name") == "gz_ros2_control::GazeboSimROS2ControlPlugin"
    )
    assert params.endswith("aha_test_description/config/controllers.yaml")


def test_controllers_are_the_robot_ones_with_the_lift_overrides():
    yaml = pytest.importorskip("yaml")
    share = Path(get_package_share_directory("aha_test_description"))
    robot = Path(get_package_share_directory("aha_description"))
    test = yaml.safe_load((share / "config" / "controllers.yaml").read_text())
    base = yaml.safe_load((robot / "config" / "controllers.yaml").read_text())
    assert test.keys() == base.keys()
    for name in base:
        if name != "lift_controller":
            assert test[name] == base[name], name
    lift = test["lift_controller"]["ros__parameters"]
    base_lift = base["lift_controller"]["ros__parameters"]
    assert lift["allow_partial_joints_goal"] is True
    assert lift["constraints"]["joint_r1"] == {"goal": 0.01}
    for key in base_lift.keys() - {"constraints"}:
        assert lift[key] == base_lift[key], key
