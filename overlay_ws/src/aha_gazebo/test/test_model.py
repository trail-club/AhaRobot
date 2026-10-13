"""Gazebo-only model adjustments (no Gazebo server or ROS transport)."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_gazebo.model import LIFT_FRICTION, PHYSICS_LIMIT_MARGIN, gazebo_model

URDF = """<robot name="r">
  <joint name="joint_r1" type="prismatic">
    <limit lower="0" upper="1.2" effort="400"/>
  </joint>
  <joint name="arm" type="revolute"><limit lower="-1.57" upper="1.57"/></joint>
  <joint name="wheel" type="continuous"/>
  <joint name="mount" type="fixed"/>
  <ros2_control name="hw" type="system">
    <joint name="joint_r1"><command_interface name="position"/></joint>
  </ros2_control>
</robot>"""


def limits(urdf):
    return {
        joint.get("name"): (
            float(joint.find("limit").get("lower")),
            float(joint.find("limit").get("upper")),
        )
        for joint in ET.fromstring(urdf).findall("joint")
        if joint.find("limit") is not None
    }


def test_limits_widen_by_the_margin_on_both_sides():
    m = PHYSICS_LIMIT_MARGIN
    assert limits(gazebo_model(URDF)) == {
        "joint_r1": (-m, 1.2 + m),
        "arm": (-1.57 - m, 1.57 + m),
    }


def test_other_elements_are_kept():
    root = ET.fromstring(gazebo_model(URDF))
    assert root.find("joint[@name='joint_r1']/limit").get("effort") == "400"
    assert root.find("joint[@name='wheel']").find("limit") is None
    assert (
        root.find("ros2_control/joint[@name='joint_r1']/command_interface") is not None
    )


def test_only_lifts_get_friction():
    root = ET.fromstring(gazebo_model(URDF))
    friction = root.find("joint[@name='joint_r1']/dynamics").get("friction")
    assert float(friction) == LIFT_FRICTION
    assert root.find("joint[@name='arm']/dynamics") is None
