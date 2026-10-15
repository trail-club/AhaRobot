"""Gazebo-only model adjustments (no Gazebo server or ROS transport)."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_gazebo.model import LIFT_EFFORT, LIMIT_MARGIN, gazebo_model

URDF = """<robot name="r">
  <joint name="joint_r1" type="prismatic">
    <limit lower="0" upper="1.2" effort="400" velocity="0.3"/>
  </joint>
  <joint name="joint_r2" type="revolute"><limit lower="-1.57" upper="1.57"/></joint>
  <joint name="joint_r7r" type="prismatic"><limit lower="0" upper="0.06"/></joint>
  <joint name="joint_l7l" type="prismatic"><limit lower="-0.06" upper="0"/></joint>
  <joint name="wheel" type="continuous"/>
  <joint name="mount" type="fixed"/>
  <ros2_control name="hw" type="system">
    <joint name="joint_r1"><command_interface name="position"/></joint>
  </ros2_control>
</robot>"""


def joints(urdf):
    return {joint.get("name"): joint for joint in ET.fromstring(urdf).findall("joint")}


def bounds(joint):
    limit = joint.find("limit")
    return float(limit.get("lower")), float(limit.get("upper"))


def test_limits_widen_by_the_margin_on_both_sides():
    model = joints(gazebo_model(URDF))
    m = LIMIT_MARGIN
    assert bounds(model["joint_r1"]) == (-m, 1.2 + m)
    assert bounds(model["joint_r2"]) == (-1.57 - m, 1.57 + m)
    assert bounds(model["joint_r7r"]) == (-m, 0.06 + m)
    assert bounds(model["joint_l7l"]) == (-0.06 - m, m)


def test_margin_is_small_against_the_finger_stroke():
    # Keeps a crossing or over-opened finger within a few millimetres.
    assert LIMIT_MARGIN <= 0.06 / 10


def test_only_lifts_get_the_holding_effort():
    model = joints(gazebo_model(URDF))
    assert float(model["joint_r1"].find("limit").get("effort")) == LIFT_EFFORT
    assert model["joint_r1"].find("limit").get("velocity") == "0.3"
    assert model["joint_r2"].find("limit").get("effort") is None


def test_other_elements_are_kept():
    root = ET.fromstring(gazebo_model(URDF))
    assert root.find("joint[@name='wheel']").find("limit") is None
    assert (
        root.find("ros2_control/joint[@name='joint_r1']/command_interface") is not None
    )
