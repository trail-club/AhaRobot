"""frame_robot on a minimal URDF with the joints it rewrites (ROS free)."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from aha_description.frame import (
    GROOVE_DEPTH,
    frame_robot,
    load_frame,
    profile_boxes,
    rpy_matrix,
)

URDF = """
<robot name="aha_robot">
  <link name="base_footprint"/>
  <link name="base_link"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <link name="left_wheel_link"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <link name="link_head_pan"/>
  <link name="link_head_tilt"/>
  <link name="camera_link"/>
  <link name="link_l1"/><link name="link_l2"/>
  <link name="link_r1"/><link name="link_r2"/>
  <joint name="base_footprint_to_base_link" type="fixed">
    <parent link="base_footprint"/><child link="base_link"/><origin xyz="0 0 0.042"/>
  </joint>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/><child link="left_wheel_link"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint_head_pan" type="revolute">
    <parent link="base_link"/><child link="link_head_pan"/>
    <origin xyz="0.05 0 1.429" rpy="-3.1416 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_head_tilt" type="revolute">
    <parent link="link_head_pan"/><child link="link_head_tilt"/>
    <origin xyz="0 0 0" rpy="1.5708 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="camera_joint" type="fixed">
    <parent link="link_head_tilt"/><child link="camera_link"/>
    <origin xyz="0.052525 0 0" rpy="1.5708 0 0"/>
  </joint>
  <joint name="joint_l1" type="prismatic">
    <parent link="base_link"/><child link="link_l1"/>
    <origin xyz="-0.0405 0.065 0.227" rpy="0 0 1.5708"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1.2" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_l2" type="revolute">
    <parent link="link_l1"/><child link="link_l2"/>
    <origin xyz="0.068 0 0" rpy="0 0 -0.18555"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_r1" type="prismatic">
    <parent link="base_link"/><child link="link_r1"/>
    <origin xyz="-0.0405 -0.065 0.227" rpy="0 0 -1.5708"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1.2" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_r2" type="revolute">
    <parent link="link_r1"/><child link="link_r2"/>
    <origin xyz="0.068 0 0" rpy="0 0 0.18555"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
</robot>
"""

CONFIG = {
    "base_height": 0.38,
    "head": {"pan_xyz": [0.09, 0.0, 0.418], "tilt_offset": [0.005, 0.0, -0.01]},
    "arms": {
        "left": {"shoulder_xyz": [0.03, 0.11, 0.244], "yaw": math.pi / 2},
        "right": {"shoulder_xyz": [0.03, -0.11, 0.222], "yaw": -math.pi / 2},
    },
    "extrusions": [
        {"name": "column", "from": [0, 0, 0], "to": [0, 0, 0.5]},
        {"name": "bar", "from": [0.03, -0.13, 0.015], "to": [0.03, 0.13, 0.015]},
        {"name": "tower", "from": [0.045, 0.145, 0], "to": [0.045, 0.145, 0.32],
         "section": [0.03, 0.12]},
    ],
}


def values(element, key):
    return np.array([float(v) for v in element.get(key).split()])


def joint(root, name):
    return root.find(f"joint[@name='{name}']")


def transform(element):
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(values(element, "rpy") if element.get("rpy") else (0, 0, 0))
    T[:3, 3] = values(element, "xyz")
    return T


@pytest.fixture
def robot():
    return ET.fromstring(frame_robot(URDF, CONFIG))


def test_base_stands_on_the_table_without_the_astra_body(robot):
    origin = joint(robot, "base_footprint_to_base_link").find("origin")
    assert values(origin, "xyz")[2] == pytest.approx(0.38)
    assert robot.find("link[@name='base_link']/visual") is None
    assert robot.find("link[@name='left_wheel_link']/visual") is None
    wheel = joint(robot, "left_wheel_joint")
    assert wheel.get("type") == "fixed" and wheel.find("axis") is None


def test_shoulder_axes_land_at_the_measured_points(robot):
    for letter, side in (("l", "left"), ("r", "right")):
        lift = joint(robot, f"joint_{letter}1")
        assert lift.get("type") == "fixed" and lift.find("limit") is None
        shoulder = transform(lift.find("origin")) @ transform(
            joint(robot, f"joint_{letter}2").find("origin")
        )
        assert shoulder[:3, 3] == pytest.approx(CONFIG["arms"][side]["shoulder_xyz"])


def test_tilt_axis_is_offset_in_base_axes(robot):
    pan = transform(joint(robot, "joint_head_pan").find("origin"))
    tilt = pan @ transform(joint(robot, "joint_head_tilt").find("origin"))
    expected = np.add(CONFIG["head"]["pan_xyz"], CONFIG["head"]["tilt_offset"])
    assert tilt[:3, 3] == pytest.approx(expected, abs=1e-6)
    # The camera keeps its design mount without a camera entry.
    camera = joint(robot, "camera_joint").find("origin")
    assert values(camera, "xyz")[0] == pytest.approx(0.052525)


def test_extrusions_are_30_mm_bars_between_their_ends(robot):
    for name, start, end in (
        ("column", (0, 0, 0), (0, 0, 0.5)),
        ("bar", (0.03, -0.13, 0.015), (0.03, 0.13, 0.015)),
    ):
        link = robot.find(f"link[@name='frame_{name}']")
        size = values(link.find("collision/geometry/box"), "size")
        assert size == pytest.approx([np.linalg.norm(np.subtract(end, start)), 0.03, 0.03])
        T = transform(link.find("collision/origin"))
        half = size[0] / 2
        assert (T @ [-half, 0, 0, 1])[:3] == pytest.approx(start, abs=1e-9)
        assert (T @ [half, 0, 0, 1])[:3] == pytest.approx(end, abs=1e-9)
        parent = joint(robot, f"frame_{name}_joint").find("parent").get("link")
        assert parent == "base_link"


def test_upright_section_width_is_y_and_depth_is_x(robot):
    link = robot.find("link[@name='frame_tower']")
    size = values(link.find("collision/geometry/box"), "size")
    R = transform(link.find("collision/origin"))[:3, :3]
    extents = np.abs(R) @ size  # box extents along base_link x, y, z
    assert extents == pytest.approx([0.12, 0.03, 0.32], abs=1e-6)


def test_profile_has_a_slot_per_module_and_face():
    # 30x120: 4 slots on each wide face, 1 on each narrow face.
    boxes = profile_boxes(0.32, 0.03, 0.12)
    strips = boxes[1:]
    assert len(strips) == 2 * (2 + 5)
    core = boxes[0][1]
    assert core == pytest.approx((0.32, 0.03 - 2 * GROOVE_DEPTH, 0.12 - 2 * GROOVE_DEPTH))
    # Strips and slots cover each face exactly.
    wide = [b for b in strips if b[1][1] == pytest.approx(GROOVE_DEPTH) and b[0][1] > 0]
    assert sum(b[1][2] for b in wide) == pytest.approx(0.12 - 4 * 0.008)


def test_visuals_stay_inside_the_collision_box(robot):
    link = robot.find("link[@name='frame_tower']")
    T = transform(link.find("collision/origin"))
    half = values(link.find("collision/geometry/box"), "size") / 2
    for visual in link.findall("visual"):
        V = np.linalg.inv(T) @ transform(visual.find("origin"))
        size = values(visual.find("geometry/box"), "size")
        extent = np.abs(V[:3, :3]) @ size / 2 + np.abs(V[:3, 3])
        assert np.all(extent <= half + 1e-6)


def test_head_base_hangs_at_the_pan_joint(robot):
    link = robot.find("link[@name='frame_head_base']")
    mesh = link.find("visual/geometry/mesh")
    assert mesh.get("filename").endswith("meshes/head_base.stl")
    assert values(link.find("visual/origin"), "xyz") == pytest.approx(CONFIG["head"]["pan_xyz"])
    top = values(link.find("collision/origin"), "xyz")[2] + values(
        link.find("collision/geometry/box"), "size"
    )[2] / 2
    assert top == pytest.approx(CONFIG["head"]["pan_xyz"][2] + 0.0976, abs=1e-4)
    config = dict(CONFIG, head=dict(CONFIG["head"], base=False))
    assert ET.fromstring(frame_robot(URDF, config)).find("link[@name='frame_head_base']") is None


def test_camera_mount_can_be_replaced():
    config = dict(CONFIG, camera={"xyz": [0.06, 0.0, -0.01]})
    root = ET.fromstring(frame_robot(URDF, config))
    origin = joint(root, "camera_joint").find("origin")
    assert values(origin, "xyz") == pytest.approx([0.06, 0.0, -0.01])
    assert values(origin, "rpy")[0] == pytest.approx(1.5708)


def test_installed_config_loads():
    path = Path(__file__).resolve().parents[1] / "config" / "frame_table_202610.yaml"
    config = load_frame(path)
    assert set(config["arms"]) == {"left", "right"}
    ET.fromstring(frame_robot(URDF, config))


def test_unknown_arm_is_an_error(tmp_path):
    path = tmp_path / "frame.yaml"
    path.write_text("base_height: 0.4\nhead: {pan_xyz: [0, 0, 0]}\narms: {middle: {}}\n")
    with pytest.raises(ValueError, match="unknown arm"):
        load_frame(path)
