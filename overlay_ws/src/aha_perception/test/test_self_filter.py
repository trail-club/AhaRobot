"""aha_perception.self_filter: link boxes from a URDF and the point test."""

import math
import struct

import numpy as np
import pytest

from aha_perception.self_filter import (
    link_boxes,
    points_in_boxes,
    read_stl_vertices,
)

PAD = 0.02


def write_stl(path, triangles):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            f.write(struct.pack("<3f", 0, 0, 0))
            for vertex in triangle:
                f.write(struct.pack("<3f", *vertex))
            f.write(struct.pack("<H", 0))


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "part.stl"
    write_stl(path, [[(0, 0, 0), (100, 0, 0), (0, 50, 20)]])
    return str(path)


def urdf(mesh_path):
    return f"""<robot name="r">
  <link name="base"/>
  <link name="meshed">
    <collision>
      <origin xyz="1 0 0" rpy="0 0 {math.pi / 2}"/>
      <geometry><mesh filename="file://{mesh_path}" scale="0.001 0.001 0.001"/></geometry>
    </collision>
  </link>
  <link name="shapes">
    <collision><geometry><box size="0.2 0.4 0.6"/></geometry></collision>
    <collision>
      <origin xyz="0 0 1"/>
      <geometry><cylinder radius="0.1" length="0.2"/></geometry>
    </collision>
  </link>
  <link name="ball"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
</robot>"""


def test_read_stl_vertices(mesh):
    vertices = read_stl_vertices(mesh)
    np.testing.assert_allclose(vertices, [[0, 0, 0], [100, 0, 0], [0, 50, 20]])


def test_ascii_stl_is_rejected(tmp_path):
    path = tmp_path / "ascii.stl"
    path.write_text("solid x\nendsolid x\n")
    with pytest.raises(ValueError):
        read_stl_vertices(str(path))


def test_link_boxes(mesh):
    boxes, warnings = link_boxes(urdf(mesh), PAD)
    assert warnings == []
    by_name = {box.link: box for box in boxes}
    assert sorted(by_name) == ["ball", "meshed", "shapes"]
    # Mesh in mm, rotated 90 deg about z and moved 1 m along x.
    meshed = by_name["meshed"]
    np.testing.assert_allclose(meshed.lower, [0.95 - PAD, 0 - PAD, 0 - PAD], atol=1e-9)
    np.testing.assert_allclose(
        meshed.upper, [1.0 + PAD, 0.1 + PAD, 0.02 + PAD], atol=1e-9
    )
    shapes = by_name["shapes"]
    np.testing.assert_allclose(shapes.lower, [-0.1 - PAD, -0.2 - PAD, -0.3 - PAD])
    np.testing.assert_allclose(shapes.upper, [0.1 + PAD, 0.2 + PAD, 1.1 + PAD])
    np.testing.assert_allclose(by_name["ball"].upper, [0.05 + PAD] * 3)


def test_unreadable_mesh_is_skipped_with_a_warning(tmp_path):
    boxes, warnings = link_boxes(urdf(str(tmp_path / "missing.stl")), PAD)
    assert "meshed" not in {box.link for box in boxes}
    assert len(warnings) == 1 and warnings[0].startswith("meshed:")


def test_points_in_boxes_at_link_pose(mesh):
    boxes, _ = link_boxes(urdf(mesh), 0.0)
    ball = [box for box in boxes if box.link == "ball"]
    pose = np.eye(4)
    pose[:3, 3] = (2.0, 0.0, 0.0)
    points = np.array([[2.0, 0.0, 0.04], [2.0, 0.0, 0.06], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(points_in_boxes(points, ball, [pose]), [1, 0, 0])
    assert not points_in_boxes(np.empty((0, 3)), ball, [pose]).any()
