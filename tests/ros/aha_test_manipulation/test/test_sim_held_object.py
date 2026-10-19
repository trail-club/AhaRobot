"""Object SDF of the simulated held objects (aha_test_manipulation.sim_held_object)."""

import xml.etree.ElementTree as ET

from aha_test_manipulation.sim_held_object import object_sdf, ycb_mesh

YCB_SDF = """<?xml version="1.0" ?>
<sdf version='1.8'><model name='ycb_013_apple'><link name='body'>
<inertial><mass>0.068</mass></inertial>
<visual name='v'><pose>-0.0009 0.0038 0.0002 0 0 0</pose><geometry><mesh>
<uri>model://ycb_013_apple/meshes/textured.dae</uri></mesh></geometry></visual>
<collision name='c'><pose>-0.0009 0.0038 0.0002 0 0 0</pose><geometry><mesh>
<uri>model://ycb_013_apple/meshes/nontextured.stl</uri></mesh></geometry></collision>
</link></model></sdf>"""


def test_ycb_mesh_is_centered_in_a_link_named_link(tmp_path):
    (tmp_path / "model-1_4.sdf").write_text(YCB_SDF)
    mesh = ycb_mesh(tmp_path, 0.072)
    assert mesh.uri == "model://ycb_013_apple/meshes/textured.dae"
    assert mesh.collision == "model://ycb_013_apple/meshes/nontextured.stl"
    assert mesh.offset == (-0.0009, 0.0038, 0.0002 - 0.036)
    assert mesh.mass == 0.068
    size = (0.075, 0.075, 0.072)
    link = ET.fromstring(object_sdf("apple_1", "cylinder", size, mesh=mesh)).find(
        "model/link"
    )
    assert link.get("name") == "link"
    assert link.find("visual/geometry/mesh/uri").text == mesh.uri
    assert link.find("collision/geometry/mesh/uri").text == mesh.collision
    for element in ("visual", "collision"):
        z = float(link.find(f"{element}/pose").text.split()[2])
        assert abs(z + 0.0358) < 1e-9
    # Without the collision mesh the cylinder collides, centered.
    shape = ycb_mesh(tmp_path, 0.072, collision=False)
    link = ET.fromstring(object_sdf("apple_1", "cylinder", size, mesh=shape)).find(
        "model/link"
    )
    assert link.find("collision/geometry/cylinder/length").text == "0.072"
    assert link.find("collision/pose") is None


def test_primitive_object_keeps_its_shape():
    model = ET.fromstring(object_sdf("box_1", "box", (0.06, 0.1, 0.08)))
    assert model.find("model/link/collision/geometry/box/size").text == "0.06 0.1 0.08"
    assert model.find("model/link/visual/material") is not None
