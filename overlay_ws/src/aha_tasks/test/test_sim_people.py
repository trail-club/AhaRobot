"""Posed sim people: skinning on a tiny rig, texture recolouring and scenes."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks import sim_people  # noqa: E402
from aha_tasks.sim_people import (  # noqa: E402
    COLLADA,
    LABELS,
    POSES,
    Rig,
    X,
    load_scene,
    models,
    person_sdf,
    recolour_shirt,
    shown_frame,
    spawn_missing,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sim" / "people"


def translation(x, y, z):
    return " ".join(str(v) for v in (1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1))


# Spine1 at 0.9 m, the left upper arm from (0, 0.2, 1.4) and the forearm from
# (0, 0.6, 1.4); one vertex on each, plus a foot on the floor.
TINY_RIG = f"""<?xml version="1.0"?>
<COLLADA xmlns="{COLLADA}" version="1.4.1">
  <library_images><image id="tex"><init_from>tex.png</init_from></image></library_images>
  <library_geometries><geometry id="body-mesh"><mesh>
    <source id="body-positions"><float_array id="body-positions-array" count="12">0 0 1 0 0.5 1.4 0 0.8 1.4 0 0 0</float_array></source>
    <source id="body-normals"><float_array id="body-normals-array" count="3">1 0 0</float_array></source>
    <vertices id="body-vertices"><input semantic="POSITION" source="#body-positions"/></vertices>
    <polylist material="m" count="1">
      <input semantic="VERTEX" source="#body-vertices" offset="0"/>
      <input semantic="NORMAL" source="#body-normals" offset="1"/>
      <vcount>4</vcount><p>0 0 1 0 2 0 3 0</p>
    </polylist>
  </mesh></geometry></library_geometries>
  <library_controllers><controller id="skin"><skin source="#body-mesh">
    <bind_shape_matrix>{translation(0, 0, 0)}</bind_shape_matrix>
    <source id="joints"><Name_array id="joints-array" count="3">Spine1 UpperArm_L LowerArm_L</Name_array></source>
    <source id="binds"><float_array id="binds-array" count="48">{translation(0, 0, -0.9)} {translation(0, -0.2, -1.4)} {translation(0, -0.6, -1.4)}</float_array></source>
    <source id="weights"><float_array id="weights-array" count="1">1</float_array></source>
    <joints><input semantic="JOINT" source="#joints"/><input semantic="INV_BIND_MATRIX" source="#binds"/></joints>
    <vertex_weights count="4">
      <input semantic="JOINT" source="#joints" offset="0"/><input semantic="WEIGHT" source="#weights" offset="1"/>
      <vcount>1 1 1 1</vcount><v>0 0 1 0 2 0 0 0</v>
    </vertex_weights>
  </skin></controller></library_controllers>
  <library_visual_scenes><visual_scene id="Scene">
    <node id="Armature" type="NODE">
      <node id="a_Spine1" sid="Spine1" type="JOINT">
        <node id="a_UpperArm_L" sid="UpperArm_L" type="JOINT">
          <node id="a_LowerArm_L" sid="LowerArm_L" type="JOINT"/>
        </node>
      </node>
      <node id="Body" type="NODE">
        <instance_controller url="#skin"><bind_material><technique_common/></bind_material></instance_controller>
      </node>
    </node>
  </visual_scene></library_visual_scenes>
</COLLADA>
"""


class RigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "rig.dae"
        self.path.write_text(TINY_RIG)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_joint_turns_with_the_joints_below_it(self):
        rig = Rig(self.path)
        positions, normals = rig.pose([("UpperArm_L", X, -90.0)])
        np.testing.assert_allclose(positions[0], (0.0, 0.0, 1.0), atol=1e-9)
        np.testing.assert_allclose(positions[1], (0.0, 0.2, 1.1), atol=1e-9)
        np.testing.assert_allclose(positions[2], (0.0, 0.2, 0.8), atol=1e-9)
        np.testing.assert_allclose(normals[0], (1.0, 0.0, 0.0), atol=1e-9)

    def test_whole_body_rotation_puts_the_lowest_point_on_the_floor(self):
        positions, _ = Rig(self.path).pose([], root_rotation=((0, 1, 0), -90.0))
        self.assertAlmostEqual(positions[:, 2].min(), 0.0)
        # Head (+z) towards -x.
        self.assertLess(positions[0, 0], positions[3, 0])

    def test_static_mesh_has_no_skin(self):
        rig = Rig(self.path)
        positions, normals = rig.pose([("LowerArm_L", X, 90.0)])
        root = ET.fromstring(rig.static_dae(positions, normals, "shirt.png"))
        ns = {"c": COLLADA}
        self.assertIsNone(root.find("c:library_controllers", ns))
        self.assertEqual(root.find(".//c:init_from", ns).text, "shirt.png")
        nodes = root.findall(".//c:visual_scene/c:node", ns)
        self.assertEqual([n.get("id") for n in nodes], ["Body"])
        body = nodes[0]
        self.assertEqual(body.find("c:instance_geometry", ns).get("url"), "#body-mesh")
        self.assertIsNotNone(body.find(".//c:bind_material", ns))
        written = np.array(
            root.find(".//c:float_array", ns).text.split(), dtype=float
        ).reshape(-1, 3)
        np.testing.assert_allclose(written, positions, atol=1e-5)


class TextureTests(unittest.TestCase):
    def test_only_the_purple_shirt_is_recoloured(self):
        pixels = np.array([[[110, 80, 170], [220, 170, 140]]], dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        out = np.asarray(
            Image.open(io.BytesIO(recolour_shirt(buffer.getvalue(), (200, 0, 0))))
        )
        self.assertGreater(out[0, 0, 0], 150)
        self.assertLess(out[0, 0, 2], 20)
        np.testing.assert_array_equal(out[0, 1], pixels[0, 1])


class SceneTests(unittest.TestCase):
    def test_every_label_has_a_pose(self):
        for label in LABELS:
            for frame in sim_people.mesh_frames(label):
                self.assertTrue(frame in POSES or frame in sim_people.ROOT_ROTATIONS)

    def test_waving_alternates(self):
        self.assertEqual(shown_frame("waving", 0.1, 0.8), "waving_a")
        self.assertEqual(shown_frame("waving", 0.5, 0.8), "waving_b")
        self.assertEqual(shown_frame("sitting", 0.5, 0.8), "sitting")

    def test_shipped_scenes_load(self):
        for path in sorted(CONFIG.glob("*.json")):
            with self.subTest(path.name):
                scene = load_scene(path)
                paths = {
                    (frame, p["shirt"]): Path(f"/m/{frame}.dae")
                    for p in scene["people"].values()
                    for frame in sim_people.mesh_frames(p["label"])
                }
                names = [model for model, *_ in models(scene, paths)]
                self.assertEqual(len(names), len(set(names)))

    def test_bad_scenes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            person = {
                "label": "dancing",
                "shirt": "red",
                "position": [0, 0, 0],
                "yaw": 0,
            }
            path.write_text(
                json.dumps({"world": "w", "wave_period": 0.8, "people": {"a": person}})
            )
            with self.assertRaises(ValueError):
                load_scene(path)

    def test_spawning_moves_left_models_and_skips_failures(self):
        missing = ["new", "left", "broken", "later"]
        created = {"new", "later"}  # "left" exists from an earlier run
        spawned = spawn_missing(missing, lambda e: e in created, lambda e: e == "left")
        self.assertEqual(spawned, ["new", "left", "later"])
        self.assertEqual(missing, ["broken"])

    def test_sdf_points_at_the_mesh(self):
        sdf = person_sdf("p", Path("/tmp/a b.dae"), 1.0, 2.0, 0.0, 1.5)
        self.assertIn("file:///tmp/a%20b.dae", sdf)
        self.assertIn("<pose>1.0 2.0 0.0 0 0 1.5</pose>", sdf)


if __name__ == "__main__":
    unittest.main()
