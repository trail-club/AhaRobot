"""Simulated objects: the model catalog, the task scenes and generated models."""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_perception.furniture import Furniture, load_config  # noqa: E402
from aha_perception.object_list import load_object_list  # noqa: E402
from aha_tasks.sim_objects import (  # noqa: E402
    SIZE_TOLERANCE,
    Catalog,
    SceneError,
    SceneObject,
    displacement,
    footprint,
    footprints_overlap,
    furniture_config,
    load_scene,
    stale_models,
)

PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE.parent
SCENES = PACKAGE / "config" / "sim" / "objects"
CATALOG = PACKAGE / "config" / "sim" / "object_models.json"
PERCEPTION = SRC / "aha_perception" / "config"
COMMON_OBJECTS = {"red bull", "pringles", "rubiks cube", "toothpaste"}


def ycb_root():
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("tmc_wrs_gz_worlds")) / "models"
    except (ImportError, LookupError):
        source = SRC.parents[1] / "upstream" / "tmc_wrs_gz" / "tmc_wrs_gz_worlds"
        return source / "models"


def singular(category):
    category = category.strip().lower()
    return category[:-1] if category.endswith("s") else category


class SceneFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ycb_root()
        if not root.is_dir():
            raise unittest.SkipTest(f"no tmc_wrs_gz_worlds models at {root}")
        cls.objects = load_object_list(PERCEPTION / "objects" / "robocup_objects.yaml")
        cls.catalog = Catalog.load(CATALOG, cls.objects, root)
        furniture = load_config(PERCEPTION / "furniture" / "rcjo2026_arena.yaml")
        cls.places = {f.name: f for f in furniture.furniture}

    def scene(self, name):
        return load_scene(SCENES / f"{name}.json", self.catalog, self.places)

    def write_scene(self, objects, world="rcjo2026_arena"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "scene.json"
        path.write_text(json.dumps({"world": world, "objects": objects}))
        return path


class CatalogTests(SceneFixture):
    def test_models_have_the_listed_size(self):
        for label in self.catalog.models:
            with self.subTest(label=label):
                np.testing.assert_allclose(
                    self.catalog.size(label),
                    self.objects.get(label).size,
                    rtol=0,
                    atol=SIZE_TOLERANCE,
                )

    def test_ycb_sdf_moves_the_origin_to_the_box_center(self):
        obj = self.scene("pick_and_place").get("mug")
        root = ET.fromstring(self.catalog.sdf(obj, Path("/unused")))
        model = root.find("model")
        self.assertEqual(model.get("name"), "mug")
        pose = [float(v) for v in model.findtext("pose").split()]
        np.testing.assert_allclose(
            pose[:3], [*obj.position[:2], obj.position[2] + 0.002], atol=1e-6
        )
        self.assertAlmostEqual(pose[5], obj.yaw)
        for collision in model.findall("link/collision"):
            self.assertEqual(collision.findtext("surface/friction/ode/mu"), "1.0")
            uri = collision.findtext("geometry/mesh/uri")
            # Absolute files: the built scans are not on Gazebo's resource path.
            self.assertTrue(uri.startswith("file://"), uri)
            self.assertTrue(Path(uri[len("file://") :]).is_file(), uri)
            self.assertIn("/ycb_025_mug/", uri)


class SceneTests(SceneFixture):
    def test_scenes_load(self):
        for name in ("pick_and_place", "gpsr"):
            with self.subTest(scene=name):
                scene = self.scene(name)
                self.assertEqual(scene.world, "rcjo2026_arena")
                for obj in scene.objects:
                    # The detector reports the model under its label.
                    self.assertEqual(self.objects.for_model(obj.name).label, obj.label)

    def test_objects_rest_on_their_surface(self):
        scene = self.scene("pick_and_place")
        for obj in scene.objects:
            height = (
                0.0
                if obj.on == "floor"
                else self.places[obj.on].level(obj.level).height
            )
            self.assertAlmostEqual(obj.position[2] - obj.size[2] / 2, height)

    def test_pick_and_place_follows_the_rulebook(self):
        tasks = json.loads((PACKAGE / "config" / "pick_and_place.json").read_text())
        trash = singular(tasks["initial_state"]["ppc.trash_category"])
        roles = self.scene("pick_and_place").by_role()
        table = {r: o for r, o in roles.items() if r.startswith("table.")}
        self.assertEqual(
            set(table),
            {
                f"table.{r}"
                for r in ("cutlery", "cup", "plate", "trash", "other1", "other2")
            },
        )
        self.assertTrue(all(o.on == "dining_table" for o in table.values()))
        self.assertEqual(table["table.cutlery"].category, "cutlery")
        self.assertEqual(table["table.cup"].label, "mug")
        self.assertEqual(table["table.plate"].label, "plate")
        self.assertEqual(singular(table["table.trash"].category), trash)
        for other in ("table.other1", "table.other2"):
            self.assertNotIn(table[other].category, ("cutlery", "tableware"))
            self.assertNotEqual(singular(table[other].category), trash)
        self.assertEqual(roles["trash.floor"].on, "floor")
        self.assertEqual(singular(roles["trash.floor"].category), trash)
        for role in ("breakfast.bowl", "breakfast.spoon", "dishwasher.tab"):
            self.assertEqual(roles[role].on, "tall_table")
        self.assertEqual(roles["dishwasher.tab"].label, "dishwasher tab")
        for role in ("extra.obj1", "extra.obj2"):
            self.assertEqual(roles[role].on, "long_table")
            self.assertIn(roles[role].label, COMMON_OBJECTS)

    def test_cabinet_groups_one_category_per_level(self):
        scene = self.scene("pick_and_place")
        levels = {}
        for obj in scene.objects:
            if obj.on == "shelf":
                levels.setdefault(obj.level, set()).add(obj.category)
        self.assertTrue(all(len(c) == 1 for c in levels.values()), levels)
        categories = {c for cs in levels.values() for c in cs}
        roles = scene.by_role()
        # Milk and cereal stand with their category; the table's other
        # objects have a level of their category.
        for role in ("breakfast.milk", "breakfast.cereal"):
            self.assertEqual(roles[role].on, "shelf")
            self.assertEqual(levels[roles[role].level], {roles[role].category})
        for role in ("table.other1", "table.other2"):
            self.assertIn(roles[role].category, categories)

    def test_furniture_of_the_scene_world(self):
        directory = PERCEPTION / "furniture"
        path = furniture_config(directory, SCENES / "pick_and_place.json")
        self.assertEqual(path.name, "rcjo2026_arena.yaml")
        other = self.write_scene([], world="rcjo2026")
        with self.assertRaises(SceneError):  # not placed on the arena's furniture
            furniture_config(directory, other)

    def test_no_model_of_the_world_is_stale(self):
        # sim_objects removes stale models; none of the scene's world may be one.
        try:
            from ament_index_python.packages import get_package_share_directory
            import xacro

            sobits = Path(get_package_share_directory("sobits_gazebo_worlds"))
        except (ImportError, LookupError):
            self.skipTest("no xacro or sobits_gazebo_worlds")
        for name in ("pick_and_place", "gpsr"):
            scene = self.scene(name)
            source = sobits / "worlds" / f"{scene.world}.world.xacro"
            root = ET.fromstring(xacro.process_file(str(source)).toxml())
            names = [m.get("name") for m in root.iter("model")]
            for include in root.iter("include"):
                uri = (include.findtext("uri") or "").rstrip("/")
                names.append((include.findtext("name") or uri.split("/")[-1]).strip())
            with self.subTest(scene=name):
                self.assertGreater(len(names), 10)  # the furniture
                self.assertEqual(stale_models(names, scene, self.objects), [])

    def test_only_objects_of_another_scene_are_stale(self):
        scene = self.scene("pick_and_place")
        present = [
            *(o.name for o in self.scene("gpsr").objects),
            *(o.name for o in scene.objects),
            # World, robot, guests, people and held objects are never objects.
            "dining_table",
            "shelf",
            "trash_box",
            "aha_robot",
            "guest1",
            "bedroom_standing",
            "office_waving__waving_a",
            "aha_held_object_right",
            "link",
            "",
        ]
        stale = stale_models(present, scene, self.objects)
        self.assertEqual(
            stale,
            sorted(["banana", "cup_noodle", "green_tea", "plum", "strawberry_jello"]),
        )
        self.assertEqual(
            stale_models(present, self.scene("gpsr"), self.objects)[:3],
            ["cereal", "chocolate_jello", "cleanser"],
        )
        self.assertEqual(
            stale_models([o.name for o in scene.objects], scene, self.objects), []
        )

    def test_settled_pose_check(self):
        obj = self.scene("pick_and_place").get("milk")
        pose = np.eye(4)
        pose[:3, 3] = obj.position
        self.assertEqual(displacement(obj, pose), (0.0, 0.0))
        pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # fallen over
        self.assertAlmostEqual(displacement(obj, pose)[1], np.pi / 2)

    def test_invalid_scenes_are_refused(self):
        base = {
            "name": "milk",
            "label": "milk",
            "on": "tall_table",
            "position": [0.75, -0.5],
            "yaw": 0.0,
        }
        cases = {
            "wrong name": [dict(base, name="carton")],
            "unknown label": [dict(base, label="pizza", name="pizza")],
            "over the edge": [dict(base, position=[0.93, -0.5])],
            "overlap": [base, dict(base, name="milk_2", position=[0.8, -0.5])],
            "duplicate": [base, dict(base, position=[0.62, -0.62])],
            "unknown furniture": [dict(base, on="sofa")],
            "shelf without level": [dict(base, on="shelf", position=[3.2, 3.45])],
            "floor under furniture": [dict(base, on="floor")],
            "extra key": [dict(base, colour="red")],
        }
        for reason, objects in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaises(SceneError):
                    load_scene(self.write_scene(objects), self.catalog, self.places)
        scene = load_scene(self.write_scene([base]), self.catalog, self.places)
        self.assertEqual(scene.objects[0].level, 1)

    def test_objects_fit_under_the_next_board(self):
        low = {"rack": Furniture("rack", "shelf", 0.0, 0.0, 0.0, 0.4, 0.6, (0.3, 0.45))}
        milk = {"name": "milk", "label": "milk", "on": "rack", "level": 1}
        milk.update(position=[0.0, 0.0], yaw=0.0)
        with self.assertRaises(SceneError):  # 0.19 m under 0.13 m
            load_scene(self.write_scene([milk]), self.catalog, low)
        milk["level"] = 2  # the top level has no board above
        load_scene(self.write_scene([milk]), self.catalog, low)


class FakeGazebo:
    """gz.transport13.Node stand-in: models (name -> x, y, z, yaw) published
    on pose/info at 20 Hz; removing a ``stuck`` model fails."""

    def __init__(self, models, stuck=()):
        self.models, self.stuck, self.calls = dict(models), set(stuck), []
        self.lock, self.stop = threading.Lock(), threading.Event()

    def subscribe(self, _type, _topic, callback, _options):
        from gz.msgs10.pose_v_pb2 import Pose_V

        def publish():
            while not self.stop.wait(0.05):
                message = Pose_V()
                with self.lock:
                    for name, (x, y, z, yaw) in self.models.items():
                        pose = message.pose.add(name=name)
                        pose.position.x, pose.position.y, pose.position.z = x, y, z
                        pose.orientation.z = math.sin(yaw / 2)
                        pose.orientation.w = math.cos(yaw / 2)
                callback(message)

        threading.Thread(target=publish, daemon=True).start()
        return True

    def request(self, service, message, _type, reply_type, _timeout):
        kind = service.rsplit("/", 1)[1]
        with self.lock:
            if kind == "create":
                model = ET.fromstring(message.sdf).find("model")
                name = model.get("name")
                x, y, z, _, _, yaw = (float(v) for v in model.findtext("pose").split())
                ok = name not in self.models
            else:
                name = message.name
                ok = name in self.models and not (
                    kind == "remove" and name in self.stuck
                )
            if ok and kind == "remove":
                del self.models[name]
            elif ok and kind == "set_pose":
                p, q = message.position, message.orientation
                x, y, z, yaw = p.x, p.y, p.z, 2 * math.atan2(q.z, q.w)
            if ok and kind != "remove":
                self.models[name] = (x, y, z, yaw)
            self.calls.append((kind, name, ok))
        return True, reply_type(data=ok)


class NodeTests(SceneFixture):
    def test_objects_that_stay_in_gazebo_are_moved_back(self):
        try:
            import gz.transport13  # noqa: F401

            from aha_tasks import sim_objects_node
        except ImportError:
            self.skipTest("no gz-transport Python bindings")
        source = json.loads((SCENES / "pick_and_place.json").read_text())
        objects = [
            o for o in source["objects"] if o["name"] in ("plate", "fork", "mug")
        ]
        path = self.write_scene(objects)
        # A moved mug that cannot be removed, a plate left from a run before
        # and a banana of another scene.
        world = {"aha_robot": (0, 0, 0, 0), "plate": (-2.0, -1.6, 0.8, 0)}
        world.update(mug=(-1.0, -1.0, 0.0, 1.0), banana=(-1.9, -1.8, 0.8, 0))
        fake = FakeGazebo(world, stuck={"mug"})
        self.addCleanup(fake.stop.set)
        with mock.patch("gz.transport13.Node", lambda: fake), mock.patch.object(
            sim_objects_node, "REMOVE_TIMEOUT", 0.5
        ):
            code = sim_objects_node.main([f"--config={path}", "--settle=0"])
        self.assertEqual(code, 0)
        self.assertIn(("remove", "banana", True), fake.calls)
        self.assertIn(("create", "plate", True), fake.calls)
        self.assertIn(("create", "fork", True), fake.calls)
        self.assertIn(("set_pose", "mug", True), fake.calls)
        self.assertNotIn("create", [k for k, name, _ in fake.calls if name == "mug"])
        mug = load_scene(path, self.catalog, self.places).get("mug")
        np.testing.assert_allclose(
            fake.models["mug"], [*mug.spawn_pose()[:3], mug.yaw], atol=1e-6
        )


class FootprintTests(unittest.TestCase):
    def test_separating_axes(self):
        square = np.array([[0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]])
        self.assertTrue(footprints_overlap(square, square + [0.09, 0.0]))
        self.assertFalse(footprints_overlap(square, square + [0.11, 0.0]))
        self.assertTrue(footprints_overlap(square, square + [0.105, 0.0], gap=0.01))
        c = np.cos(np.pi / 4)
        diamond = square @ np.array([[c, c], [-c, c]])
        # Bounding boxes overlap, the shapes do not.
        self.assertFalse(footprints_overlap(diamond, square + [0.123, 0.0]))

    def test_footprint_turns_with_the_yaw(self):
        obj = SceneObject(
            "fork",
            "fork",
            "cutlery",
            "t",
            1,
            "",
            (1.0, 2.0, 0.8),
            np.pi / 2,
            (0.2, 0.02, 0.01),
        )
        corners = footprint(obj)
        np.testing.assert_allclose(
            corners.max(0) - corners.min(0), [0.02, 0.2], atol=1e-9
        )
        np.testing.assert_allclose(corners.mean(0), [1.0, 2.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
