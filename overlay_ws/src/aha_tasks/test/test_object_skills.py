"""Object skills: clustering, surfaces, roles and the shelf, and the skills on
fake detection messages (no ROS graph)."""

from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.objects import (  # noqa: E402
    PLACE_GAP,
    PLACE_WIDTH,
    Objects,
    Observation,
    Percept,
    PerceiveObjects,
    PerceiveShelf,
    assign_roles,
    categories_by_level,
    cluster,
    floor_region,
    furniture_region,
    half_width,
    norm_category,
    placement,
    same_category,
    shelf_groups,
)
from aha_perception.furniture import load_config  # noqa: E402
from aha_perception.object_list import parse_object_list  # noqa: E402

FURNITURE = load_config(
    Path(__file__).resolve().parents[2]
    / "aha_perception"
    / "config"
    / "furniture"
    / "rcjo2026_arena.yaml"
)
OBJECT_LIST = parse_object_list(
    {
        "objects": [
            {"label": label, "category": category, "shape": "box", "size": [0.1] * 3}
            for label, category in (
                ("fork", "cutlery"),
                ("spoon", "cutlery"),
                ("mug", "tableware"),
                ("plate", "tableware"),
                ("bowl", "tableware"),
                ("cookies", "snack"),
                ("pringles", "snack"),
                ("apple", "fruit"),
                ("orange", "fruit"),
                ("lemon", "fruit"),
                ("peach", "fruit"),
                ("cola", "drink"),
                ("milk", "drink"),
                ("cereal", "food"),
                ("tuna can", "food"),
                ("rubiks cube", "toy"),
            )
        ]
    }
)
CATEGORY = OBJECT_LIST.category
TABLE_ROLES = {
    "cutlery": {"category": "cutlery"},
    "cup": {"labels": ["mug", "cup"]},
    "plate": {"labels": ["plate", "dish"]},
    "trash": {"category_key": "ppc.trash_category"},
    "other1": {},
    "other2": {},
}
TABLE = 0.80  # dining table top (m)


def seen(label, xyz, size=(0.1, 0.1, 0.1), count=5, confidence=1.0, track=1):
    return Percept(
        label, CATEGORY(label), tuple(xyz), tuple(size), track, confidence, count
    )


def on_top(x, y, height, size):
    """Centre of an object of box `size` standing at height."""
    return (x, y, height + size[2] / 2)


class LabelTests(unittest.TestCase):
    def test_categories_ignore_case_and_plural(self):
        self.assertEqual(norm_category("Snacks"), "snack")
        self.assertTrue(same_category("snacks", "snack"))
        self.assertTrue(same_category("Cleaning supplies", "cleaning supplies"))
        self.assertFalse(same_category("", ""))
        self.assertFalse(same_category("fruit", "food"))


class ClusterTests(unittest.TestCase):
    def test_tracks_are_objects_with_the_majority_label(self):
        messages = [
            [Observation("mug", 3, (1.0, 0.0, 0.8), (0.1, 0.1, 0.1), 0.9)],
            [Observation("cup", 3, (1.01, 0.0, 0.8), (0.1, 0.1, 0.1), 0.5)],
            [Observation("mug", 3, (0.99, 0.0, 0.8), (0.1, 0.1, 0.1), 0.7)],
        ]
        (percept,) = cluster(messages, CATEGORY)
        self.assertEqual(percept.label, "mug")
        self.assertEqual(percept.category, "tableware")
        self.assertEqual(percept.count, 3)
        self.assertAlmostEqual(percept.confidence, 0.8)
        self.assertAlmostEqual(percept.position[0], 1.0)

    def test_a_new_track_of_the_same_object_is_merged(self):
        messages = [
            [Observation("mug", 3, (1.0, 0.0, 0.8), (0.1,) * 3, 1.0)],
            [Observation("mug", 9, (1.03, 0.0, 0.8), (0.1,) * 3, 1.0)],
        ]
        (percept,) = cluster(messages, CATEGORY)
        self.assertEqual((percept.count, percept.track), (2, 9))

    def test_objects_side_by_side_stay_apart(self):
        # Two cans (0.066 m) touching, seen together with their own tracks.
        can = (0.066, 0.066, 0.12)
        messages = [
            [
                Observation("cola", 1, (1.0, 0.0, 0.86), can, 1.0),
                Observation("cola", 2, (1.066, 0.0, 0.86), can, 1.0),
            ]
        ] * 2
        self.assertEqual(len(cluster(messages, CATEGORY)), 2)
        untracked = [[replace(o, track=-1) for o in messages[0]]] * 2
        self.assertEqual([p.count for p in cluster(untracked, CATEGORY)], [2, 2])

    def test_close_objects_of_other_labels_stay_apart(self):
        # A fork lying on a plate.
        messages = [
            [
                Observation("plate", 1, (1.0, 0.0, 0.81), (0.26, 0.26, 0.02), 1.0),
                Observation("fork", 2, (1.02, 0.0, 0.82), (0.02, 0.2, 0.02), 1.0),
            ]
        ] * 2
        labels = sorted(p.label for p in cluster(messages, CATEGORY))
        self.assertEqual(labels, ["fork", "plate"])

    def test_yaw_is_the_circular_mean_of_the_label(self):
        size = (0.195, 0.027, 0.015)
        messages = [
            [Observation("fork", 1, (1.0, 0.0, 0.8), size, 1.0, math.pi - 0.02)],
            [Observation("fork", 1, (1.0, 0.0, 0.8), size, 1.0, -math.pi + 0.04)],
            [Observation("spoon", 1, (1.0, 0.0, 0.8), size, 1.0, 1.0)],
            [Observation("fork", 1, (1.0, 0.0, 0.8), size, 1.0, None)],
        ]
        (percept,) = cluster(messages, CATEGORY)
        self.assertAlmostEqual(abs(percept.yaw), math.pi - 0.01, places=6)
        (unknown,) = cluster([[Observation("fork", 1, (0, 0, 0), size, 1.0)]], CATEGORY)
        self.assertIsNone(unknown.yaw)

    def test_untracked_detections_cluster_by_distance(self):
        messages = [
            [Observation("apple", -1, (1.0, 0.0, 0.8), (0.1,) * 3, 1.0)],
            [
                Observation("apple", -1, (1.02, 0.01, 0.8), (0.1,) * 3, 1.0),
                Observation("apple", -1, (1.5, 0.0, 0.8), (0.1,) * 3, 1.0),
            ],
        ]
        counts = sorted(p.count for p in cluster(messages, CATEGORY))
        self.assertEqual(counts, [1, 2])


class RegionTests(unittest.TestCase):
    def setUp(self):
        identity = (0.0, 0.0, 0.0, 0.0)
        self.table = furniture_region(FURNITURE.get("dining_table"), identity, 0.5)
        self.shelf = furniture_region(FURNITURE.get("shelf"), identity, 0.1)

    def test_objects_on_the_table_and_stacked_ones(self):
        cup = seen("mug", on_top(-2.0, -2.0, TABLE, (0.1, 0.1, 0.1)))
        fork_on_plate = seen(
            "fork", on_top(-2.0, -2.0, TABLE + 0.02, (0.02, 0.2, 0.02))
        )
        lying = seen("cola", (-1.8, -2.0, TABLE + 0.03), (0.066, 0.066, 0.12))
        for percept in (cup, fork_on_plate, lying):
            self.assertEqual(self.table.level_of(percept), 1, percept.label)

    def test_floor_chair_and_far_objects_are_not_on_the_table(self):
        floor = seen("apple", on_top(-2.0, -1.2, 0.0, (0.075,) * 3))
        chair = seen("apple", on_top(-1.2, -2.0, 0.45, (0.075,) * 3))
        beyond = seen("apple", on_top(-2.0, -0.6, TABLE, (0.075,) * 3))
        for percept in (floor, chair, beyond):
            self.assertIsNone(self.table.level_of(percept))

    def test_a_moved_table_is_within_the_margin(self):
        # The table's near edge is y = -1.35; moved 0.4 m towards the robot.
        moved = seen("mug", on_top(-2.0, -1.0, TABLE, (0.1,) * 3))
        self.assertEqual(self.table.level_of(moved), 1)

    def test_shelf_levels(self):
        heights = FURNITURE.get("shelf").heights
        for number, height in enumerate(heights, start=1):
            tall = seen("cereal", on_top(3.2, 3.5, height, (0.06, 0.16, 0.26)))
            self.assertEqual(self.shelf.level_of(tall), number)
            self.assertEqual(self.shelf.level_of(tall, level=number), number)
        self.assertIsNone(self.shelf.level_of(tall, level=1))

    def test_look_points_cover_a_long_table_and_each_shelf_level(self):
        points = self.table.look_points(robot_xy=(-2.0, -0.7))
        self.assertEqual(len(points), 3)  # along the 1.8 m side, nearest first
        self.assertEqual([round(p[1], 2) for p in points], [-1.65, -2.25, -2.85])
        self.assertTrue(all(p[2] == TABLE for p in points))
        shelf = self.shelf.look_points()
        self.assertEqual(len(shelf), len(FURNITURE.get("shelf").heights))
        tall_table = furniture_region(FURNITURE.get("tall_table"), (0, 0, 0, 0), 0.1)
        self.assertEqual(len(tall_table.look_points()), 1)

    def test_furniture_is_moved_into_the_detection_frame(self):
        tall = FURNITURE.get("tall_table")
        region = furniture_region(tall, (1.0, 2.0, 0.1, math.pi / 2), 0.0)
        self.assertAlmostEqual(region.x, 1.0 - tall.y)
        self.assertAlmostEqual(region.y, 2.0 + tall.x)
        self.assertAlmostEqual(region.heights[0], tall.heights[0] + 0.1)

    def test_floor_within_the_radius(self):
        floor = floor_region((4.0, -3.0), 1.5)
        near = seen("cookies", on_top(4.5, -3.5, 0.0, (0.06, 0.1, 0.15)))
        far = seen("cookies", on_top(6.0, -3.0, 0.0, (0.06, 0.1, 0.15)))
        raised = seen("cookies", on_top(4.5, -3.5, 0.4, (0.06, 0.1, 0.15)))
        self.assertEqual(floor.level_of(near), 1)
        self.assertIsNone(floor.level_of(far))
        self.assertIsNone(floor.level_of(raised))


class RoleTests(unittest.TestCase):
    def table(self):
        return [
            seen("fork", (-2.2, -1.8, 0.81)),
            seen("mug", (-1.8, -1.8, 0.84)),
            seen("plate", (-2.0, -1.9, 0.81)),
            seen("cookies", (-2.3, -2.2, 0.87)),
            seen("apple", (-1.7, -2.4, 0.84)),
            seen("cola", (-2.0, -1.6, 0.86)),
        ]

    def test_the_dining_table_roles(self):
        memory = {"ppc.trash_category": "snacks"}
        assigned, missing = assign_roles(
            self.table(), TABLE_ROLES, memory, (-2.0, -0.7)
        )
        labels = {role: p.label for role, p in assigned.items()}
        self.assertEqual(missing, [])
        self.assertEqual(
            labels,
            {
                "cutlery": "fork",
                "cup": "mug",
                "plate": "plate",
                "trash": "cookies",
                "other1": "cola",  # nearer the robot
                "other2": "apple",
            },
        )

    def test_matching_objects_never_become_others(self):
        # A second piece of cutlery and an unknown object; the trash is missing.
        objects = self.table()[:3] + [
            seen("spoon", (-2.1, -2.0, 0.81)),
            seen("vase", (-2.0, -2.5, 0.9)),
        ]
        memory = {"ppc.trash_category": "snacks"}
        assigned, missing = assign_roles(objects, TABLE_ROLES, memory, (-2.0, -0.7))
        self.assertEqual(missing, ["trash", "other1", "other2"])
        self.assertEqual(set(assigned), {"cutlery", "cup", "plate"})

    def test_the_best_seen_object_wins(self):
        objects = [
            seen("fork", (-2.2, -1.8, 0.81), count=1),
            seen("spoon", (-2.0, -2.0, 0.81), count=6),
        ]
        assigned, _ = assign_roles(
            objects, {"cutlery": {"category": "cutlery"}}, {}, (0, 0)
        )
        self.assertEqual(assigned["cutlery"].label, "spoon")

    def test_without_the_trash_category_no_trash(self):
        assigned, missing = assign_roles(self.table(), TABLE_ROLES, {}, (-2.0, -0.7))
        self.assertEqual(missing, ["trash"])
        # Not known as trash, the snack is one of the two nearest others.
        self.assertEqual(assigned["other2"].label, "cookies")


class ShelfTests(unittest.TestCase):
    def setUp(self):
        self.shelf = furniture_region(FURNITURE.get("shelf"), (0, 0, 0, 0), 0.1)
        self.heights = FURNITURE.get("shelf").heights
        # Seen from the open side (world -y), the robot's left is world -x.
        left, right = 3.0, 3.35
        self.placed = [
            self.item("apple", left, 2, (0.075,) * 3),
            self.item("orange", left + 0.08, 2, (0.07,) * 3),
            self.item("pringles", right, 2, (0.075, 0.075, 0.23)),
            self.item("cereal", left, 3, (0.06, 0.16, 0.26)),
            self.item("tuna can", right, 3, (0.085,) * 3),
            self.item("milk", left, 4, (0.07, 0.07, 0.19)),
        ]

    def item(self, label, x, level, size, yaw=None):
        """(percept, level) of an object at world x on the shelf level."""
        xyz = on_top(x, 3.5, self.heights[level - 1], size)
        return (replace(seen(label, xyz, size), yaw=yaw), level)

    def taken(self, placed, level):
        """Local y (low, high) of the objects on a level."""
        out = []
        for percept, number in placed:
            if number == level:
                y = self.shelf.local(*percept.position[:2])[1]
                half = half_width(self.shelf, percept)
                out.append((y - half, y + half))
        return out

    def assertClear(self, spot, width, taken):
        """The spot keeps PLACE_GAP from the side panels and the `taken` spans;
        returns its span."""
        y = self.shelf.local(*spot["position"][:2])[1]
        low, high = y - width / 2, y + width / 2
        edge = self.shelf.width / 2 - PLACE_GAP + 1e-3  # positions are rounded
        self.assertTrue(-edge <= low and high <= edge, spot)
        for a, b in taken:
            clear = high <= a - PLACE_GAP + 1e-3 or low >= b + PLACE_GAP - 1e-3
            self.assertTrue(clear, (spot, (a, b)))
        return (low, high)

    def test_groups_by_level_and_side(self):
        groups = shelf_groups(self.placed, self.shelf)
        self.assertEqual(
            [(g.level, g.side, g.category) for g in groups],
            [
                (2, "left", "fruit"),
                (2, "right", "snack"),
                (3, "left", "food"),
                (3, "right", "food"),
                (4, "left", "drink"),
            ],
        )
        self.assertEqual(groups[0].labels, ["apple", "orange"])
        self.assertAlmostEqual(groups[0].position[2], self.heights[1])
        self.assertEqual(
            categories_by_level(groups, len(self.heights)),
            [[], ["fruit", "snack"], ["food", "food"], ["drink"], []],
        )

    def test_placement_by_category_then_empty_parts(self):
        groups = shelf_groups(self.placed, self.shelf)
        where = placement(
            [
                ("table.other1", "orange", "fruit", 0.07),
                ("table.other2", "rubiks cube", "toy", 0.058),
                ("extra.obj1", "dice", "toy", 0.05),
            ],
            groups,
            self.shelf,
        )
        fruit = where["table.other1"]
        self.assertEqual(
            (fruit["level"], fruit["side"], fruit["reason"]), (2, "left", "category")
        )
        # Beside the fruit (no room at the panel), clear of the pringles.
        self.assertClear(fruit, 0.07, self.taken(self.placed, 2))
        toy = where["table.other2"]
        self.assertEqual(
            (toy["level"], toy["side"], toy["reason"]), (4, "right", "empty")
        )
        self.assertAlmostEqual(toy["position"][2], self.heights[3])
        span = self.assertClear(toy, 0.058, self.taken(self.placed, 4))
        # The second toy joins the first one, beside it.
        dice = where["extra.obj1"]
        self.assertEqual(
            (dice["level"], dice["side"], dice["reason"]), (4, "right", "category")
        )
        self.assertClear(dice, 0.05, self.taken(self.placed, 4) + [span])

    def test_the_spot_is_beside_the_group_not_on_it(self):
        # Level 5 of the simulated shelf: orange and lemon leave no room on the
        # left, a peach stands right of the middle.
        placed = [
            self.item(label, x, 5, size, yaw=0.0)
            for label, x, size in (
                ("orange", 3.0, (0.07, 0.07, 0.07)),
                ("lemon", 3.11, (0.055, 0.055, 0.06)),
                ("peach", 3.22, (0.07, 0.07, 0.065)),
            )
        ]
        groups = shelf_groups(placed, self.shelf)
        items = [("a", "apple", "fruit", 0.075), ("b", "orange", "fruit", 0.07)]
        where = placement(items, groups, self.shelf)
        taken = self.taken(placed, 5)
        for key, _, _, width in items:
            spot = where[key]
            self.assertEqual(
                (spot["level"], spot["side"], spot["reason"]), (5, "right", "category")
            )
            taken.append(self.assertClear(spot, width, taken))
        # Next to the peach (x 3.22 + 0.035), then next to the apple (rounded).
        x = 3.255 + PLACE_GAP + 0.0375
        self.assertAlmostEqual(where["a"]["position"][0], x, delta=1e-3)
        x += 0.0375 + PLACE_GAP + 0.035
        self.assertAlmostEqual(where["b"]["position"][0], x, delta=1e-3)

    def test_a_full_shelf_takes_the_least_occupied_part(self):
        placed = [
            self.item("apple", x, level, (0.075,) * 3)
            for level in range(1, 6)
            for x in (3.05, 3.35)
        ]
        groups = shelf_groups(placed, self.shelf)
        where = placement([("k", "cola", "drink", 0.066)], groups, self.shelf)
        self.assertEqual(where["k"]["reason"], "least occupied")
        self.assertIn(where["k"]["level"], (2, 3, 4))
        # Between the apples, not on one.
        self.assertClear(where["k"], 0.066, self.taken(placed, where["k"]["level"]))


def detection(label, xyz, track, size=(0.1, 0.1, 0.1), yaw=None):
    """A DetectedObject; no yaw gives the identity orientation of a camera detector."""
    half = 0.0 if yaw is None else yaw / 2
    return SimpleNamespace(
        class_name=label,
        track_id=track,
        confidence=1.0,
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)
                ),
            )
        ),
        size=SimpleNamespace(x=size[0], y=size[1], z=size[2]),
    )


def stamp(seconds):
    return SimpleNamespace(sec=int(seconds), nanosec=int(seconds % 1 * 1e9))


class FakeObjects(Objects):
    """Objects without ROS: every mark() delivers the scene's detections, of an
    image taken at that time (the fake image clock counts the marks)."""

    look_settle = 0.0
    dwell = 0.01
    look_timeout = 0.5
    now = staticmethod(time.monotonic)

    def __init__(self, scene, robot=(-2.0, -0.7, -math.pi / 2), frame="map"):
        self.furniture = FURNITURE
        self.object_list = OBJECT_LIST
        self.lock = threading.Lock()
        self.history = []
        self.sequence = 0
        self.scene, self.robot, self.frame = scene, robot, frame
        self.looked = []
        self.clock = 0.0

    def image_time(self):
        return self.clock

    def deliver(self, objects, taken):
        self._on_objects(
            SimpleNamespace(
                header=SimpleNamespace(frame_id=self.frame, stamp=stamp(taken)),
                objects=list(objects),
            )
        )

    def mark(self):
        self.clock += 1.0
        if self.scene is not None:
            self.deliver(self.scene, self.clock)
        return super().mark()

    def transform(self, target, source):
        if source == "base_footprint":
            return (self.robot[0], self.robot[1], 0.0, self.robot[2])
        return (0.0, 0.0, 0.0, 0.0)  # world == map

    def look_at(self, frame, xyz):
        self.looked.append((frame, tuple(round(v, 3) for v in xyz)))


def run(skill_class, objects, params, memory=None):
    spec = SimpleNamespace(subgoals=())
    runtime = SimpleNamespace(objects=objects, spec=spec, skills={})
    skill_class.validate(params, spec)
    skill = skill_class(runtime)
    logs = []
    ctx = SkillContext(
        "test", params, dict(memory or {}), threading.Event(), logs.append
    )
    return skill.execute(ctx), logs


def table_scene():
    return [
        detection("fork", (-2.2, -1.8, 0.81), 1, (0.02, 0.2, 0.02)),
        detection("mug", (-1.8, -1.8, 0.85), 2),
        detection("plate", (-2.0, -1.9, 0.81), 3, (0.26, 0.26, 0.02)),
        detection("cookies", (-2.3, -2.2, 0.875), 4, (0.06, 0.1, 0.15)),
        detection("apple", (-1.7, -2.4, 0.84), 5, (0.075,) * 3),
        detection("cola", (-2.0, -2.7, 0.86), 6, (0.066, 0.066, 0.12)),
        detection("orange", (-2.0, -1.0, 0.035), 7, (0.07,) * 3),  # on the floor
        detection("milk", (3.1, 3.45, 1.005), 8, (0.07, 0.07, 0.19)),  # elsewhere
    ]


class SkillTests(unittest.TestCase):
    TABLE_PARAMS = {
        "surface": "dining_table",
        "output": "table",
        "margin": 0.5,
        "exclude": [],
        "roles": TABLE_ROLES,
    }

    def test_perceive_the_dining_table(self):
        objects = FakeObjects(table_scene())
        result, logs = run(
            PerceiveObjects,
            objects,
            self.TABLE_PARAMS,
            {"ppc.trash_category": "snacks"},
        )
        self.assertTrue(result.success, result.message)
        outputs = result.outputs
        self.assertEqual(
            {k: v["label"] for k, v in outputs.items()},
            {
                "table.cutlery": "fork",
                "table.cup": "mug",
                "table.plate": "plate",
                "table.trash": "cookies",
                "table.other1": "apple",
                "table.other2": "cola",
            },
        )
        self.assertEqual(
            outputs["table.cup"],
            {
                "label": "mug",
                "category": "tableware",
                "position": [-1.8, -1.8, 0.85],
                "frame": "map",
                "size": [0.1, 0.1, 0.1],
                "track": 2,
                "confidence": 1.0,
                "yaw": 0.0,
            },
        )
        # Three head poses along the table, then back to its near end.
        self.assertEqual(len(objects.looked), 4)
        self.assertEqual(objects.looked[0], objects.looked[-1])
        self.assertIn("2 other objects seen", logs[0])

    def test_yaw_from_the_detection_orientation(self):
        scene = table_scene()
        scene[0] = detection(
            "fork", (-2.2, -1.8, 0.81), 1, (0.195, 0.027, 0.015), 1.5708
        )
        result, _ = run(
            PerceiveObjects,
            FakeObjects(scene),
            self.TABLE_PARAMS,
            {"ppc.trash_category": "snacks"},
        )
        self.assertEqual(result.outputs["table.cutlery"]["yaw"], 1.571)
        # Identity: the box of a camera detector is along the map axes.
        self.assertEqual(result.outputs["table.cup"]["yaw"], 0.0)

    def test_detections_of_images_before_settling_are_ignored(self):
        class Stale(FakeObjects):
            """A slow detector: late answers for images from before the first
            look and from the head's motion, with an extra object each."""

            def look_at(self, frame, xyz):
                if not self.looked:  # taken at the end of the move
                    self.deliver([detection("cookies", (-2.1, -1.7, 0.87), 11)], 0.5)
                super().look_at(frame, xyz)
                # Taken while the head turns, before it settles at this point.
                self.deliver(
                    [detection("orange", (-2.0, -1.5, 0.835), 12)], self.clock + 0.5
                )

        objects = Stale(table_scene())
        result, _ = run(
            PerceiveObjects,
            objects,
            self.TABLE_PARAMS,
            {"ppc.trash_category": "snacks"},
        )
        self.assertTrue(result.success, result.message)
        labels = {k: v["label"] for k, v in result.outputs.items()}
        self.assertNotIn("orange", labels.values())
        self.assertEqual(
            (labels["table.trash"], labels["table.other1"], labels["table.other2"]),
            ("cookies", "apple", "cola"),
        )
        self.assertNotEqual(result.outputs["table.trash"]["track"], 11)

    def test_missing_roles_get_no_key(self):
        scene = [d for d in table_scene() if d.class_name not in ("cookies", "cola")]
        result, _ = run(
            PerceiveObjects,
            FakeObjects(scene),
            self.TABLE_PARAMS,
            {"ppc.trash_category": "snacks"},
        )
        self.assertTrue(result.success)
        self.assertNotIn("table.trash", result.outputs)
        self.assertNotIn("table.other2", result.outputs)
        self.assertIn("missing trash, other2", result.message)

    def test_nothing_on_the_surface_fails(self):
        params = dict(self.TABLE_PARAMS, surface="long_table", margin=0.1)
        result, _ = run(PerceiveObjects, FakeObjects(table_scene()), params)
        self.assertFalse(result.success)
        self.assertIn("long_table", result.message)

    def test_no_detector_fails(self):
        result, _ = run(PerceiveObjects, FakeObjects(None), self.TABLE_PARAMS)
        self.assertFalse(result.success)
        self.assertIn("objects:=true", result.message)

    def test_unknown_surface_fails(self):
        params = dict(self.TABLE_PARAMS, surface="dishwasher")
        result, _ = run(PerceiveObjects, FakeObjects(table_scene()), params)
        self.assertFalse(result.success)
        self.assertIn("Unknown surface", result.message)

    def test_floor_trash_near_the_robot(self):
        scene = [
            detection("cookies", (4.1, -3.4, 0.075), 1, (0.06, 0.1, 0.15)),
            detection("pringles", (6.0, -3.4, 0.115), 2, (0.075, 0.075, 0.23)),
        ]
        objects = FakeObjects(scene, robot=(3.65, -3.1, -0.44))
        result, _ = run(
            PerceiveObjects,
            objects,
            self.FLOOR_PARAMS,
            {"ppc.trash_category": "snacks"},
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["trash.floor"]["label"], "cookies")
        self.assertTrue(all(z == 0.0 for _, (_, _, z) in objects.looked[:3]))
        # Then at the trash, not the last floor pan.
        self.assertEqual(objects.looked[3:], [("map", (4.1, -3.4, 0.075))])

    FLOOR_PARAMS = {
        "surface": "floor",
        "output": "trash",
        "margin": 1.5,
        "exclude": ["trash_box"],
        "roles": {"floor": {"category_key": "ppc.trash_category"}},
    }

    def test_trash_in_the_bin_is_not_floor_trash(self):
        # The table's trash, put in the bin (4.5, -3.5) just before, is nearer
        # the robot than the floor trash and lies near the floor.
        scene = [
            detection("pringles", (4.5, -3.45, 0.12), 1, (0.075, 0.075, 0.23)),
            detection("cookies", (4.9, -3.0, 0.075), 2, (0.06, 0.1, 0.15)),
        ]
        memory = {"ppc.trash_category": "snacks"}
        robot = (3.65, -3.1, -0.44)
        result, logs = run(
            PerceiveObjects, FakeObjects(scene, robot), self.FLOOR_PARAMS, memory
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs["trash.floor"]["label"], "cookies")
        self.assertTrue(any("in trash_box" in line for line in logs))
        # Without the exclusion the binned object would be taken.
        params = dict(self.FLOOR_PARAMS, exclude=[])
        result, _ = run(PerceiveObjects, FakeObjects(scene, robot), params, memory)
        self.assertEqual(result.outputs["trash.floor"]["label"], "pringles")

    def test_on_the_shelf_the_head_ends_at_the_first_role_level(self):
        heights = FURNITURE.get("shelf").heights
        cereal, milk = (0.06, 0.16, 0.26), (0.07, 0.07, 0.19)
        scene = [
            detection("cereal", on_top(3.05, 3.5, heights[2], cereal), 1, cereal),
            detection("milk", on_top(3.35, 3.5, heights[3], milk), 2, milk),
        ]
        shelf = furniture_region(FURNITURE.get("shelf"), (0.0, 0.0, 0.0, 0.0), 0.1)
        levels = [("map", tuple(round(v, 3) for v in p)) for p in shelf.look_points()]
        cases = (
            (["cereal", "milk"], scene, 3),
            (["milk", "cereal"], scene, 4),
            (["cereal", "milk"], scene[1:], 4),  # no cereal: the milk's level
        )
        for roles, detections, level in cases:
            with self.subTest(roles=roles, objects=len(detections)):
                objects = FakeObjects(detections, robot=(3.2, 2.62, math.pi / 2))
                params = {
                    "surface": "shelf",
                    "output": "breakfast",
                    "margin": 0.1,
                    "exclude": [],
                    "roles": {role: {"labels": [role]} for role in roles},
                }
                result, _ = run(PerceiveObjects, objects, params)
                self.assertTrue(result.success, result.message)
                # Each level from the lowest, then back to the first role's.
                self.assertEqual(objects.looked[:-1], levels)
                self.assertEqual(objects.looked[-1], levels[level - 1])

    def test_cancel_ends_the_scan(self):
        objects = FakeObjects(table_scene())
        objects.look_settle = 30.0
        spec = SimpleNamespace(subgoals=())
        skill = PerceiveObjects(SimpleNamespace(objects=objects, spec=spec, skills={}))
        cancelled = threading.Event()
        threading.Timer(0.1, cancelled.set).start()
        ctx = SkillContext("t", self.TABLE_PARAMS, {}, cancelled, lambda m: None)
        started = time.monotonic()
        result = skill.execute(ctx)
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Cancelled")
        self.assertLess(time.monotonic() - started, 5.0)

    def test_perceive_the_shelf_and_where_to_put_objects(self):
        heights = FURNITURE.get("shelf").heights
        scene = [
            detection(
                "apple", on_top(3.05, 3.5, heights[1], (0.075,) * 3), 1, (0.075,) * 3
            ),
            detection(
                "cereal",
                on_top(3.05, 3.5, heights[2], (0.06, 0.16, 0.26)),
                2,
                (0.06, 0.16, 0.26),
            ),
            detection(
                "milk",
                on_top(3.35, 3.5, heights[3], (0.07, 0.07, 0.19)),
                3,
                (0.07, 0.07, 0.19),
            ),
        ]
        memory = {
            "table.other1": {"label": "orange", "category": "fruit"},
            "table.other2": "cola",  # a label from an older mock
        }
        params = {
            "surface": "shelf",
            "objects": ["table.other1", "table.other2", "table.other3"],
            "output": "cabinet",
        }
        objects = FakeObjects(scene, robot=(3.2, 2.62, math.pi / 2))
        result, _ = run(PerceiveShelf, objects, params, memory)
        self.assertTrue(result.success, result.message)
        out = result.outputs
        self.assertEqual(
            out["cabinet.categories"], [[], ["fruit"], ["food"], ["drink"], []]
        )
        self.assertEqual(len(out["cabinet.groups"]), 3)
        self.assertEqual(out["cabinet.groups"][0]["frame"], "map")
        placement_ = out["cabinet.placement"]
        self.assertEqual(set(placement_), {"table.other1", "table.other2"})
        self.assertEqual(placement_["table.other1"]["level"], 2)
        # Beside the apple (x 3.05, 0.075 m), not on it; no size in the record.
        x = 3.05 + 0.0375 + PLACE_GAP + PLACE_WIDTH / 2
        self.assertAlmostEqual(placement_["table.other1"]["position"][0], x, delta=1e-3)
        self.assertEqual(
            (placement_["table.other2"]["level"], placement_["table.other2"]["side"]),
            (4, "right"),
        )
        self.assertEqual(placement_["table.other2"]["frame"], "map")
        # One head pose per shelf level.
        self.assertEqual(len(objects.looked), len(heights))


class ValidationTests(unittest.TestCase):
    def test_bad_params_are_refused(self):
        good = SkillTests.TABLE_PARAMS
        bad = [
            {k: v for k, v in good.items() if k != "margin"},
            dict(good, margin=-1),
            dict(good, roles={}),
            dict(good, roles={"cup": {"labels": []}}),
            dict(good, roles={"cup": {"labels": ["mug"], "category": "tableware"}}),
            dict(good, roles={"cup": {"colour": "red"}}),
            dict(good, roles={"a b": {}}),
            dict(good, surface=""),
            {k: v for k, v in good.items() if k != "exclude"},
            dict(good, exclude="trash_box"),
            dict(good, exclude=[""]),
            dict(good, exclude=["floor"]),
        ]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                PerceiveObjects.validate(params, None)
        with self.assertRaises(ValueError):
            PerceiveShelf.validate(
                {"surface": "floor", "objects": [], "output": "c"}, None
            )
        with self.assertRaises(ValueError):
            PerceiveShelf.validate(
                {"surface": "shelf", "objects": "x", "output": "c"}, None
            )

    def test_ready_checks_the_surfaces(self):
        subgoal = SimpleNamespace(
            id="s", skill="perceive_objects", params={"surface": "sofa"}
        )
        runtime = SimpleNamespace(
            objects=FakeObjects([]), spec=SimpleNamespace(subgoals=(subgoal,))
        )
        skill = PerceiveObjects(runtime)
        runtime.skills = {"perceive_objects": skill}
        with self.assertRaises(ValueError):
            skill.ready()
        subgoal.params["surface"] = "floor"
        skill.ready()
        subgoal.params["exclude"] = ["dustbin"]
        with self.assertRaises(ValueError):
            skill.ready()
        subgoal.params["exclude"] = ["trash_box"]
        skill.ready()


if __name__ == "__main__":
    unittest.main()
