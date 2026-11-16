"""GPSR object steps: names, superlatives, room order and the skills on fake
detections of the GPSR object scene (no ROS graph)."""

import json
import math
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core import parse_spec  # noqa: E402
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.object_queries import (  # noqa: E402
    CountObjects,
    DescribeObject,
    FindObject,
    count_answer,
    forms,
    is_person,
    parse_superlative,
    places_of,
    plural,
    rank,
    resolve_target,
    singular,
    visit_order,
)
from aha_tasks.objects import Cancelled, Objects, Percept  # noqa: E402
from aha_tasks.plan import validate_plan  # noqa: E402
from aha_tasks.runner import load_task  # noqa: E402
from aha_perception.furniture import load_config  # noqa: E402
from aha_perception.object_list import load_object_list  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
PERCEPTION = PACKAGE.parent / "aha_perception" / "config"
FURNITURE = load_config(PERCEPTION / "furniture" / "rcjo2026_arena.yaml")
OBJECT_LIST = load_object_list(PERCEPTION / "objects" / "robocup_objects.yaml")
GPSR = PACKAGE / "config" / "gpsr.json"
SCENE = PACKAGE / "config" / "sim" / "objects" / "gpsr.json"
SPEC, _ = load_task(GPSR)


def labels_of(text):
    target = resolve_target(text, OBJECT_LIST)
    if target.kind != "objects":
        return target.kind
    return {o.label for o in OBJECT_LIST.objects if target.matches(o.label, o.category)}


class NameTests(unittest.TestCase):
    def test_generator_object_names_find_the_labels(self):
        # RoboCup Japan Open 2026 GPSR objects and YCB names.
        for text, label in [
            ("tomato soup can", "tomato soup"),
            ("Tuna", "tuna can"),
            ("red bull can", "red bull"),
            ("toothpaste box", "toothpaste"),
            ("jell-o strawberry", "strawberry jello"),
            ("Jell-O pudding box", "chocolate jello"),
            ("rubik's cube", "rubiks cube"),
            ("rubiks_cube", "rubiks cube"),
            ("a can of cola", "cola"),
            ("cokes", "cola"),
            ("the cookie", "cookies"),
            ("strawberries", "strawberry"),
            ("peaches", "peach"),
            ("knives", "knife"),
            ("cup noodles", "cup noodle"),
            ("dishwasher tab", "dishwasher tab"),
        ]:
            with self.subTest(text):
                self.assertEqual(labels_of(text), {label})

    def test_categories_singular_and_plural(self):
        drinks = {"green tea", "lemonade", "cola", "milk", "red bull"}
        self.assertEqual(labels_of("drinks"), drinks)
        self.assertEqual(labels_of("drink"), drinks)
        self.assertEqual(labels_of("Drinks"), drinks)
        self.assertIn("cleanser", labels_of("cleaning supply"))
        self.assertEqual(labels_of("foods"), labels_of("food"))
        self.assertIn("pringles", labels_of("snacks"))

    def test_groups_and_generic_words(self):
        kitchen = labels_of("kitchen items")
        self.assertTrue({"fork", "mug", "sponge", "dishwasher tab"} <= kitchen)
        self.assertNotIn("toothpaste", kitchen)
        self.assertEqual(labels_of("dishes"), labels_of("dish"))
        for text in ("objects", "object", "items", "the thing", "anything"):
            self.assertEqual(labels_of(text), "any", text)
        self.assertEqual(labels_of("fanta"), "unknown")
        self.assertEqual(labels_of("can"), "unknown")

    def test_forms_ignore_order_and_containers(self):
        self.assertTrue(forms("strawberry jello") & forms("jello strawberry"))
        self.assertTrue(forms("cereal box") & forms("cereal"))
        self.assertFalse(forms("strawberry") & forms("strawberry jello"))

    def test_people(self):
        for text in (
            "people",
            "persons raising their left arm",
            "name of the person",
            "pose",
            "woman",
        ):
            self.assertTrue(is_person(text), text)
        for text in ("drinks", "biggest object", "kitchen items"):
            self.assertFalse(is_person(text), text)

    def test_singular_and_plural_for_answers(self):
        for one, many in [
            ("drink", "drinks"),
            ("cleaning supply", "cleaning supplies"),
            ("kitchen item", "kitchen items"),
            ("dish", "dishes"),
            ("strawberry", "strawberries"),
            ("cookie", "cookies"),
        ]:
            with self.subTest(one):
                self.assertEqual(singular(many), one)
                self.assertEqual(plural(one), many)
                self.assertEqual(plural(many), many)

    def test_count_answers(self):
        place = "on the desk"
        self.assertEqual(
            count_answer("drinks", ["cola", "milk"], place),
            "There are 2 drinks on the desk: cola and milk.",
        )
        self.assertEqual(
            count_answer("drinks", ["cola"], place),
            "There is one drink on the desk: the cola.",
        )
        self.assertEqual(
            count_answer("drinks", [], place), "There are no drinks on the desk."
        )
        self.assertEqual(
            count_answer("fruits", ["apple"] * 6, place),
            "There are 6 fruits on the desk.",
        )


class SuperlativeTests(unittest.TestCase):
    def test_generator_and_rephrased_targets(self):
        for text, expected in [
            ("biggest object", ("biggest", "object")),
            ("the heaviest drink", ("heaviest", "drink")),
            ("largest kitchen item", ("largest", "kitchen item")),
            (
                "what is the smallest item on the side tables",
                ("smallest", "item"),
            ),
            ("tiniest thing", ("smallest", "thing")),
            ("thing that weighs the most", ("heaviest", "thing")),
            ("most slender snack", ("thinnest", "snack")),
            ("slimmest fruit", ("thinnest", "fruit")),
            ("lightest", ("lightest", "object")),
            ("the heaviest of the drinks", ("heaviest", "drinks")),
            ("biggest can of cola", ("biggest", "can of cola")),
        ]:
            with self.subTest(text):
                self.assertEqual(parse_superlative(text), expected)

    def test_without_one_superlative(self):
        for text in ("colour of the cup", "biggest and lightest object", "drinks"):
            with self.subTest(text), self.assertRaises(ValueError):
                parse_superlative(text)

    def test_ranking(self):
        def seen(label, size, count=3):
            return (Percept(label, "", (0, 0, 0), size, 1, 1.0, count),)

        weights = {o.label: o.weight for o in OBJECT_LIST.objects}
        desk = [
            seen("green tea", (0.073, 0.073, 0.27)),
            seen("lemonade", (0.067, 0.067, 0.22)),
            seen("red bull", (0.053, 0.054, 0.135)),
            seen("rubiks cube", (0.058, 0.059, 0.058)),
            seen("plate", (0.26, 0.26, 0.027)),
        ]

        def best(name):
            return rank(desk, name, weights.get)[0][0].label

        self.assertEqual(best("biggest"), "plate")
        self.assertEqual(best("smallest"), "rubiks cube")
        self.assertEqual(best("heaviest"), "green tea")
        self.assertEqual(best("lightest"), "red bull")
        self.assertEqual(best("thinnest"), "plate")
        # Without a weight an object is left out.
        self.assertEqual(rank(desk, "heaviest", lambda label: None), [])
        # Ties go to the object seen more often.
        twins = [seen("cola", (0.066,) * 3, 2), seen("lemonade", (0.066,) * 3, 5)]
        self.assertEqual(rank(twins, "biggest", weights.get)[0][0].label, "lemonade")


class PlaceTests(unittest.TestCase):
    def test_rooms_of_the_gpsr_task(self):
        self.assertEqual(
            places_of(SPEC, "living_room"),
            [("desk", "long_table"), ("side_tables", "tall_table"), ("trashbin", None)],
        )
        self.assertEqual(places_of(SPEC, "desk"), [("desk", "long_table")])
        self.assertEqual(places_of(SPEC, "instruction_point"), [])
        for places in SPEC.rooms.values():
            for furniture in places.values():
                if furniture is not None:
                    self.assertIsNotNone(FURNITURE.get(furniture), furniture)

    def test_visit_order_is_the_shortest_path(self):
        points = {"a": (1.0, 0.0), "b": (2.0, 0.0), "c": (-0.5, 0.0)}
        # Going to c first (0.5 + 2.5) beats a, b, c (1 + 1 + 2.5).
        self.assertEqual(visit_order((0.0, 0.0), points), ["c", "a", "b"])
        self.assertEqual(visit_order((3.0, 0.0), points), ["b", "a", "c"])
        self.assertEqual(visit_order((0.0, 0.0), {}), [])
        many = {str(i): (float(i), 0.0) for i in range(9)}
        self.assertEqual(
            visit_order((9.0, 0.0), many), [str(i) for i in range(8, -1, -1)]
        )

    def test_rooms_in_the_task_definition(self):
        base = {
            "task": "t",
            "locations": {k: [0.0, 0.0, 0.0] for k in ("room", "table", "bed")},
            "skills": {"navigate": "aha_tasks.skills:Navigate"},
            "subgoals": [
                {"id": "a", "skill": "navigate", "params": {"location": "room"}}
            ],
        }
        spec = parse_spec(dict(base, rooms={"room": {"table": "t1", "bed": None}}))
        self.assertEqual(spec.rooms["room"], {"table": "t1", "bed": None})
        self.assertEqual(parse_spec(base).rooms, {})
        for rooms in (
            {"hall": {"table": "t1"}},
            {"room": {"chair": "t1"}},
            {"room": {"table": ""}},
            {"room": ["table"]},
            {"room": {"room": "t1"}},
            {"room": {"table": "t1"}, "bed": {"table": "t1"}},
        ):
            with self.subTest(rooms), self.assertRaises(ValueError):
                parse_spec(dict(base, rooms=rooms))


# -- skills on fake detections ----------------------------------------------


def scene_detections():
    """Detections of the GPSR object scene as the ground-truth detector gives
    them: centre, object-list size and yaw, one track per object."""
    data = json.loads(SCENE.read_text())
    out = []
    for track, item in enumerate(data["objects"], start=1):
        entry = OBJECT_LIST.get(item["label"])
        furniture = FURNITURE.get(item["on"])
        height = furniture.heights[item.get("level", 1) - 1]
        x, y = item["position"]
        out.append(
            detection(
                item["label"],
                (x, y, height + entry.size[2] / 2),
                track,
                entry.size,
                item.get("yaw", 0.0),
            )
        )
    return out


def detection(label, xyz, track, size, yaw=0.0):
    return SimpleNamespace(
        class_name=label,
        track_id=track,
        confidence=1.0,
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
                ),
            )
        ),
        size=SimpleNamespace(x=size[0], y=size[1], z=size[2]),
    )


class FakeObjects(Objects):
    """Every mark() delivers the scene's detections (a camera that sees all)."""

    def __init__(self, scene):
        self.furniture = FURNITURE
        self.object_list = OBJECT_LIST
        self.lock = threading.Lock()
        self.history = []
        self.sequence = 0
        self.scene = scene
        self.robot = (-2.0, 1.5, 0.0)
        self.looked = []
        self.clock = 0.0

    def image_time(self):
        return self.clock

    def mark(self):
        self.clock += 1.0
        stamp = SimpleNamespace(sec=int(self.clock), nanosec=0)
        self._on_objects(
            SimpleNamespace(
                header=SimpleNamespace(frame_id="map", stamp=stamp),
                objects=list(self.scene),
            )
        )
        return super().mark()

    def pause(self, seconds, cancelled, until=None):
        """No waiting: a new image comes at once."""
        if cancelled.is_set():
            raise Cancelled("Cancelled")
        self.mark()
        return True

    def transform(self, target, source):
        if source == "base_footprint":
            return (self.robot[0], self.robot[1], 0.0, self.robot[2])
        return (0.0, 0.0, 0.0, 0.0)

    def look_at(self, frame, xyz):
        self.looked.append(tuple(round(v, 3) for v in xyz))


class FakeNavigation:
    def __init__(self, objects):
        self.objects = objects
        self.goals = []

    def navigate(self, pose, cancelled=None):
        self.goals.append(pose)
        self.objects.robot = tuple(pose)
        return True, ""


class FakeInteraction:
    def __init__(self):
        self.said = []

    def say(self, text, cancelled):
        self.said.append(text)
        return True, "said"


def runtime_for(scene):
    objects = FakeObjects(scene)
    return SimpleNamespace(
        spec=SPEC,
        objects=objects,
        navigation=FakeNavigation(objects),
        interaction=FakeInteraction(),
        skills={},
    )


def run(skill_class, params, location="instruction_point", scene=None):
    runtime = runtime_for(scene_detections() if scene is None else scene)
    skill_class.validate(params, SPEC)
    skill = skill_class(runtime)
    skill.ready()
    why = skill.unsupported(params)
    if why:
        return why, runtime
    logs = []
    ctx = SkillContext(
        "test",
        params,
        {"robot.location": location},
        threading.Event(),
        logs.append,
    )
    result = skill.execute(ctx)
    result.logs = logs
    return result, runtime


def goals(runtime):
    poses = {tuple(v): k for k, v in SPEC.locations.items()}
    return [poses[tuple(g)] for g in runtime.navigation.goals]


class CountTests(unittest.TestCase):
    def test_drinks_on_the_desk(self):
        params = {"target": "drinks", "location": "desk", "output": "cmd2.count"}
        result, runtime = run(CountObjects, params)
        self.assertTrue(result.success, result.message)
        answer = "There are 3 drinks on the desk: green tea, lemonade and red bull."
        self.assertEqual(
            result.outputs,
            {
                "cmd2.count": 3,
                "cmd2.count.labels": ["green tea", "lemonade", "red bull"],
                "cmd2.count.answer": answer,
                "robot.location": "desk",
            },
        )
        self.assertEqual(runtime.interaction.said, [answer])
        self.assertEqual(goals(runtime), ["desk"])

    def test_no_move_when_there_and_other_targets(self):
        cases = [
            ("fruits", "kitchen_table", 2),
            ("foods", "shelf", 3),
            ("drinks", "shelf", 2),
            ("objects", "shelf", 8),
            ("snacks", "desk", 0),
            ("cola", "kitchen", 1),
            ("drinks", "office", 2),
        ]
        for target, location, number in cases:
            with self.subTest(target=target, location=location):
                params = {"target": target, "location": location, "output": "n"}
                result, runtime = run(CountObjects, params, location)
                self.assertTrue(result.success, result.message)
                self.assertEqual(result.outputs["n"], number)
        result, runtime = run(
            CountObjects,
            {"target": "drinks", "location": "desk", "output": "n"},
            "desk",
        )
        self.assertEqual(runtime.navigation.goals, [])
        self.assertEqual(result.outputs["robot.location"], "desk")

    def test_unknown_names_fail_and_people_are_not_counted_here(self):
        result, _ = run(
            CountObjects, {"target": "fantas", "location": "desk", "output": "n"}
        )
        self.assertFalse(result.success)
        self.assertIn("not in the object list", result.message)
        why, _ = run(
            CountObjects, {"target": "people", "location": "desk", "output": "n"}
        )
        self.assertIn("person skills", why)


class DescribeTests(unittest.TestCase):
    def describe(self, target, location):
        params = {"target": target, "location": location, "output": "answer"}
        result, runtime = run(DescribeObject, params)
        self.assertTrue(result.success, result.message)
        return result, runtime

    def test_superlatives_on_the_gpsr_scene(self):
        for target, location, label in [
            ("biggest object", "desk", "green tea"),
            ("smallest object", "desk", "rubiks cube"),
            ("heaviest drink", "desk", "green tea"),
            ("lightest object", "desk", "red bull"),
            ("thinnest drink", "desk", "red bull"),
            ("largest object", "kitchen_table", "bowl"),
            ("lightest fruit", "kitchen_table", "banana"),
            ("smallest fruit", "shelf", "plum"),
            ("heaviest food", "shelf", "tomato soup"),
            ("thinnest object", "side_tables", "strawberry jello"),
            ("heaviest object", "living_room", "green tea"),
        ]:
            with self.subTest(target=target, location=location):
                result, _ = self.describe(target, location)
                self.assertEqual(result.outputs["answer"], label)

    def test_outputs_and_answer(self):
        result, runtime = self.describe("the heaviest drink", "desk")
        answer = "The heaviest drink on the desk is the green tea."
        self.assertEqual(result.outputs["answer.answer"], answer)
        self.assertEqual(runtime.interaction.said, [answer])
        record = result.outputs["answer.object"]
        self.assertEqual((record["label"], record["surface"]), ("green tea", "desk"))
        self.assertEqual(result.outputs["robot.location"], "desk")
        # The head ends on the answer.
        self.assertEqual(runtime.objects.looked[-1][:2], (1.72, -0.5))

    def test_people_and_bad_targets(self):
        params = {"target": "name of the person", "location": "bed", "output": "a"}
        why, _ = run(DescribeObject, params)
        self.assertIn("person skills", why)
        with self.assertRaises(ValueError):
            DescribeObject.validate(dict(params, target="colour of the cup"), SPEC)
        result, _ = run(
            DescribeObject,
            {"target": "biggest snack", "location": "desk", "output": "a"},
        )
        self.assertFalse(result.success)
        self.assertIn("No snack on the desk", result.message)


class FindTests(unittest.TestCase):
    def find(self, params, location="instruction_point", scene=None):
        return run(FindObject, params, location, scene)

    def test_find_in_a_room_stops_where_it_is(self):
        result, runtime = self.find(
            {"object": "rubik's cube", "location": "living_room"}
        )
        self.assertTrue(result.success, result.message)
        found = result.outputs["object"]
        self.assertEqual(
            (found["label"], found["surface"], found["level"]),
            ("rubiks cube", "desk", 1),
        )
        self.assertEqual(found["frame"], "map")
        self.assertEqual(result.outputs["robot.location"], "desk")
        # From the instruction point the side tables are nearer: looked there first.
        self.assertEqual(goals(runtime), ["side_tables", "desk"])
        self.assertEqual(
            runtime.interaction.said, ["I found the rubiks cube on the desk."]
        )

    def test_the_robot_stays_at_the_furniture_where_it_found_it(self):
        # Not the room: the next step must not drive to the room's location.
        result, runtime = self.find(
            {"object": "cola", "location": "kitchen"}, "kitchen_table"
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.navigation.goals, [])
        self.assertEqual(result.outputs["robot.location"], "kitchen_table")

    def test_find_a_category_with_an_output_key(self):
        params = {"object": "snack", "location": "living_room", "output": "cmd1.obj"}
        result, runtime = self.find(params)
        self.assertTrue(result.success, result.message)
        self.assertIn(
            result.outputs["cmd1.obj"]["label"], {"cookies", "strawberry jello"}
        )
        self.assertEqual(goals(runtime), ["side_tables"])
        result, _ = self.find({"object": "drink", "location": "office"})
        self.assertEqual(
            (result.outputs["object"]["surface"], result.outputs["object"]["level"]),
            ("shelf", 4),
        )

    def test_not_found(self):
        result, runtime = self.find({"object": "apple", "location": "living_room"})
        self.assertFalse(result.success)
        self.assertEqual(goals(runtime), ["side_tables", "desk"])
        self.assertEqual(
            runtime.interaction.said, ["I could not find any apple in the living room."]
        )
        result, _ = self.find({"object": "fanta", "location": "kitchen"})
        self.assertFalse(result.success)

    def test_places_without_a_top_are_unsupported(self):
        why, _ = self.find({"object": "apple", "location": "bedroom"})
        self.assertIn("no furniture top", why)
        why, _ = self.find({"object": "apple", "location": "sofa"})
        self.assertIn("no furniture top", why)

    def test_validation(self):
        good = {"object": "cola", "location": "kitchen"}
        FindObject.validate(good, SPEC)
        for params in (
            {"object": "cola"},
            dict(good, location="instruction_point"),
            dict(good, location="garage"),
            dict(good, object=""),
            dict(good, output=""),
            dict(good, margin=-1),
            dict(good, colour="red"),
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                FindObject.validate(params, SPEC)
        with self.assertRaises(ValueError):
            CountObjects.validate({"target": "drinks", "location": "desk"}, SPEC)

    def test_plan_steps_validate(self):
        plan = [
            {
                "command": 1,
                "action": "find_object",
                "params": {"object": "cola", "location": "kitchen"},
            },
            {
                "command": 2,
                "action": "describe",
                "params": {
                    "target": "biggest object",
                    "location": "desk",
                    "output": "a",
                },
            },
            {
                "command": 3,
                "action": "describe",
                "params": {
                    "target": "pose of the person",
                    "location": "bed",
                    "output": "b",
                },
            },
        ]
        validate_plan(plan, SPEC)


if __name__ == "__main__":
    unittest.main()
