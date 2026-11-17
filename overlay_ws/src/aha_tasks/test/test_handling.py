"""pick_object, place_object and deliver_object on fake detections, servers,
navigation and screen (no ROS graph), and their geometry."""

import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

TEST = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TEST))
sys.path.insert(0, TEST)
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.handling import (  # noqa: E402
    HANDOVER_ACTION,
    PICK_ACTION,
    Unstopped,
    HELD,
    HOLDING,
    DeliverObject,
    Hands,
    PickObject,
    PlaceObject,
    choose_object,
    free_spot,
    furniture_ahead,
    in_reach,
    matches_text,
    object_target,
    reach_pose,
    spot_from,
)
from aha_tasks.objects import Cancelled, Percept, furniture_region  # noqa: E402
from aha_tasks.runner import guard_navigation, load_task  # noqa: E402
from aha_tasks.skills import ARM_OUT, LOCATION, go_to  # noqa: E402
from aha_perception.object_list import load_object_list  # noqa: E402
from test_object_skills import FURNITURE, FakeObjects, detection  # noqa: E402

PACKAGE = Path(TEST).parent
OBJECT_LIST = load_object_list(
    PACKAGE.parent / "aha_perception" / "config" / "objects" / "robocup_objects.yaml"
)
SPEC, _ = load_task(PACKAGE / "config" / "gpsr.json")
COLA = (0.066, 0.066, 0.12)


def pose_of(name):
    return tuple(SPEC.locations[name])


def region(name, margin=0.0):
    return furniture_region(FURNITURE.get(name), (0.0, 0.0, 0.0, 0.0), margin)


def percept(label, xyz, size=(0.1, 0.1, 0.1), count=3, confidence=1.0):
    return Percept(
        label=label,
        category=OBJECT_LIST.category(label),
        position=xyz,
        size=size,
        track=1,
        confidence=confidence,
        count=count,
    )


def kitchen_scene():
    return [
        detection("cola", (-1.8, -1.55, 0.86), 1, COLA),
        detection("apple", (-2.02, -1.55, 0.8375), 2, (0.075,) * 3),
        detection("bowl", (-1.98, -1.8, 0.8275), 3, (0.16, 0.16, 0.055)),
    ]


def desk_scene():
    return [
        detection("green tea", (1.72, -0.5, 0.5), 1, (0.065, 0.065, 0.2)),
        detection("lemonade", (1.88, -0.5, 0.46), 2, (0.066, 0.066, 0.12)),
        detection("red bull", (2.03, -0.5, 0.4675), 3, (0.053, 0.053, 0.135)),
        detection("rubiks cube", (2.22, -0.48, 0.429), 4, (0.058, 0.059, 0.058)),
    ]


class Result(SimpleNamespace):
    STATUS_RELEASED, STATUS_TIMEOUT, STATUS_UNREACHABLE = 0, 1, 2
    STATUS_ERROR, STATUS_CANCELED = 3, 4


def ok(**kw):
    return Result(success=True, status=0, message="done", **kw)


class PlaceResult(SimpleNamespace):
    STATUS_OK, STATUS_NO_SURFACE, STATUS_NO_SPACE, STATUS_UNREACHABLE = 0, 1, 2, 3
    STATUS_EXECUTION_FAILED, STATUS_ERROR, STATUS_CANCELED = 4, 5, 6
    STATUS_NOT_AT_SPOT = 9


class PickResult(SimpleNamespace):
    """PickObject.Result: no status."""


def place_failed(status, message):
    return PlaceResult(success=False, status=status, message=message)


def failed(status=4, message="no grasp"):
    return Result(success=False, status=status, message=message)


def goal(**kw):
    return SimpleNamespace(object_size=SimpleNamespace(), **kw)


class FakeHands:
    """The pick, place and handover servers: results in order, by action name.
    A None result never finishes (until the caller gives up)."""

    PickObject = PlaceObject = HandOver = SimpleNamespace(Goal=goal)

    def __init__(self, on_cancel=None, **results):
        self.results = {name: list(r) for name, r in results.items()}
        self.goals = []  # (action, goal)
        self.cancelled = 0
        # The result of a cancelled goal that never finished by itself.
        self.on_cancel = on_cancel or failed(4, "cancelled")

    def point(self, frame, xyz):
        return (frame, tuple(round(v, 3) for v in xyz))

    def target(self, entry):
        return entry

    def start(self, name, request, cancelled):
        kind = name.rsplit("/", 1)[-1]
        self.goals.append((kind, SimpleNamespace(**vars(request))))
        return SimpleNamespace(result=self.results[kind].pop(0))

    def finish(self, handle, timeout, cancelled, until=None):
        deadline = time.monotonic() + 2.0
        while handle.result is None:
            if until is not None and until():
                return None
            assert time.monotonic() < deadline, "the goal never finished"
            time.sleep(0.01)
        return handle.result

    def cancel(self, handle):
        self.cancelled += 1
        return handle.result or self.on_cancel

    def run(self, name, request, timeout, cancelled):
        handle = self.start(name, request, cancelled)
        if isinstance(handle.result, Exception):  # e.g. Unstopped
            raise handle.result
        return self.finish(handle, timeout, cancelled), ""


class FakeNavigation:
    """Stops `off` m off the goal in y (Nav2 stops within its tolerance)."""

    def __init__(self, objects, fail=False, off=0.0):
        self.objects, self.fail, self.off, self.goals = objects, fail, off, []

    def navigate(self, pose, cancelled=None):
        self.goals.append([round(v, 3) for v in pose])
        if self.fail:
            return False, "blocked"
        self.objects.robot = (pose[0], pose[1] + self.off, pose[2])
        return True, ""


class FakeInteraction:
    def __init__(self, touch=False):
        self.said, self.asked, self.touch = [], [], touch

    def say(self, text, cancelled):
        self.said.append(text)
        return True, "said"

    def ask(self, prompt, choices, cancelled, timeout=0.0):
        self.asked.append(prompt)
        if self.touch:
            return 0, choices[0]
        cancelled.wait(5.0)
        return None, "Cancelled"


class FakePeople:
    def __init__(self, people=()):
        self.people = people

    def wait_for(self, choose, timeout, cancelled):
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            people=[
                SimpleNamespace(visible=True, position=SimpleNamespace(x=x, y=y, z=z))
                for x, y, z in self.people
            ],
        )
        found = choose(message)
        return (found, "") if found is not None else (None, "nobody")


class FakeGaze:
    def __init__(self):
        self.looked = []

    def look_at(self, frame, point):
        self.looked.append((frame, (point.x, point.y, point.z)))

    def follow_path(self, pose):
        pass

    def hold(self):
        pass


def make_runtime(scene=None, robot="kitchen_table", hands=None, **kw):
    objects = FakeObjects(scene, robot=pose_of(robot))
    objects.object_list = OBJECT_LIST
    runtime = SimpleNamespace(
        node=None,
        spec=SPEC,
        objects=objects,
        hands=hands or FakeHands(),
        interaction=kw.get("interaction") or FakeInteraction(),
        people=kw.get("people") or FakePeople(),
        gaze=FakeGaze(),
        skills={},
    )
    runtime.navigation = FakeNavigation(
        objects, kw.get("blocked", False), kw.get("off", 0.0)
    )
    return runtime


def run(skill_class, runtime, params, memory=None):
    skill_class.validate(params, SPEC)
    logs = []
    ctx = SkillContext(
        "test", params, dict(memory or {}), threading.Event(), logs.append
    )
    return skill_class(runtime).execute(ctx), logs


class ObjectWordsTests(unittest.TestCase):
    def test_labels_plurals_categories_and_generic_words(self):
        self.assertTrue(matches_text("the Cola", "cola", "drink"))
        self.assertTrue(matches_text("colas", "cola", "drink"))
        self.assertTrue(matches_text("peaches", "peach", "fruit"))
        self.assertTrue(matches_text("a drink", "cola", "drink"))
        self.assertTrue(matches_text("drinks", "cola", "drink"))
        self.assertTrue(matches_text("object", "cola", "drink"))
        self.assertTrue(matches_text("red_bull", "red bull", "drink"))
        self.assertFalse(matches_text("apple", "cola", "drink"))
        self.assertFalse(matches_text("fruit", "cola", "drink"))

    def test_a_memory_key_gives_its_record(self):
        record = {"label": "cola", "position": [0.0, 0.0, 0.8]}
        self.assertEqual(object_target("object", {"object": record}), (record, "cola"))
        self.assertEqual(object_target("The Apple", {}), (None, "apple"))
        self.assertEqual(object_target("object", {"object": "cola"}), (None, "object"))

    def test_the_remembered_object_wins_then_the_best_seen(self):
        far = percept("cola", (1.0, 0.0, 0.8), count=9)
        near = percept("cola", (2.0, 0.0, 0.8), count=2)
        apple = percept("apple", (2.0, 0.1, 0.8), count=9)
        remembered = {"label": "cola", "position": [2.1, 0.0, 0.8]}
        self.assertIs(
            choose_object([far, near, apple], "cola", remembered, (0, 0)), near
        )
        self.assertIs(choose_object([far, near, apple], "cola", None, (2, 0)), far)
        moved = {"label": "cola", "position": [5.0, 0.0, 0.8]}
        self.assertIs(choose_object([far, near], "cola", moved, (0, 0)), far)
        self.assertIsNone(choose_object([apple], "milk", None, (0, 0)))
        self.assertIs(choose_object([apple, near], "fruit", None, (0, 0)), apple)


class GeometryTests(unittest.TestCase):
    def test_gpsr_locations_stand_for_the_furniture_ahead(self):
        regions = [region(name) for name in FURNITURE.names]
        expected = {
            "kitchen_table": "dining_table",
            "desk": "long_table",
            "side_tables": "tall_table",
            "trashbin": "trash_box",
            "shelf": "shelf",
            "bedroom": None,
            "instruction_point": None,
        }
        for location, furniture in expected.items():
            with self.subTest(location):
                found = furniture_ahead(pose_of(location), regions)
                self.assertEqual(found and found.name, furniture)

    def test_the_base_faces_the_table_with_the_object_at_the_right_arm(self):
        robot = pose_of("kitchen_table")
        pose = reach_pose(robot, (-1.8, -1.55), region("dining_table"))
        self.assertEqual([round(v, 3) for v in pose], [-1.6, -0.7, -1.571])
        # The base keeps STANDOFF from the edge at y -1.35.
        self.assertAlmostEqual(pose[1] + 1.35, 0.65)
        self.assertTrue(in_reach(pose, (-1.8, -1.55), reach=0.9))
        self.assertFalse(in_reach(robot, (-1.8, -1.55)))
        left = reach_pose(robot, (-1.8, -1.55), region("dining_table"), arm="left")
        self.assertAlmostEqual(left[0], -2.0)
        floor = reach_pose((0.0, 0.0, 0.0), (2.0, 0.0))
        self.assertEqual([round(v, 3) for v in floor], [1.55, 0.2, 0.0])

    def test_objects_behind_are_out_of_reach(self):
        self.assertFalse(in_reach((0.0, 0.0, 0.0), (-0.3, 0.0)))
        self.assertTrue(in_reach((0.0, 0.0, 0.0), (0.5, -0.2)))

    def test_free_spot_keeps_clear_of_objects_and_edges(self):
        table = region("long_table")
        objects = [
            percept("rubiks cube", (2.22, -0.48, 0.429), (0.058, 0.059, 0.058)),
            percept("red bull", (2.03, -0.5, 0.4675), (0.053, 0.053, 0.135)),
        ]
        spot = free_spot(table, 1, objects, COLA, (2.2, -0.92))
        self.assertEqual(spot[2], 0.40)
        radius = math.hypot(*COLA[:2]) / 2
        for other in objects:
            gap = math.dist(spot[:2], other.position[:2])
            self.assertGreaterEqual(
                gap, radius + math.hypot(*other.size[:2]) / 2 + 0.05
            )
        self.assertLessEqual(abs(spot[1] + 0.5), 0.2 - radius - 0.03 + 1e-9)
        self.assertAlmostEqual(spot[0], 2.2)
        # A table too small for the object has no spot.
        self.assertIsNone(free_spot(table, 1, [], (0.5, 0.5, 0.1), (2.0, -0.9)))

    def test_spots_from_memory(self):
        placement = {"table.other2": {"level": 4, "position": [3.3, 3.4, 0.91]}}
        self.assertEqual(
            spot_from(placement, "table.other2"), (3.3, 3.4, 0.91, 4, "map")
        )
        self.assertEqual(spot_from([1, 2, 0.4], "x"), (1.0, 2.0, 0.4, 0, "map"))
        self.assertIsNone(spot_from(placement, "table.other1"))
        self.assertIsNone(spot_from("shelf", "x"))


class PickTests(unittest.TestCase):
    def test_pick_the_cola_from_a_gpsr_location(self):
        runtime = make_runtime(kitchen_scene(), hands=FakeHands(pick=[ok()]))
        result, logs = run(
            PickObject, runtime, {"object": "cola", "surface": "kitchen_table"}
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.navigation.goals, [[-1.6, -0.7, -1.571]])
        ((kind, sent),) = runtime.hands.goals
        self.assertEqual(kind, "pick")
        self.assertEqual(sent.arm, "right")
        self.assertEqual(sent.target["label"], "cola")
        self.assertEqual(sent.target["position"], [-1.8, -1.55, 0.86])
        self.assertEqual(result.outputs[HOLDING], "cola")
        self.assertEqual(result.outputs[HELD]["surface"], "kitchen_table")
        self.assertEqual(result.outputs[HELD]["size"], list(COLA))
        self.assertEqual(result.outputs[LOCATION], [-1.6, -0.7, -1.571])
        self.assertEqual(
            PickObject(None).expected({"object": "cola"}), {HOLDING: "cola"}
        )

    def test_a_category_names_an_object_of_it(self):
        runtime = make_runtime(kitchen_scene(), hands=FakeHands(pick=[ok()]))
        result, _ = run(
            PickObject,
            runtime,
            {"object": "a drink", "surface": "dining_table", "approach": False},
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.hands.goals[0][1].target["label"], "cola")
        self.assertEqual(runtime.navigation.goals, [])
        self.assertEqual(result.outputs[HOLDING], "a drink")

    def test_a_record_of_find_object_names_the_object_and_its_surface(self):
        scene = [
            detection("cola", (3.02, 3.42, 0.97), 1, COLA),
            detection("cola", (3.27, 3.42, 0.97), 2, COLA),
        ]
        runtime = make_runtime(scene, robot="shelf", hands=FakeHands(pick=[ok()]))
        found = {
            "label": "cola",
            "position": [3.26, 3.40, 0.97],
            "frame": "map",
            "surface": "shelf",
        }
        result, _ = run(PickObject, runtime, {"object": "object"}, {"object": found})
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.hands.goals[0][1].target["position"][0], 3.27)
        self.assertEqual(result.outputs[HOLDING], "object")

    def test_objects_in_the_bin_are_not_on_the_floor(self):
        # Put into the bin (4.5, -3.5) before, it is nearer than the floor one.
        scene = [
            detection("pringles", (4.5, -3.45, 0.12), 1, (0.075, 0.075, 0.23)),
            detection("cookies", (4.9, -3.0, 0.075), 2, (0.06, 0.1, 0.15)),
        ]
        runtime = make_runtime(scene, robot="trashbin", hands=FakeHands(pick=[ok()]))
        runtime.objects.robot = (3.65, -3.1, -0.44)
        result, _ = run(PickObject, runtime, {"object": "snack", "surface": "floor"})
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.hands.goals[0][1].target["label"], "cookies")

    def test_pick_from_the_floor_around_the_robot(self):
        scene = [detection("cola", (2.0, 2.5, 0.06), 1, COLA)]
        runtime = make_runtime(scene, robot="office", hands=FakeHands(pick=[ok()]))
        result, _ = run(PickObject, runtime, {"object": "cola", "surface": "floor"})
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.hands.goals[0][1].target["position"], [2.0, 2.5, 0.06])
        # Facing it, the cola at the right arm's point (0.45, -0.2).
        self.assertEqual(runtime.navigation.goals, [[1.8, 2.05, 1.571]])

    def test_no_goal_after_one_that_may_still_run(self):
        unstopped = Unstopped("timed out, and the server did not confirm")
        for skill, kind, memory in (
            (PickObject, "pick", {}),
            (PlaceObject, "place", PlaceTests.HELD_COLA),
        ):
            with self.subTest(skill.__name__):
                hands = FakeHands(**{kind: [unstopped, ok()]})
                runtime = make_runtime(kitchen_scene(), hands=hands)
                params = {"object": "cola", "surface": "kitchen_table"}
                result, _ = run(skill, runtime, params, memory)
                self.assertFalse(result.success)
                self.assertIn("did not confirm", result.message)
                self.assertEqual(len(hands.goals), 1)

    def test_a_success_after_the_cancel_says_the_memory_misses_it(self):
        class CancelledMeanwhile(FakeHands):
            def run(self, name, request, timeout, cancelled):
                cancelled.set()  # the task is cancelled; the goal ends well
                return super().run(name, request, timeout, cancelled)

        hands = CancelledMeanwhile(pick=[ok()])
        runtime = make_runtime(kitchen_scene(), hands=hands)
        params = {"object": "cola", "surface": "kitchen_table"}
        result, logs = run(PickObject, runtime, params)
        self.assertTrue(result.success, result.message)
        self.assertIn("its memory does not keep robot.holding='", result.message)
        self.assertTrue(any("check the hand" in line for line in logs))

    def test_the_base_comes_closer_after_a_place_before_the_arm_moved(self):
        for status, moves in ((3, 2), (2, 2), (4, 1)):  # unreachable, no space, exec
            with self.subTest(status=status):
                hands = FakeHands(place=[place_failed(status, "no"), ok()])
                runtime = make_runtime([], robot="kitchen_table", hands=hands, off=0.15)
                runtime.objects.robot = (-2.0, -0.35, -math.pi / 2)
                params = {"object": "cola", "surface": "kitchen_table"}
                result, _ = run(PlaceObject, runtime, params, PlaceTests.HELD_COLA)
                self.assertTrue(result.success, result.message)
                self.assertEqual(len(runtime.navigation.goals), moves)

    def test_unreachable_moves_the_base_also_within_reach(self):
        hands = FakeHands(place=[place_failed(3, "unreachable"), ok()])
        runtime = make_runtime([], robot="kitchen_table", hands=hands)
        # In REACH of the spot, but not at the arm's point of it.
        runtime.objects.robot = (-1.75, -0.8, -math.pi / 2)
        params = {"object": "cola", "surface": "kitchen_table"}
        result, _ = run(PlaceObject, runtime, params, PlaceTests.HELD_COLA)
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(runtime.navigation.goals), 1)

    def test_a_retry_does_not_move_the_base_with_the_arm_maybe_out(self):
        for skill, scene, robot, params, memory, kind in (
            (
                PickObject,
                kitchen_scene(),
                "kitchen_table",
                {"object": "cola", "surface": "kitchen_table"},
                {},
                "pick",
            ),
            (
                PlaceObject,
                [],
                "kitchen_table",
                {"object": "cola", "surface": "kitchen_table"},
                PlaceTests.HELD_COLA,
                "place",
            ),
        ):
            with self.subTest(skill.__name__):
                hands = FakeHands(**{kind: [place_failed(4, "stuck"), ok()]})
                runtime = make_runtime(scene, robot=robot, hands=hands, off=0.15)
                # Farther from the table than its location, so that both move.
                runtime.objects.robot = (-2.0, -0.35, -math.pi / 2)
                result, logs = run(skill, runtime, params, memory)
                self.assertTrue(result.success, result.message)
                self.assertEqual(len(hands.goals), 2)
                self.assertEqual(len(runtime.navigation.goals), 1)
                self.assertTrue(
                    any("the arm may still be out" in line for line in logs)
                )

    def test_a_failed_pick_is_tried_again(self):
        hands = FakeHands(pick=[failed(), ok()])
        runtime = make_runtime(kitchen_scene(), hands=hands)
        result, logs = run(
            PickObject, runtime, {"object": "cola", "surface": "kitchen_table"}
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(hands.goals), 2)
        self.assertTrue(any("Pick failed: no grasp" in line for line in logs))

    def test_failures(self):
        cases = [
            ({"object": "milk", "surface": "kitchen_table"}, {}, "No milk on"),
            ({"object": "cola", "surface": "garage"}, {}, "No furniture for surface"),
            ({"object": "cola"}, {}, "No surface"),
            (
                {"object": "cola", "surface": "kitchen_table"},
                {HOLDING: "apple"},
                "Already holding apple",
            ),
        ]
        for params, memory, message in cases:
            with self.subTest(message):
                hands = FakeHands(pick=[failed(), failed()])
                runtime = make_runtime(kitchen_scene(), hands=hands)
                result, _ = run(PickObject, runtime, params, memory)
                self.assertFalse(result.success)
                self.assertIn(message, result.message)
                self.assertEqual(hands.goals, [])

    def test_the_last_failure_keeps_where_the_base_went(self):
        runtime = make_runtime(kitchen_scene(), hands=FakeHands(pick=[failed()]))
        result, _ = run(
            PickObject,
            runtime,
            {"object": "cola", "surface": "kitchen_table", "retries": 0},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outputs, {LOCATION: [-1.6, -0.7, -1.571]})


class PlaceTests(unittest.TestCase):
    HELD_COLA = {
        HOLDING: "cola",
        HELD: {"label": "cola", "size": list(COLA), "surface": "kitchen_table"},
    }

    def test_place_on_a_free_spot_of_the_desk(self):
        runtime = make_runtime(
            desk_scene(), robot="desk", hands=FakeHands(place=[ok()])
        )
        result, _ = run(
            PlaceObject, runtime, {"object": "cola", "surface": "desk"}, self.HELD_COLA
        )
        self.assertTrue(result.success, result.message)
        ((kind, sent),) = runtime.hands.goals
        self.assertEqual(kind, "place")
        self.assertEqual(sent.near, ("map", (2.2, -0.62, 0.4)))
        self.assertEqual((sent.furniture, sent.furniture_level), ("long_table", 1))
        self.assertFalse(sent.drop)
        self.assertEqual(
            (sent.object_size.x, sent.object_size.y, sent.object_size.z), COLA
        )
        # Out of REACH, but the desk location is where the base would go.
        self.assertEqual(runtime.navigation.goals, [])
        self.assertEqual(result.outputs[HOLDING], None)
        self.assertEqual(result.outputs[HELD], None)

    def test_throw_it_in_the_trash(self):
        runtime = make_runtime(None, robot="trashbin", hands=FakeHands(place=[ok()]))
        result, _ = run(
            PlaceObject,
            runtime,
            {"object": "cola", "surface": "trashbin"},
            self.HELD_COLA,
        )
        self.assertTrue(result.success, result.message)
        sent = runtime.hands.goals[0][1]
        self.assertTrue(sent.drop)
        self.assertEqual(sent.furniture, "trash_box")
        self.assertEqual(sent.near, ("map", (4.5, -3.5, 0.375)))
        self.assertIn("into trashbin", result.message)

    def test_on_the_floor_ahead(self):
        runtime = make_runtime(None, robot="office", hands=FakeHands(place=[ok()]))
        result, _ = run(
            PlaceObject, runtime, {"object": "cola", "surface": "floor"}, self.HELD_COLA
        )
        self.assertTrue(result.success, result.message)
        sent = runtime.hands.goals[0][1]
        self.assertEqual(sent.near, ("map", (2.0, 2.0, 0.0)))
        self.assertEqual((sent.furniture, sent.furniture_level), ("", 0))
        self.assertEqual(runtime.navigation.goals, [])

    def test_on_the_shelf_next_to_its_category(self):
        scene = [
            detection("milk", (3.02, 3.42, 1.005), 1, (0.07, 0.07, 0.19)),
            detection("lemonade", (3.14, 3.42, 0.97), 2, (0.066, 0.066, 0.12)),
            detection("apple", (3.1, 3.43, 1.2375), 3, (0.075,) * 3),
        ]
        runtime = make_runtime(scene, robot="shelf", hands=FakeHands(place=[ok()]))
        result, logs = run(
            PlaceObject, runtime, {"object": "cola", "surface": "shelf"}, self.HELD_COLA
        )
        self.assertTrue(result.success, result.message)
        sent = runtime.hands.goals[0][1]
        self.assertEqual(sent.furniture_level, 4)
        self.assertEqual(sent.near[1][2], 0.91)
        self.assertTrue(any("(category)" in line for line in logs))

    def test_a_spot_from_memory(self):
        memory = {
            HOLDING: "table.other2",
            "table.other2": {"label": "cola", "size": list(COLA)},
            "cabinet.placement": {
                "table.other2": {"level": 4, "position": [3.3, 3.42, 0.91]}
            },
        }
        runtime = make_runtime(None, robot="shelf", hands=FakeHands(place=[ok()]))
        result, _ = run(
            PlaceObject,
            runtime,
            {"object": "table.other2", "surface": "shelf", "spot": "cabinet.placement"},
            memory,
        )
        self.assertTrue(result.success, result.message)
        sent = runtime.hands.goals[0][1]
        self.assertEqual(
            (sent.near, sent.furniture_level), (("map", (3.3, 3.42, 0.91)), 4)
        )

    def test_a_failed_place_is_tried_again_and_nothing_held_fails(self):
        hands = FakeHands(place=[place_failed(3, "unreachable"), ok()])
        runtime = make_runtime(desk_scene(), robot="desk", hands=hands)
        result, _ = run(
            PlaceObject, runtime, {"object": "cola", "surface": "desk"}, self.HELD_COLA
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(hands.goals), 2)
        result, _ = run(PlaceObject, runtime, {"object": "cola", "surface": "desk"})
        self.assertFalse(result.success)
        self.assertIn("Not holding", result.message)

    def test_a_failure_another_spot_cannot_change_is_not_tried_again(self):
        hands = FakeHands(place=[place_failed(5, "Nothing held"), ok()])
        runtime = make_runtime(desk_scene(), robot="desk", hands=hands)
        result, _ = run(
            PlaceObject, runtime, {"object": "cola", "surface": "desk"}, self.HELD_COLA
        )
        self.assertFalse(result.success)
        self.assertIn("Nothing held", result.message)
        self.assertEqual(len(hands.goals), 1)

    def test_let_go_but_not_seen_at_the_spot_ends_holding(self):
        hands = FakeHands(place=[place_failed(9, "no object seen"), ok()])
        runtime = make_runtime(desk_scene(), robot="desk", hands=hands)
        result, _ = run(
            PlaceObject, runtime, {"object": "cola", "surface": "desk"}, self.HELD_COLA
        )
        # Only a success clears robot.holding: the engine drops a failure's outputs.
        self.assertTrue(result.success, result.message)
        self.assertIn("not seen there: no object seen", result.message)
        self.assertEqual((result.outputs[HOLDING], result.outputs[HELD]), (None, None))
        self.assertEqual(len(hands.goals), 1)


class DeliverTests(unittest.TestCase):
    HELD_COLA = PlaceTests.HELD_COLA

    def test_hand_over_to_the_found_person(self):
        runtime = make_runtime(None, robot="office", hands=FakeHands(handover=[ok()]))
        runtime.objects.robot = (2.0, 0.0, math.pi / 2)
        memory = {
            **self.HELD_COLA,
            "person.position": [2.0, 2.0, 1.3],
            "person.frame": "map",
        }
        result, _ = run(
            DeliverObject,
            runtime,
            {"object": "cola", "person": "waving person"},
            memory,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.navigation.goals, [[2.0, 1.0, 1.571]])
        ((kind, sent),) = runtime.hands.goals
        self.assertEqual(kind, "handover")
        self.assertEqual(sent.target, ("map", (2.0, 2.0, 1.1)))  # held out at 1.1 m
        self.assertEqual((sent.release, sent.timeout), ("take", 20.0))
        self.assertEqual(
            runtime.interaction.said, ["Here is the cola. Please take it from my hand."]
        )
        self.assertEqual(result.outputs[HOLDING], None)
        self.assertEqual(result.outputs[LOCATION], [2.0, 1.0, 1.571])

    def test_no_handover_from_afar_while_the_arm_may_be_out(self):
        memory = {
            **self.HELD_COLA,
            "person.position": [2.0, 2.0, 1.3],
            "person.frame": "map",
        }
        params = {"object": "cola", "person": "waving person"}
        for robot, sent in (((2.0, 0.0, math.pi / 2), 0), ((2.0, 1.0, math.pi / 2), 1)):
            with self.subTest(robot=robot):
                runtime = make_runtime(
                    None, robot="office", hands=FakeHands(handover=[ok()])
                )
                runtime.objects.robot = robot
                runtime.arm_out = "the arm may be out (test)"
                result, _ = run(DeliverObject, runtime, params, memory)
                # 2 m away: no handover from there; 1 m, facing: as it is.
                self.assertEqual(result.success, bool(sent), result.message)
                self.assertEqual(len(runtime.hands.goals), sent)
                self.assertEqual(runtime.navigation.goals, [])
                if not sent:
                    self.assertIn("2.0 m away: not moving the base", result.message)

    def test_the_operator_at_the_instruction_point(self):
        runtime = make_runtime(None, robot="office", hands=FakeHands(handover=[ok()]))
        result, logs = run(
            DeliverObject,
            runtime,
            {"object": "cola", "person": "operator"},
            {**self.HELD_COLA, LOCATION: "office"},
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.navigation.goals, [[-1.0, 1.5, 3.142]])
        self.assertEqual(runtime.hands.goals[0][1].target, ("map", (-2.0, 1.5, 1.1)))
        self.assertEqual(result.outputs[LOCATION], "instruction_point")
        self.assertTrue(any("No operator seen" in line for line in logs))

    def test_the_robot_turns_to_the_operator_it_sees(self):
        runtime = make_runtime(
            None,
            robot="instruction_point",
            hands=FakeHands(handover=[ok()]),
            people=FakePeople([(-1.5, 2.5, 0.9), (5.0, 5.0, 0.9)]),
        )
        result, _ = run(
            DeliverObject,
            runtime,
            {"object": "cola", "person": "me"},
            {**self.HELD_COLA, LOCATION: "instruction_point"},
        )
        self.assertTrue(result.success, result.message)
        # Already there; it only turns to the person on its right.
        turn = math.atan2(1.0, -0.5)
        self.assertEqual(runtime.navigation.goals, [[-1.0, 1.5, round(turn, 3)]])
        self.assertEqual(runtime.hands.goals[0][1].target, ("map", (-1.5, 2.5, 0.9)))

    def test_a_touch_on_the_screen_releases(self):
        hands = FakeHands(handover=[None, ok()])
        runtime = make_runtime(
            None,
            robot="office",
            hands=hands,
            interaction=FakeInteraction(touch=True),
        )
        memory = {**self.HELD_COLA, "person.position": [2.0, 2.4, 0.9]}
        result, _ = run(
            DeliverObject,
            runtime,
            {"object": "cola", "person": "Jane", "touch": True},
            memory,
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("released by touch", result.message)
        self.assertEqual([g.release for _, g in hands.goals], ["take", "now"])
        self.assertEqual(hands.cancelled, 1)
        self.assertEqual(
            runtime.interaction.asked, ["Touch Release when you hold the cola"]
        )

    def test_a_touch_after_the_take_sends_nothing_more(self):
        # The server let go and was bringing the arm back when the touch came.
        for ended, success, goals in ((ok(), True, 1), (failed(1, "nobody"), False, 1)):
            with self.subTest(status=ended.status):
                hands = FakeHands(handover=[None, ok()], on_cancel=ended)
                runtime = make_runtime(
                    None,
                    robot="office",
                    hands=hands,
                    interaction=FakeInteraction(touch=True),
                )
                memory = {**self.HELD_COLA, "person.position": [2.0, 2.4, 0.9]}
                params = {
                    "object": "cola",
                    "person": "Jane",
                    "touch": True,
                    "retries": 0,
                }
                result, _ = run(DeliverObject, runtime, params, memory)
                self.assertEqual(result.success, success, result.message)
                self.assertEqual([g.release for _, g in hands.goals], ["take"] * goals)
                self.assertEqual(hands.cancelled, 1)

    def test_no_second_goal_while_the_take_goal_may_still_run(self):
        hands = FakeHands(handover=[None, ok()])
        hands.on_cancel = None  # no result within the wait after the cancel
        runtime = make_runtime(
            None, robot="office", hands=hands, interaction=FakeInteraction(touch=True)
        )
        memory = {**self.HELD_COLA, "person.position": [2.0, 2.4, 0.9]}
        params = {"object": "cola", "person": "Jane", "touch": True}
        result, _ = run(DeliverObject, runtime, params, memory)
        self.assertFalse(result.success)
        self.assertIn("did not stop the take goal", result.message)
        self.assertEqual([g.release for _, g in hands.goals], ["take"])

    def test_the_screen_is_not_used_unless_touch_is_set(self):
        hands = FakeHands(handover=[ok()])
        runtime = make_runtime(
            None, robot="office", hands=hands, interaction=FakeInteraction(touch=True)
        )
        memory = {**self.HELD_COLA, "person.position": [2.0, 2.4, 0.9]}
        result, _ = run(
            DeliverObject, runtime, {"object": "cola", "person": "Jane"}, memory
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(runtime.interaction.asked, [])

    def test_asked_again_when_nobody_takes_it(self):
        hands = FakeHands(handover=[failed(1, "nobody"), ok()])
        runtime = make_runtime(None, robot="office", hands=hands)
        memory = {**self.HELD_COLA, "person.position": [2.0, 2.4, 0.9]}
        result, _ = run(
            DeliverObject, runtime, {"object": "cola", "person": "Jane"}, memory
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            runtime.interaction.said,
            [
                "Here is the cola. Please take it from my hand.",
                "Please take the cola from my hand.",
            ],
        )
        hands = FakeHands(handover=[failed(1, "nobody"), failed(1, "nobody")])
        runtime = make_runtime(None, robot="office", hands=hands)
        result, _ = run(
            DeliverObject, runtime, {"object": "cola", "person": "Jane"}, memory
        )
        self.assertFalse(result.success)
        self.assertIn("nobody", result.message)

    def test_failures(self):
        runtime = make_runtime(None, robot="office", hands=FakeHands(handover=[]))
        result, _ = run(
            DeliverObject, runtime, {"object": "cola", "person": "Jane"}, self.HELD_COLA
        )
        self.assertFalse(result.success)
        self.assertIn("No person.position", result.message)
        result, _ = run(DeliverObject, runtime, {"object": "cola", "person": "Jane"})
        self.assertIn("Not holding", result.message)
        hands = FakeHands(handover=[failed(2, "too far")])
        runtime = make_runtime(None, robot="office", hands=hands)
        memory = {**self.HELD_COLA, "guest.position": [2.0, 2.4, 0.9]}
        result, _ = run(
            DeliverObject,
            runtime,
            {"object": "cola", "person": "Jane", "role": "guest"},
            memory,
        )
        self.assertFalse(result.success)
        self.assertEqual(len(hands.goals), 1, "no second try unless timed out")


class FakeFuture:
    def __init__(self):
        self.value, self.callbacks = None, []

    def done(self):
        return self.value is not None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def set_result(self, value):
        self.value = value
        for callback in self.callbacks:
            callback(self)


class FakeClient:
    def __init__(self):
        self.sent = []

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal):
        self.sent.append(FakeFuture())
        return self.sent[-1]


class FakeHandle:
    """A goal that ends with `on_cancel` once it is cancelled."""

    def __init__(self, on_cancel):
        self.future, self.on_cancel = FakeFuture(), on_cancel

    def get_result_async(self):
        return self.future

    def cancel_goal_async(self):
        self.future.set_result(SimpleNamespace(result=self.on_cancel))


class HandsTests(unittest.TestCase):
    def test_a_goal_that_succeeded_before_the_cancel_counts(self):
        hands = Hands.__new__(Hands)
        hands.runtime = SimpleNamespace()
        cancelled = threading.Event()
        cancelled.set()
        # The object in the hand, the arm stopped on its way to carry.
        done = PickResult(success=True, message="Picked; not at carry", returned=False)
        self.assertIs(hands.finish(FakeHandle(done), 1.0, cancelled), done)
        self.assertIn("not back: Picked; not at carry", getattr(hands.runtime, ARM_OUT))
        with self.assertRaises(Cancelled):
            hands.finish(FakeHandle(failed(4, "cancelled")), 1.0, cancelled)
        # A timeout likewise.
        self.assertIs(hands.finish(FakeHandle(done), 0.0, threading.Event()), done)

    def test_the_arm_state_follows_the_ends_of_the_goals(self):
        runtime = SimpleNamespace()
        hands = Hands.__new__(Hands)
        hands.runtime = runtime
        place = PlaceResult(success=False, status=4, message="stuck", returned=False)
        hands.arm_after(place)
        self.assertIn("the arm may be out (not back: stuck)", getattr(runtime, ARM_OUT))
        hands.arm_after(Result(success=False, status=0, message="no", returned=True))
        self.assertIsNone(getattr(runtime, ARM_OUT))
        # A pick: `returned` false where a cancel or failure left the arm.
        hands.arm_after(PickResult(success=False, message="stuck", returned=False))
        self.assertIn("not back: stuck", getattr(runtime, ARM_OUT))
        hands.arm_after(PickResult(success=True, message="Picked", returned=True))
        self.assertIsNone(getattr(runtime, ARM_OUT))
        # Without `returned` (an older server), a cancel or failure may leave
        # the arm out.
        for pick in (
            SimpleNamespace(success=True, message="Picked"),
            SimpleNamespace(success=False, message="cancelled"),
        ):
            with self.subTest(stopped=pick.message):
                hands.arm_after(pick, stopped=True)
                self.assertIn("ended without returned", getattr(runtime, ARM_OUT))
        hands.arm_after(SimpleNamespace(success=False, message="no grasp"))
        self.assertIn("ended without returned", getattr(runtime, ARM_OUT))
        hands.arm_after(SimpleNamespace(success=True, message="Picked"))
        self.assertIsNone(getattr(runtime, ARM_OUT))
        with self.assertRaises(Unstopped):
            hands.finish(FakeHandle(None), 0.0, threading.Event())
        self.assertIn("did not confirm that it stopped", getattr(runtime, ARM_OUT))

    def test_failures_before_any_motion_leave_the_arm_as_it_was(self):
        # `returned` is false by the message default; the status says more.
        # The arm is where earlier goals left it, e.g. out after a cancel.
        runtime = SimpleNamespace(arm_out="earlier")
        hands = Hands.__new__(Hands)
        hands.runtime = runtime
        before = [
            *(
                PlaceResult(success=False, status=status, message="", returned=False)
                for status in (1, 2, 3, 5)  # no surface, no space, unreachable, error
            ),
            *(
                Result(success=False, status=status, message="", returned=False)
                for status in (2, 3)  # unreachable, error
            ),
            # No status: maybe before any motion, else back to its pose before.
            PickResult(success=False, message="too far", returned=True),
        ]
        for result in before:
            for earlier in ("earlier", None):
                with self.subTest(result=result, earlier=earlier):
                    runtime.arm_out = earlier
                    hands.arm_after(result, stopped=True)
                    self.assertEqual(runtime.arm_out, earlier)
        # After a motion, `returned` decides: execution failed, cancelled.
        for result in (
            PlaceResult(success=False, status=4, message="stuck", returned=False),
            Result(success=False, status=4, message="cancelled", returned=False),
        ):
            with self.subTest(after_motion=result.status):
                runtime.arm_out = None
                hands.arm_after(result, stopped=True)
                self.assertIn("not back", runtime.arm_out)

    def test_every_base_motion_waits_for_the_arm(self):
        try:
            from aha_tasks.navigation import Navigation
        except ImportError as error:  # host without ROS
            self.skipTest(f"no ROS: {error}")
        navigation = Navigation.__new__(Navigation)  # no client: nothing is sent
        runtime = SimpleNamespace(navigation=navigation, spec=SPEC)
        guard_navigation(navigation, runtime)
        runtime.arm_out = "the arm may be out (test)"
        refused = (False, "Not moving the base: the arm may be out (test)")
        self.assertEqual(navigation.navigate([0.0, 0.0, 0.0]), refused)
        # navigate, the restore of robot.location and plan mocks use go_to.
        moved = go_to(runtime, "desk", threading.Event())
        self.assertEqual((moved.success, moved.message), refused)

    def test_the_guard_reads_the_runtime(self):
        navigation, runtime = SimpleNamespace(), SimpleNamespace()
        guard_navigation(navigation, runtime)
        self.assertIsNone(navigation.blocked())
        runtime.arm_out = "out"
        self.assertEqual(navigation.blocked(), "out")

    def test_the_base_stays_while_the_arm_may_be_out(self):
        runtime = make_runtime(kitchen_scene(), hands=FakeHands(pick=[ok()]))
        runtime.arm_out = "the arm may be out (test)"
        result, logs = run(
            PickObject, runtime, {"object": "cola", "surface": "kitchen_table"}
        )
        self.assertTrue(result.success, result.message)  # from where it is
        self.assertEqual(runtime.navigation.goals, [])
        self.assertTrue(any("the arm may still be out" in line for line in logs))

    def test_a_timed_out_goal_that_did_not_confirm_its_stop(self):
        hands = Hands.__new__(Hands)
        with self.assertRaises(Unstopped):
            hands.finish(FakeHandle(None), 0.0, threading.Event())
        # On a cancel of the task the skill ends anyway.
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            hands.finish(FakeHandle(None), 1.0, cancelled)

    def test_a_goal_accepted_after_a_cancel_is_cancelled(self):
        client = FakeClient()
        hands = Hands.__new__(Hands)
        hands.clients = {HANDOVER_ACTION: (None, client)}
        cancelled = threading.Event()
        cancelled.set()
        result, why = hands.run(HANDOVER_ACTION, goal(), 1.0, cancelled)
        self.assertEqual((result, why), (None, "Cancelled"))
        handle = SimpleNamespace(accepted=True, cancels=0)
        handle.cancel_goal_async = lambda: setattr(handle, "cancels", 1)
        client.sent[0].set_result(handle)  # the late acceptance
        self.assertEqual(handle.cancels, 1)

    def test_a_goal_without_an_answer_is_not_followed(self):
        # The acceptance may come late or be lost: the arm may move.
        client = FakeClient()
        hands = Hands.__new__(Hands)
        hands.runtime = SimpleNamespace(arm_out=None)
        hands.clients = {PICK_ACTION: (None, client)}
        with mock.patch("aha_tasks.handling.SERVER_TIMEOUT", 0.0):
            with self.assertRaises(Unstopped):
                hands.run(PICK_ACTION, goal(), 1.0, threading.Event())
        self.assertIn("did not confirm", hands.runtime.arm_out)
        client.sent[0].set_result(SimpleNamespace(accepted=False))
        self.assertIsNone(hands.runtime.arm_out)
        # Two unanswered goals: the late answer of the first one does not
        # give back the arm state while the second may still move the arm.
        with mock.patch("aha_tasks.handling.SERVER_TIMEOUT", 0.0):
            for _ in range(2):
                with self.assertRaises(Unstopped):
                    hands.run(PICK_ACTION, goal(), 1.0, threading.Event())
        second = hands.runtime.arm_out
        client.sent[1].set_result(SimpleNamespace(accepted=False))
        self.assertEqual(hands.runtime.arm_out, second)
        client.sent[2].set_result(SimpleNamespace(accepted=False))
        self.assertIsNone(hands.runtime.arm_out)

    def test_a_late_accepted_goal_keeps_the_base_until_its_result(self):
        hands = Hands.__new__(Hands)
        hands.runtime = SimpleNamespace(arm_out=None)
        cancelled = threading.Event()
        cancelled.set()
        stuck = PickResult(success=False, message="cancelled", returned=False)
        back = PickResult(success=True, message="Picked", returned=True)
        before = PickResult(success=False, message="too far", returned=True)
        for on_cancel, out in ((stuck, True), (back, False), (before, False)):
            with self.subTest(result=on_cancel.message):
                client = FakeClient()
                hands.clients = {PICK_ACTION: (None, client)}
                self.assertEqual(
                    hands.run(PICK_ACTION, goal(), 1.0, cancelled), (None, "Cancelled")
                )
                handle = FakeHandle(on_cancel)
                handle.accepted = True
                calls = []
                handle.cancel_goal_async = lambda h=handle: calls.append(h)
                self.assertIn("did not confirm", hands.runtime.arm_out)
                client.sent[0].set_result(handle)  # the late acceptance
                self.assertEqual(len(calls), 1)
                self.assertIn("did not confirm", hands.runtime.arm_out)
                handle.future.set_result(SimpleNamespace(result=on_cancel))
                self.assertEqual(hands.runtime.arm_out is not None, out)
        # A late rejection changes nothing.
        client = FakeClient()
        hands.clients = {PICK_ACTION: (None, client)}
        hands.run(PICK_ACTION, goal(), 1.0, cancelled)
        client.sent[0].set_result(SimpleNamespace(accepted=False))
        self.assertIsNone(hands.runtime.arm_out)


class ValidationTests(unittest.TestCase):
    def test_bad_params_are_refused(self):
        bad = [
            (PickObject, {}),
            (PickObject, {"object": ""}),
            (PickObject, {"object": "cola", "surface": 3}),
            (PickObject, {"object": "cola", "arm": "middle"}),
            (PickObject, {"object": "cola", "retries": 9}),
            (PickObject, {"object": "cola", "margin": -1}),
            (PickObject, {"object": "cola", "colour": "red"}),
            (PlaceObject, {"object": "cola"}),
            (PlaceObject, {"object": "cola", "surface": "desk", "spot": ""}),
            (PlaceObject, {"object": "cola", "surface": "desk", "clearance": -0.1}),
            (DeliverObject, {"object": "cola"}),
            (DeliverObject, {"object": "cola", "person": "me", "distance": 0}),
            (DeliverObject, {"object": "cola", "person": "me", "role": ""}),
        ]
        for cls, params in bad:
            with self.subTest(cls=cls.__name__, params=params):
                with self.assertRaises(ValueError):
                    cls.validate(params, SPEC)

    def test_the_mock_plan_validates(self):
        plan = next(s for s in SPEC.subgoals if s.id == "plan.generate").mock["plan"]
        skills = {"pick_object": PickObject, "deliver_object": DeliverObject}
        steps = [s for s in plan if s["action"] in skills]
        self.assertEqual(len(steps), 2)
        for step in steps:
            skills[step["action"]].validate(step["params"], SPEC)


if __name__ == "__main__":
    unittest.main()
