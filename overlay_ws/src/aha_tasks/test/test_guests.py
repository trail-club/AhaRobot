"""Guest phases and motion against task statuses (no ROS, no Gazebo)."""

import math
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core import load_spec  # noqa: E402
from aha_tasks.guests import (  # noqa: E402
    ARRIVE,
    AWAY,
    FOLLOW,
    SEAT,
    TALK,
    TRIGGERS,
    DOOR_TRIGGER,
    Guest,
    attempts_of,
    chair_sdf,
    door_open,
    executing,
    guest_phase,
    guest_sdf,
    load_config,
    seated_models,
    seated_pose,
    statuses_of,
)
from aha_tasks.seats import RADIUS, nearest_seat  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config"
SCENE = load_config(CONFIG / "sim" / "human_interaction.json")
WALL_X = -5.0  # the front wall of the arena; the door is in it
ENTRANCE = (-4.2, 1.5)


def guest1():
    return Guest("guest1", SCENE["guests"]["guest1"], SCENE)


def run(guest, seconds, robot=None, dt=0.1):
    for _ in range(round(seconds / dt)):
        guest.step(dt, robot)


class PhaseTests(unittest.TestCase):
    def test_pending_subgoals_leave_a_guest_away(self):
        statuses = {"guest1.doorbell": "pending", "guest1.ask": "pending"}
        self.assertEqual(guest_phase("guest1", statuses), AWAY)

    def test_the_latest_reached_subgoal_gives_the_phase(self):
        cases = {
            "guest1.doorbell": ARRIVE,
            "guest1.ask": TALK,
            "guest1.guide": FOLLOW,
            "guest1.seat": SEAT,
        }
        for subgoal, phase in cases.items():
            for status in ("running", "succeeded", "unsupported", "bypassed"):
                with self.subTest(subgoal, status=status):
                    self.assertEqual(guest_phase("guest1", {subgoal: status}), phase)

    def test_a_rewound_subgoal_lowers_the_phase(self):
        statuses = {"guest1.doorbell": "succeeded", "guest1.ask": "succeeded"}
        statuses["guest1.guide"] = "pending"
        self.assertEqual(guest_phase("guest1", statuses), TALK)

    def test_another_guest_has_its_own_phase(self):
        self.assertEqual(guest_phase("guest2", {"guest1.ask": "running"}), AWAY)

    def test_door_opens_with_either_guest(self):
        guests = ["guest1", "guest2"]
        self.assertFalse(door_open(guests, {"guest1.open_door": "pending"}))
        self.assertTrue(door_open(guests, {"guest2.open_door": "unsupported"}))

    def test_statuses_of_a_status_message(self):
        message = {
            "current": "a",
            "phase": "executing",
            "run": {"number": 3},
            "subgoals": [{"id": "a", "status": "running", "attempts": 2}],
        }
        self.assertEqual(statuses_of(message), {"a": "running"})
        self.assertEqual(attempts_of(message), {"a": (3, 2)})
        self.assertEqual(executing(message), "a")
        message["phase"] = "restoring"  # the skill has not started yet
        self.assertIsNone(executing(message))
        message["run"] = None  # before the first run
        self.assertEqual(attempts_of(message), {"a": (0, 2)})


class MotionTests(unittest.TestCase):
    def test_arrives_at_the_door_and_stays_outside(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        run(guest, 10.0, ENTRANCE)
        self.assertEqual((guest.x, guest.y), tuple(guest.config["door"]))
        self.assertLess(guest.x, WALL_X)
        self.assertAlmostEqual(guest.yaw, 0.0)  # looks in at the door

    def test_rings_once_on_reaching_the_door(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_bell((1, 1), True)
        rings = []
        for _ in range(100):
            guest.step(0.1, ENTRANCE)
            if guest.ring is not None:
                rings.append(((guest.x, guest.y), guest.ring))
                guest.ring = None
        self.assertEqual(rings, [(tuple(guest.config["door"]), "chime")])
        guest.set_phase(AWAY)
        guest.set_phase(ARRIVE)  # the doorbell subgoal again: rings again
        run(guest, 10.0, ENTRANCE)
        self.assertEqual(guest.ring, "chime")

    def test_rings_only_while_the_robot_waits(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_bell((1, 1), False)  # e.g. the runner still restores the robot pose
        run(guest, 10.0, ENTRANCE)
        self.assertIsNone(guest.ring)
        guest.set_bell((1, 1), True)
        guest.step(0.1, ENTRANCE)
        self.assertEqual(guest.ring, "chime")

    def test_rings_again_when_the_doorbell_is_retried(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_bell((1, 1), True)
        run(guest, 10.0, ENTRANCE)
        guest.ring = None
        guest.set_bell((1, 1), True)
        run(guest, 1.0, ENTRANCE)
        self.assertIsNone(guest.ring)
        guest.set_bell((1, 2), False)  # failed (e.g. missed), retried: restoring first
        run(guest, 1.0, ENTRANCE)
        self.assertIsNone(guest.ring)
        guest.set_bell((1, 2), True)
        guest.step(0.1, ENTRANCE)
        self.assertEqual(guest.ring, "chime")

    def test_rings_again_in_a_new_run_of_the_same_subgoals(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_bell((1, 1), True)
        run(guest, 10.0, ENTRANCE)
        guest.ring = None
        # Run 2 starts the attempts from 0 again; the guest may miss the pending
        # status in between, so it stays in ARRIVE.
        guest.set_bell((2, 1), True)
        guest.step(0.1, ENTRANCE)
        self.assertEqual(guest.ring, "chime")

    def test_walks_at_the_configured_speed(self):
        guest = guest1()
        start = (guest.x, guest.y)
        guest.set_phase(ARRIVE)
        run(guest, 1.0)
        self.assertAlmostEqual(
            math.dist(start, (guest.x, guest.y)), SCENE["walk_speed"]
        )

    def test_paces_during_the_conversation_without_coming_in(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        run(guest, 10.0)
        guest.set_phase(TALK)
        ys, xs = [], []
        for _ in range(60):
            guest.step(0.1, ENTRANCE)
            xs.append(guest.x)
            ys.append(guest.y)
        amplitude = SCENE["pace"]["amplitude"]
        self.assertAlmostEqual(max(ys) - 1.5, amplitude, delta=0.01)
        self.assertAlmostEqual(1.5 - min(ys), amplitude, delta=0.01)
        self.assertLess(max(xs), WALL_X)
        self.assertAlmostEqual(math.cos(guest.yaw), 1.0, delta=0.1)  # faces the robot

    def test_does_not_come_in_before_the_robot_makes_room(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        run(guest, 10.0)
        guest.set_phase(FOLLOW)
        robot = (guest.x + SCENE["follow_gap"] - 0.1, guest.y)
        run(guest, 5.0, robot)
        self.assertLess(guest.x, WALL_X)

    def test_follows_the_way_the_robot_went_through_the_door(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        run(guest, 10.0)
        guest.set_phase(FOLLOW)
        # The robot leaves east, then turns north at x = -2.
        path = [(-4.2 + 0.05 * i, 1.5) for i in range(45)]
        path += [(-2.0, 1.5 + 0.05 * i) for i in range(30)]
        trace = []
        for robot in path:
            guest.step(0.1, robot)
            trace.append((guest.x, guest.y))
            self.assertGreaterEqual(
                math.dist((guest.x, guest.y), robot), SCENE["follow_gap"] - 0.15
            )
        run(guest, 5.0, path[-1])
        self.assertGreater(guest.x, WALL_X)
        self.assertAlmostEqual(
            math.dist((guest.x, guest.y), path[-1]), SCENE["follow_gap"], delta=0.15
        )
        for x, y in trace:
            if x > WALL_X - 0.05 and x < -2.1:  # in through the 1 m gap, y from 1 to 2
                self.assertAlmostEqual(y, 1.5, delta=0.3)

    def test_goes_to_the_offered_seat_and_faces_the_room(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_phase(FOLLOW)
        guest.x, guest.y, guest.route = 0.6, 2.5, []
        guest.set_phase(SEAT, "chair_extra")
        run(guest, 6.0, (1.25, 2.5))
        self.assertEqual((guest.x, guest.y), (2.55, 3.05))
        self.assertAlmostEqual(guest.yaw, -math.pi / 2, places=3)

    def test_without_an_offered_seat_it_takes_the_configured_one(self):
        for seat in (None, "sofa_that_the_sim_does_not_have"):
            guest = guest1()
            guest.set_phase(SEAT, seat)
            self.assertEqual(guest.seat, guest.config["seat"])

    def test_another_seat_offered_meanwhile_turns_the_guest(self):
        guest = guest1()
        guest.set_phase(FOLLOW)
        guest.x, guest.y, guest.route = 0.75, 2.6, []
        guest.set_phase(SEAT, "chair_right")
        run(guest, 0.2)
        guest.set_phase(SEAT, "chair_left")
        run(guest, 5.0)
        self.assertEqual((guest.x, guest.y), (1.75, 3.05))

    def test_seat_without_the_walk_is_a_jump(self):
        """guest2.* alone: guest1's subgoals are bypassed at once."""
        guest = guest1()
        guest.set_phase(SEAT, "chair_right")
        self.assertEqual((guest.x, guest.y), (0.75, 3.05))

    def test_rewind_goes_back_without_walking(self):
        guest = guest1()
        guest.set_phase(SEAT)
        guest.set_phase(TALK)
        self.assertEqual((guest.x, guest.y), tuple(guest.config["door"]))
        guest.set_phase(AWAY)
        self.assertEqual((guest.x, guest.y), tuple(guest.config["away"]))

    def test_model_yaw_turns_the_mesh_to_the_heading(self):
        guest = guest1()
        guest.yaw = 0.0  # facing +x; the mesh faces -y at model yaw 0
        self.assertAlmostEqual(guest.model_yaw, math.pi / 2)


class SceneTests(unittest.TestCase):
    def test_every_trigger_is_a_subgoal_of_the_task(self):
        spec = load_spec(CONFIG / "human_interaction.json")
        ids = {subgoal.id for subgoal in spec.subgoals}
        for name in SCENE["guests"]:
            for trigger in [*TRIGGERS.values(), DOOR_TRIGGER]:
                self.assertIn(f"{name}.{trigger}", ids)

    def test_guests_wait_outside_the_house(self):
        for name, guest in SCENE["guests"].items():
            with self.subTest(name):
                self.assertLess(guest["door"][0], WALL_X)
                self.assertLess(guest["away"][0], WALL_X)

    def test_a_guest_is_hidden_until_it_arrives(self):
        guest = guest1()
        self.assertLess(guest.z, 0.0)
        self.assertIn(
            f" {guest.z} 0 0 0</pose>", guest_sdf("g", guest.config["away"], {})
        )
        guest.set_phase(ARRIVE)
        self.assertEqual(guest.z, 0.0)
        guest.set_phase(AWAY)
        self.assertLess(guest.z, 0.0)

    def test_guests_look_different(self):
        people = SCENE["guests"].values()
        shirts = {tuple(person["colors"]["shirt"]) for person in people}
        self.assertEqual(len(shirts), len(people))

    def test_seats_match_the_task_and_a_person_there_occupies_it(self):
        """The sim seats the host and stands guests in front of a seat.

        find_free_seat must count both. person_perception measures the front
        of the body, nearer the robot.
        """
        spec = load_spec(CONFIG / "human_interaction.json")
        keys = {s.params["seats"] for s in spec.subgoals if s.skill == "find_free_seat"}
        self.assertEqual(len(keys), 1)  # both guests share the seats of the task
        seats = spec.initial_state[keys.pop()]
        self.assertEqual(set(SCENE["seats"]), set(seats))
        for name, seat in SCENE["seats"].items():
            with self.subTest(name):
                self.assertEqual(seat["sit"], seats[name][:2])
                x, y, yaw = seat["pose"]  # faces the room
                for ahead in (0.0, 0.25):
                    xy = (x + ahead * math.cos(yaw), y + ahead * math.sin(yaw))
                    self.assertEqual(nearest_seat(seats, xy, RADIUS), name)
                if "chair" in seat:  # the added chair is where the task expects
                    self.assertEqual(seat["chair"][:2], seats[name][:2])
        mocks = [s.mock for s in spec.subgoals if s.skill == "find_free_seat"]
        for guest, mock in zip(("guest1", "guest2"), mocks):
            seat = mock[f"{guest}.seat"]
            self.assertEqual(mock[f"{guest}.seat_position"], seats[seat])
            # The mock is what the sim's guest is offered with the host seated.
            self.assertEqual(seat, SCENE["guests"][guest]["seat"])
        taken = {
            p["seat"] for p in [*SCENE["seated"].values(), *SCENE["guests"].values()]
        }
        self.assertEqual(len(taken), len(SCENE["seated"]) + len(SCENE["guests"]))

    def test_the_host_sits_at_its_seat_facing_the_room(self):
        x, y, z, yaw = seated_pose(SCENE, "host")
        seat = SCENE["seats"][SCENE["seated"]["host"]["seat"]]
        self.assertEqual((x, y, z), (*seat["sit"], 0.0))
        self.assertAlmostEqual(yaw, -math.pi / 2, places=5)  # sim_people faces +x

    def test_an_added_chair_is_static_and_renamed(self):
        model = (
            '<?xml version="1.0" ?><sdf version="1.8"><model name="chair">'
            '<link name="link"/></model></sdf>'
        )
        sdf = ET.fromstring(chair_sdf(model, "chair_extra", [2.5, 3.5, -1.5]))
        chair = sdf.find("model")
        self.assertEqual(chair.get("name"), "chair_extra")
        self.assertEqual(chair.findtext("static"), "true")
        self.assertEqual(chair.findtext("pose"), "2.5 3.5 0 0 0 -1.5")
        with self.assertRaises(ValueError):
            chair_sdf("<sdf/>", "chair_extra", [0, 0, 0])

    def test_guest_model_has_no_collision_and_every_mesh_part(self):
        sdf = guest_sdf("g", (1.0, 2.0), {"shirt": [1.0, 0.0, 0.0]})
        model = ET.fromstring(sdf).find("model")
        self.assertEqual(model.findtext("static"), "true")
        self.assertEqual(model.findall(".//collision"), [])
        names = [e.text for e in model.findall(".//submesh/name")]
        self.assertIn("v2_tshirt02", names)
        self.assertEqual(len(names), 7)
        self.assertEqual(len(model.findall(".//material")), 1)

    def test_a_broken_scene_is_rejected(self):
        import json
        import tempfile

        def broken(change):
            scene = json.loads(json.dumps(SCENE))
            change(scene)
            return scene

        for scene in (
            broken(lambda s: s["guests"]["guest1"]["colors"].update(hat=[0, 0, 0])),
            broken(lambda s: s["guests"]["guest1"].update(seat="sofa")),
            broken(lambda s: s["seated"]["host"].update(seat="sofa")),
            broken(lambda s: s["seated"]["host"].update(shirt="pink")),
            broken(lambda s: s["seats"]["chair_left"].pop("sit")),
            broken(lambda s: s["seats"]["chair_left"].pop("via")),
            broken(lambda s: s["seated"].update(guest1=s["seated"]["host"])),
        ):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "scene.json"
                path.write_text(json.dumps(scene))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_unknown_bell_is_rejected(self):
        import json
        import tempfile

        scene = json.loads(json.dumps(SCENE))
        scene["guests"]["guest2"]["bell"] = "siren"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            path.write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                load_config(path)


class SeatedModelsTests(unittest.TestCase):
    def test_the_host_sits_hidden_until_moved(self):
        asked = []

        def build(needed):
            asked.append(needed)
            return {("sitting", "blue"): Path("/cache/sitting__blue.dae")}

        logged = []
        models = seated_models(SCENE, build, logged.append)
        self.assertEqual(asked, [[("sitting", "blue")]])
        self.assertEqual([name for name, _ in models], ["host"])
        model = ET.fromstring(models[0][1]).find("model")
        self.assertEqual(model.get("name"), "host")
        x, y, _, yaw = seated_pose(SCENE, "host")
        self.assertEqual(model.findtext("pose"), f"{x} {y} -5.0 0 0 {yaw}")
        self.assertIn("file:///cache/sitting__blue.dae", models[0][1])
        self.assertEqual(logged, [])

    def test_a_broken_cache_leaves_nobody_seated_but_the_guests_run(self):
        import xml.etree.ElementTree
        import zipfile

        for error in (
            zipfile.BadZipFile("truncated download"),
            xml.etree.ElementTree.ParseError("half-written DAE"),
            RuntimeError("hash mismatch"),
            OSError("no network"),
            KeyError("UpperLeg_L"),
        ):
            with self.subTest(type(error).__name__):

                def build(needed, error=error):
                    raise error

                logged = []
                self.assertEqual(seated_models(SCENE, build, logged.append), [])
                self.assertEqual(len(logged), 1)
                self.assertIn(type(error).__name__, logged[0])

    def test_nobody_seated_builds_nothing(self):
        scene = {**SCENE, "seated": {}}
        self.assertEqual(seated_models(scene, None, None), [])


if __name__ == "__main__":
    unittest.main()
