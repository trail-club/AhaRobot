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
    door_open,
    guest_phase,
    guest_sdf,
    load_config,
    statuses_of,
)

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
        message = {"subgoals": [{"id": "a", "status": "running", "title": "x"}]}
        self.assertEqual(statuses_of(message), {"a": "running"})


class MotionTests(unittest.TestCase):
    def test_arrives_at_the_door_and_stays_outside(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        run(guest, 10.0, ENTRANCE)
        self.assertEqual((guest.x, guest.y), tuple(guest.config["door"]))
        self.assertLess(guest.x, WALL_X)
        self.assertAlmostEqual(guest.yaw, 0.0)  # looks in at the door

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

    def test_goes_to_the_seat_and_faces_the_room(self):
        guest = guest1()
        guest.set_phase(ARRIVE)
        guest.set_phase(FOLLOW)
        guest.x, guest.y, guest.route = 0.6, 2.5, []
        guest.set_phase(SEAT)
        run(guest, 5.0, (1.75, 2.5))
        self.assertEqual((guest.x, guest.y), (1.75, 3.05))
        self.assertAlmostEqual(guest.yaw, -math.pi / 2, places=3)

    def test_seat_without_the_walk_is_a_jump(self):
        """guest2.* alone: guest1's subgoals are bypassed at once."""
        guest = guest1()
        guest.set_phase(SEAT)
        self.assertEqual((guest.x, guest.y), (1.75, 3.05))

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
        colors = [guest["colors"] for guest in SCENE["guests"].values()]
        self.assertNotEqual(colors[0]["shirt"], colors[1]["shirt"])

    def test_guest_model_has_no_collision_and_every_mesh_part(self):
        sdf = guest_sdf("g", (1.0, 2.0), {"shirt": [1.0, 0.0, 0.0]})
        model = ET.fromstring(sdf).find("model")
        self.assertEqual(model.findtext("static"), "true")
        self.assertEqual(model.findall(".//collision"), [])
        names = [e.text for e in model.findall(".//submesh/name")]
        self.assertIn("v2_tshirt02", names)
        self.assertEqual(len(names), 7)
        self.assertEqual(len(model.findall(".//material")), 1)

    def test_unknown_color_is_rejected(self):
        import json
        import tempfile

        scene = json.loads(json.dumps(SCENE))
        scene["guests"]["guest1"]["colors"]["hat"] = [0, 0, 0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            path.write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
