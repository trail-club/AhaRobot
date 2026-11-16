"""GPSR sim walkers against task statuses (no ROS, no Gazebo)."""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.walkers import (  # noqa: E402
    PASSED,
    RUN,
    WAIT,
    SimWalker,
    load_config,
    stage_of,
    step_status,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
SCENE = load_config(CONFIG / "sim" / "walkers" / "gpsr.json")
GEOMETRY = CONFIG.parents[1] / "aha_navigation" / "maps" / "rcjo2026_geometry.json"


def walker(action):
    name = next(n for n, w in SCENE["walkers"].items() if w["action"] == action)
    return SimWalker(name, SCENE["walkers"][name], SCENE)


def status(plan_actions, slot_statuses, plan=True):
    subgoals = [{"id": "plan.generate", "skill": "plan_commands", "status": "x"}]
    subgoals += [
        {"id": f"exec.step{i + 1:02d}", "skill": "run_plan_step", "status": s}
        for i, s in enumerate(slot_statuses)
    ]
    memory = {}
    if plan:
        memory["plan"] = [{"action": a, "params": {}} for a in plan_actions]
    return {"subgoals": subgoals, "memory": memory}


def run(person, seconds, robot=None, dt=0.1):
    for _ in range(round(seconds / dt)):
        person.step(dt, robot)


class StatusTests(unittest.TestCase):
    def test_the_slot_of_the_first_step_with_the_action(self):
        message = status(
            ["navigate", "follow_person", "follow_person"],
            ["succeeded", "running", "pending"],
        )
        self.assertEqual(step_status(message, "follow_person"), "running")
        self.assertIsNone(step_status(message, "guide_person"))
        self.assertIsNone(step_status(status([], [], plan=False), "follow_person"))

    def test_stages(self):
        self.assertEqual(stage_of(None, WAIT), WAIT)
        self.assertEqual(stage_of("pending", RUN), WAIT)  # rewound
        self.assertEqual(stage_of("running", WAIT), RUN)
        self.assertEqual(stage_of("bypassed", WAIT), PASSED)
        self.assertEqual(stage_of("skipped", RUN), RUN)  # failed after walking
        self.assertEqual(stage_of("succeeded", RUN), RUN)


class LeaderTests(unittest.TestCase):
    def test_waits_until_the_step_then_walks_its_route(self):
        leader = walker("follow_person")
        start = tuple(leader.config["start"])
        robot = (start[0], start[1] - 1.2)
        run(leader, 5.0, robot)
        self.assertEqual((leader.x, leader.y), start)
        leader.set_status("running")
        run(leader, 1.0, robot)
        self.assertAlmostEqual(
            math.dist(start, (leader.x, leader.y)), SCENE["walk_speed"], places=6
        )

    def test_stops_when_the_step_ends_without_success(self):
        leader = walker("follow_person")
        start = tuple(leader.config["start"])
        robot = (start[0], start[1] - 1.2)
        leader.set_status("running")
        run(leader, 1.0, robot)
        leader.set_status("failed")
        stopped = (leader.x, leader.y)
        run(leader, 3.0, robot)
        self.assertEqual((leader.x, leader.y), stopped)

    def test_waits_for_a_robot_left_behind(self):
        leader = walker("follow_person")
        leader.set_status("running")
        far = (leader.x + SCENE["wait_gap"] + 1.0, leader.y)
        before = (leader.x, leader.y)
        run(leader, 3.0, far)
        self.assertEqual((leader.x, leader.y), before)
        # Following closely, it reaches the end of the route and faces the robot.
        for _ in range(600):
            robot = (leader.x + 1.0, leader.y)
            leader.step(0.1, robot)
        self.assertEqual([leader.x, leader.y], leader.config["route"][-1])
        self.assertAlmostEqual(leader.yaw, 0.0)

    def test_a_passed_step_moves_to_the_end_and_a_rewind_back(self):
        leader = walker("follow_person")
        leader.set_status("bypassed")
        self.assertEqual([leader.x, leader.y], leader.config["route"][-1])
        leader.set_status("pending")
        self.assertEqual([leader.x, leader.y], leader.config["start"])
        self.assertEqual(leader.yaw, leader.config["yaw"])


class FollowerTests(unittest.TestCase):
    def test_follows_the_robot_path_while_the_step_runs(self):
        person = walker("guide_person")
        start = tuple(person.config["start"])
        robot = (start[0], start[1] - 1.1)
        person.set_status("running")
        run(person, 3.0, robot)
        self.assertEqual((person.x, person.y), start)  # the robot is close
        path = [(robot[0] - 0.05 * i, robot[1]) for i in range(60)]
        for point in path:
            person.step(0.1, point)
        run(person, 5.0, path[-1])
        gap = math.dist((person.x, person.y), path[-1])
        self.assertAlmostEqual(gap, SCENE["follow_gap"], delta=0.16)
        # Stands once the step is over.
        person.set_status("succeeded")
        before = (person.x, person.y)
        run(person, 3.0, (path[-1][0] - 3.0, path[-1][1]))
        self.assertEqual((person.x, person.y), before)


class SceneTests(unittest.TestCase):
    def test_walkers_stand_and_walk_in_free_space(self):
        geometry = json.loads(GEOMETRY.read_text())
        bounds = geometry["free_bounds"]
        boxes = [o for o in geometry["obstacles"] if o["name"] != "door"]

        def inside(point, box, margin=0.2):
            x, y, yaw = box["pose"]
            dx, dy = point[0] - x, point[1] - y
            u = dx * math.cos(yaw) + dy * math.sin(yaw)
            v = -dx * math.sin(yaw) + dy * math.cos(yaw)
            sx, sy = box["size"]
            return abs(u) <= sx / 2 + margin and abs(v) <= sy / 2 + margin

        for name, config in SCENE["walkers"].items():
            points = [config["start"], *config["route"]]
            for a, b in zip(points, points[1:] or points):
                for i in range(21):
                    p = (a[0] + (b[0] - a[0]) * i / 20, a[1] + (b[1] - a[1]) * i / 20)
                    with self.subTest(name, point=p):
                        self.assertTrue(bounds[0] < p[0] < bounds[2])
                        self.assertTrue(bounds[1] < p[1] < bounds[3])
                        self.assertFalse(any(inside(p, box) for box in boxes))

    def test_one_walker_per_action(self):
        actions = [w["action"] for w in SCENE["walkers"].values()]
        self.assertEqual(sorted(actions), ["follow_person", "guide_person"])

    def test_invalid_scenes_are_rejected(self):
        bad = [
            {**SCENE, "extra": 1},
            {**SCENE, "wait_gap": 0},
            {
                **SCENE,
                "walkers": {"w": {**SCENE["walkers"]["walker_leader"], "route": []}},
            },
            {
                **SCENE,
                "walkers": {
                    "w": {**SCENE["walkers"]["walker_leader"], "action": "dance"}
                },
            },
        ]
        for scene in bad:
            with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
                json.dump(scene, file)
                file.flush()
                with self.subTest(scene=scene), self.assertRaises(ValueError):
                    load_config(file.name)


if __name__ == "__main__":
    unittest.main()
