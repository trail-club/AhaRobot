"""Task definitions under config/: loadable, and consistent with each other."""

import json
import math
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core import load_spec, select  # noqa: E402
from aha_tasks.runner import load_task  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config"
TASKS = sorted(CONFIG.glob("*.json"))
# Total Score of each score sheet (rulebook 2027 Rev-1, 2026-10-14).
SCORE_SHEET = {"pick_and_place": 3515, "gpsr": 1490, "doing_laundry": 4415}
MAP = CONFIG.parents[1] / "aha_navigation" / "maps"
# Nav2 needs room to reach a goal: robot_radius 0.40 m, inflation_radius 0.65 m.
MIN_CLEARANCE = 0.6  # m
PICK_KEYS = {"pick_object": "object", "pick_cloth": "cloth"}
PLACE_KEYS = {"place_object": "object", "place_cloth": "cloth"}


def occupied_cells():
    """Centres of the non-free cells of the RCJO2026 map, in map coordinates."""
    resolution, (origin_x, origin_y) = 0.05, (-5.5, -4.5)
    raw = (MAP / "rcjo2026.pgm").read_bytes()
    _, width, height, _ = raw.split(maxsplit=4)[:4]
    width, height = int(width), int(height)
    pixels = raw[-width * height :]
    return [
        (origin_x + (i + 0.5) * resolution, origin_y + (j + 0.5) * resolution)
        for j in range(height)
        for i in range(width)
        if pixels[(height - 1 - j) * width + i] < 250
    ]


def steps_of(spec):
    """(skill, params) of every subgoal and of the steps of mock plans."""
    for subgoal in spec.subgoals:
        yield subgoal.skill, subgoal.params
        for step in subgoal.mock.get("plan", []):
            yield step["action"], step["params"]


class TaskDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.specs = {path.stem: load_spec(path) for path in TASKS}

    def test_every_task_loads_with_its_skills(self):
        self.assertLessEqual(set(SCORE_SHEET), set(self.specs))
        for path in TASKS:
            with self.subTest(path.stem):
                load_task(path)

    def test_task_matches_its_file_name(self):
        for name, spec in self.specs.items():
            self.assertEqual(spec.task, name)

    def test_points_match_the_score_sheet(self):
        for name, total in SCORE_SHEET.items():
            with self.subTest(name):
                points = sum(s.points for s in self.specs[name].subgoals)
                self.assertEqual(points, total)

    def test_a_skill_name_has_one_implementation(self):
        bindings = {}
        for spec in self.specs.values():
            for name, target in spec.skills.items():
                if target is not None:
                    bindings.setdefault(name, set()).add(target)
        for name, targets in bindings.items():
            self.assertEqual(len(targets), 1, name)

    def test_a_skill_name_has_one_set_of_params(self):
        """Skills are shared between tasks once implemented."""
        keys = {}
        for spec in self.specs.values():
            for skill, params in steps_of(spec):
                keys.setdefault(skill, {}).setdefault(frozenset(params), spec.task)
        # Which keys are given depends on the description in the command.
        optional = {"find_described_person"}
        for skill in set(keys) - optional:
            self.assertEqual(len(keys[skill]), 1, f"{skill}: {keys[skill]}")

    def test_declared_skills_are_used(self):
        for spec in self.specs.values():
            if any(s.skill == "run_plan_step" for s in spec.subgoals):
                continue  # A plan may use any of them.
            used = {s.skill for s in spec.subgoals}
            self.assertEqual(set(spec.skills) - used, set(), spec.task)

    def test_objects_are_held_from_pick_to_place(self):
        for spec in self.specs.values():
            holding = spec.initial_state.get("robot.holding")
            for subgoal in spec.subgoals:
                skill, mock = subgoal.skill, subgoal.mock
                if skill in PICK_KEYS:
                    self.assertIsNone(holding, subgoal.id)
                    self.assertEqual(
                        mock["robot.holding"], subgoal.params[PICK_KEYS[skill]]
                    )
                if skill in PLACE_KEYS:
                    self.assertEqual(
                        holding, subgoal.params[PLACE_KEYS[skill]], subgoal.id
                    )
                    self.assertIsNone(mock["robot.holding"], subgoal.id)
                holding = mock.get("robot.holding", holding)
            self.assertIsNone(holding, spec.task)

    def test_groups_select_for_partial_runs(self):
        selections = {
            "pick_and_place": ("dishwasher.*", 5),
            "gpsr": ("exec.*", 20),
            "doing_laundry": ("start.*,shirt1.*", 8),
        }
        for name, (expression, count) in selections.items():
            with self.subTest(name):
                self.assertEqual(len(select(self.specs[name], expression)), count)

    def test_locations_keep_clear_of_obstacles(self):
        occupied = occupied_cells()
        for spec in self.specs.values():
            for label, (x, y, _) in spec.locations.items():
                clearance = min(math.dist((x, y), cell) for cell in occupied)
                self.assertGreaterEqual(
                    clearance, MIN_CLEARANCE, f"{spec.task}: {label}"
                )

    def test_rulebook_is_the_same_revision(self):
        revisions = {
            json.dumps(
                {k: spec.rulebook[k] for k in ("name", "version", "revision", "url")}
            )
            for spec in self.specs.values()
        }
        self.assertEqual(len(revisions), 1)


if __name__ == "__main__":
    unittest.main()
