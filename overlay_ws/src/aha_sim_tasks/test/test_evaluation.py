"""Scoring regressions: sustained evidence, release, history, and deadlines."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.evaluation import Evaluator, WorldState, load_tasks  # noqa: E402

TASKS = load_tasks(Path(__file__).resolve().parents[1] / "config/tasks.json")


def state(t, *, contacts=2, position=(0.962, -0.426, 0.28), speed=0.0):
    return WorldState(t, (0.49, 0, 0), position, speed, contacts)


def test_pick_requires_grasp_and_continuous_height():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    assert evaluator.update(state(1, contacts=0)) is None
    assert evaluator.update(state(2)) is None
    assert evaluator.update(state(2.8, position=(0.962, -0.426, 0.16))) is None
    assert evaluator.update(state(3)) is None
    assert evaluator.update(state(3.5)) is None
    assert evaluator.update(state(4)) == "success"


def test_duplicate_and_out_of_order_samples_do_not_finish_hold():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    evaluator.update(state(1))
    for _ in range(20):
        assert evaluator.update(state(1)) is None
    assert evaluator.update(state(0.5)) is None
    assert evaluator.update(state(1.5)) is None
    assert evaluator.update(state(2)) == "success"


def test_place_requires_previous_pick_release_and_settling():
    evaluator = Evaluator(TASKS["place_apple"], 0)
    target = (1.362, -0.426, 0.16)
    assert evaluator.update(state(1, position=target, contacts=0)) is None
    assert evaluator.update(state(4, position=target, contacts=0)) is None
    assert evaluator.update(state(5)) is None
    assert evaluator.update(state(6, position=target, contacts=2)) is None
    assert evaluator.update(state(7, position=target, contacts=0, speed=0.1)) is None
    assert evaluator.update(state(8, position=target, contacts=0)) is None
    assert evaluator.update(state(8.5, position=target, contacts=0)) is None
    assert evaluator.update(state(9, position=target, contacts=0)) is None
    assert evaluator.update(state(9.5, position=target, contacts=0)) is None
    assert evaluator.update(state(10, position=target, contacts=0)) == "success"


@pytest.mark.parametrize(
    "position", [(1.1, -0.426, 0.16), (1.362, -0.2, 0.16), (1.362, -0.426, 0.04)]
)
def test_place_rejects_wrong_table_or_floor(position):
    evaluator = Evaluator(TASKS["place_apple"], 0)
    evaluator.update(state(1))
    evaluator.update(state(2, position=position, contacts=0))
    assert evaluator.update(state(6, position=position, contacts=0)) is None


@pytest.mark.parametrize(
    "edge_x, edge_y",
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
)
def test_place_accepts_all_tabletop_edges_and_corners(edge_x, edge_y):
    evaluator = Evaluator(TASKS["place_apple"], 0)
    evaluator.update(state(1))
    position = (1.362 + edge_x * 0.12, -0.426 + edge_y * 0.14, 0.16)
    for t in (1.5, 2, 2.5, 3):
        assert evaluator.update(state(t, position=position, contacts=0)) is None
    assert evaluator.update(state(3.5, position=position, contacts=0)) == "success"


@pytest.mark.parametrize(
    "offset_x, offset_y", [(-0.1201, 0), (0.1201, 0), (0, -0.1401), (0, 0.1401)]
)
def test_place_rejects_just_outside_tabletop_and_restarts_hold(offset_x, offset_y):
    evaluator = Evaluator(TASKS["place_apple"], 0)
    evaluator.update(state(1))
    target = (1.362, -0.426, 0.16)
    assert evaluator.update(state(1.5, position=target, contacts=0)) is None
    assert evaluator.update(state(2, position=target, contacts=0)) is None
    outside = (target[0] + offset_x, target[1] + offset_y, target[2])
    for t in (2.5, 3, 3.5, 4, 4.5):
        assert evaluator.update(state(t, position=outside, contacts=0)) is None
    for t in (5, 5.5, 6, 6.5):
        assert evaluator.update(state(t, position=target, contacts=0)) is None
    assert evaluator.update(state(7, position=target, contacts=0)) == "success"


def test_custom_task_can_still_use_scalar_xy_tolerance(tmp_path):
    tasks_path = Path(__file__).resolve().parents[1] / "config/tasks.json"
    entries = json.loads(tasks_path.read_text())
    entries["place_apple"]["xy_tolerance"] = 0.08
    configuration = tmp_path / "tasks.json"
    configuration.write_text(json.dumps(entries))
    scene = json.loads(tasks_path.with_name("scene.json").read_text())
    task = load_tasks(configuration, scene)["place_apple"]
    evaluator = Evaluator(task, 0)
    evaluator.update(state(1))
    for t in (1.5, 2, 2.5, 3, 3.5):
        assert (
            evaluator.update(state(t, position=(1.462, -0.426, 0.16), contacts=0))
            is None
        )


def test_nonfinite_state_resets_confirmation():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    evaluator.update(state(1))
    evaluator.update(state(1.5, position=(0, 0, float("nan"))))
    assert evaluator.update(state(2)) is None
    assert evaluator.update(state(2.5)) is None
    assert evaluator.update(state(3)) == "success"


def test_approach_uses_world_pose_and_heading():
    evaluator = Evaluator(TASKS["approach_apple"], 0)
    assert evaluator.update(replace(state(1), robot_pose=(0, 0, 0))) is None
    assert evaluator.update(replace(state(2), robot_pose=(0.49, 0, 0.2))) is None
    assert evaluator.update(state(3)) is None
    assert evaluator.update(state(3.5)) is None
    assert evaluator.update(state(4)) == "success"


def test_timeout_is_terminal_and_wins_at_deadline():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    evaluator.update(state(59.5))
    assert evaluator.update(state(60.5)) == "timeout"
    assert evaluator.update(state(62)) == "timeout"


def test_gap_in_scoring_samples_restarts_hold():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    assert evaluator.update(state(1)) is None
    assert evaluator.update(state(2)) is None
    assert evaluator.update(state(2.5)) is None
    assert evaluator.update(state(3)) == "success"


def test_pick_rejects_one_finger_contact_and_resets_hold_after_slip():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    for t in (1, 1.5, 2):
        assert evaluator.update(state(t, contacts=1)) is None
    assert evaluator.update(state(2.5)) is None
    assert evaluator.update(state(3, contacts=1)) is None
    for t in (3.5, 4):
        assert evaluator.update(state(t)) is None
    assert evaluator.update(state(4.5)) == "success"


def test_place_requires_release_from_both_fingers():
    evaluator = Evaluator(TASKS["place_apple"], 0)
    evaluator.update(state(1))
    target = (1.362, -0.426, 0.16)
    for t in (1.5, 2, 2.5, 3, 3.5):
        assert evaluator.update(state(t, position=target, contacts=1)) is None
    for t in (4, 4.5, 5, 5.5):
        assert evaluator.update(state(t, position=target, contacts=0)) is None
    assert evaluator.update(state(6, position=target, contacts=0)) == "success"


def test_approach_must_stop_before_success():
    evaluator = Evaluator(TASKS["approach_apple"], 0)
    for t in (1, 1.5, 2):
        assert evaluator.update(replace(state(t), robot_speed=0.1)) is None
    for t in (2.5, 3):
        assert evaluator.update(state(t)) is None
    assert evaluator.update(state(3.5)) == "success"
