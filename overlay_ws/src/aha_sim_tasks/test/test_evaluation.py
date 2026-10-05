"""Scoring regressions: sustained evidence, release, history, and deadlines."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.evaluation import Evaluator, WorldState, load_tasks  # noqa: E402

TASKS = load_tasks(Path(__file__).resolve().parents[1] / "config/tasks.json")


def state(t, *, grasped=True, position=(0.962, -0.426, 0.28), speed=0.0):
    return WorldState(t, (0.49, 0, 0), position, speed, grasped)


def test_pick_requires_grasp_and_continuous_height():
    evaluator = Evaluator(TASKS["pick_apple"], 0)
    assert evaluator.update(state(1, grasped=False)) is None
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
    assert evaluator.update(state(1, position=target, grasped=False)) is None
    assert evaluator.update(state(4, position=target, grasped=False)) is None
    assert evaluator.update(state(5)) is None
    assert evaluator.update(state(6, position=target, grasped=True)) is None
    assert evaluator.update(state(7, position=target, grasped=False, speed=0.1)) is None
    assert evaluator.update(state(8, position=target, grasped=False)) is None
    assert evaluator.update(state(8.5, position=target, grasped=False)) is None
    assert evaluator.update(state(9, position=target, grasped=False)) is None
    assert evaluator.update(state(9.5, position=target, grasped=False)) is None
    assert evaluator.update(state(10, position=target, grasped=False)) == "success"


@pytest.mark.parametrize(
    "position", [(1.1, -0.426, 0.16), (1.362, -0.2, 0.16), (1.362, -0.426, 0.04)]
)
def test_place_rejects_wrong_table_or_floor(position):
    evaluator = Evaluator(TASKS["place_apple"], 0)
    evaluator.update(state(1))
    evaluator.update(state(2, position=position, grasped=False))
    assert evaluator.update(state(6, position=position, grasped=False)) is None


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


def test_approach_must_stop_before_success():
    evaluator = Evaluator(TASKS["approach_apple"], 0)
    for t in (1, 1.5, 2):
        assert evaluator.update(replace(state(t), robot_speed=0.1)) is None
    for t in (2.5, 3):
        assert evaluator.update(state(t)) is None
    assert evaluator.update(state(3.5)) == "success"
