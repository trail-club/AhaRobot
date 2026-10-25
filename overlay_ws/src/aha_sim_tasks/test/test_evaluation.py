"""Scoring needs sustained grasp, whole-object placement, support, and settling."""

from dataclasses import replace
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.evaluation import Evaluator, WorldState, load_tasks  # noqa: E402
from aha_sim_tasks.scene import bounds_corners, load_scene, sample_scene  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config/tasks.json"
SCENE = load_scene(CONFIG.with_name("scene.json"))
TASKS = load_tasks(CONFIG, SCENE)
TARGET = tuple(TASKS["place_object"].parameters["object_target"])


def state(
    t,
    *,
    contacts=2,
    position=(0.962, -0.426, 1.05),
    speed=0.0,
    supported=True,
    **kwargs,
):
    return WorldState(
        t,
        (0.49, 0, 0),
        position,
        speed,
        contacts,
        destination_contact=supported,
        **kwargs,
    )


def picked(evaluator):
    for t in (0.5, 1.0, 1.5):
        evaluator.update(state(t))
    assert evaluator.was_lifted


def settle(evaluator, start=2, **kwargs):
    result = None
    for step in range(5):
        result = evaluator.update(
            state(start + 0.5 * step, contacts=0, position=TARGET, **kwargs)
        )
    return result


def test_pick_requires_continuous_grasp_and_clearance():
    evaluator = Evaluator(TASKS["pick_object"], 0)
    assert evaluator.update(state(1, contacts=0)) is None
    assert evaluator.update(state(2)) is None
    assert (
        evaluator.update(state(2.5, position=tuple(SCENE["source_position"]))) is None
    )
    assert evaluator.update(state(3)) is None
    assert evaluator.update(state(3.5)) is None
    assert evaluator.update(state(4)) == "success"


def test_duplicate_out_of_order_and_missing_samples_do_not_finish_hold():
    evaluator = Evaluator(TASKS["pick_object"], 0)
    evaluator.update(state(1))
    for _ in range(20):
        assert evaluator.update(state(1)) is None
    assert evaluator.update(state(0.5)) is None
    assert evaluator.update(state(2)) is None  # Gap restarts the hold.
    assert evaluator.update(state(2.5)) is None
    assert evaluator.update(state(3)) == "success"


def test_place_requires_sustained_pick_release_and_settling():
    evaluator = Evaluator(TASKS["place_object"], 0)
    assert settle(evaluator, start=0) is None  # Already at destination isn't a pick.
    evaluator.update(state(3))
    evaluator.update(state(3.5))
    assert not evaluator.was_lifted
    evaluator.update(state(4))
    assert evaluator.was_lifted
    assert evaluator.update(state(4.5, position=TARGET, contacts=2)) is None
    assert evaluator.update(state(5, position=TARGET, contacts=0, speed=0.1)) is None
    assert settle(evaluator, start=5.5) == "success"


@pytest.mark.parametrize(
    "supported,contacts,speed,angular",
    [
        (False, 0, 0, 0),
        (True, 1, 0, 0),
        (True, 0, 0.1, 0),
        (True, 0, 0, 0.3),
    ],
)
def test_place_rejects_missing_support_contact_motion_or_partial_release(
    supported, contacts, speed, angular
):
    evaluator = Evaluator(TASKS["place_object"], 0)
    picked(evaluator)
    for step in range(6):
        assert (
            evaluator.update(
                state(
                    2 + step * 0.5,
                    position=TARGET,
                    contacts=contacts,
                    supported=supported,
                    speed=speed,
                    object_angular_speed=angular,
                )
            )
            is None
        )


@pytest.mark.parametrize("axis,direction", [(0, -1), (0, 1), (1, -1), (1, 1)])
def test_placement_requires_the_whole_footprint_inside_table(axis, direction):
    evaluator = Evaluator(TASKS["place_object"], 0)
    picked(evaluator)
    position = list(TARGET)
    # Its centre is still on the table, but half the object overhangs.
    position[axis] += direction * SCENE["table_size"][axis] / 2
    for t in (2, 2.5, 3, 3.5, 4):
        assert evaluator.update(state(t, contacts=0, position=tuple(position))) is None
    assert not evaluator.metrics["footprint_inside"]
    assert settle(evaluator, start=4.5) == "success"


def test_rotation_changes_the_placement_footprint():
    parameters = {
        **TASKS["place_object"].parameters,
        "object_size": [0.04, 0.10, 0.04],
        "object_points": bounds_corners([0.04, 0.10, 0.04]),
    }
    evaluator = Evaluator(replace(TASKS["place_object"], parameters=parameters), 0)
    picked(evaluator)
    position = (TARGET[0] + 0.215, TARGET[1], TARGET[2])
    assert evaluator.update(state(2, contacts=0, position=position)) is None
    assert evaluator.metrics["footprint_inside"]
    angle = math.pi / 4
    orientation = (0, 0, math.sin(angle / 2), math.cos(angle / 2))
    evaluator.update(
        state(2.5, contacts=0, position=position, object_orientation=orientation)
    )
    assert not evaluator.metrics["footprint_inside"]


def test_rotated_round_object_is_scored_from_mesh_instead_of_box_corners():
    scene = sample_scene(SCENE, "orange")
    task = load_tasks(CONFIG, scene)["place_object"]
    evaluator = Evaluator(task, 0)
    picked(evaluator)
    target = tuple(task.parameters["object_target"])
    orientation = (
        0.06623920175569079,
        -0.6709919155966912,
        0.4468046373380889,
        0.5880032596915006,
    )
    for t in (2, 2.5, 3, 3.5):
        assert (
            evaluator.update(
                state(t, position=target, contacts=0, object_orientation=orientation)
            )
            is None
        )
    assert (
        evaluator.update(
            state(4, position=target, contacts=0, object_orientation=orientation)
        )
        == "success"
    )


@pytest.mark.parametrize(
    "position", [(0.962, -0.426, 0.78), (1.922, -0.426, 0.04), (1.922, 0.2, 0.78)]
)
def test_wrong_table_floor_or_outside_region_cannot_succeed(position):
    evaluator = Evaluator(TASKS["place_object"], 0)
    picked(evaluator)
    for t in (2, 2.5, 3, 3.5, 4):
        assert evaluator.update(state(t, position=position, contacts=0)) is None


@pytest.mark.parametrize("object_id", ["apple", "orange", "soup_can", "gelatin_box"])
def test_same_criteria_work_for_different_object_sizes_and_table_heights(object_id):
    scene = sample_scene(SCENE, object_id)
    for prefix in ("source", "destination"):
        scene[prefix + "_table_position"][2] += 0.15
        scene[prefix + "_position"][2] += 0.15
    task = load_tasks(CONFIG, scene)["place_object"]
    evaluator = Evaluator(task, 0)
    for t in (0.5, 1, 1.5):
        evaluator.update(state(t, position=(0.962, -0.426, 1.25)))
    target = tuple(task.parameters["object_target"])
    for t in (2, 2.5, 3, 3.5):
        assert evaluator.update(state(t, contacts=0, position=target)) is None
    assert evaluator.update(state(4, contacts=0, position=target)) == "success"


def test_alternating_finger_contacts_do_not_count_as_a_grasp():
    evaluator = Evaluator(TASKS["pick_object"], 0)
    for t in (1, 1.5, 2, 2.5):
        assert evaluator.update(state(t, simultaneous_grasp=False)) is None


def test_nonfinite_state_and_slip_restart_confirmation():
    evaluator = Evaluator(TASKS["pick_object"], 0)
    evaluator.update(state(1))
    evaluator.update(state(1.5, position=(0, 0, float("nan"))))
    assert evaluator.update(state(2)) is None
    assert evaluator.update(state(2.5, contacts=1)) is None
    for t in (3, 3.5):
        assert evaluator.update(state(t)) is None
    assert evaluator.update(state(4)) == "success"


def test_timeout_wins_at_deadline():
    evaluator = Evaluator(TASKS["pick_object"], 0)
    deadline = evaluator.task.timeout
    evaluator.update(state(deadline - 0.5))
    assert evaluator.update(state(deadline)) == "timeout"
    assert evaluator.update(state(deadline + 2)) == "timeout"


def test_approach_requires_correct_pose_heading_and_stopping():
    evaluator = Evaluator(TASKS["approach_object"], 0)
    assert evaluator.update(replace(state(1), robot_pose=(0, 0, 0))) is None
    assert evaluator.update(replace(state(1.5), robot_pose=(0.49, 0, 0.2))) is None
    assert evaluator.update(replace(state(2), robot_speed=0.1)) is None
    for t in (2.5, 3):
        assert evaluator.update(state(t)) is None
    assert evaluator.update(state(3.5)) == "success"


def test_legacy_names_keep_the_generalized_criteria():
    assert TASKS["pick_apple"].parameters == TASKS["pick_object"].parameters
