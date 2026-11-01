"""Step planning of base_motion (needs the ROS messages: colcon test)."""

import math

import pytest

from aha_test_orchestration.base_motion import plan_steps, wrap


def test_forward_and_sideways_move_turns_drives_and_turns_back():
    # Facing +y (the shelf); 0.09 m closer and 0.04 m to the right (+x).
    start, goal = (2.95, 2.80, math.pi / 2), (2.99, 2.89, math.pi / 2)
    (turn, heading), (drive, distance), (last, yaw) = plan_steps(start, goal)
    assert (turn, drive, last) == ("turn", "drive", "turn")
    assert heading == pytest.approx(math.atan2(0.09, 0.04))
    assert distance == pytest.approx(math.hypot(0.04, 0.09))
    assert yaw == pytest.approx(math.pi / 2)


def test_target_behind_drives_backwards():
    start, goal = (0.0, 0.0, 0.0), (-0.2, 0.0, 0.0)
    (_, heading), (_, distance), _ = plan_steps(start, goal)
    assert wrap(heading) == pytest.approx(0.0)
    assert distance == pytest.approx(-0.2)


def test_tiny_move_only_turns():
    assert plan_steps((0.0, 0.0, 0.0), (0.005, 0.0, 0.3)) == [("turn", 0.3)]


def test_a_step_that_times_out_reports_that_the_base_was_commanded():
    from aha_test_orchestration.base_motion import BaseMover, MoveOptions

    mover = BaseMover.__new__(BaseMover)
    mover.options = MoveOptions()
    mover.settled_pose = lambda: (0.0, 0.0, 0.0)
    mover.timed = lambda linear, angular, duration: linear == 0.0  # drive stalls
    mover.stop = lambda: None
    report = mover.move_to(0.2, 0.0, 0.0)
    assert not report["ok"] and "timed out" in report["message"]
    assert report["rounds"]  # the caller returns the base from where it stopped
