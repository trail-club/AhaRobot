"""Slow inference must consume wall time only; scoring follows physics samples."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.api import Action  # noqa: E402
from aha_sim_tasks.evaluation import Evaluator, Task, WorldState  # noqa: E402
from aha_sim_tasks import stepping  # noqa: E402


def test_slow_policy_is_called_each_step_and_commands_are_refreshed(monkeypatch):
    rclpy = pytest.importorskip("rclpy")
    wall = 0.0
    received = 0.0
    pending_samples = 0
    observations, commands = [], []
    node = SimpleNamespace(
        world=WorldState(0, (0, 0, 0), (0, 0, 0), 0, 0, 0),
        observation_ready=lambda: True,
        observation=lambda: node.world.sim_time,
        apply=commands.append,
        start_episode=lambda: None,
        fresh=lambda: wall - received < 2,
    )

    def act(observation):
        nonlocal wall
        observations.append(observation)
        policy.pending = True
        wall += 3  # Longer than the continuous evaluator's stream freshness limit.

    def poll():
        if policy.pending:
            policy.pending = False
            return Action(linear_velocity=0.05)

    def advance():
        nonlocal pending_samples
        pending_samples = 2

    def spin_once(node, timeout_sec):
        nonlocal wall, received, pending_samples
        wall += 0.01
        if pending_samples:
            pending_samples -= 1
            received = wall
            node.world = WorldState(
                round(node.world.sim_time + 0.05, 3), (0, 0, 0), (0, 0, 0), 0, 0, 0
            )

    policy = SimpleNamespace(pending=False, act=act, poll=poll)
    monkeypatch.setattr(stepping.time, "monotonic", lambda: wall)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(rclpy, "spin_once", spin_once)
    task = Task(
        "approach",
        "Approach",
        "approach",
        10,
        0.25,
        {
            "robot_target": (0, 0, 0),
            "position_tolerance": 0.04,
            "yaw_tolerance": 0.08,
            "max_robot_speed": 0.02,
        },
    )
    evaluator = Evaluator(task, 0)
    args = SimpleNamespace(step_duration=0.2, wall_timeout=60, camera_timeout=10)
    status = stepping.run_steps(
        args,
        node,
        policy,
        SimpleNamespace(poll=lambda: None),
        evaluator,
        SimpleNamespace(quantum=0.1, advance=advance),
    )
    assert status == "success"
    assert observations == [0, 0.2]
    assert len(commands) == 3
    assert wall >= 6 and node.world.sim_time == 0.3


def test_world_control_rejection_is_explicit():
    stepper = stepping.GazeboStepper.__new__(stepping.GazeboStepper)
    stepper.control_type = lambda **kwargs: kwargs
    stepper.response_type = object
    stepper.node = SimpleNamespace(
        request=lambda *args: (True, SimpleNamespace(data=False))
    )
    with pytest.raises(RuntimeError, match="world control"):
        stepper.advance()
