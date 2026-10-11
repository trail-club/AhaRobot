"""Policy adapter and command validation failures must be explicit."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.api import Action, Observation  # noqa: E402
from aha_sim_tasks.policies import load_policy, policy_factory  # noqa: E402


@pytest.mark.parametrize(
    "action",
    [
        Action(linear_velocity=float("nan")),
        Action(angular_velocity=0.6),
        Action(joint_positions={"joint_r1": 1.3}),
        Action(joint_positions={"joint_r7l": 0.02}),
        Action(joint_positions={"missing_joint": 0}),
    ],
)
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError):
        action.validate()


def test_scripted_policy_waits_for_lift_and_gripper_before_driving():
    policy = load_policy("scripted")
    policy.reset("place_apple", "Place the apple.")
    observation = Observation(
        0,
        "place_apple",
        "Place the apple.",
        {
            "joint_r1": 0,
            "joint_r7r": 0,
            "joint_r7l": 0,
        },
        {},
        (0, 0, 0),
    )
    action = policy.act(observation)
    action.validate()
    assert action.linear_velocity == 0
    assert action.joint_positions["joint_r1"] > 0
    assert action.joint_positions["joint_r7r"] == 0.06


def test_module_factory_loading_and_protocol(monkeypatch):
    import types

    monkeypatch.setitem(
        sys.modules,
        "example_policy",
        types.SimpleNamespace(
            create=lambda: load_policy("noop"),
            bad=lambda: object(),
        ),
    )
    policy = load_policy("example_policy:create")
    policy.reset("pick_apple", "Pick.")
    assert policy.act(None) == Action()
    with pytest.raises(TypeError, match="implement reset"):
        load_policy("example_policy:bad")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="module:factory"):
        load_policy("unknown")


def test_factory_validation_does_not_initialize_policy(monkeypatch):
    import types

    calls = []

    def factory():
        calls.append(True)
        return load_policy("noop")

    monkeypatch.setitem(
        sys.modules, "deferred_policy", types.SimpleNamespace(create=factory)
    )
    assert policy_factory("deferred_policy:create") is factory
    assert calls == []
    load_policy("deferred_policy:create")
    assert calls == [True]


def test_scripted_pick_accepts_finger_stopped_by_object_contact():
    policy = load_policy("scripted")
    policy.reset("pick_apple", "Pick.")
    policy.phase = "close"
    policy.phase_time = 0.0
    policy.start_pose = (0, 0, 0)
    observation = Observation(
        2,
        "pick_apple",
        "Pick.",
        {
            "joint_r1": 0.05,
            "joint_r7r": 0.045,
            "joint_r7l": -0.035,
        },
        {},
        (0.49, 0, 0),
    )
    policy.act(observation)
    assert policy.phase == "lift"
    assert policy.act(observation).joint_positions["joint_r1"] == 0.18
