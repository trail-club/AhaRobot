"""place_server.py's real-robot wiring (frozen readings, gripper ramp, fixed
TCP height), with the methods called on a stand-in node: no ROS spin."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from aha_test_manipulation.frozen_watch import FrozenWatch
from aha_test_manipulation.place_selection import PlaceError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "place_server.py"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("place_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Logger:
    def __init__(self):
        self.lines = []

    def error(self, text, **_):
        self.lines.append(("error", text))

    def warn(self, text, **_):
        self.lines.append(("warn", text))

    def info(self, text, **_):
        self.lines.append(("info", text))


def node(server, **params):
    """A stand-in for PlaceServer with the given parameters."""
    defaults = {
        "gripper_interface": "arm_node",
        "gripper_ramp_time": 1.0,
        "tcp_height": 0.0,
        "max_release_height": 0.04,
    }
    defaults.update(params)
    logger = Logger()
    stub = SimpleNamespace(
        frozen=None,
        frozen_readings={},
        fingers={},
        joints={},
        param=lambda name: defaults[name],
        get_logger=lambda: logger,
        logger=logger,
    )
    for name in (
        "on_bridge_feedback",
        "on_finger_states",
        "finger_state",
        "release_at",
    ):
        method = getattr(server.PlaceServer, name)
        setattr(stub, name, method.__get__(stub))
    return stub


def feedback(goal, names, desired, actual):
    return SimpleNamespace(
        goal_id=SimpleNamespace(uuid=np.array([goal] * 16, dtype=np.uint8)),
        feedback=SimpleNamespace(
            joint_names=list(names),
            desired=SimpleNamespace(positions=list(desired)),
            actual=SimpleNamespace(positions=list(actual)),
        ),
    )


NAMES = ["joint_r2", "joint_r5"]


@pytest.fixture
def clock(server, monkeypatch):
    """Feedback 50 ms apart (place_server stamps it with time.monotonic)."""
    now = [100.0]

    def tick():
        now[0] += 0.05
        return now[0]

    monkeypatch.setattr(server.time, "monotonic", tick)


def freeze_wrist(stub, watch, goal=1):
    """Feed a trajectory whose wrist reading stays at 0.0015 rad."""
    for k in range(30):
        stub.on_bridge_feedback(
            "right",
            watch,
            feedback(goal, NAMES, [0.02 * k, 0.03 * k], [0.02 * k, 0.0015]),
        )


def test_frozen_wrist_is_kept_until_its_reading_changes(server, clock):
    stub = node(server)
    watch = FrozenWatch(0.5, 0.05)
    freeze_wrist(stub, watch)
    assert stub.frozen and "joint_r5" in stub.frozen
    assert stub.frozen_readings == {"joint_r5": 0.0015}
    # The other joints reading live values do not clear it.
    stub.on_finger_states(SimpleNamespace(name=NAMES, position=[0.7, 0.0015]))
    assert stub.frozen
    # The wrist reads a new value (the arm_node restarted): cleared.
    stub.on_finger_states(SimpleNamespace(name=NAMES, position=[0.7, 0.2]))
    assert stub.frozen is None and stub.frozen_readings == {}


def test_a_new_bridge_goal_starts_the_watch_afresh(server, clock):
    stub = node(server)
    watch = FrozenWatch(0.5, 0.05)
    # At rest at the end of goal 1: constant readings and command.
    for k in range(20):
        stub.on_bridge_feedback(
            "right", watch, feedback(1, NAMES, [0.5, 0.3], [0.5, 0.3])
        )
    # Goal 2 starts from there: its first samples must not count the rest.
    stub.on_bridge_feedback("right", watch, feedback(2, NAMES, [0.6, 0.3], [0.5, 0.3]))
    assert stub.frozen is None


def test_motions_are_refused_while_frozen(server):
    stub = node(server)
    stub.frozen = "right arm: frozen readings"
    stub.check_cancel = lambda: None
    run_action = server.PlaceServer.run_action.__get__(stub)
    with pytest.raises(PlaceError, match="frozen readings"):
        run_action(client=None, goal=None, what="pre-place")


class Publisher:
    topic_name = "/right/arm/gripper_joint_command"

    def __init__(self):
        self.sent = []

    def get_subscription_count(self):
        return 1

    def publish(self, msg):
        self.sent.append(msg)


def gripper_stub(server, start, **params):
    stub = node(server, **params)
    stub.gripper_pubs = {"right": Publisher()}
    stub.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))
    if start is not None:
        stub.on_finger_states(SimpleNamespace(name=["joint_r7r"], position=[start]))
    stub.set_gripper = server.PlaceServer.set_gripper.__get__(stub)
    return stub


def test_gripper_ramps_from_the_measured_finger(server, monkeypatch):
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    stub = gripper_stub(server, 0.02)
    stub.set_gripper("right", 0.05)
    sent = stub.gripper_pubs["right"].sent
    values = [m.position_cmd[0] for m in sent]
    assert len(values) == round(1.0 * server.GRIPPER_RAMP_RATE)
    assert values == sorted(values) and values[0] > 0.02
    assert values[-1] == pytest.approx(0.05)
    assert all(list(m.name) == ["joint_r7r"] for m in sent)


def test_gripper_without_a_finger_state_sends_one_command(server, monkeypatch):
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    monkeypatch.setattr(server, "FINGER_STATE_TIMEOUT", 0.0)
    stub = gripper_stub(server, None)
    stub.set_gripper("right", 0.05)
    assert [m.position_cmd[0] for m in stub.gripper_pubs["right"].sent] == [0.05]
    assert any("not ramped" in text for _, text in stub.logger.lines)


def test_release_height_is_the_goals_without_a_fixed_tcp(server):
    stub = node(server, tcp_height=0.0)
    req = SimpleNamespace(release_height=0.01)
    assert stub.release_at(req, (0.5, 0.0, 0.35)) == (0.01, None)
