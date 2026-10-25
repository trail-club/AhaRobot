"""real.launch.py builds its description and nodes (needs the installed
aha_description, aha_servo and aha_perception: colcon test)."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument

SCRIPT = Path(__file__).resolve().parents[1] / "launch" / "real.launch.py"


@pytest.fixture
def module(monkeypatch):
    spec = importlib.util.spec_from_file_location("real_launch", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def context_with_defaults(module, **overrides):
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations.update(overrides)
    return context


def recorded_nodes(module, monkeypatch, function, context):
    monkeypatch.setattr(module, "Node", lambda **kwargs: SimpleNamespace(**kwargs))
    return function(context)


def test_description_is_the_frame_model(module, monkeypatch):
    context = context_with_defaults(module)
    (rsp,) = recorded_nodes(module, monkeypatch, module._robot, context)
    root = ET.fromstring(rsp.parameters[0]["robot_description"])
    assert root.find("joint[@name='joint_r1']").get("type") == "fixed"
    assert root.find("link[@name='frame_column']") is not None
    base = root.find("joint[@name='base_footprint_to_base_link']/origin")
    assert float(base.get("xyz").split()[2]) > 0.3  # on the table


def test_head_state_joins_the_joint_states(module, monkeypatch):
    for head in ("servo", "manual"):
        context = context_with_defaults(module, head=head)
        (node,) = recorded_nodes(module, monkeypatch, module._head, context)
        assert node.remappings == [(module.HEAD_STATES, "/joint_states")]
    context = context_with_defaults(module, head="none")
    assert recorded_nodes(module, monkeypatch, module._head, context) == []


def test_head_port_overrides_the_servo_config(module, monkeypatch):
    context = context_with_defaults(module, head="servo", head_port="/dev/ttyUSB7")
    (node,) = recorded_nodes(module, monkeypatch, module._head, context)
    assert node.parameters[1] == {"port": "/dev/ttyUSB7"}
    assert node.parameters[0].endswith("aha_servo/config/head.yaml")
