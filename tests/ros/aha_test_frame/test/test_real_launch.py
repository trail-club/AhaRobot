"""real.launch.py builds its description and nodes (needs the installed
aha_test_frame, aha_description, aha_servo and aha_perception: colcon test)."""

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


def test_head_and_arm_states_are_the_joint_state_sources(module, monkeypatch):
    monkeypatch.setattr(module, "Node", lambda **kwargs: SimpleNamespace(**kwargs))
    entities = module.generate_launch_description().entities
    (jsp,) = [
        e for e in entities if getattr(e, "executable", None) == "joint_state_publisher"
    ]
    assert jsp.parameters[0]["source_list"] == [
        "/head_controller/joint_states",
        "/left/arm/joint_states",
        "/right/arm/joint_states",
    ]


def test_head_modes(module, monkeypatch):
    for head, executable in (
        ("servo", "servo_trajectory_bridge.py"),
        ("manual", "manual_head.py"),
        ("imu", "imu_head.py"),
    ):
        context = context_with_defaults(module, head=head)
        (node,) = recorded_nodes(module, monkeypatch, module._head, context)
        assert node.executable == executable and not hasattr(node, "remappings")
    context = context_with_defaults(module, head="none")
    assert recorded_nodes(module, monkeypatch, module._head, context) == []


def test_camera_imu_only_for_the_imu_head(module):
    for head, imu in (("imu", "true"), ("servo", "false")):
        (include,) = module._camera(context_with_defaults(module, head=head))
        assert dict(include.launch_arguments)["imu"] == imu
    assert module._camera(context_with_defaults(module, camera="false")) == []


def test_head_port_overrides_the_servo_config(module, monkeypatch):
    context = context_with_defaults(module, head="servo", head_port="/dev/ttyUSB7")
    (node,) = recorded_nodes(module, monkeypatch, module._head, context)
    assert node.parameters[1] == {"port": "/dev/ttyUSB7"}
    assert node.parameters[0].endswith("aha_servo/config/head.yaml")


def test_bridges_run_by_python3_for_the_used_arms(module):
    context = context_with_defaults(module, use_left="false")
    (bridge,) = module._bridges(context)
    cmd = [
        part[0].perform(context) if isinstance(part, list) else str(part)
        for part in bridge.cmd
    ]
    joined = " ".join(cmd)
    assert joined.startswith("python3 ") and "arm_trajectory_bridge.py" in joined
    assert "__node:=aha_right_arm_bridge" in joined and "arm.yaml" in joined
    assert (
        module._bridges(context_with_defaults(module, trajectory_bridge="false")) == []
    )
    assert module._bridges(context_with_defaults(module, arms="false")) == []
