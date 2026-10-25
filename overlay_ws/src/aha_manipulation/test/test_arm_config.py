"""config/arm.yaml and lift.yaml against the launch and the drivers' limits (no ROS)."""

import math
import re
from pathlib import Path
import importlib.util

import pytest
import yaml

from aha_manipulation.arm_map import parse_config

PKG = Path(__file__).resolve().parents[1]
SOURCE = PKG.parent
REPO = SOURCE.parents[1]
CONFIG = yaml.safe_load((PKG / "config/arm.yaml").read_text())
LIFT = yaml.safe_load((PKG / "config/lift.yaml").read_text())["aha_lift_bridge"][
    "ros__parameters"
]


def node_params(node):
    return {**CONFIG["/**"]["ros__parameters"], **CONFIG[node]["ros__parameters"]}


def arms_launch():
    path = SOURCE / "aha_bringup/launch/arms.launch.py"
    spec = importlib.util.spec_from_file_location("arms_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_profiles():
    """{side: (min, max)} of the arm joints from the arm_node patch, in rad."""
    text = (REPO / "patches/astra_controller-trail-club-ver1004.patch").read_text()
    profiles = {}
    for side in ("left", "right"):
        block = text[text.index(f'+        "{side}": {{') :]
        bounds = [
            [
                math.radians(float(v))
                for v in re.search(
                    rf'"{key}": np\.array\(\[\*np\.radians\(\[([^\]]+)\]\)', block
                )
                .group(1)
                .split(",")
            ]
            for key in ("min", "max")
        ]
        profiles[side] = tuple(bounds)
    return profiles


@pytest.mark.parametrize("side", ["left", "right"])
def test_bridge_config_matches_the_arm_node(side):
    cfg = parse_config(node_params(f"aha_{side}_arm_bridge"))
    joints, _grippers = arms_launch().ARMS[side]
    assert cfg.joint_names == joints
    assert cfg.controller_name == f"{side}_arm_controller"
    assert cfg.command_topic == f"/{side}/arm/joint_command"
    assert cfg.state_topic == arms_launch().state_topic(side)
    # One JointCommand with all five joints, as the arm_node requires.
    assert [(g.topic, list(g.names)) for g in cfg.groups] == [
        (cfg.command_topic, joints)
    ]
    assert cfg.state_topics == (cfg.state_topic,)


def test_lift_config_matches_the_lift_nodes():
    cfg = parse_config(LIFT)
    launch = arms_launch()
    assert cfg.controller_name == "lift_controller"
    assert cfg.joint_names == [launch.LIFTS[side] for side in ("left", "right")]
    # lift_node takes a JointCommand with only its own joint.
    assert [(g.topic, g.names) for g in cfg.groups] == [
        (f"/{side}/lift/joint_command", (launch.LIFTS[side],))
        for side in ("left", "right")
    ]
    assert cfg.state_topics == tuple(
        launch.lift_state_topic(side) for side in ("left", "right")
    )


def test_lift_limits_are_inside_the_lift_firmware():
    # lift_node: 0 <= pos <= RAIL_MAX_LENGTH_MM / 1000; AstraLiftController:
    # MAX_SPEED_MM 300, ACCELERATION_MM 100.
    for joint in parse_config(LIFT).joints:
        assert 0.0 <= joint.min < joint.max <= 1.19, joint.name
        assert joint.max_velocity <= 0.3, joint.name
        assert joint.max_acceleration <= 0.1, joint.name
        assert joint.unit == "m", joint.name
    # The board plans its own stop with ACCELERATION_MM; the bridge leads by it.
    cfg = parse_config(LIFT)
    assert cfg.board_accelerations == {j.name: 0.1 for j in cfg.joints}


def test_head_launch_merges_the_arm_states():
    # The bridge reads only the arm_node's topic; /joint_states gets the arms
    # from joint_state_publisher instead of zeros.
    path = SOURCE / "aha_perception/launch/real_head_camera.launch.py"
    spec = importlib.util.spec_from_file_location("real_head_camera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = arms_launch()
    assert module.ARM_STATES == [
        *(launch.state_topic(side) for side in launch.ARMS),
        *(launch.lift_state_topic(side) for side in launch.LIFTS),
    ]


@pytest.mark.parametrize("side", ["left", "right"])
def test_position_limits_are_inside_the_arm_node_limits(side):
    cfg = parse_config(node_params(f"aha_{side}_arm_bridge"))
    lows, highs = patch_profiles()[side]
    for joint, lo, hi in zip(cfg.joints, lows, highs):
        assert lo <= joint.min < lo + 1e-4, joint.name
        assert hi - 1e-4 < joint.max <= hi, joint.name


def test_parse_config_rejects_bad_values():
    params = node_params("aha_left_arm_bridge")
    with pytest.raises(ValueError, match="joints"):
        parse_config({**params, "joints": []})
    with pytest.raises(ValueError, match="joints"):
        parse_config({**params, "joints": ["joint_l2", "joint_l2"]})
    with pytest.raises(ValueError, match="below max"):
        parse_config({**params, "joint_l2": {"min": 1.0, "max": 0.0}})
    with pytest.raises(ValueError, match="positive"):
        parse_config({**params, "rate_hz": 0})
    with pytest.raises(ValueError, match="board_acceleration"):
        parse_config({**params, "joint_l2": {"board_acceleration": -1.0}})
    with pytest.raises(ValueError, match="over `board_acceleration`"):
        parse_config(
            {**params, "joint_l2": {"max_acceleration": 2.0, "board_acceleration": 1.0}}
        )
