"""config/arm.yaml against the arm_node launch and its joint limits (no ROS)."""

import math
import re
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import yaml

from aha_manipulation.arm_map import parse_config

PKG = Path(__file__).resolve().parents[1]
SOURCE = PKG.parent
REPO = SOURCE.parents[1]
CONFIG = yaml.safe_load((PKG / "config/arm.yaml").read_text())


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


def test_head_launch_merges_the_arm_states():
    # The bridge reads only the arm_node's topic; /joint_states gets the arms
    # from joint_state_publisher instead of zeros.
    path = SOURCE / "aha_perception/launch/real_head_camera.launch.py"
    spec = importlib.util.spec_from_file_location("real_head_camera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = arms_launch()
    assert module.ARM_STATES == [launch.state_topic(side) for side in launch.ARMS]


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
        parse_config({**params, "joints": params["joints"][:4]})
    with pytest.raises(ValueError, match="below max"):
        parse_config({**params, "joint_l2": {"min": 1.0, "max": 0.0}})
    with pytest.raises(ValueError, match="positive"):
        parse_config({**params, "rate_hz": 0})
    with pytest.raises(ValueError, match="negative"):
        parse_config({**params, "frozen_timeout": -1.0})


def test_frozen_readings_are_detected_but_not_stale():
    path = PKG / "scripts/arm_trajectory_bridge.py"
    spec = importlib.util.spec_from_file_location("arm_trajectory_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = parse_config(node_params("aha_right_arm_bridge"))
    assert cfg.frozen_timeout > 0
    names = cfg.joint_names
    bridge = SimpleNamespace(cfg=cfg, names=names, state_times={}, change_times={})
    bridge._frozen = lambda now: module.ArmTrajectoryBridge._frozen(bridge, now)
    stale = lambda now: module.ArmTrajectoryBridge._stale(bridge, now)  # noqa: E731
    bridge.state_times = {n: 10.0 for n in names}
    bridge.change_times = {n: 10.0 - cfg.frozen_timeout - 0.1 for n in names}
    bridge.change_times[names[0]] = 9.5  # one joint changed recently
    assert not bridge._frozen(10.0)
    bridge.change_times[names[0]] = 10.0 - cfg.frozen_timeout - 0.1
    assert bridge._frozen(10.0)
    assert stale(10.0) == []  # an arm at rest reads the same values
    off = SimpleNamespace(
        **{
            **vars(bridge),
            "cfg": parse_config(
                {**node_params("aha_right_arm_bridge"), "frozen_timeout": 0.0}
            ),
        }
    )
    assert not module.ArmTrajectoryBridge._frozen(off, 10.0)
