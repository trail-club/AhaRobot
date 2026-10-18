"""joint_state_bounds.py helpers (no ROS transport)."""

import importlib.util
from pathlib import Path

from sensor_msgs.msg import JointState

SHARE = Path(__file__).resolve().parents[1]
SCRIPTS = SHARE / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bounds = load("joint_state_bounds")


def test_position_limits_from_urdf():
    urdf = """<robot name="r">
      <joint name="lift" type="prismatic"><limit lower="0" upper="1.2"/></joint>
      <joint name="arm" type="revolute"><limit lower="-1.57" upper="1.57"/></joint>
      <joint name="wheel" type="continuous"/>
      <joint name="mount" type="fixed"/>
    </robot>"""
    assert bounds.position_limits(urdf) == {"lift": (0.0, 1.2), "arm": (-1.57, 1.57)}


def test_clamp_near_limit_only_within_the_tolerance():
    assert bounds.clamp_near_limit(-1.3e-14, 0.0, 1.2, bounds.ROUNDOFF) == 0.0
    assert bounds.clamp_near_limit(1.2 + 1e-7, 0.0, 1.2, bounds.ROUNDOFF) == 1.2
    assert bounds.clamp_near_limit(-0.001, 0.0, 1.2, bounds.ROUNDOFF) == -0.001
    # A simulator margin widens the range; larger excursions still pass.
    assert bounds.clamp_near_limit(-0.005, 0.0, 1.2, 0.005 + 1e-6) == 0.0
    assert bounds.clamp_near_limit(-7e-3, 0.0, 1.2, 0.005 + 1e-6) == -7e-3
    assert bounds.clamp_near_limit(0.5, 0.0, 1.2, 0.005 + 1e-6) == 0.5


def test_clamp_positions_uses_each_joint_limit():
    limits = {"lift": (0.0, 1.2), "arm": (-1.57, 1.57)}
    msg = JointState(name=["lift", "arm", "wheel"], position=[-1e-14, 1.5700001, 9.0])
    clamped = bounds.clamp_positions(msg, limits, bounds.ROUNDOFF).position
    assert list(clamped) == [0.0, 1.57, 9.0]
    # Velocity-only messages are relayed unchanged.
    msg = JointState(name=["lift"], velocity=[0.1])
    assert list(bounds.clamp_positions(msg, limits, bounds.ROUNDOFF).position) == []


def test_launch_default_tolerance_is_roundoff_only():
    spec = importlib.util.spec_from_file_location(
        "manip_launch", SHARE / "launch/manip.launch.py"
    )
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    (argument,) = [
        entity
        for entity in launch.generate_launch_description().entities
        if getattr(entity, "name", None) == "limit_tolerance"
    ]
    assert float(argument.default_value[0].text) == bounds.ROUNDOFF
