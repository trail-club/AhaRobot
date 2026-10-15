"""joint_state_bounds.py helpers (no ROS transport)."""

import importlib.util
from pathlib import Path

from sensor_msgs.msg import JointState
import yaml

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


def test_clamp_roundoff_only_near_limits():
    assert bounds.clamp_roundoff(-1.3e-14, 0.0, 1.2) == 0.0
    # Up to Gazebo's physics limit (0.005 outside) with its roundoff.
    assert bounds.clamp_roundoff(-0.005 - 1e-11, 0.0, 1.2) == 0.0
    assert bounds.clamp_roundoff(1.2 + 0.005 + 1e-11, 0.0, 1.2) == 1.2
    # Larger excursions and values inside the range pass unchanged.
    assert bounds.clamp_roundoff(-7e-3, 0.0, 1.2) == -7e-3
    assert bounds.clamp_roundoff(0.5, 0.0, 1.2) == 0.5


def test_clamp_positions_uses_each_joint_limit():
    limits = {"lift": (0.0, 1.2), "arm": (-1.57, 1.57)}
    msg = JointState(name=["lift", "arm", "wheel"], position=[-1e-14, 1.571, 9.0])
    assert list(bounds.clamp_positions(msg, limits).position) == [0.0, 1.57, 9.0]
    # Velocity-only messages are relayed unchanged.
    msg = JointState(name=["lift"], velocity=[0.1])
    assert list(bounds.clamp_positions(msg, limits).position) == []


def test_tolerance_stays_inside_the_execution_start_tolerance():
    controllers = yaml.safe_load((SHARE / "config/controllers.yaml").read_text())
    start = controllers["trajectory_execution"]["allowed_start_tolerance"]
    assert 0 < bounds.TOLERANCE < start
