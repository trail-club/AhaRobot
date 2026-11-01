"""Head pan/tilt limits, speed and presets from config/head.yaml.

Shared by head_look_at.py, sim_control_panel.py and teleop_head.py.
The yaml is a ROS parameter file (`/**: ros__parameters: head.*`), so nodes
receive it via launch parameters or `--ros-args --params-file`.
"""

import math
import os
from dataclasses import dataclass, field

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

PAN_JOINT = "joint_head_pan"
TILT_JOINT = "joint_head_tilt"
TRAJECTORY_TOPIC = "/head_controller/joint_trajectory"


def default_params_file():
    return os.path.join(
        get_package_share_directory("aha_perception"), "config", "head.yaml"
    )


@dataclass
class Preset:
    label: str
    pan: float
    tilt: float


@dataclass
class HeadConfig:
    pan_min: float
    pan_max: float
    tilt_min: float
    tilt_max: float
    max_speed: float
    min_duration: float
    presets: list = field(default_factory=list)

    def clamp(self, pan, tilt):
        """Return (pan, tilt, clamped) limited to the configured range."""
        cpan = min(max(pan, self.pan_min), self.pan_max)
        ctilt = min(max(tilt, self.tilt_min), self.tilt_max)
        return cpan, ctilt, (cpan != pan or ctilt != tilt)

    def duration(self, current, target):
        """Move duration in seconds; current may be None (unknown)."""
        if current is None:
            return max(math.pi / 2 / self.max_speed, self.min_duration)
        delta = max(abs(t - c) for c, t in zip(current, target))
        return max(delta / self.max_speed, self.min_duration)

    def trajectory(self, pan, tilt, duration):
        """JointTrajectory for head_controller; zero stamp means start now."""
        sec = int(duration)
        point = JointTrajectoryPoint(
            positions=[pan, tilt],
            time_from_start=Duration(sec=sec, nanosec=int((duration - sec) * 1e9)),
        )
        return JointTrajectory(joint_names=[PAN_JOINT, TILT_JOINT], points=[point])


def declare_head_config(node):
    """Declare the head.* parameters on node and return a HeadConfig."""
    defaults = {
        "head.pan_limits": [-1.57, 1.57],
        "head.tilt_limits": [-0.524, 1.047],
        "head.max_speed": 1.0,
        "head.min_duration": 0.3,
        "head.preset_labels": ["正面"],
        "head.preset_pan": [0.0],
        "head.preset_tilt": [0.0],
    }
    values = {
        name: node.declare_parameter(name, default).value
        for name, default in defaults.items()
    }
    labels = values["head.preset_labels"]
    pans = values["head.preset_pan"]
    tilts = values["head.preset_tilt"]
    if not len(labels) == len(pans) == len(tilts):
        raise ValueError("head.preset_labels/preset_pan/preset_tilt lengths differ")
    return HeadConfig(
        pan_min=values["head.pan_limits"][0],
        pan_max=values["head.pan_limits"][1],
        tilt_min=values["head.tilt_limits"][0],
        tilt_max=values["head.tilt_limits"][1],
        max_speed=values["head.max_speed"],
        min_duration=values["head.min_duration"],
        presets=[Preset(lb, p, t) for lb, p, t in zip(labels, pans, tilts)],
    )


def head_position(joint_state):
    """(pan, tilt) from a sensor_msgs/JointState, or None if absent."""
    try:
        pan = joint_state.position[joint_state.name.index(PAN_JOINT)]
        tilt = joint_state.position[joint_state.name.index(TILT_JOINT)]
    except (ValueError, IndexError):
        return None
    return pan, tilt


def describe(pan, tilt):
    """Human-readable direction, e.g. '右 30° / 下 10°'."""
    pan_deg = math.degrees(pan)
    tilt_deg = math.degrees(tilt)
    pan_txt = (
        "正面"
        if abs(pan_deg) < 0.5
        else f"{'右' if pan_deg > 0 else '左'} {abs(pan_deg):.0f}°"
    )
    tilt_txt = (
        "水平"
        if abs(tilt_deg) < 0.5
        else f"{'下' if tilt_deg > 0 else '上'} {abs(tilt_deg):.0f}°"
    )
    return f"{pan_txt} / {tilt_txt}"
