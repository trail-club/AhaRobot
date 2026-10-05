"""The policy API contains robot observations and commands, without scoring state."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol


CONTROLLER_JOINTS = {
    "left_arm_controller": tuple(f"joint_l{i}" for i in range(2, 7)),
    "right_arm_controller": tuple(f"joint_r{i}" for i in range(2, 7)),
    "lift_controller": ("joint_l1", "joint_r1"),
    "head_controller": ("joint_head_pan", "joint_head_tilt"),
}
GRIPPER_JOINTS = {
    "left_gripper_controller": ("joint_l7r", "joint_l7l"),
    "right_gripper_controller": ("joint_r7r", "joint_r7l"),
}
JOINT_LIMITS = {
    **{
        f"joint_{side}{i}": (-3.14, 3.14) if i == 4 else (-1.57, 1.57)
        for side in ("l", "r")
        for i in range(2, 7)
    },
    "joint_l1": (0.0, 1.2),
    "joint_r1": (0.0, 1.2),
    "joint_head_pan": (-1.57, 1.57),
    "joint_head_tilt": (-3.14, 3.14),
    **{f"joint_{side}7r": (0.0, 0.06) for side in ("l", "r")},
    **{f"joint_{side}7l": (-0.06, 0.0) for side in ("l", "r")},
}


@dataclass(frozen=True)
class Observation:
    sim_time: float
    task_id: str
    instruction: str
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    odometry: tuple[float, float, float]  # x, y, yaw in odom
    head_image: Any | None = None  # Optional sensor_msgs/msg/Image


@dataclass(frozen=True)
class Action:
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    joint_positions: Mapping[str, float] = field(default_factory=dict)

    def validate(self):
        for name, value, bound in (
            ("linear_velocity", self.linear_velocity, 0.3),
            ("angular_velocity", self.angular_velocity, 0.5),
        ):
            if not math.isfinite(value) or abs(value) > bound:
                raise ValueError(f"{name} must be finite and within +/-{bound}")
        for name, value in self.joint_positions.items():
            if name not in JOINT_LIMITS:
                raise ValueError(f"Unknown commanded joint: {name}")
            lower, upper = JOINT_LIMITS[name]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


class Policy(Protocol):
    def reset(self, task_id: str, instruction: str) -> None: ...

    def act(self, observation: Observation) -> Action: ...
