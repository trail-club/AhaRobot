"""Robot observations and commands exchanged with a separate policy process.

Normal policies exclude scoring state. The explicit ground_truth_ik baseline
receives Gazebo poses and joint positions for simulator-only IK verification.
"""

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


def clamp_joint_near_limits(name, value, tolerance):
    """Move a value at most tolerance outside its limits onto them for MoveIt."""
    if name in JOINT_LIMITS:
        lower, upper = JOINT_LIMITS[name]
        if lower - tolerance <= value <= upper + tolerance:
            return max(lower, min(upper, value))
    return value


@dataclass(frozen=True)
class CameraObservation:
    image: Any  # sensor_msgs/msg/Image (RGB)
    camera_info: Any  # sensor_msgs/msg/CameraInfo
    base_transform: (
        Any  # geometry_msgs/msg/TransformStamped: base_link -> optical frame
    )
    depth_image: Any | None = None  # Optional aligned depth, 32FC1 in metres


@dataclass(frozen=True)
class GroundTruth:
    # Pose of the URDF root (base_footprint) in world: xyz, quaternion xyzw.
    robot_world_pose: tuple[float, float, float, float, float, float, float]
    object_position: tuple[float, float, float]
    finger_contacts: int


@dataclass(frozen=True)
class Observation:
    sim_time: float
    task_id: str
    instruction: str
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    odometry: tuple[float, float, float]  # x/y/yaw: odom normally, world for truth IK
    head_image: Any | None = None  # Optional sensor_msgs/msg/Image
    cameras: Mapping[str, CameraObservation] = field(default_factory=dict)
    ground_truth: GroundTruth | None = None


@dataclass(frozen=True)
class Action:
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    joint_positions: Mapping[str, float] = field(default_factory=dict)
    # Ends the episode now, as a team telling the referee to stop. Scoring
    # tasks rule on the state at that moment; others count it as a failure.
    end_episode: bool = False

    def validate(self):
        if type(self.end_episode) is not bool:
            raise ValueError("end_episode must be a bool")
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
