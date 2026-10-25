"""Arm bus configuration for arm_trajectory_bridge.py, parsed from ROS parameters.

The shape mirrors aha_servo's joint_map (one block per joint, defaults filled
in) but keeps only what a hardware-driven arm needs: no Feetech ids/zero/sign,
no encoder steps. Joint limits are radians.
"""

from dataclasses import dataclass, field

from .arm_plan import ArmJoint

BUS_DEFAULTS = {
    "controller_name": "left_arm_controller",
    "command_topic": "joint_command",
    "state_topic": "joint_states",
    "rate_hz": 30.0,
    "max_speed": 1.0,
    "goal_tolerance": 0.03,
    "goal_timeout": 2.0,
    "settle_time": 1.5,
    "state_timeout": 0.3,
    "closed_loop": True,
}
JOINT_DEFAULTS = {"min": -3.14, "max": 3.14}


@dataclass
class ArmBusConfig:
    controller_name: str
    command_topic: str
    state_topic: str
    rate_hz: float
    max_speed: float
    goal_tolerance: float
    goal_timeout: float
    settle_time: float
    state_timeout: float
    closed_loop: bool
    # Full joint list used for the arm_node JointCommand `name`; arm joints
    # beyond `joints` are held at `command_hold` (e.g. the gripper degree).
    command_joint_names: list = field(default_factory=list)
    command_hold: float = 0.0
    hold_interval: float = 1.0
    joints: list[ArmJoint] = field(default_factory=list)

    @property
    def joint_names(self) -> list:
        return [j.name for j in self.joints]


def parse_config(params: dict) -> ArmBusConfig:
    bus = {**BUS_DEFAULTS, **params}
    for key in (
        "rate_hz",
        "max_speed",
        "goal_tolerance",
        "goal_timeout",
        "settle_time",
        "state_timeout",
    ):
        bus[key] = float(bus[key])
    bus["closed_loop"] = bool(bus["closed_loop"])

    joints = []
    for name in bus["joint_names"]:
        joint = {**JOINT_DEFAULTS, **params.get(name, {})}
        joint["min"] = float(joint["min"])
        joint["max"] = float(joint["max"])
        if joint["min"] >= joint["max"]:
            raise ValueError(f"{name}: min must be < max")
        joints.append(ArmJoint(name, joint["min"], joint["max"]))
    command_names = bus.get("command_joint_names") or [j.name for j in joints]
    return ArmBusConfig(
        controller_name=str(bus["controller_name"]),
        command_topic=str(bus["command_topic"]),
        state_topic=str(bus["state_topic"]),
        rate_hz=float(bus["rate_hz"]),
        max_speed=float(bus["max_speed"]),
        goal_tolerance=float(bus["goal_tolerance"]),
        goal_timeout=float(bus["goal_timeout"]),
        settle_time=float(bus["settle_time"]),
        state_timeout=float(bus["state_timeout"]),
        closed_loop=bool(bus["closed_loop"]),
        command_joint_names=list(command_names),
        command_hold=float(bus.get("command_hold", 0.0)),
        hold_interval=float(bus.get("hold_interval", 1.0)),
        joints=joints,
    )
