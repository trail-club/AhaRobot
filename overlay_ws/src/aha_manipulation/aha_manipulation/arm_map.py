"""Configuration of arm_trajectory_bridge.py (config/arm.yaml), from ROS parameters.

One bridge drives one arm. `joints` lists the five arm joints in the order of
the arm_node's `joint_names`: the arm_node accepts a JointCommand only with
exactly that name list. Each joint has a block (`<joint>.min`, ...) with its
position limits (rad), `max_velocity` (rad/s) and `max_acceleration`
(rad/s^2).
"""

from dataclasses import dataclass

from .arm_plan import JointLimits

BUS_DEFAULTS = {
    "controller_name": "right_arm_controller",
    "command_topic": "/right/arm/joint_command",
    "state_topic": "/joint_states",
    "rate_hz": 50.0,  # command samples per second
    "start_tolerance": 0.05,  # rad, a point at time 0 vs. the measured pose
    "path_tolerance": 0.3,  # rad, sampled command vs. measured while moving
    "goal_tolerance": 0.03,  # rad, final point vs. measured
    "goal_time": 2.0,  # s after the last point to be within goal_tolerance
    "state_timeout": 0.3,  # s, older joint states count as missing
}
JOINT_DEFAULTS = {
    "min": -1.57,
    "max": 1.57,
    "max_velocity": 1.0,
    "max_acceleration": 2.0,
}
JOINT_COUNT = 5


@dataclass(frozen=True)
class ArmConfig:
    controller_name: str
    command_topic: str
    state_topic: str
    rate_hz: float
    start_tolerance: float
    path_tolerance: float
    goal_tolerance: float
    goal_time: float
    state_timeout: float
    joints: tuple  # JointLimits in command order

    @property
    def joint_names(self):
        return [j.name for j in self.joints]


def parse_config(params):
    """ArmConfig from {key: value, joint: {key: value}}; raises ValueError."""
    bus = {**BUS_DEFAULTS, **{k: params[k] for k in BUS_DEFAULTS if k in params}}
    for key in ("controller_name", "command_topic", "state_topic"):
        if not bus[key]:
            raise ValueError(f"`{key}` is empty")
    for key in ("rate_hz", "goal_tolerance", "goal_time", "state_timeout"):
        if not float(bus[key]) > 0:
            raise ValueError(f"`{key}` must be positive, got {bus[key]}")
    for key in ("start_tolerance", "path_tolerance"):
        if not float(bus[key]) >= 0:
            raise ValueError(f"`{key}` must not be negative, got {bus[key]}")

    names = list(params.get("joints", []))
    if len(names) != JOINT_COUNT or len(set(names)) != JOINT_COUNT:
        raise ValueError(
            f"`joints` must name the {JOINT_COUNT} arm joints once each, got {names}"
        )
    joints = []
    for name in names:
        joint = {**JOINT_DEFAULTS, **params.get(name, {})}
        lo, hi = float(joint["min"]), float(joint["max"])
        if not lo < hi:
            raise ValueError(f"{name}: min {lo} must be below max {hi}")
        for key in ("max_velocity", "max_acceleration"):
            if not float(joint[key]) > 0:
                raise ValueError(f"{name}: `{key}` must be positive, got {joint[key]}")
        joints.append(
            JointLimits(
                name,
                lo,
                hi,
                float(joint["max_velocity"]),
                float(joint["max_acceleration"]),
            )
        )
    return ArmConfig(
        controller_name=str(bus["controller_name"]),
        command_topic=str(bus["command_topic"]),
        state_topic=str(bus["state_topic"]),
        rate_hz=float(bus["rate_hz"]),
        start_tolerance=float(bus["start_tolerance"]),
        path_tolerance=float(bus["path_tolerance"]),
        goal_tolerance=float(bus["goal_tolerance"]),
        goal_time=float(bus["goal_time"]),
        state_timeout=float(bus["state_timeout"]),
        joints=tuple(joints),
    )
