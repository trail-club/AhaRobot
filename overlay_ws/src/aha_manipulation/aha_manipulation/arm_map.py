"""Configuration of arm_trajectory_bridge.py (config/arm.yaml, config/lift.yaml).

One bridge serves one trajectory controller. `joints` lists its joints. Each
joint has a block (`<joint>.min`, ...) with its position limits,
`max_velocity` and `max_acceleration` in `unit` ("rad" or "m"), per s
and s^2, and `command_topic` / `state_topic`. Joints with the same command topic form one command group:
the astra_controller nodes accept a JointCommand only with exactly their
`joint_names`, so a group is always sent whole, in `joints` order. An empty
`command_topic` uses the bridge's.

`board_acceleration` > 0 is for a board that plans its own stop at that
acceleration (the lift's AccelStepper): it only moves as fast as it can still
stop at the goal, so a goal at the sample would lag the trajectory by
v^2 / (2 a). For trajectories with velocities the bridge sends the sample
plus that distance in the direction of motion instead, and it holds at the
point where the board can stop.
`max_acceleration` must not exceed it.

  arm  (arm.yaml):  one group, the five arm joints of one arm_node
  lift (lift.yaml): one group per lift_node, one joint each
"""

from dataclasses import dataclass

from .arm_plan import JointLimits

BUS_DEFAULTS = {
    "controller_name": "right_arm_controller",
    "command_topic": "/right/arm/joint_command",
    "state_topic": "/right/arm/joint_states",  # the node's own, not /joint_states
    "rate_hz": 50.0,  # command samples per second
    "start_tolerance": 0.05,  # a point at time 0 vs. the measured pose
    "path_tolerance": 0.3,  # sampled command vs. measured while moving
    "goal_tolerance": 0.03,  # final point vs. measured
    "goal_time": 2.0,  # s after the last point to be within goal_tolerance
    "state_timeout": 0.3,  # s, older joint states count as missing
}
JOINT_DEFAULTS = {
    "min": -1.57,
    "max": 1.57,
    "max_velocity": 1.0,
    "max_acceleration": 2.0,
    "unit": "rad",  # "m" for a prismatic joint, used in messages
    "command_topic": "",  # empty: the bridge's command_topic
    "state_topic": "",  # empty: the bridge's state_topic
    "board_acceleration": 0.0,  # > 0: commands lead the samples (see above)
}


@dataclass(frozen=True)
class CommandGroup:
    topic: str
    names: tuple  # joint names in command order


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
    joints: tuple  # JointLimits in `joints` order
    groups: tuple  # CommandGroup in order of their first joint
    state_topics: tuple  # every topic to read joint states from
    board_accelerations: dict  # joint -> board_acceleration, only those > 0

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
    if not names or "" in names or len(set(names)) != len(names):
        raise ValueError(f"`joints` must name each joint once, got {names}")
    joints = []
    groups = {}  # topic -> [names], in insertion order
    state_topics = []
    boards = {}
    for name in names:
        joint = {**JOINT_DEFAULTS, **params.get(name, {})}
        lo, hi = float(joint["min"]), float(joint["max"])
        if not lo < hi:
            raise ValueError(f"{name}: min {lo} must be below max {hi}")
        for key in ("max_velocity", "max_acceleration"):
            if not float(joint[key]) > 0:
                raise ValueError(f"{name}: `{key}` must be positive, got {joint[key]}")
        board = float(joint["board_acceleration"])
        if board < 0:
            raise ValueError(f"{name}: `board_acceleration` must not be negative")
        if board > 0:
            if float(joint["max_acceleration"]) > board:
                raise ValueError(
                    f"{name}: `max_acceleration` {joint['max_acceleration']} is over "
                    f"`board_acceleration` {board}"
                )
            boards[name] = board
        joints.append(
            JointLimits(
                name,
                lo,
                hi,
                float(joint["max_velocity"]),
                float(joint["max_acceleration"]),
                str(joint["unit"]),
            )
        )
        groups.setdefault(
            str(joint["command_topic"] or bus["command_topic"]), []
        ).append(name)
        state = str(joint["state_topic"] or bus["state_topic"])
        if state not in state_topics:
            state_topics.append(state)
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
        groups=tuple(CommandGroup(t, tuple(n)) for t, n in groups.items()),
        state_topics=tuple(state_topics),
        board_accelerations=boards,
    )
