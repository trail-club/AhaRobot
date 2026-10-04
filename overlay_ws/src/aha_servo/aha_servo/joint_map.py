"""Joint <-> servo mapping for one STS bus, parsed from a plain dict.

The config is a ROS parameter file (config/head.yaml) so launch can pass it to
servo_trajectory_bridge.py; servo_calibrate.py reads and edits the same file.
Layout of the `ros__parameters` block:

  controller_name: head_controller   # topics/action live under /<name>/
  port: /dev/ttyUSB0
  baud: 115200
  ...                                # see BUS_DEFAULTS
  joints: [joint_head_pan, joint_head_tilt]
  joint_head_pan:                    # one block per joint, see JOINT_DEFAULTS
    id: 12
    zero: 2048                       # servo steps at joint position 0
    sign: -1                         # +1: steps grow with the joint position
    min: -1.570                      # rad
    max: 1.570

rad = sign * (steps - zero) / steps_per_rev * 2 pi
(upstream astra_controller head: zero 2048, sign -1).
"""

import math
import re
from dataclasses import dataclass, field

import yaml

BUS_DEFAULTS = {
    "controller_name": "servo_controller",
    "port": "/dev/ttyUSB0",
    "baud": 115200,
    "rate_hz": 30.0,  # state read / command loop
    "max_speed": 1.0,  # rad/s, caps every commanded move
    "acc": 30,  # STS acceleration register (0 = unlimited)
    "torque_limit": 500,  # 0..1000 (0.1 %), written at start; 0 = leave as is
    "timeout": 0.05,  # s per reply
    "goal_tolerance": 0.03,  # rad, action success when no per-joint tolerance given
    "goal_timeout": 2.0,  # s after the last point before the action aborts
    "zero_pose": "",  # calibration prompt for the all-zero pose
}

JOINT_DEFAULTS = {
    "id": None,  # required
    "zero": 2048,
    "sign": -1,
    "min": -math.pi,
    "max": math.pi,
    "steps_per_rev": 4096,
    "positive": "",  # calibration prompt for the + direction
}


@dataclass
class JointConfig:
    name: str
    id: int
    zero: int = 2048
    sign: int = -1
    min: float = -math.pi
    max: float = math.pi
    steps_per_rev: int = 4096
    positive: str = ""

    @property
    def rad_per_step(self):
        return 2.0 * math.pi / self.steps_per_rev

    def to_steps(self, rad):
        return round(self.zero + self.sign * rad / self.rad_per_step)

    def to_rad(self, steps):
        return self.sign * (steps - self.zero) * self.rad_per_step

    def speed_to_steps(self, rad_per_s):
        """Unsigned speed in steps/s."""
        return abs(rad_per_s) / self.rad_per_step

    def speed_to_rad(self, steps_per_s):
        return self.sign * steps_per_s * self.rad_per_step

    def clamp(self, rad):
        """(rad limited to [min, max], clamped)."""
        value = min(max(rad, self.min), self.max)
        return value, value != rad

    def step_range(self):
        """Servo steps covered by [min, max], low to high."""
        a, b = self.to_steps(self.min), self.to_steps(self.max)
        return min(a, b), max(a, b)


@dataclass
class BusConfig:
    controller_name: str
    port: str
    baud: int
    rate_hz: float
    max_speed: float
    acc: int
    torque_limit: int
    timeout: float
    goal_tolerance: float
    goal_timeout: float
    zero_pose: str
    joints: list = field(default_factory=list)

    @property
    def joint_names(self):
        return [j.name for j in self.joints]

    def joint(self, name):
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)


def parse_config(params):
    """BusConfig from the nested `ros__parameters` dict; raises ValueError."""
    names = list(params.get("joints") or [])
    if not names:
        raise ValueError("config: `joints` must list at least one joint")
    if len(set(names)) != len(names):
        raise ValueError(f"config: duplicate joint names in {names}")

    bus = {}
    for key, default in BUS_DEFAULTS.items():
        value = params.get(key, default)
        bus[key] = type(default)(value) if value is not None else default

    joints = []
    for name in names:
        block = params.get(name)
        if not isinstance(block, dict):
            raise ValueError(f"config: missing block for joint {name}")
        values = {k: block.get(k, d) for k, d in JOINT_DEFAULTS.items()}
        if values["id"] is None:
            raise ValueError(f"config: {name}.id is required")
        joint = JointConfig(
            name=name,
            id=int(values["id"]),
            zero=int(values["zero"]),
            sign=int(values["sign"]),
            min=float(values["min"]),
            max=float(values["max"]),
            steps_per_rev=int(values["steps_per_rev"]),
            positive=str(values["positive"]),
        )
        if joint.sign not in (-1, 1):
            raise ValueError(f"config: {name}.sign must be 1 or -1, got {joint.sign}")
        if not joint.min < joint.max:
            raise ValueError(f"config: {name}.min must be below {name}.max")
        if not 0 <= joint.id <= 253:
            raise ValueError(f"config: {name}.id {joint.id} is out of range 0..253")
        joints.append(joint)

    ids = [j.id for j in joints]
    if len(set(ids)) != len(ids):
        raise ValueError(f"config: duplicate servo ids in {ids}")
    return BusConfig(joints=joints, **bus)


def ros_parameters(doc):
    """The first `ros__parameters` block of a ROS parameter file document."""
    if isinstance(doc, dict):
        for value in doc.values():
            if isinstance(value, dict):
                if "ros__parameters" in value:
                    return value["ros__parameters"]
    raise ValueError("no `<node>: ros__parameters:` block found")


def load_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_config(ros_parameters(yaml.safe_load(f)))


def wrap_steps(delta, steps_per_rev=4096):
    """Fold a step difference into [-steps_per_rev/2, steps_per_rev/2)."""
    half = steps_per_rev // 2
    return (delta + half) % steps_per_rev - half


_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*)$")


def update_yaml_text(text, joint, values):
    """Set `values` ({key: value}) in the block of `joint`, keeping comments.

    Only replaces scalar lines like `  zero: 2048  # note`; a missing key is
    added as the first line of the block.
    """
    lines = text.splitlines(keepends=True)
    header = None
    for i, line in enumerate(lines):
        m = _KEY_LINE.match(line.rstrip("\n"))
        if m and m["key"] == joint and m["rest"].split("#")[0].strip() == "":
            header = i
            break
    if header is None:
        raise ValueError(f"joint block `{joint}:` not found")
    parent_indent = len(_KEY_LINE.match(lines[header])["indent"])

    child_indent = None
    end = len(lines)
    for i in range(header + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= parent_indent:
            end = i
            break
        if child_indent is None:
            child_indent = indent
    if child_indent is None:
        child_indent = parent_indent + 2

    pending = dict(values)
    for i in range(header + 1, end):
        m = _KEY_LINE.match(lines[i].rstrip("\n"))
        if not m or len(m["indent"]) != child_indent or m["key"] not in pending:
            continue
        comment = re.search(r"\s+#.*$", m["rest"])
        newline = "\n" if lines[i].endswith("\n") else ""
        lines[i] = (
            f"{m['indent']}{m['key']}: {pending.pop(m['key'])}"
            f"{comment.group(0) if comment else ''}{newline}"
        )
    for key, value in pending.items():
        lines.insert(header + 1, f"{' ' * child_indent}{key}: {value}\n")
    return "".join(lines)
