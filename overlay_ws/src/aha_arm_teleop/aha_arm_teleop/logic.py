"""両腕キーボード教示の純粋ロジック。

`tools/servo/keyboard_teleop.py` の「1 回叩くと step、押しっぱなしは rate、
目標は実位置から lead 以上進まない、可動域とエンコーダ原点の手前で止める」
を、サーボ step のまま持つ。ROS へ出す瞬間だけ `arm_controller.to_si_unit` と
同じ式でラジアン / メートルに換算する。

符号: シリアル教示の +goal は +1 サーボの raw を増やす。ファームが返す raw は
`si = -(raw - 2048) / 4096 * 2π` なので、同じ物理方向は ROS の位置を減らす。
キーの向きをシリアル教示と揃えるため、count が正のとき指令 SI は減る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

COUNTS_PER_REV = 4096
RAD_PER_COUNT = 2.0 * math.pi / COUNTS_PER_REV
# arm_controller.ArmController.GRIPPER_GEAR_R と同じ。ここだけ変えると
# グリッパのメートル指令がファームの生値とずれる。
GRIPPER_GEAR_R = 0.027 / 2
GRIPPER_ZERO_M = 0.03

ORIGIN_MARGIN = 150
MAX_ROM = 1950
HOLD_GRACE = 0.15
REPEAT_WINDOW = 0.25
STALL_MOVE = 3
BLOCK_COOLDOWN = 0.6

# arm_controller.ArmController.JOINT_MIN/MAX の先頭 5 要素。範囲外の 1 成分でも
# set_pos はパケットごと捨てるので、出す前にこちらで収める。
_FW_ABS = (
    math.pi / 2,
    math.pi / 2,
    math.pi * 0.999,
    math.pi / 2,
    math.pi * 0.875,
)

# (名前, 腕の何番目か, 実測可動域の片側 step, 左腕のキー)
# 右腕は同じ文字の大文字。可動域は docs/servo-bringup.md。
_JOINTS = (
    ("joint0", 0, 920, ("w", "s")),
    ("joint1", 1, 990, ("e", "d")),
    ("wrist12", 2, 2362, ("r", "f")),
    ("wrist13", 3, 1360, ("t", "g")),
    ("wrist14", 4, 1650, ("u", "j")),
    ("gripper", 5, 1310, ("y", "h")),
)

_ARM_NAMES = {
    "left": ("joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"),
    "right": ("joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6"),
}
_GRIP_NAMES = {"left": "joint_l7r", "right": "joint_r7r"}
_ARM_TOPICS = {
    "left": "/left/arm/joint_command",
    "right": "/right/arm/joint_command",
}
_GRIP_TOPICS = {
    "left": "/left/arm/gripper_joint_command",
    "right": "/right/arm/gripper_joint_command",
}
_TORQUE_TOPICS = {
    "left": "/left/arm/torque_enable",
    "right": "/right/arm/torque_enable",
}

HELP = """
  左腕 (小文字)                         右腕 (大文字)
  w/s  joint0 (joint_*2)                W/S
  e/d  joint1 (joint_*3)                E/D
  r/f  wrist12 (joint_*4)               R/F
  t/g  wrist13 (joint_*5)               T/G
  u/j  wrist14 (joint_*6)               U/J
  y/h  gripper (joint_*7r)              Y/H
  [ ]  ステップ幅 -/+        space  その場で停止        0  両腕トルクOFF
  ?    このヘルプ            q      終了 (両腕トルクOFF)
"""


def si_from_raw(raw: float, is_gripper: bool) -> float:
    """ファーム raw count を arm_node が配る SI に戻す。"""
    angle = -(raw - 2048.0) / COUNTS_PER_REV * (2.0 * math.pi)
    if is_gripper:
        return angle * GRIPPER_GEAR_R + GRIPPER_ZERO_M
    return angle


def encoder_raw(si: float, is_gripper: bool) -> int:
    """SI をファーム raw に戻す。原点マージンの判定に使う。"""
    angle = si
    if is_gripper:
        angle = (si - GRIPPER_ZERO_M) / GRIPPER_GEAR_R
    raw = 2048.0 - angle / (2.0 * math.pi) * COUNTS_PER_REV
    return int(round(raw)) % COUNTS_PER_REV


@dataclass
class Command:
    topic: str
    names: tuple[str, ...]
    positions: tuple[float, ...]


@dataclass
class TorqueCommand:
    topic: str
    value: int


@dataclass
class Outputs:
    commands: list[Command] = field(default_factory=list)
    torque: list[TorqueCommand] = field(default_factory=list)
    quit: bool = False
    message: str = ""
    status: str = ""
    banner: str = ""


class _Joint:
    def __init__(self, label: str, index: int, rom: int, keys: tuple[str, str], ros_name: str):
        self.label = label
        self.index = index
        self.is_gripper = label == "gripper"
        self.rom = min(rom, MAX_ROM)
        self.keys = keys
        self.ros_name = ros_name
        self.scale = RAD_PER_COUNT * (GRIPPER_GEAR_R if self.is_gripper else 1.0)
        if self.is_gripper:
            # set_pos はグリッパも ±π/2 で判定する。単位はメートルだが、
            # この外に出ると腕の指令ごと捨てられる。
            self.fw_min = -math.pi / 2
            self.fw_max = math.pi / 2
        else:
            self.fw_min = -_FW_ABS[index]
            self.fw_max = _FW_ABS[index]
        self.start_si: float | None = None
        self.last_si: float | None = None
        self.goal = 0
        self.actual = 0.0
        self.lim_pos = self.rom
        self.lim_neg = self.rom
        self.blocked_dir = 0
        self.blocked_at = 0.0
        self.hold_dir = 0
        self.hold_until = 0.0
        self.key_dir = 0
        self.key_at = 0.0
        self.faulted = False
        self.jumped = 0.0
        self.stall_ref = 0.0
        self.stall_at = 0.0
        self.seen = False

    def count_bounds(self) -> tuple[float, float]:
        if self.start_si is None:
            return (-self.rom, self.rom)
        c_at_min = (self.start_si - self.fw_min) / self.scale
        c_at_max = (self.start_si - self.fw_max) / self.scale
        return (min(c_at_min, c_at_max), max(c_at_min, c_at_max))

    def command_si(self) -> float | None:
        if self.start_si is None:
            return None
        return self.start_si - self.goal * self.scale


class _Arm:
    def __init__(self, side: str):
        self.side = side
        self.torque_on = True
        names = _ARM_NAMES[side]
        grip = _GRIP_NAMES[side]
        self.joints: list[_Joint] = []
        for label, index, rom, keys in _JOINTS:
            ros_name = grip if label == "gripper" else names[index]
            keyed = keys if side == "left" else tuple(k.upper() for k in keys)
            self.joints.append(_Joint(label, index, rom, keyed, ros_name))
        self.by_ros = {j.ros_name: j for j in self.joints}

    def ready(self) -> bool:
        return all(j.seen for j in self.joints)

    def arm_joints(self) -> list[_Joint]:
        return [j for j in self.joints if not j.is_gripper]

    def gripper(self) -> _Joint:
        return self.joints[-1]


class Teleop:
    """キー列と joint 位置から、出す JointCommand を決める。"""

    def __init__(
        self,
        step: int = 8,
        rate: int = 250,
        lead: int = 80,
        stall_time: float = 0.5,
        max_jump: int = 250,
        sides: tuple[str, ...] = ("left", "right"),
    ):
        self.step = step
        self.rate = rate
        self.lead = lead
        self.stall_time = stall_time
        self.max_jump = max_jump
        self.sides = tuple(sides)
        self.arms = {side: _Arm(side) for side in ("left", "right")}
        self.message = "トルクON"
        self.banner = ""
        self.quit = False
        self._arm_dirty = {"left": False, "right": False}
        self._grip_dirty = {"left": False, "right": False}
        self._torque: list[TorqueCommand] = []
        self._by_key: dict[str, tuple[str, _Joint, int]] = {}
        for side, arm in self.arms.items():
            for joint in arm.joints:
                self._by_key[joint.keys[0]] = (side, joint, +1)
                self._by_key[joint.keys[1]] = (side, joint, -1)

    def update_feedback(self, name: str, position: float) -> None:
        for arm in self.arms.values():
            joint = arm.by_ros.get(name)
            if joint is None:
                continue
            joint.last_si = float(position)
            if joint.start_si is None:
                joint.start_si = float(position)
                joint.actual = 0.0
                joint.goal = 0
                joint.seen = True
                self._apply_limits(joint)
                return
            new_actual = (joint.start_si - float(position)) / joint.scale
            if abs(new_actual - joint.actual) > self.max_jump:
                joint.jumped = abs(new_actual - joint.actual)
            joint.actual = new_actual
            return

    def tick(self, keys: list[str], now: float, dt: float) -> Outputs:
        self.handle_keys(keys, now)
        if not self.quit:
            self.integrate(now, max(0.0, dt))
        return self.take_outputs()

    def handle_keys(self, keys: list[str], now: float) -> None:
        for key in keys:
            if key == "q":
                self.quit = True
                self._torque_all(0)
                self.message = "終了"
                return
            if key == "?":
                self.banner = HELP
                continue
            if key in ("[", "]"):
                delta = 5 if key == "]" else -5
                self.step = max(1, min(200, self.step + delta))
                self.rate = max(20, min(2000, self.rate + delta * 20))
                self.message = f"ステップ {self.step} / 速度 {self.rate}"
                continue
            if key == " ":
                for side in self.sides:
                    for joint in self.arms[side].joints:
                        self._set_goal(side, joint, int(joint.actual), force=True)
                        joint.hold_dir = 0
                self.message = "停止"
                continue
            if key == "0":
                self._torque_all(0)
                for side in self.sides:
                    self.arms[side].torque_on = False
                    for joint in self.arms[side].joints:
                        joint.hold_dir = 0
                self.message = "トルクOFF (方向キーで再投入)"
                continue
            mapped = self._by_key.get(key)
            if mapped is None:
                continue
            side, joint, direction = mapped
            if side not in self.sides:
                self.message = f"{side} は無効"
                continue
            arm = self.arms[side]
            if not joint.seen:
                self.message = f"{joint.ros_name} の joint_states がまだ来ていません"
                continue
            if not arm.torque_on:
                self._resync(side)
                self._torque_side(side, 1)
                arm.torque_on = True
                self.message = "トルクON"
                continue
            if joint.faulted:
                self.message = (
                    f"{side} {joint.label}: 異常検出で切り離し中。"
                    "0 でトルクを入れ直してください"
                )
                continue
            if joint.blocked_dir and (direction > 0) == (joint.blocked_dir > 0):
                self.message = f"{side} {joint.label}: この向きは停止中"
                continue
            if direction > 0 and joint.lim_pos == 0:
                self.message = f"{side} {joint.label}: 原点に近すぎるため +方向は無効"
                continue
            if direction < 0 and joint.lim_neg == 0:
                self.message = f"{side} {joint.label}: 原点に近すぎるため -方向は無効"
                continue
            held = joint.key_dir == direction and now - joint.key_at < REPEAT_WINDOW
            if held:
                joint.hold_dir = direction
                joint.hold_until = now + HOLD_GRACE
            else:
                self._set_goal(
                    side,
                    joint,
                    self._clamp(joint, joint.goal + direction * self.step),
                )
                joint.hold_dir = 0
                joint.hold_until = 0.0
                joint.stall_ref = joint.actual
                joint.stall_at = now
            joint.key_dir = direction
            joint.key_at = now
            self.message = f"{side} {joint.label} goal={joint.goal:+d}"

    def integrate(self, now: float, dt: float) -> None:
        for side in self.sides:
            arm = self.arms[side]
            for joint in arm.joints:
                moving = bool(
                    joint.hold_dir and now < joint.hold_until and arm.torque_on and not joint.faulted
                )
                if not moving:
                    if joint.hold_dir and now >= joint.hold_until:
                        joint.hold_dir = 0
                    joint.stall_ref = joint.actual
                    joint.stall_at = now
                    if joint.blocked_dir and now - joint.blocked_at > BLOCK_COOLDOWN:
                        joint.blocked_dir = 0
                    continue
                self._set_goal(
                    side,
                    joint,
                    self._clamp(joint, joint.goal + joint.hold_dir * self.rate * dt),
                )
                if joint.jumped:
                    joint.faulted = True
                    joint.hold_dir = 0
                    self._set_goal(side, joint, int(joint.actual), force=True)
                    self.message = (
                        f"★{side} {joint.label}: 実位置が {joint.jumped:.0f} step 飛んだので切り離しました"
                    )
                    joint.jumped = 0.0
                    continue
                pushing = abs(joint.goal - joint.actual) > self.lead * 0.6
                if pushing and abs(joint.actual - joint.stall_ref) < STALL_MOVE:
                    stalled = now - joint.stall_at > self.stall_time
                else:
                    joint.stall_ref = joint.actual
                    joint.stall_at = now
                    stalled = False
                if stalled:
                    joint.blocked_dir = 1 if joint.goal >= joint.actual else -1
                    joint.blocked_at = now
                    joint.hold_dir = 0
                    self._set_goal(side, joint, int(joint.actual), force=True)
                    self.message = f"★{side} {joint.label}: 動かないので停止"
                elif joint.blocked_dir and now - joint.blocked_at > BLOCK_COOLDOWN:
                    joint.blocked_dir = 0

    def take_outputs(self) -> Outputs:
        commands: list[Command] = []
        for side in self.sides:
            arm = self.arms[side]
            if not arm.ready():
                # 5 関節とグリッパが揃う前に消すと、先に動いた分の指令が消える。
                continue
            if self._arm_dirty[side]:
                names = tuple(j.ros_name for j in arm.arm_joints())
                positions = tuple(float(j.command_si()) for j in arm.arm_joints())
                commands.append(Command(_ARM_TOPICS[side], names, positions))
                self._arm_dirty[side] = False
            if self._grip_dirty[side]:
                grip = arm.gripper()
                commands.append(
                    Command(
                        _GRIP_TOPICS[side],
                        (grip.ros_name,),
                        (float(grip.command_si()),),
                    )
                )
                self._grip_dirty[side] = False
        torque = self._torque
        self._torque = []
        banner = self.banner
        self.banner = ""
        quit_now = self.quit
        return Outputs(commands, torque, quit_now, self.message, self.status(), banner)

    def status(self) -> str:
        parts = []
        for side in self.sides:
            arm = self.arms[side]
            chunks = []
            for joint in arm.joints:
                if not joint.seen:
                    chunks.append(f"{joint.label}:----")
                    continue
                mark = " "
                if joint.faulted:
                    mark = "X"
                elif joint.blocked_dir:
                    mark = "!"
                elif joint.hold_dir:
                    mark = ">"
                chunks.append(f"{joint.label}:{joint.goal:+5d}/{joint.actual:+5.0f}{mark}")
            parts.append(side[0].upper() + " " + " ".join(chunks))
        return f"{'  '.join(parts)}  step={self.step:<3} | {self.message}"

    def _clamp(self, joint: _Joint, want: float) -> int:
        want = max(joint.actual - self.lead, min(joint.actual + self.lead, want))
        want = max(-joint.lim_neg, min(joint.lim_pos, want))
        lo, hi = joint.count_bounds()
        # 今の姿勢自体が arm_node の受付範囲の外なら、範囲の端まで引き戻して
        # 飛ばさない。その場に留める。
        if joint.actual < lo or joint.actual > hi:
            return int(joint.actual)
        want = max(lo, min(hi, want))
        return int(want)

    def _apply_limits(self, joint: _Joint) -> None:
        if joint.start_si is None:
            return
        raw = encoder_raw(joint.start_si, joint.is_gripper)
        pos = 4095 - raw
        neg = raw
        joint.lim_pos = min(joint.rom, pos)
        joint.lim_neg = min(joint.rom, neg)
        if pos < ORIGIN_MARGIN:
            joint.lim_pos = 0
        if neg < ORIGIN_MARGIN:
            joint.lim_neg = 0

    def _set_goal(self, side: str, joint: _Joint, goal: int, force: bool = False) -> None:
        goal = int(goal)
        if goal == joint.goal and not force:
            return
        joint.goal = goal
        if joint.is_gripper:
            self._grip_dirty[side] = True
        else:
            self._arm_dirty[side] = True

    def _resync(self, side: str) -> None:
        for joint in self.arms[side].joints:
            if joint.last_si is None:
                continue
            joint.start_si = joint.last_si
            joint.actual = 0.0
            joint.goal = 0
            joint.jumped = 0.0
            joint.faulted = False
            joint.hold_dir = 0
            joint.blocked_dir = 0
            joint.key_dir = 0
            self._apply_limits(joint)
            self._set_goal(side, joint, 0, force=True)

    def _torque_side(self, side: str, value: int) -> None:
        self._torque.append(TorqueCommand(_TORQUE_TOPICS[side], value))

    def _torque_all(self, value: int) -> None:
        for side in self.sides:
            self._torque_side(side, value)
