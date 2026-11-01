"""押しっぱなしの速度と、実位置が古いときの扱い。

2026-10-18 の実機で、グリッパの押しっぱなしが止まったり逆に跳んだりした。
教示ノードが /joint_states を 1 周 1 本しか捌かず、グリッパの実位置が
数百 ms 古くなっていたため。あわせて、rate * dt を整数に切り捨てていたので
開く向きと閉じる向きで速度が 2 倍違った。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aha_arm_teleop.logic import FEEDBACK_STALE, Teleop

DT = 0.0055
NAMES = ("joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6", "joint_l7r")


def _ready(now=0.0):
    t = Teleop(sides=("left",))
    for name in NAMES:
        t.update_feedback(name, 0.0, now)
    return t


def _hold(t, joint, direction, seconds, follow=True, start=0.0):
    """direction の向きに押しっぱなしにして積分する。follow なら実位置を指令に追従させる。"""
    joint.hold_dir = direction
    now = start
    for _ in range(int(seconds / DT)):
        now += DT
        joint.hold_until = now + 1.0
        t.integrate(now, DT)
        if follow:
            t.update_feedback(joint.ros_name, joint.command_si(), now)
    return now


def test_hold_speed_matches_rate_in_both_directions():
    for direction in (+1, -1):
        t = _ready()
        joint = t.arms["left"].joints[0]
        _hold(t, joint, direction, 1.0)
        assert abs(abs(joint.goal) - t.rate) <= 3, joint.goal


def test_hold_speed_same_from_negative_goal():
    t = _ready()
    joint = t.arms["left"].joints[0]
    now = _hold(t, joint, -1, 1.0)
    start = joint.goal
    _hold(t, joint, +1, 1.0, start=now)
    assert abs((joint.goal - start) - t.rate) <= 3, (start, joint.goal)


def test_stale_feedback_does_not_advance_or_stall():
    t = _ready()
    joint = t.arms["left"].joints[0]
    now = _hold(t, joint, +1, 0.2)
    goal = joint.goal
    # 実位置が来ないまま 1 秒押し続ける。
    _hold(t, joint, +1, 1.0, follow=False, start=now)
    assert joint.blocked_dir == 0
    assert joint.goal - goal <= FEEDBACK_STALE * t.rate + 3
