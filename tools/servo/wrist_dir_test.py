#!/usr/bin/env python3
"""腕部 (ID12-14) の方向確認: 上位機から +DEG° の小さな指令を出し、実機が URDF の + 方向へ回るかを目視で確認する。
 
- side を指定して ArmController を作る (腕ごとの符号・可動域を適用)
- set_torque(1) のみで set_pid は呼ばない → joint0/1 は位置ループなし (動かない)
- 腕部を 1 関節ずつ: 1 秒かけて +DEG° → 1.5 秒保持 → 1 秒かけて戻す
- グリッパは現在値を保持 (可動域内にクリップ)
- 終了・Ctrl+C・例外で必ず set_torque(0)
 
使い方:
  uv run wrist_dir_test.py <port> <left|right> [deg=10]
"""
 
import math
import os
import sys
import time
 
import numpy as np
 
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "upstream", "astra_controller"))
 
from astra_controller.arm_controller import ArmController  # noqa: E402
 
if len(sys.argv) < 3 or sys.argv[2] not in ("left", "right"):
    sys.exit(__doc__)
port, side = sys.argv[1], sys.argv[2]
deg = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
grip_mm = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
 
URDF_PLUS = {
    "left": {2: "ID12 (joint_l4): 向右 (内侧)", 3: "ID13 (joint_l5): 向上", 4: "ID14 (joint_l6): 向右 (内侧)"},
    "right": {2: "ID12 (joint_r4): 向右 (外侧)", 3: "ID13 (joint_r5): 向下", 4: "ID14 (joint_r6): 向右 (外侧)"},
}
 
_real = ArmController.set_torque
ArmController.set_torque = lambda self, *a, **k: None
ctrl = ArmController(port, do_init=True, side=side)
ArmController.set_torque = _real
if getattr(ctrl, "side", None) != side:
    sys.exit("ArmController が side を受け付けていない (apply_arm_profiles.py 未適用?)")
 
print("=" * 60)
print(f"{side} 腕: 腕部を 1 関節ずつ +{deg:.0f}° 動かして戻す。期待する回転方向:")
for k in (2, 3, 4):
    print(f"  {URDF_PLUS[side][k]}")
print("  - 周囲に障害物なし、12V スイッチのそばに手。異常時は Ctrl+C (自動で脱力)")
if input("確認したら yes: ").strip() != "yes":
    sys.exit("中止")
 
RATE = 50
 
 
def send_ramp(hold, k, a0, a1, sec):
    n = max(1, int(sec * RATE))
    for i in range(n + 1):
        tgt = hold.copy()
        tgt[k] = hold[k] + math.radians(a0 + (a1 - a0) * i / n)
        ctrl.set_pos(tgt)
        time.sleep(1.0 / RATE)
 
 
try:
    ctrl.set_torque(1)
    time.sleep(0.5)
    p = ctrl.last_position
    if p is None:
        raise RuntimeError("位置フィードバックがない")
    hold = np.array(p, dtype=float)
    hold[5] = float(np.clip(hold[5], ctrl.joint_min[5], ctrl.joint_max[5]))
    print("hold: " + "  ".join(f"{math.degrees(v):+6.1f}°" for v in hold[:5]) + f"  grip={hold[5]*1000:.1f}mm")
    for k in (2, 3, 4):
        print(f"\n→ {URDF_PLUS[side][k]}  を +{deg:.0f}°")
        if hold[k] + math.radians(deg) > ctrl.joint_max[k]:
            print(f"   スキップ: {math.degrees(hold[k]):+.1f}° + {deg:.0f}° が上限 "
                  f"{math.degrees(ctrl.joint_max[k]):+.0f}° を超える。零点付近に戻してから再実行")
            continue
        send_ramp(hold, k, 0, deg, 1.0)
        t_end = time.time() + 1.5
        while time.time() < t_end:
            ctrl.set_pos(np.concatenate([hold[:k], [hold[k] + math.radians(deg)], hold[k + 1:]]))
            time.sleep(1.0 / RATE)
        now = ctrl.last_position
        print(f"   読み値: {math.degrees(hold[k]):+6.1f}° → {math.degrees(now[k]):+6.1f}°")
        send_ramp(hold, k, deg, 0, 1.0)
        time.sleep(0.5)
 
    # グリッパ: 現在値から grip_mm 開いて戻す (全閉側へは動かさない)
    print(f"\n→ グリッパ: {grip_mm:.0f} mm 開く (片顎)。開く方向に動けば OK")
    g0 = hold[5]
    if g0 + grip_mm / 1000 > ctrl.joint_max[5]:
        raise RuntimeError(f"グリッパ {g0*1000:.1f} + {grip_mm:.0f} mm が上限を超える")
    n = RATE
    for i in list(range(n + 1)) + [n] * int(1.5 * RATE) + list(range(n, -1, -1)):
        tgt = hold.copy()
        tgt[5] = g0 + grip_mm / 1000 * i / n
        ctrl.set_pos(tgt)
        time.sleep(1.0 / RATE)
    print(f"   読み値: {g0*1000:.1f} mm → (開いて戻した)")
except KeyboardInterrupt:
    print("\n中断")
finally:
    ctrl.set_torque(0)
    time.sleep(0.5)
    print("set_torque(0) 送信済み")