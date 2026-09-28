#!/usr/bin/env python3
"""第一次闭环动作测试：单关节、小幅度正弦往复，带软件限幅和自动脱力。

流程:
  1. 连接（不发送任何命令）
  2. set_torque(1) → 读取当前姿态作为 hold（其余关节、腕部、夹爪都保持这个值）
  3. set_pid()（上游默认参数 p=30）
  4. 原地保持 2 秒，检查误差
  5. 被测关节做 hold ± 振幅 的正弦往复
  6. 回到 hold 保持 1 秒 → set_torque(0)

保护:
  - 软件限幅：joint0 ±85°，joint1 ±70°（相对零点）
  - 被测关节跟随误差 > ERR_ABORT 持续 0.3 秒，或其他关节偏离 hold > OTHER_ABORT → 立即脱力
  - Ctrl+C / 任何异常 → 脱力

用法:
  uv run motion_test.py <关节 0|1> [振幅deg=10] [周期s=8] [次数=2] [端口]
例:
  uv run motion_test.py 0            # joint0，±10°，8 秒一个来回，2 次
  uv run motion_test.py 1 20 8 2
"""

import csv
import datetime
import math
import os
import sys
import time

REPO = os.path.expanduser("~/aharobot/AhaRobot")
sys.path.insert(0, os.path.join(REPO, "upstream", "astra_controller"))

from astra_controller.arm_controller import ArmController  # noqa: E402

LIMIT_DEG = {0: 85.0, 1: 70.0}   # 相对零点的软件限幅（实测可动范围约 ±91.9° / ±78.2°）
ERR_ABORT_DEG = 15.0              # 被测关节跟随误差上限
OTHER_ABORT_DEG = 10.0            # 其他关节偏离 hold 的上限
RATE_HZ = 50

if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
    sys.exit(__doc__)
j = int(sys.argv[1])
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
period = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
cycles = int(sys.argv[4]) if len(sys.argv) > 4 else 2
port = sys.argv[5] if len(sys.argv) > 5 else "/dev/ttyUSB0"
other = 1 - j

# 连接时不发送任何命令（上游 __init__ 普通模式会自动 set_torque(1) + set_pid）
_real_set_torque = ArmController.set_torque
ArmController.set_torque = lambda self, *a, **k: None
ctrl = ArmController(port, do_init=True)
ArmController.set_torque = _real_set_torque

print("=" * 60)
print(f"joint{j}: 振幅 ±{amp:.1f}°，周期 {period:.1f} 秒，{cycles} 次；软件限幅 ±{LIMIT_DEG[j]:.0f}°")
print("  - 腕部大致在中间，夹爪闭合，周围没有障碍物")
print("  - 手放在 12V 电源开关旁；异常时 Ctrl+C（自动脱力），必要时断电")
if input("确认后输入 yes: ").strip() != "yes":
    sys.exit("已取消")


def pos_now():
    p = ctrl.last_position
    return None if p is None else list(p)


rows = []
out = f"motion_test_j{j}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
aborted = None

try:
    ctrl.set_torque(1)
    time.sleep(0.5)
    hold = pos_now()
    if hold is None:
        raise RuntimeError("没有收到位置反馈")
    print(f"hold: j0={math.degrees(hold[0]):+.1f}°  j1={math.degrees(hold[1]):+.1f}°")

    lim = math.radians(LIMIT_DEG[j])
    if abs(hold[j]) + math.radians(amp) > lim:
        raise RuntimeError(f"hold ± 振幅 超出软件限幅 ±{LIMIT_DEG[j]:.0f}°，请先把关节放回零点附近或减小振幅")

    ctrl.set_pid()
    time.sleep(0.1)

    t_hold = 2.0
    t_move = period * cycles
    t_back = 1.0
    t_total = t_hold + t_move + t_back
    t0 = time.time()
    over_since = None
    last_print = -1.0

    while True:
        t = time.time() - t0
        if t > t_total:
            break

        target = list(hold)
        if t_hold <= t < t_hold + t_move:
            tm = t - t_hold
            target[j] = hold[j] + math.radians(amp) * math.sin(2 * math.pi * tm / period)
        target[j] = max(-lim, min(lim, target[j]))
        ctrl.set_pos(target)

        p = pos_now()
        if p is not None:
            err = math.degrees(p[j] - target[j])
            dev_other = math.degrees(p[other] - hold[other])
            rows.append([round(t, 3), math.degrees(target[j]), math.degrees(p[j]), err,
                         math.degrees(p[other]), p[5] * 1000])

            if abs(err) > ERR_ABORT_DEG:
                over_since = over_since or time.time()
                if time.time() - over_since > 0.3:
                    aborted = f"跟随误差过大 ({err:+.1f}°)"
                    break
            else:
                over_since = None
            if abs(dev_other) > OTHER_ABORT_DEG:
                aborted = f"joint{other} 偏离 hold {dev_other:+.1f}°"
                break

            if t - last_print >= 0.5:
                last_print = t
                print(f"[{t:5.1f}s] target={math.degrees(target[j]):+7.1f}°  "
                      f"actual={math.degrees(p[j]):+7.1f}°  err={err:+6.1f}°  "
                      f"joint{other}={math.degrees(p[other]):+6.1f}°")

        time.sleep(1.0 / RATE_HZ)

except KeyboardInterrupt:
    aborted = "Ctrl+C"
except Exception as e:  # noqa: BLE001
    aborted = f"异常: {e}"
finally:
    ctrl.set_torque(0)
    time.sleep(0.5)
    print("已发送 set_torque(0)")

if aborted:
    print(f"*** 中止: {aborted}")

if rows:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", f"j{j}_target_deg", f"j{j}_actual_deg", "err_deg", f"j{other}_deg", "grip_mm"])
        w.writerows(rows)
    moving = [r for r in rows if 2.0 <= r[0] < 2.0 + period * cycles]
    if moving:
        errs = [abs(r[3]) for r in moving]
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"运动段误差: 最大 {max(errs):.1f}°，RMS {rms:.1f}°")
    print(f"{len(rows)} 个采样已保存: {out}")
