#!/usr/bin/env python3

"""
用法:
  uv run check_zero.py [端口] [秒数，默认 10]
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

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

# 连接时不发送任何命令（上游 __init__ 普通模式会自动 set_torque(1) + set_pid）
_real_set_torque = ArmController.set_torque
ArmController.set_torque = lambda self, *a, **k: None
ctrl = ArmController(port, do_init=True)
ArmController.set_torque = _real_set_torque

print("=" * 60)
print(f"静置测试：set_torque(1)（不设置 PID）→ 保持 {duration:.0f} 秒 → set_torque(0)")
print("  - 全程不要用手碰机械臂，只观察：上力矩的瞬间各关节有没有转动、转了多少")
print("  - 异响、自行持续转动、明显发热：Ctrl+C（会自动脱力），必要时断电")
if input("确认后输入 yes: ").strip() != "yes":
    sys.exit("已取消")

names = ["j0", "j1", "w12", "w13", "w14"]
out = f"check_zero_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
rows = []

ctrl.set_torque(1)
t_on = time.time()
time.sleep(0.3)  # 丢弃 set_torque(1) 之前的旧反馈
try:
    last_print = 0.0
    while time.time() - t_on < duration:
        p = ctrl.last_position
        t = time.time() - t_on
        if p is not None:
            p = list(p)
            rows.append([round(t, 3)] + [math.degrees(v) for v in p[:5]] + [p[5] * 1000])
            if t - last_print >= 0.5:
                last_print = t
                s = "  ".join(f"{n}={math.degrees(v):+7.1f}°" for n, v in zip(names, p[:5]))
                print(f"[{t:5.1f}s] {s}  grip={p[5] * 1000:5.1f}mm")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("中断")
finally:
    ctrl.set_torque(0)
    time.sleep(0.5)
    print("已发送 set_torque(0)")

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s"] + [f"{n}_deg" for n in names] + ["grip_mm"])
    w.writerows(rows)
print(f"{len(rows)} 个采样已保存: {out}")