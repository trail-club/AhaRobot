#!/usr/bin/env python3

"""
Usage:
  uv run check_zero.py [port] [seconds, default 10]
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

# Do not send any command on connect (upstream __init__ in normal mode auto-calls set_torque(1) + set_pid).
_real_set_torque = ArmController.set_torque
ArmController.set_torque = lambda self, *a, **k: None
ctrl = ArmController(port, do_init=True)
ArmController.set_torque = _real_set_torque

print("=" * 60)
print(
    f"Static test: set_torque(1) (no PID set) -> hold for {duration:.0f} seconds -> set_torque(0)"
)
print(
    "  - Do not touch the arm. Just observe: do joints rotate the moment torque is enabled, and by how much?"
)
print(
    "  - On abnormal noise, continuous rotation, or obvious heating: Ctrl+C (auto-releases torque); cut power if needed"
)
if input("Type yes to confirm: ").strip() != "yes":
    sys.exit("Cancelled")

names = ["j0", "j1", "w12", "w13", "w14"]
out = f"check_zero_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
rows = []

ctrl.set_torque(1)
t_on = time.time()
time.sleep(0.3)  # discard stale feedback from before set_torque(1)
try:
    last_print = 0.0
    while time.time() - t_on < duration:
        p = ctrl.last_position
        t = time.time() - t_on
        if p is not None:
            p = list(p)
            rows.append(
                [round(t, 3)] + [math.degrees(v) for v in p[:5]] + [p[5] * 1000]
            )
            if t - last_print >= 0.5:
                last_print = t
                s = "  ".join(
                    f"{n}={math.degrees(v):+7.1f}°" for n, v in zip(names, p[:5])
                )
                print(f"[{t:5.1f}s] {s}  grip={p[5] * 1000:5.1f}mm")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Interrupted")
finally:
    ctrl.set_torque(0)
    time.sleep(0.5)
    print("set_torque(0) sent")

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s"] + [f"{n}_deg" for n in names] + ["grip_mm"])
    w.writerows(rows)
print(f"{len(rows)} samples saved: {out}")
