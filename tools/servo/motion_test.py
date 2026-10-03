#!/usr/bin/env python3
"""First closed-loop motion test: single joint, small-amplitude sinusoidal sweep, with software limits and auto torque-off.

Flow:
  1. Connect (do not send any command)
  2. set_torque(1) -> read current pose as `hold` (other joints, wrist, and gripper all stay at this value)
  3. set_pid() (upstream default p=30)
  4. Hold in place for 2 seconds, check error
  5. The target joint performs sinusoidal sweep of hold ± amplitude
  6. Return to hold for 1 second -> set_torque(0)

Protection:
  - Software limits: joint0 ±85°, joint1 ±70° (relative to zero)
  - If target joint tracking error > ERR_ABORT for 0.3 s, or other joints deviate from hold > OTHER_ABORT -> release torque immediately
  - Ctrl+C / any exception -> release torque

Usage:
  uv run motion_test.py <joint 0|1> [amplitude_deg=10] [period_s=8] [cycles=2] [port]
Examples:
  uv run motion_test.py 0            # joint0, ±10°, 8-second period, 2 cycles
  uv run motion_test.py 1 20 8 2
"""

import csv
import datetime
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "upstream", "astra_controller"))

from astra_controller.arm_controller import ArmController  # noqa: E402

LIMIT_DEG = {
    0: 85.0,
    1: 70.0,
}  # Software limits relative to zero (measured range of motion ~ ±91.9° / ±78.2°)
ERR_ABORT_DEG = 15.0  # Max tracking error for target joint
OTHER_ABORT_DEG = 10.0  # Max deviation from hold for other joints
RATE_HZ = 50

if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
    sys.exit(__doc__)
j = int(sys.argv[1])
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
period = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
cycles = int(sys.argv[4]) if len(sys.argv) > 4 else 2
port = sys.argv[5] if len(sys.argv) > 5 else "/dev/ttyUSB0"
other = 1 - j

# Do not send any command on connect (upstream __init__ in normal mode auto-calls set_torque(1) + set_pid).
_real_set_torque = ArmController.set_torque
ArmController.set_torque = lambda self, *a, **k: None
ctrl = ArmController(port, do_init=True)
ArmController.set_torque = _real_set_torque

print("=" * 60)
print(
    f"joint{j}: amplitude ±{amp:.1f}°, period {period:.1f} s, {cycles} cycles; software limit ±{LIMIT_DEG[j]:.0f}°"
)
print("  - Wrist roughly centered, gripper closed, no obstacles around")
print(
    "  - Keep your hand near the 12V power switch; on anomaly press Ctrl+C (auto-releases torque), cut power if needed"
)
if input("Type yes to confirm: ").strip() != "yes":
    sys.exit("Cancelled")


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
        raise RuntimeError("No position feedback received")
    print(f"hold: j0={math.degrees(hold[0]):+.1f}°  j1={math.degrees(hold[1]):+.1f}°")

    lim = math.radians(LIMIT_DEG[j])
    if abs(hold[j]) + math.radians(amp) > lim:
        raise RuntimeError(
            f"hold ± amplitude exceeds software limit ±{LIMIT_DEG[j]:.0f}°; return joint to near zero or reduce amplitude"
        )

    ctrl.set_pid()
    time.sleep(0.1)

    t_hold = 2.0
    t_move = period * cycles
    t_back = 1.0
    t_total = t_hold + t_move + t_back
    t0 = time.time()
    over_since = None
    last_print = -1.0
    # compare feedback against previous command to avoid timing-induced error
    prev_target = list(hold)

    while True:
        t = time.time() - t0
        if t > t_total:
            break

        target = list(hold)
        if t_hold <= t < t_hold + t_move:
            tm = t - t_hold
            target[j] = hold[j] + math.radians(amp) * math.sin(
                2 * math.pi * tm / period
            )
        target[j] = max(-lim, min(lim, target[j]))
        ctrl.set_pos(target)

        p = pos_now()
        if p is not None:
            err = math.degrees(p[j] - prev_target[j])
            dev_other = math.degrees(p[other] - hold[other])
            rows.append(
                [
                    round(t, 3),
                    math.degrees(prev_target[j]),
                    math.degrees(p[j]),
                    err,
                    math.degrees(p[other]),
                    p[5] * 1000,
                ]
            )

            if abs(err) > ERR_ABORT_DEG:
                over_since = over_since or time.time()
                if time.time() - over_since > 0.3:
                    aborted = f"tracking error too large ({err:+.1f}°)"
                    break
            else:
                over_since = None
            if abs(dev_other) > OTHER_ABORT_DEG:
                aborted = f"joint{other} deviated from hold by {dev_other:+.1f}°"
                break

            if t - last_print >= 0.5:
                last_print = t
                print(
                    f"[{t:5.1f}s] target={math.degrees(target[j]):+7.1f}°  "
                    f"actual={math.degrees(p[j]):+7.1f}°  err={err:+6.1f}°  "
                    f"joint{other}={math.degrees(p[other]):+6.1f}°"
                )

        prev_target = list(target)
        time.sleep(1.0 / RATE_HZ)

except KeyboardInterrupt:
    aborted = "Ctrl+C"
except Exception as e:  # noqa: BLE001
    aborted = f"exception: {e}"
finally:
    ctrl.set_torque(0)
    time.sleep(0.5)
    print("set_torque(0) sent")

if aborted:
    print(f"*** Aborted: {aborted}")

if rows:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_s",
                f"j{j}_target_deg",
                f"j{j}_actual_deg",
                "err_deg",
                f"j{other}_deg",
                "grip_mm",
            ]
        )
        w.writerows(rows)
    moving = [r for r in rows if 2.0 <= r[0] < 2.0 + period * cycles]
    if moving:
        errs = [abs(r[3]) for r in moving]
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"Motion segment error: max {max(errs):.1f}°, RMS {rms:.1f}°")
    print(f"{len(rows)} samples saved: {out}")
