#!/usr/bin/env python3
"""オープンループの一定速度回転中に、PLL帯域ごとの vel_estimate のばらつきを比べる。

lockin の真の速度は指令値なので、vel_estimate の標準偏差がホールのノイズによる
推定誤差になる。終了時に bandwidth を元へ戻し、設定は保存しない。

Usage:  pll_bw_test.py AXIS [--current 5] [--vel 20] [--bw 20 50 100 200]
"""

import argparse
import math
import statistics
import time

import odrive
from odrive.enums import AXIS_STATE_IDLE, AXIS_STATE_LOCKIN_SPIN
from odrive.utils import dump_errors

p = argparse.ArgumentParser()
p.add_argument("axis", type=int, choices=[0, 1])
p.add_argument("--current", type=float, default=5.0)
p.add_argument("--vel", type=float, default=20.0)
p.add_argument("--accel", type=float, default=10.0)
p.add_argument("--bw", type=float, nargs="+", default=[20, 50, 100, 200])
p.add_argument("--window", type=float, default=5.0)
args = p.parse_args()

d = odrive.find_any(timeout=10)
ax = getattr(d, f"axis{args.axis}")
enc = ax.encoder
pp = ax.motor.config.pole_pairs
bw0 = enc.config.bandwidth
d.clear_errors()

lk = ax.config.general_lockin
lk.current = args.current
lk.vel = args.vel
# fw 0.5.6: accel is a ramp magnitude and must be > 0 (negative makes the
# field accelerate forward without bound); direction comes from vel only.
# ramp_distance is unused by 0.5.6.
lk.accel = abs(args.accel)
lk.finish_on_vel = lk.finish_on_distance = lk.finish_on_enc_idx = False

# vel_estimate is already in the direction-corrected frame (lockin + -> +)
true_vel = args.vel / (2 * math.pi * pp)  # turn/s
print(f"bw0={bw0} expected vel_estimate={true_vel:+.4f} turn/s")
try:
    ax.requested_state = AXIS_STATE_LOCKIN_SPIN
    time.sleep(lk.ramp_time + abs(args.vel / args.accel) + 1.0)
    for bw in args.bw:
        enc.config.bandwidth = bw
        time.sleep(0.5)  # let the PLL settle
        v, glitch, prev, n = [], 0, enc.hall_state, 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.window:
            v.append(enc.vel_estimate)
            h = enc.hall_state
            glitch += h in (0, 7)
            n += 1
            if ax.current_state != AXIS_STATE_LOCKIN_SPIN:
                raise RuntimeError("axis left LOCKIN_SPIN")
        m = statistics.mean(v)
        sd = statistics.pstdev(v)
        dev = max(abs(x - true_vel) for x in v)
        print(
            f"bw={bw:6.0f}: mean={m:+.4f} sd={sd:.4f} ({sd / abs(true_vel) * 100:5.1f}%) "
            f"max|err|={dev:.4f} ({dev / abs(true_vel) * 100:5.1f}%) n={n} illegal={glitch}"
        )
finally:
    ax.requested_state = AXIS_STATE_IDLE
    enc.config.bandwidth = bw0
    time.sleep(0.2)
    print(f"restored bandwidth={enc.config.bandwidth}")
    dump_errors(d)
