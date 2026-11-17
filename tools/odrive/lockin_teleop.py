#!/usr/bin/env python3
"""キーボードでオープンループ (lockin) 回転を操作し、ホール信号の乱れを数える。

ホールの値はホストから USB で読める最速 (約 0.5ms 間隔) で読み続け、1 秒ごとに
次を集計して表示する。設定は保存しない。

- illegal: 0 / 7 を読んだ回数
- skip:    隣り合わない状態へ飛んだ回数
- pulse:   状態が変わって `PULSE_MS` 以内に元へ戻った回数 (ノイズの典型)
- trans:   状態の変化回数。正常なら 6 × |vel| / 2π 回/秒

lockin の速度・電流は回転中に書き換える。0.5.6 は目標速度を上げると accel で
加速し、下げると即座にその速度へ落とすので、刻みは小さく (既定 10 rad/s) する。
回転方向は停止中しか変えられない。

キー:
  s      開始 / 停止           space  即停止 (IDLE)
  ↑ / ↓  速度 ±--vel-step       ← / → 電流 ±1A
  r      回転方向反転 (停止中)  c      累計カウンタをリセット
  q      終了 (IDLE にして抜ける)

Usage:  lockin_teleop.py AXIS [--vel 20] [--current 5] [--vel-max 300]
"""

from __future__ import annotations

import argparse
import math
import select
import signal
import sys
import termios
import time
import tty

import odrive
from odrive.enums import AXIS_STATE_IDLE, AXIS_STATE_LOCKIN_SPIN

# 手回しで観測した正転方向の並び (shadow_count が増える向き)
NEXT = {3: 2, 2: 6, 6: 4, 4: 5, 5: 1, 1: 3}
PREV = {v: k for k, v in NEXT.items()}
PULSE_MS = 3.0  # これより短い滞在で元に戻った状態はノイズとみなす
CURRENT_MAX = 10.0
REPORT_S = 1.0


class Hangup(Exception):
    pass


def on_signal(signum, frame):
    raise Hangup(signum)


def read_keys(fd) -> list[str]:
    keys = []
    while select.select([fd], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # 矢印キーは ESC [ A..D
            seq = sys.stdin.read(2)
            keys.append(
                {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "")
            )
        else:
            keys.append(ch)
    return keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("axis", type=int, choices=[0, 1])
    p.add_argument("--vel", type=float, default=20.0, help="電気角 rad/s")
    p.add_argument("--vel-step", type=float, default=10.0)
    p.add_argument("--vel-max", type=float, default=300.0)
    p.add_argument("--accel", type=float, default=10.0, help="電気角 rad/s^2")
    p.add_argument("--current", type=float, default=5.0)
    args = p.parse_args()

    d = odrive.find_any(timeout=10)
    ax = getattr(d, f"axis{args.axis}")
    enc = ax.encoder
    lk = ax.config.general_lockin
    pp = ax.motor.config.pole_pairs
    d.clear_errors()
    lk.finish_on_vel = lk.finish_on_distance = lk.finish_on_enc_idx = False

    vel = abs(args.vel)
    sign = 1.0
    current = min(args.current, CURRENT_MAX)

    def apply() -> None:
        lk.current = current
        lk.vel = sign * vel
        # fw 0.5.6 の accel は加速度の大きさ。負にすると磁界が正方向へ際限なく
        # 加速する。向きは vel の符号だけで決まり、ramp_distance は使われない。
        lk.accel = abs(args.accel)

    apply()
    running = False
    total = {"illegal": 0, "skip": 0, "pulse": 0, "trans": 0}
    win = dict.fromkeys(total, 0)
    msg = ""

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    for s in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(s, on_signal)
    try:
        tty.setcbreak(fd)
        print(__doc__.split("キー:")[1].split("Usage")[0])
        prev_h = enc.hall_state
        prev_t = time.monotonic()
        before_prev = None  # 1 つ前の状態 (pulse 判定用)
        t_report = prev_t
        shadow0 = enc.shadow_count
        n = 0
        while True:
            for k in read_keys(fd):
                if k == "q":
                    return
                if k == " ":
                    ax.requested_state = AXIS_STATE_IDLE
                    running, msg = False, "停止"
                elif k == "s":
                    if running:
                        ax.requested_state = AXIS_STATE_IDLE
                        running, msg = False, "停止"
                    else:
                        d.clear_errors()
                        apply()
                        ax.requested_state = AXIS_STATE_LOCKIN_SPIN
                        running, msg = True, "開始"
                elif k in ("up", "down"):
                    step = args.vel_step if k == "up" else -args.vel_step
                    vel = min(max(vel + step, args.vel_step), args.vel_max)
                    apply()
                elif k in ("right", "left"):
                    current = min(
                        max(current + (1 if k == "right" else -1), 1.0), CURRENT_MAX
                    )
                    apply()
                elif k == "r":
                    if running:
                        msg = "反転は停止中のみ"
                    else:
                        sign = -sign
                        apply()
                        msg = "方向反転"
                elif k == "c":
                    total = dict.fromkeys(total, 0)
                    msg = "リセット"

            h = enc.hall_state
            t = time.monotonic()
            n += 1
            if h != prev_h:
                win["trans"] += 1
                if h in (0, 7):
                    win["illegal"] += 1
                elif prev_h not in (0, 7) and h not in (NEXT[prev_h], PREV[prev_h]):
                    win["skip"] += 1
                if h == before_prev and (t - prev_t) * 1000 < PULSE_MS:
                    win["pulse"] += 1
                before_prev, prev_h, prev_t = prev_h, h, t

            if t - t_report >= REPORT_S:
                dt = t - t_report
                shadow = enc.shadow_count
                state = ax.current_state
                if running and state != AXIS_STATE_LOCKIN_SPIN:
                    running = False
                    msg = (
                        f"軸が止まった err axis={ax.error:#x} motor={ax.motor.error:#x}"
                    )
                for key in total:
                    total[key] += win[key]
                cmd = sign * vel
                ideal = 6 * vel / (2 * math.pi) if running else 0.0
                sys.stdout.write(
                    f"\r{'RUN ' if running else 'IDLE'} cmd={cmd:+6.0f}rad/s(e)"
                    f"={cmd / pp / (2 * math.pi):+5.2f}rev/s I={current:4.1f}A | "
                    f"/s: trans={win['trans'] / dt:4.0f}(理想{ideal:3.0f}) "
                    f"illegal={win['illegal'] / dt:3.0f} skip={win['skip'] / dt:3.0f} "
                    f"pulse={win['pulse'] / dt:3.0f} | 累計 i={total['illegal']} "
                    f"s={total['skip']} p={total['pulse']} | "
                    f"shadow/s={(shadow - shadow0) / dt:+5.0f} hall={h} "
                    f"ibus={d.ibus:4.2f}A fet={ax.motor.fet_thermistor.temperature:3.0f}C "
                    f"{n / dt:4.0f}Hz {msg}\x1b[K"
                )
                sys.stdout.flush()
                win = dict.fromkeys(total, 0)
                t_report, shadow0, n, msg = t, shadow, 0, ""
    except (Hangup, KeyboardInterrupt):
        pass
    finally:
        ax.requested_state = AXIS_STATE_IDLE
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print(f"\nIDLE。累計 {total}")


if __name__ == "__main__":
    main()
