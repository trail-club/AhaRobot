#!/usr/bin/env python3
"""キーボードで閉ループ制御を操作し、追従と電流を表示する。

既定の `--mode ros` は `base_controller.py` と同じ方式で動かす。位置制御 + 台形軌道で、
0.1 秒ごとに目標速度 × 0.1 秒を目標位置へ足して送り、現在位置から ±2 turn で頭打ちにする。
`--mode vel` は速度制御 (VEL_RAMP) で直接速度を指令する。

ゲインと Kt はデバイスの値を使い、引数で指定したものだけ RAM で上書きする。
設定は保存せず、終了時に変えた値を元へ戻して IDLE にする。
Iq が電流制限の 90% を 0.3 秒超えたら IDLE にする。車輪を浮かせ、電源をすぐ切れる状態で使う。

速度は車輪の turn/s (モーター軸の正方向)。1 turn/s は直径 0.139 m の車輪で 0.437 m/s。

キー:
  s      閉ループ開始 / 停止       space  即停止 (IDLE)
  ↑ / ↓  速度 ±--vel-step         x / 0  速度 0
  r      カウンタ・最大値リセット   q      終了 (IDLE にして抜ける)

Usage:  closed_loop_teleop.py AXIS [--mode ros|vel] [--vel-step 0.1] [--vel-max 1.0]
          [--pos-gain G] [--vel-gain G] [--vel-int G] [--kt KT] [--current-lim 5]
"""

from __future__ import annotations

import argparse
import math
import select
import signal
import statistics
import sys
import termios
import time
import tty

import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_POSITION_CONTROL,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_TRAP_TRAJ,
    INPUT_MODE_VEL_RAMP,
)

WHEEL_D = 0.139  # base_controller.py と同じ [m]
ROS_DT = 0.1  # base_controller.py が目標位置を送る周期 [s]
ROS_MAX_POS_DIFF = 2.0  # base_controller.py の頭打ち [turn]
REPORT_S = 0.2
START_GRACE_S = 1.0  # 閉ループへ切り替わるまで待つ時間
OVERCURRENT_S = 0.3


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
    p.add_argument("--mode", choices=["ros", "vel"], default="ros")
    p.add_argument("--vel-step", type=float, default=0.1, help="turn/s")
    p.add_argument("--vel-max", type=float, default=1.0, help="turn/s")
    p.add_argument("--accel", type=float, default=1.0, help="turn/s^2 (ROS と同じ)")
    p.add_argument("--pos-gain", type=float)
    p.add_argument("--vel-gain", type=float)
    p.add_argument("--vel-int", type=float)
    p.add_argument("--kt", type=float)
    p.add_argument("--current-lim", type=float, default=5.0)
    args = p.parse_args()

    d = odrive.find_any(timeout=10)
    ax = getattr(d, f"axis{args.axis}")
    mot, enc, ctrl, tt = ax.motor, ax.encoder, ax.controller, ax.trap_traj
    cc = mot.current_control

    # 変える設定を (オブジェクト, 属性) で覚え、終了時に戻す
    changes = {
        (mot.config, "current_lim"): args.current_lim,
        (ctrl.config, "vel_limit"): args.vel_max + 0.5,
        (ctrl.config, "vel_ramp_rate"): args.accel,
        (tt.config, "vel_limit"): args.vel_max,
        (tt.config, "accel_limit"): args.accel,
        (tt.config, "decel_limit"): args.accel,
        (ctrl.config, "control_mode"): CONTROL_MODE_POSITION_CONTROL
        if args.mode == "ros"
        else CONTROL_MODE_VELOCITY_CONTROL,
        (ctrl.config, "input_mode"): INPUT_MODE_TRAP_TRAJ
        if args.mode == "ros"
        else INPUT_MODE_VEL_RAMP,
    }
    for (obj, name), val in (
        ((ctrl.config, "pos_gain"), args.pos_gain),
        ((ctrl.config, "vel_gain"), args.vel_gain),
        ((ctrl.config, "vel_integrator_gain"), args.vel_int),
        ((mot.config, "torque_constant"), args.kt),
    ):
        if val is not None:
            changes[(obj, name)] = val
    saved = {k: getattr(*k) for k in changes}
    for (obj, name), val in changes.items():
        setattr(obj, name, val)

    i_abort = 0.9 * args.current_lim
    gains = (
        f"Kt={mot.config.torque_constant:.3f} pos_gain={ctrl.config.pos_gain:g} "
        f"vel_gain={ctrl.config.vel_gain:g} "
        f"({ctrl.config.vel_gain / mot.config.torque_constant:.2f}A/(turn/s)) "
        f"vel_int={ctrl.config.vel_integrator_gain:g}"
    )
    has_filter = hasattr(enc, "hall_rejected_count")

    vel_cmd = 0.0
    setpos = enc.pos_estimate
    running = False
    msg = ""
    iq_peak = 0.0
    over_since = None
    started_at = 0.0

    def stop(reason: str) -> None:
        nonlocal running, msg, vel_cmd
        ax.requested_state = AXIS_STATE_IDLE
        running, vel_cmd, msg = False, 0.0, reason

    fd = sys.stdin.fileno()
    term = termios.tcgetattr(fd)
    for s in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(s, on_signal)
    try:
        tty.setcbreak(fd)
        print(f"axis{args.axis} mode={args.mode} {gains}")
        print(__doc__.split("キー:")[1].split("Usage")[0])
        t_ros = t_report = time.monotonic()
        rej0 = enc.hall_rejected_count if has_filter else 0
        win_iq: list[float] = []
        while True:
            now = time.monotonic()
            for k in read_keys(fd):
                if k == "q":
                    return
                if k == " ":
                    stop("停止")
                elif k == "s":
                    if running:
                        stop("停止")
                    else:
                        d.clear_errors()
                        vel_cmd = 0.0
                        setpos = enc.pos_estimate
                        ctrl.input_pos = setpos
                        ctrl.input_vel = 0.0
                        ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                        running, msg = True, "開始"
                        started_at = now
                elif k in ("up", "down") and running:
                    step = args.vel_step if k == "up" else -args.vel_step
                    vel_cmd = round(
                        min(max(vel_cmd + step, -args.vel_max), args.vel_max), 3
                    )
                elif k in ("x", "0"):
                    vel_cmd = 0.0
                elif k == "r":
                    iq_peak, msg = 0.0, "リセット"

            pos = enc.pos_estimate
            if running and now - t_ros >= ROS_DT:
                if args.mode == "ros":
                    setpos += vel_cmd * (now - t_ros)
                    setpos = min(
                        max(setpos, pos - ROS_MAX_POS_DIFF), pos + ROS_MAX_POS_DIFF
                    )
                    ctrl.input_pos = setpos
                else:
                    ctrl.input_vel = vel_cmd
                t_ros = now
            elif not running:
                t_ros = now

            iq = cc.Iq_measured
            win_iq.append(iq)
            iq_peak = max(iq_peak, abs(iq))
            if running:
                if (
                    ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL
                    and now - started_at > START_GRACE_S
                ):
                    stop(
                        f"軸が止まった axis={ax.error:#x} motor={mot.error:#x} "
                        f"enc={enc.error:#x} ctrl={ctrl.error:#x}"
                    )
                elif abs(iq) > i_abort:
                    over_since = over_since or now
                    if now - over_since > OVERCURRENT_S:
                        stop(f"過電流 Iq={iq:+.2f}A")
                else:
                    over_since = None

            if now - t_report >= REPORT_S:
                dt = now - t_report
                rej = enc.hall_rejected_count if has_filter else 0
                vel = enc.vel_estimate
                err = pos - ctrl.pos_setpoint if args.mode == "ros" else 0.0
                iq_sd = statistics.pstdev(win_iq) if len(win_iq) > 1 else 0.0
                sys.stdout.write(
                    f"\r{'RUN ' if running else 'IDLE'} cmd={vel_cmd:+5.2f}turn/s"
                    f"({vel_cmd * math.pi * WHEEL_D:+5.2f}m/s) "
                    f"vel={vel:+6.3f} "
                    + (
                        f"lag={setpos - pos:+6.3f} err={err:+6.3f}turn "
                        if args.mode == "ros"
                        else ""
                    )
                    + f"| Iq={iq:+5.2f} sd={iq_sd:4.2f} peak={iq_peak:4.2f}A "
                    f"rej/s={(rej - rej0) / dt:5.0f} "
                    f"vbus={d.vbus_voltage:4.1f}V fet={mot.fet_thermistor.temperature:3.0f}C "
                    f"{msg}\x1b[K"
                )
                sys.stdout.flush()
                t_report, rej0, win_iq, msg = now, rej, [], ""
            time.sleep(0.005)
    except (Hangup, KeyboardInterrupt):
        pass
    finally:
        ax.requested_state = AXIS_STATE_IDLE
        ctrl.input_vel = 0.0
        for (obj, name), val in saved.items():
            setattr(obj, name, val)
        termios.tcsetattr(fd, termios.TCSADRAIN, term)
        print(
            f"\nIDLE。設定を戻した。errors axis={ax.error:#x} motor={mot.error:#x} "
            f"enc={enc.error:#x} ctrl={ctrl.error:#x}"
        )


if __name__ == "__main__":
    main()
