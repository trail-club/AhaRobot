#!/usr/bin/env python3
"""接地した台車をキーボードで走らせる (ROS なし、USB 直結)。

aha_base (overlay_ws/src/aha_base) の odrive_usb.py / diff_drive.py を使い、
base_node.py と同じ方式で動かす。位置制御 + 台形軌道で、20 Hz ごとに目標位置を
速度 × 経過時間だけ進め、現在位置から ±2 turn で頭打ちにする。

走行キーは押している間だけ有効で、最後のキー入力から --hold 秒 (既定 0.6 s) で速度 0 になる。
端末のキーリピート開始までの遅れより長くする。
ゲインと Kt はデバイスの保存値を使う。current_lim は RAM で --current-lim (既定 5 A)。
ODrive の watchdog (0.5 s) を使い、このスクリプトが止まると両軸 IDLE になる。
Iq が --i-abort (既定 4 A) を 0.3 秒超えるか、軸が閉ループから外れたら両軸 IDLE にする。
終了時は両軸 IDLE にして RAM の値を戻す。設定は保存しない。

キー:
  g          閉ループ開始 (その場で保持)       b      IDLE (脱力)
  w / ↑      前進      s / ↓  後退            a / ←  左旋回    d / →  右旋回
  space      速度 0 (保持)                    1-4    速度段 25/50/75/100% (開始時 50%)
  r          odom リセット                     q      終了 (Ctrl-C も可)

Usage:  ground_teleop.py [--max-lin 0.4] [--max-ang 1.0] [--hold 0.6]
          [--current-lim 5] [--i-abort 4] [--csv PATH] [--serial HEX] [--aha-base DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import signal
import sys
import termios
import time
import tty

RATE = 20.0
REPORT_S = 0.2
OVERCURRENT_S = 0.3
SIGNS = (-1, 1)  # base.yaml left_sign / right_sign
WHEEL_D = 0.139  # base.yaml wheel_diameter [m]
WHEEL_SEP = 0.44  # base.yaml wheel_separation [m]
MAX_LEAD = 2.0  # base.yaml max_lead_turns
TRAJ_VEL = 2.0  # base.yaml traj_vel_limit [turn/s]
KEYS = {
    "w": (1, 0),
    "up": (1, 0),
    "s": (-1, 0),
    "down": (-1, 0),
    "a": (0, 1),
    "left": (0, 1),
    "d": (0, -1),
    "right": (0, -1),
}
LEVELS = {"1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0}


class Hangup(Exception):
    pass


def on_signal(signum, frame):
    raise Hangup(signum)


def import_aha_base(path: str | None):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else []
    # リポジトリ内、または aha_base/ を隣に置いたコピー
    candidates += [
        os.path.join(here, "..", "..", "overlay_ws", "src", "aha_base"),
        here,
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "aha_base", "odrive_usb.py")):
            sys.path.insert(0, os.path.abspath(c))
            break
    else:
        sys.exit(
            "aha_base が見つからない。--aha-base で aha_base/ を含むディレクトリを指定する"
        )


def read_keys(fd) -> list[str]:
    keys = []
    while select.select([fd], [], [], 0)[0]:
        ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":  # 矢印キーは ESC [ A..D
            seq = os.read(fd, 2).decode(errors="ignore")
            keys.append(
                {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "")
            )
        elif ch == "\x03":
            raise KeyboardInterrupt
        else:
            keys.append(ch)
    return keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-lin", type=float, default=0.4, help="m/s (100%%)")
    p.add_argument("--max-ang", type=float, default=1.0, help="rad/s (100%%)")
    p.add_argument(
        "--hold", type=float, default=0.6, help="キー入力が途切れて止めるまで [s]"
    )
    p.add_argument("--current-lim", type=float, default=5.0, help="A (RAM)")
    p.add_argument("--i-abort", type=float, default=4.0, help="A")
    p.add_argument("--csv", help="20 Hz の記録を書き出すパス")
    p.add_argument("--serial", default="", help="ODrive のシリアル (16 進)")
    p.add_argument("--aha-base", help="aha_base/ パッケージを含むディレクトリ")
    args = p.parse_args()
    if not (0 < args.hold <= 2.0):
        p.error("--hold は 0 < hold <= 2")

    import_aha_base(args.aha_base)
    from aha_base.diff_drive import (
        Kinematics,
        Odometry,
        advance_setpoint,
        limit_wheel_speeds,
    )
    from aha_base.odrive_usb import (
        AXIS_STATE_CLOSED_LOOP_CONTROL,
        DriveSettings,
        OdriveBase,
        OdriveError,
        find_odrive,
        normalize_serial,
    )

    kin = Kinematics(WHEEL_D, WHEEL_SEP)
    print("ODrive を探している...", flush=True)
    try:
        dev = find_odrive(normalize_serial(args.serial), 10.0)
    except OdriveError as e:
        print(e, flush=True)
        os._exit(1)
    base = OdriveBase(
        dev,
        (0, 1),
        DriveSettings(
            traj_vel_limit=TRAJ_VEL,
            current_lim=args.current_lim,
            watchdog_timeout=0.5,
        ),
    )
    iq_attr = [ax.motor.current_control for ax in base.axes]

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            "t,running,cmd_v,cmd_w,pos_l,pos_r,vel_l,vel_r,sp_l,sp_r,iq_l,iq_r,"
            "x,y,theta,odom_v,odom_w".split(",")
        )

    running = False
    level = 0.5
    direction = (0, 0)
    last_key = -math.inf
    setpoints = [0.0, 0.0]
    odom = Odometry(kin)
    over_since = None
    msg = ""
    iq_peak = 0.0

    def stop(reason: str) -> None:
        nonlocal running, direction, msg
        fails = base.stop()
        running, direction = False, (0, 0)
        msg = reason + (f" (停止時の失敗: {fails})" if fails else "")

    fd = sys.stdin.fileno()
    term = termios.tcgetattr(fd)
    for s in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(s, on_signal)
    t0 = time.monotonic()
    try:
        tty.setcbreak(fd)
        print(__doc__.split("キー:")[1].split("Usage")[0])
        print(
            f"100% = {args.max_lin:g} m/s・{args.max_ang:g} rad/s、"
            f"current_lim {args.current_lim:g} A、Iq 上限 {args.i_abort:g} A"
        )
        t_report = last = time.monotonic()
        while True:
            now = time.monotonic()
            for k in read_keys(fd):
                if k == "q":
                    return
                if k == "g" and not running:
                    try:
                        setpoints = base.start()
                    except OdriveError as e:
                        base.stop()
                        msg = f"開始できない: {e}"
                        continue
                    for line in base.log:
                        print(f"\n{line}", end="")
                    base.log.clear()
                    print()
                    running, direction, msg = True, (0, 0), "閉ループ開始"
                    over_since, iq_peak = None, 0.0
                    last = now
                elif k == "b" and running:
                    stop("IDLE")
                elif k == " ":
                    direction = (0, 0)
                elif k in LEVELS:
                    level = LEVELS[k]
                elif k == "r":
                    odom = Odometry(kin)
                    iq_peak = 0.0
                elif k in KEYS and running:
                    direction, last_key = KEYS[k], now

            if now - last_key > args.hold:
                direction = (0, 0)
            cmd_v = direction[0] * level * args.max_lin
            cmd_w = direction[1] * level * args.max_ang

            readings = base.read()
            pos = [r[0] for r in readings]
            vel = [r[1] for r in readings]
            iq = [float(c.Iq_measured) for c in iq_attr]
            iq_peak = max(iq_peak, *map(abs, iq))
            wheel_pos = (pos[0] * SIGNS[0], pos[1] * SIGNS[1])
            odom.update(*wheel_pos)
            odom_v, odom_w = kin.body_twist(vel[0] * SIGNS[0], vel[1] * SIGNS[1])

            if running:
                if any(r[2] != AXIS_STATE_CLOSED_LOOP_CONTROL for r in readings):
                    desc = base.describe()
                    stop(f"閉ループから外れた: {desc}")
                elif max(map(abs, iq)) > args.i_abort:
                    over_since = over_since or now
                    if now - over_since > OVERCURRENT_S:
                        stop(f"過電流 Iq={iq[0]:+.2f}/{iq[1]:+.2f} A")
                else:
                    over_since = None
            if running:
                left, right = limit_wheel_speeds(
                    *kin.wheel_speeds(cmd_v, cmd_w), TRAJ_VEL
                )
                speed = (left * SIGNS[0], right * SIGNS[1])
                step = min(now - last, 3.0 / RATE)
                setpoints = [
                    advance_setpoint(sp, v, step, x, MAX_LEAD)
                    for sp, v, x in zip(setpoints, speed, pos)
                ]
                base.command(setpoints)
            last = now

            if writer:
                writer.writerow(
                    [
                        f"{now - t0:.3f}",
                        int(running),
                        cmd_v,
                        cmd_w,
                        *pos,
                        *vel,
                        *setpoints,
                        *iq,
                        odom.x,
                        odom.y,
                        odom.theta,
                        odom_v,
                        odom_w,
                    ]
                )

            if now - t_report >= REPORT_S:
                t_report = now
                sys.stdout.write(
                    f"\r{'RUN ' if running else 'IDLE'} {int(level * 100):3d}% "
                    f"cmd {cmd_v:+.2f}m/s {cmd_w:+.2f}rad/s | "
                    f"odom x={odom.x:+6.2f} y={odom.y:+6.2f} θ={math.degrees(odom.theta):+6.1f}° "
                    f"v={odom_v:+.2f} w={odom_w:+.2f} | "
                    f"Iq L={iq[0]:+5.2f} R={iq[1]:+5.2f} peak={iq_peak:4.2f}A "
                    f"vbus={float(dev.vbus_voltage):4.1f}V {msg}\x1b[K"
                )
                sys.stdout.flush()
                msg = ""
            time.sleep(max(0.0, 1 / RATE - (time.monotonic() - now)))
    except (Hangup, KeyboardInterrupt):
        pass
    finally:
        fails = base.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, term)
        if csv_file:
            csv_file.close()
        print(
            f"\nIDLE。RAM の値を戻した。{base.describe()}"
            + (f" 失敗: {fails}" if fails else "")
        )
        sys.stdout.flush()
        os._exit(0)  # libfibre のスレッドで終了が止まらないように


if __name__ == "__main__":
    main()
