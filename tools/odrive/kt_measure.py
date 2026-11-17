#!/usr/bin/env python3
"""逆起電力からトルク定数 Kt を測る。

速度制御で一定速度を正逆に回し、後半の平均から (Vq − R·Iq) ÷ phase_vel を求め、
直線の傾き λ [Vs/rad (電気角)] から Kt = 1.5 × 極対数 × λ を計算する。
fw 0.5.6 は電流制御中に `Vq_setpoint` を更新しないので、電流制御の積分項
`v_current_control_integral_q` を Vq とみなす (定常では比例項がほぼ 0)。

速度ループのゲインは A/(turn/s) で指定し、現在の Kt で Nm へ換算して RAM に入れる。
設定は保存せず、終了時に元へ戻して IDLE にする。車輪を浮かせて使う。

Usage:  kt_measure.py AXIS [--speeds 0.5,1.0,1.5] [--secs 6]
"""

from __future__ import annotations

import argparse
import signal
import statistics as st
import time

import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

I_ABORT = 4.5


class Abort(Exception):
    pass


def on_signal(signum, frame):
    raise Abort(f"signal {signum}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("axis", type=int, choices=[0, 1])
    p.add_argument("--speeds", default="0.5,1.0,1.5", help="turn/s (正逆の両方を回す)")
    p.add_argument(
        "--secs", type=float, default=6.0, help="1 速度あたり。後半半分を平均する"
    )
    p.add_argument("--vel-gain-a", type=float, default=0.5, help="A/(turn/s)")
    p.add_argument("--vel-int-a", type=float, default=2.5, help="A/turn")
    args = p.parse_args()
    speeds = [float(s) for s in args.speeds.split(",")]
    speeds = speeds + [-s for s in speeds]

    d = odrive.find_any(timeout=15)
    ax = getattr(d, f"axis{args.axis}")
    mot, enc, ctrl = ax.motor, ax.encoder, ax.controller
    cc = mot.current_control
    kt0 = mot.config.torque_constant
    r_ph = mot.config.phase_resistance
    pp = mot.config.pole_pairs

    changes = {
        (mot.config, "current_lim"): 5.0,
        (ctrl.config, "vel_limit"): max(map(abs, speeds)) + 0.5,
        (ctrl.config, "control_mode"): CONTROL_MODE_VELOCITY_CONTROL,
        (ctrl.config, "input_mode"): INPUT_MODE_PASSTHROUGH,
        (ctrl.config, "vel_gain"): args.vel_gain_a * kt0,
        (ctrl.config, "vel_integrator_gain"): args.vel_int_a * kt0,
    }
    saved = {k: getattr(*k) for k in changes}
    for (obj, name), val in changes.items():
        setattr(obj, name, val)
    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)

    print(
        f"axis{args.axis} Kt(現在)={kt0:.4f} R={r_ph:.4f} pole_pairs={pp}", flush=True
    )
    pts = []
    try:
        d.clear_errors()
        ctrl.input_vel = 0.0
        ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        time.sleep(0.5)
        for v in speeds:
            ctrl.input_vel = v
            t0 = time.time()
            rows = []
            while (t := time.time() - t0) < args.secs:
                if ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    raise Abort(
                        f"閉ループを抜けた axis={ax.error:#x} motor={mot.error:#x} "
                        f"enc={enc.error:#x} ctrl={ctrl.error:#x}"
                    )
                iq = cc.Iq_measured
                if abs(iq) > I_ABORT:
                    raise Abort(f"Iq {iq:+.2f}A")
                if t > args.secs / 2:
                    rows.append(
                        (
                            enc.vel_estimate,
                            cc.phase_vel,
                            cc.v_current_control_integral_q,
                            iq,
                        )
                    )
                time.sleep(0.01)
            vel, pv, vq, iq = (st.mean(r[i] for r in rows) for i in range(4))
            pts.append((pv, vq - r_ph * iq))
            print(
                f"v={v:+.2f} vel={vel:+.3f} phase_vel={pv:+7.2f} Vq={vq:+.3f}V "
                f"Iq={iq:+.3f}A λ={(vq - r_ph * iq) / pv:.5f}",
                flush=True,
            )
        ctrl.input_vel = 0.0
        time.sleep(1.0)
    except Abort as e:
        print("中止:", e, flush=True)
    finally:
        ax.requested_state = AXIS_STATE_IDLE
        ctrl.input_vel = 0.0
        for (obj, name), val in saved.items():
            setattr(obj, name, val)

    if len(pts) == len(speeds):
        mx = st.mean(x for x, _ in pts)
        my = st.mean(y for _, y in pts)
        lam = sum((x - mx) * (y - my) for x, y in pts) / sum(
            (x - mx) ** 2 for x, _ in pts
        )
        kt = 1.5 * pp * lam
        print(
            f"λ={lam:.5f} Vs/rad 切片={my - lam * mx:+.4f}V → Kt={kt:.4f} Nm/A "
            f"(8.27/Kt={8.27 / kt:.1f})\n"
            f"保存する場合: odrv0.axis{args.axis}.motor.config.torque_constant = {kt:.3f}; "
            "odrv0.save_configuration()",
            flush=True,
        )
    print(f"IDLE errors axis={ax.error:#x} motor={mot.error:#x}", flush=True)


if __name__ == "__main__":
    main()
