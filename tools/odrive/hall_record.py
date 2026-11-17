#!/usr/bin/env python3
"""オープンループ (lockin) の条件ごとにホール信号の生データを記録する。

条件ごとに CSV を 1 つ書き、全体の条件と結果を summary.json にまとめる。
CSV の各行はホストが読んだ hall_state。`AUX_EVERY` 行ごとに shadow_count・
vel_estimate・電流・電圧・FET 温度と、フィルタ付きファームの内部カウンタ
(hall_filtered_cnt / hall_rejected_count / hall_illegal_count) を併記する。
フィルタ付きファームでは条件ごとに `encoder.hall_debounce_cycles` を設定する。
標準ファームでは debounce 列を無視して記録だけ行う。

ODrive の設定は保存しない。エラー、FET 過熱、シグナル受信で IDLE にして中止する。
lockin の accel は fw 0.5.6 では加速度の大きさで、負にすると磁界が正方向へ
際限なく加速する。向きは vel の符号だけで決まる。

Usage:  hall_record.py AXIS OUTDIR [--plan baseline|filter|idle]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import odrive
from odrive.enums import AXIS_STATE_IDLE, AXIS_STATE_LOCKIN_SPIN

FET_MAX_C = 60.0
COOLDOWN_S = 10.0
AUX_EVERY = 50
ACCEL = 20.0  # 電気角 rad/s^2。正の値のみ

# (名前, 電流 A, 電気角 rad/s, 定速区間 s, debounce cycles または None)
# 電流 0 は無通電。debounce None はファームの値を変えない
PLANS = {
    # 2026-11-15 に標準ファームで記録した条件 (逆転は不具合修正後に追加)
    "baseline": [
        ("idle_0A", 0, 0, 10, None),
        ("sweep_angle_5A_v2", 5, 2, 10, None),
        ("speed_5A_v20", 5, 20, 10, None),
        ("speed_5A_v40", 5, 40, 10, None),
        ("speed_5A_v80", 5, 80, 10, None),
        ("speed_5A_v120", 5, 120, 10, None),
        ("current_2A_v20", 2, 20, 10, None),
        ("current_3A_v20", 3, 20, 10, None),
        ("current_8A_v20", 8, 20, 10, None),
        ("reverse_5A_v-20", 5, -20, 10, None),
        ("reverse_5A_v-80", 5, -80, 10, None),
        ("idle_0A_end", 0, 0, 10, None),
    ],
    # 無通電だけの動作確認 (モーターは回さない)
    "idle": [("idle_0A", 0, 0, 3, None)],
    # フィルタ付きファームで debounce を切り替えて比べる
    "filter": [("idle_0A", 0, 0, 10, 0)]
    + [
        (f"{tag}_N{n}", cur, vel, 10, n)
        for tag, cur, vel in [
            ("5A_v20", 5, 20),
            ("5A_v80", 5, 80),
            ("5A_v-20", 5, -20),
            ("5A_v-80", 5, -80),
            ("2A_v20", 2, 20),
        ]
        for n in (0, 2, 4)
    ]
    + [("idle_0A_end", 0, 0, 10, 0)],
}


class Abort(Exception):
    pass


def on_signal(signum, frame):
    raise Abort(f"signal {signum}")


def fw_counters(enc) -> dict:
    """フィルタ付きファームの内部カウンタ。標準ファームでは空。"""
    if not hasattr(enc, "hall_rejected_count"):
        return {}
    return {
        "filtered_cnt": enc.hall_filtered_cnt,
        "rejected": enc.hall_rejected_count,
        "illegal": enc.hall_illegal_count,
    }


def record(d, ax, name, current, vel, steady, debounce, outdir):
    enc, mot = ax.encoder, ax.motor
    d.clear_errors()
    has_filter = hasattr(enc, "hall_debounce_cycles")
    if has_filter and debounce is not None:
        enc.hall_debounce_cycles = debounce
    run_s = steady
    if current > 0:
        lk = ax.config.general_lockin
        lk.current = current
        lk.vel = vel
        lk.accel = ACCEL
        lk.finish_on_vel = lk.finish_on_distance = lk.finish_on_enc_idx = False
        run_s = lk.ramp_time + abs(vel) / ACCEL + steady
        ax.requested_state = AXIS_STATE_LOCKIN_SPIN
    path = os.path.join(outdir, f"{name}.csv")
    n = 0
    fet_max = 0.0
    c_start = c_steady = c_end = {}
    t_steady = run_s - steady
    with open(path, "w") as f:
        f.write("t,hall,shadow,vel_est,ibus,iq,id,vbus,fet,filt,rejected,illegal\n")
        t0 = time.monotonic()
        c_start = fw_counters(enc)
        while (t := time.monotonic() - t0) < run_s:
            h = enc.hall_state
            if n % AUX_EVERY == 0:
                fet = mot.fet_thermistor.temperature
                fet_max = max(fet_max, fet)
                c = fw_counters(enc)
                if not c_steady and t >= t_steady:
                    c_steady = dict(c, t=t) if c else {"t": t}
                f.write(
                    f"{t:.5f},{h},{enc.shadow_count},{enc.vel_estimate:.4f},{d.ibus:.3f},"
                    f"{mot.current_control.Iq_measured:.3f},"
                    f"{mot.current_control.Id_measured:.3f},{d.vbus_voltage:.2f},{fet:.1f},"
                    f"{c.get('filtered_cnt', '')},{c.get('rejected', '')},{c.get('illegal', '')}\n"
                )
                if current > 0 and ax.current_state != AXIS_STATE_LOCKIN_SPIN:
                    raise Abort(
                        f"{name}: left LOCKIN_SPIN axis={ax.error:#x} motor={mot.error:#x}"
                    )
                if fet > FET_MAX_C:
                    raise Abort(f"{name}: FET {fet:.1f}C")
            else:
                f.write(f"{t:.5f},{h},,,,,,,,,,\n")
            n += 1
        c_end = dict(fw_counters(enc), t=time.monotonic() - t0)
    ax.requested_state = AXIS_STATE_IDLE
    return {
        "name": name,
        "current": current,
        "vel": vel,
        "debounce": enc.hall_debounce_cycles if has_filter else None,
        "steady_from": t_steady,
        "samples": n,
        "duration": run_s,
        "fet_max": fet_max,
        "fw_counters": {"start": c_start, "steady": c_steady, "end": c_end},
        "axis_error": ax.error,
        "motor_error": mot.error,
        "encoder_error": enc.error,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("axis", type=int, choices=[0, 1])
    p.add_argument("outdir")
    p.add_argument("--plan", choices=sorted(PLANS), default="baseline")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)
    d = odrive.find_any(timeout=15)
    ax = getattr(d, f"axis{args.axis}")
    enc = ax.encoder
    has_filter = hasattr(enc, "hall_debounce_cycles")
    debounce0 = enc.hall_debounce_cycles if has_filter else None
    summary = {
        "axis": args.axis,
        "plan": args.plan,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fw_version": f"{d.fw_version_major}.{d.fw_version_minor}.{d.fw_version_revision}",
        "fw_hall_filter": has_filter,
        "pole_pairs": ax.motor.config.pole_pairs,
        "phase_resistance": ax.motor.config.phase_resistance,
        "encoder_direction": enc.config.direction,
        "encoder_bandwidth": enc.config.bandwidth,
        "ignore_illegal_hall_state": enc.config.ignore_illegal_hall_state,
        "accel": ACCEL,
        "runs": [],
    }
    plan = PLANS[args.plan]
    try:
        for i, (name, cur, vel, steady, deb) in enumerate(plan):
            print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
            r = record(d, ax, name, cur, vel, steady, deb, args.outdir)
            summary["runs"].append(r)
            print(f"  samples={r['samples']} fet_max={r['fet_max']:.1f}C", flush=True)
            if cur > 0 and i < len(plan) - 1:
                time.sleep(COOLDOWN_S)
    except Abort as e:
        summary["aborted"] = str(e)
        print("ABORT:", e, flush=True)
    finally:
        ax.requested_state = AXIS_STATE_IDLE
        if has_filter:
            enc.hall_debounce_cycles = debounce0
        summary["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("done, IDLE", flush=True)


if __name__ == "__main__":
    main()
