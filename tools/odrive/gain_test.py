#!/usr/bin/env python3
"""閉ループのゲインを段階的に変え、保持・ステップ・速度追従を同じ手順で測る。

`--vel-gains` の各値について次を順に行い、条件ごとの指標を表にする。

1. 保持: 現在位置を保つ。Iq のばらつきと、後半 2 秒の位置の振れ幅 (ハンチング)。
   カウント変化回数は、止まっていてもホールの境目で増えるので参考値
2. ステップ: `--step` turn 進めて戻す (PASSTHROUGH)。行き過ぎ・整定時間・残る誤差
3. 速度追従: `base_controller.py` と同じ方式 (台形軌道、0.1 秒ごとに目標位置を進め、
   現在位置から ±2 turn で頭打ち) で、`--speeds` の各速度を前進→停止→後退→停止。
   遅れ・速度のばらつき (sd/平均、スティックスリップの目安)・Iq

`--plan ground` では複数軸を台車の前進方向 (左 axis0 は −1、右 axis1 は +1) へ同時に動かす。
後退で始点へ戻る。必要な直線距離は最大速度 × `--stream-secs` turn (1 turn = 0.437 m)。

保持の Iq sd が `--unstable-iq-sd` を超えるか、位置の振れ幅が `--unstable-pp` カウントを超えたら
発振とみなし、それ以上のゲインは試さない。Iq が電流制限の 90% を 0.3 秒超えたとき、
速度超過、エラーでも IDLE にして止める。設定は保存せず、終了時に元へ戻す。

ゲインの単位は ODrive と同じ Nm/(turn/s) で、Kt で割った A/(turn/s) も表示する。
Kt を測っていない軸では先に kt_measure.py を使う。

Usage:  gain_test.py AXES OUTDIR --vel-gains 0.1,0.15,0.2 [--pos-gain 20] [--vel-int 0]
          [--plan lifted|ground] [--speeds 0.1,0.3,1.0] [--stream-secs 3] [--step 0.25]
          [--current-lim 10] [--vel-limit 4] [--accel 1] [--bandwidth BW] [--gain-scheduling WIDTH]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics as st
import time

import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_POSITION_CONTROL,
    INPUT_MODE_PASSTHROUGH,
    INPUT_MODE_TRAP_TRAJ,
)

GROUND_SIGN = {0: -1.0, 1: 1.0}  # base_controller.py の LEFT/RIGHT_INSTALL_DIR
ROS_DT = 0.1
ROS_MAX_POS_DIFF = 2.0
CPR = 60  # 1 回転のホールカウント (極対数 10 × 6)
OVERCURRENT_S = 0.3
SAMPLE_S = 0.005


class Abort(Exception):
    pass


class Unstable(Exception):
    pass


def on_signal(signum, frame):
    raise Abort(f"signal {signum}")


class Axis:
    def __init__(self, d, n: int, sign: float):
        self.n, self.sign = n, sign
        self.ax = getattr(d, f"axis{n}")
        self.mot, self.enc, self.ctrl = (
            self.ax.motor,
            self.ax.encoder,
            self.ax.controller,
        )
        self.cc = self.mot.current_control
        self.over_since = None

    def errors(self) -> str:
        return (
            f"axis{self.n} axis={self.ax.error:#x} motor={self.mot.error:#x} "
            f"enc={self.enc.error:#x} ctrl={self.ctrl.error:#x}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("axes", help="0 / 1 / 0,1")
    p.add_argument("outdir")
    p.add_argument(
        "--vel-gains", required=True, help="Nm/(turn/s) をカンマ区切りで小さい順に"
    )
    p.add_argument("--pos-gain", type=float, default=20.0)
    p.add_argument("--vel-int", type=float, default=0.0, help="vel_integrator_gain")
    p.add_argument("--vel-int-limit", type=float, help="vel_integrator_limit")
    p.add_argument("--plan", choices=["lifted", "ground"], default="lifted")
    p.add_argument("--speeds", default="0.1,0.3,1.0", help="turn/s")
    p.add_argument("--stream-secs", type=float, default=3.0)
    p.add_argument("--step", type=float, default=0.25, help="turn")
    p.add_argument("--current-lim", type=float, default=10.0)
    p.add_argument(
        "--vel-limit",
        type=float,
        default=4.0,
        help="controller.vel_limit。位置誤差から作る速度指令の上限で、積分なしでは "
        "vel_gain × vel_limit ÷ Kt が出せる電流の上限になる。超過で overspeed エラー",
    )
    p.add_argument(
        "--accel", type=float, default=1.0, help="台形軌道 turn/s^2 (ROS と同じ)"
    )
    p.add_argument("--bandwidth", type=float, help="encoder.config.bandwidth")
    p.add_argument(
        "--gain-scheduling", type=float, help="gain_scheduling_width [turn]。0 で無効"
    )
    p.add_argument("--unstable-iq-sd", type=float, default=1.0, help="A")
    p.add_argument(
        "--unstable-pp",
        type=float,
        default=3.0,
        help="保持後半 2 秒の pos_estimate の振れ幅 [カウント]",
    )
    args = p.parse_args()

    gains = [float(g) for g in args.vel_gains.split(",")]
    speeds = [float(s) for s in args.speeds.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    d = odrive.find_any(timeout=15)
    axes = [
        Axis(d, int(n), GROUND_SIGN[int(n)] if args.plan == "ground" else 1.0)
        for n in args.axes.split(",")
    ]
    vmax = max(speeds)
    changes = {}
    for a in axes:
        cfg, mcfg, tcfg = a.ctrl.config, a.mot.config, a.ax.trap_traj.config
        changes.update(
            {
                (mcfg, "current_lim"): args.current_lim,
                (cfg, "vel_limit"): max(args.vel_limit, vmax + 1.0),
                (cfg, "enable_overspeed_error"): True,
                (cfg, "control_mode"): CONTROL_MODE_POSITION_CONTROL,
                (cfg, "input_mode"): INPUT_MODE_PASSTHROUGH,
                (cfg, "pos_gain"): args.pos_gain,
                (cfg, "vel_gain"): gains[0],
                (cfg, "vel_integrator_gain"): args.vel_int,
                (tcfg, "vel_limit"): vmax + 1.0,
                (tcfg, "accel_limit"): args.accel,
                (tcfg, "decel_limit"): args.accel,
            }
        )
        if args.vel_int_limit is not None:
            changes[(cfg, "vel_integrator_limit")] = args.vel_int_limit
        if args.bandwidth is not None:
            changes[(a.enc.config, "bandwidth")] = args.bandwidth
        if args.gain_scheduling is not None:
            changes[(cfg, "enable_gain_scheduling")] = args.gain_scheduling > 0
            if args.gain_scheduling > 0:
                changes[(cfg, "gain_scheduling_width")] = args.gain_scheduling
    saved = {k: getattr(*k) for k in changes}
    for (obj, name), val in changes.items():
        setattr(obj, name, val)
    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)

    kts = {a.n: a.mot.config.torque_constant for a in axes}
    for n, kt in kts.items():
        if abs(kt - 0.04) < 1e-6:
            print(
                f"注意: axis{n} の Kt は既定値 0.04 のまま (kt_measure.py で測る)",
                flush=True,
            )
    summary = {
        "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "kt": kts,
        "encoder_bandwidth": {a.n: a.enc.config.bandwidth for a in axes},
        "vbus": d.vbus_voltage,
        "gains": [],
    }
    i_abort = 0.9 * args.current_lim

    def segment(writer, name, secs, update=None):
        """全軸を標本化し、行を返す。監視に掛かったら Abort。"""
        rows = {a.n: [] for a in axes}
        t0 = time.monotonic()
        while (t := time.monotonic() - t0) < secs:
            if update:
                update(t)
            for a in axes:
                if a.ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    raise Abort(f"閉ループを抜けた {a.errors()}")
                iq = a.cc.Iq_measured
                r = {
                    "t": t,
                    "pos": a.enc.pos_estimate,
                    "vel": a.enc.vel_estimate,
                    "iq": iq,
                    "pos_set": a.ctrl.pos_setpoint,
                    "in_pos": a.ctrl.input_pos,
                    "shadow": a.enc.shadow_count,
                }
                rows[a.n].append(r)
                writer.writerow([name, a.n] + list(r.values()))
                if abs(iq) > i_abort:
                    a.over_since = a.over_since or time.monotonic()
                    if time.monotonic() - a.over_since > OVERCURRENT_S:
                        raise Abort(f"axis{a.n} 過電流 Iq={iq:+.2f}A")
                else:
                    a.over_since = None
                if abs(r["vel"]) > max(args.vel_limit, vmax + 1.0):
                    raise Abort(f"axis{a.n} 速度超過 {r['vel']:+.2f}turn/s")
            time.sleep(SAMPLE_S)
        return rows

    def hold_metrics(rows):
        out = {}
        for n, rs in rows.items():
            tail = [r for r in rs if r["t"] > rs[-1]["t"] - 2.0]
            iq = [r["iq"] for r in rs]
            out[n] = {
                "iq_sd": st.pstdev(iq),
                "iq_max": max(map(abs, iq)),
                "count_changes": sum(
                    1 for x, y in zip(tail, tail[1:]) if x["shadow"] != y["shadow"]
                ),
                "pos_pp_counts": CPR
                * (max(r["pos"] for r in tail) - min(r["pos"] for r in tail)),
                "err": st.mean(r["pos"] - r["in_pos"] for r in tail),
            }
        return out

    def step_metrics(rows, start, target):
        out = {}
        for n, rs in rows.items():
            span = target[n] - start[n]
            y = [(r["pos"] - start[n]) / span for r in rs]
            late = [
                i for i, r in enumerate(rs) if abs(r["pos"] - target[n]) > 1.5 / CPR
            ]
            tail = [r for r in rs if r["t"] > rs[-1]["t"] - 0.5]
            out[n] = {
                "overshoot_pct": 100 * (max(y) - 1),
                "settle_s": rs[late[-1]]["t"] if late else 0.0,
                "final_err": st.mean(r["pos"] for r in tail) - target[n],
                "iq_max": max(abs(r["iq"]) for r in rs),
            }
        return out

    def stream_metrics(rows, sign_v):
        out = {}
        for n, rs in rows.items():
            sv = sign_v * next(a.sign for a in axes if a.n == n)
            w = [r for r in rs if r["t"] > 1.0]
            vel = [r["vel"] * sv for r in w]
            mv = st.mean(vel)
            out[n] = {
                "vel_mean": mv,
                "vel_cv": st.pstdev(vel) / abs(mv) if abs(mv) > 1e-6 else float("inf"),
                "lag": st.mean((r["in_pos"] - r["pos"]) * sv for r in w),
                "iq_mean": st.mean(r["iq"] * sv for r in w),
                "iq_sd": st.pstdev(r["iq"] for r in w),
            }
        return out

    def set_mode(mode):
        for a in axes:
            a.ctrl.config.input_mode = mode

    def run_gain(vg: float) -> dict:
        res = {
            "vel_gain": vg,
            "a_per_turn_s": {a.n: vg / kts[a.n] for a in axes},
            "i_cap": {a.n: vg * a.ctrl.config.vel_limit / kts[a.n] for a in axes},
        }
        path = os.path.join(args.outdir, f"vg{vg:g}.csv")
        with open(path, "w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "segment",
                    "axis",
                    "t",
                    "pos",
                    "vel",
                    "iq",
                    "pos_set",
                    "in_pos",
                    "shadow",
                ]
            )
            d.clear_errors()
            for a in axes:
                a.ctrl.config.vel_gain = vg
                a.ctrl.input_pos = a.enc.pos_estimate
                a.ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            t0 = time.monotonic()
            while any(
                a.ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL for a in axes
            ):
                if time.monotonic() - t0 > 2.0:
                    raise Abort(
                        "閉ループへ入らない " + " ".join(a.errors() for a in axes)
                    )
                time.sleep(0.05)

            set_mode(INPUT_MODE_PASSTHROUGH)
            res["hold"] = hold_metrics(segment(w, "hold", 4.0))
            bad = [
                n
                for n, m in res["hold"].items()
                if m["iq_sd"] > args.unstable_iq_sd
                or m["pos_pp_counts"] > args.unstable_pp
            ]
            if bad:
                res["unstable"] = f"保持で発振 axis{bad}"
                return res

            start = {a.n: a.enc.pos_estimate for a in axes}
            target = {a.n: start[a.n] + a.sign * args.step for a in axes}
            for a in axes:
                a.ctrl.input_pos = target[a.n]
            res["step_fwd"] = step_metrics(segment(w, "step_fwd", 3.0), start, target)
            back = {a.n: start[a.n] for a in axes}
            for a in axes:
                a.ctrl.input_pos = back[a.n]
            res["step_back"] = step_metrics(segment(w, "step_back", 3.0), target, back)

            set_mode(INPUT_MODE_TRAP_TRAJ)
            for v in speeds:
                for sign_v, tag in ((1.0, "fwd"), (-1.0, "back")):
                    setpos = {a.n: a.enc.pos_estimate for a in axes}
                    last = [0.0]

                    def update(t, setpos=setpos, last=last, sign_v=sign_v, v=v):
                        if t - last[0] < ROS_DT:
                            return
                        dt, last[0] = t - last[0], t
                        if t > args.stream_secs:
                            return
                        for a in axes:
                            pos = a.enc.pos_estimate
                            sp = setpos[a.n] + a.sign * sign_v * v * dt
                            sp = min(
                                max(sp, pos - ROS_MAX_POS_DIFF), pos + ROS_MAX_POS_DIFF
                            )
                            setpos[a.n] = sp
                            a.ctrl.input_pos = sp

                    rows = segment(w, f"v{v:g}_{tag}", args.stream_secs, update)
                    res[f"v{v:g}_{tag}"] = stream_metrics(rows, sign_v)
                    segment(w, f"v{v:g}_{tag}_stop", 2.0)
            return res

    try:
        for vg in gains:
            print(f"--- vel_gain={vg:g} ---", flush=True)
            try:
                r = run_gain(vg)
            except Abort as e:
                for a in axes:
                    a.ax.requested_state = AXIS_STATE_IDLE
                summary["gains"].append({"vel_gain": vg, "aborted": str(e)})
                print("中止:", e, flush=True)
                break
            finally:
                for a in axes:
                    a.ax.requested_state = AXIS_STATE_IDLE
            summary["gains"].append(r)
            print_result(r)
            if "unstable" in r:
                print(r["unstable"], "→ これより大きいゲインは試さない", flush=True)
                break
            time.sleep(1.0)
    finally:
        for a in axes:
            a.ax.requested_state = AXIS_STATE_IDLE
        for (obj, name), val in saved.items():
            setattr(obj, name, val)
        summary["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        print("IDLE。設定を戻した。" + " ".join(a.errors() for a in axes), flush=True)


def print_result(r: dict) -> None:
    for n, h in r["hold"].items():
        line = (
            f"axis{n} {r['a_per_turn_s'][n]:.2f}A/(turn/s) 上限{r['i_cap'][n]:.1f}A | 保持 Iq sd={h['iq_sd']:.2f} "
            f"max={h['iq_max']:.2f} 振れ幅={h['pos_pp_counts']:.1f}カウント カウント変化={h['count_changes']} err={h['err']:+.3f}"
        )
        for k in ("step_fwd", "step_back"):
            if k in r:
                s = r[k][n]
                line += (
                    f" | {k} 行き過ぎ={s['overshoot_pct']:+.0f}% 整定={s['settle_s']:.2f}s "
                    f"残差={s['final_err']:+.3f} Iqmax={s['iq_max']:.2f}"
                )
        print(line, flush=True)
        for k, m in r.items():
            if k.startswith("v") and k[1].isdigit():
                s = m[n]
                print(
                    f"    {k:10s} vel={s['vel_mean']:+.3f} sd/平均={s['vel_cv']:.2f} "
                    f"遅れ={s['lag']:+.3f}turn Iq={s['iq_mean']:+.2f}±{s['iq_sd']:.2f}A",
                    flush=True,
                )


if __name__ == "__main__":
    main()
