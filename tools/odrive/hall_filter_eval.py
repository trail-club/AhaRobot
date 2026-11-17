#!/usr/bin/env python3
"""ホールのグリッチ除去フィルタ候補を、記録した生データに当てて比べる。

hall_record.py の出力 (summary.json と条件ごとの CSV) を読み、前後 1.5 ms の
多数決を参照系列として、各フィルタの偽エッジ・遅れ・カウントずれと、
fw 0.5.6 の PLL (8 kHz, kp = 2bw, ki = kp^2 / 4) を再現した速度推定のばらつきを出す。
フィルタはホストの読み取り時刻を基準にした時間で判定する。
ファームの 8 kHz 標本より粗いので、短いグリッチは実際より少なく見える。

Usage:  hall_filter_eval.py OUTDIR
"""

from __future__ import annotations

import json
import math
import statistics
import sys

from hall_common import CNT, fw_delta, load, reference

FW_DT = 1 / 8000  # current_meas_period


class HallFilter:
    """制御周期ごとに動かす想定の因果フィルタ。

    adjacent_only: 隣接 (±1) 以外の遷移は recover_s 続いたときだけ受け付ける。
    debounce_s:    候補がこの時間続いたら受け付ける。
    0 / 7 は無視し、候補の計時は止めない。
    """

    def __init__(self, debounce_s=0.0, adjacent_only=False, recover_s=1e-3):
        self.deb, self.adj, self.rec = debounce_s, adjacent_only, recover_s
        self.state = None
        self.cand = None
        self.cand_t = 0.0

    def update(self, t, h):
        if h not in CNT:
            return self.state
        if self.state is None:
            self.state = h
            return h
        if h == self.state:
            self.cand = None
            return self.state
        if h != self.cand:
            self.cand, self.cand_t = h, t
        need = self.deb
        if self.adj and abs(fw_delta(CNT[h], CNT[self.state])) != 1:
            need = max(self.deb, self.rec)
        if t - self.cand_t >= need:
            self.state, self.cand = h, None
        return self.state


def counts(S):
    """fw 0.5.6 と同じ規則で数えたカウント列。"""
    c, cur, out = 0, None, []
    for s in S:
        if s in CNT:
            if cur is None:
                cur = s
            elif s != cur:
                c += fw_delta(CNT[s], CNT[cur])
                cur = s
        out.append(c)
    return out


def pll(T, C, bw):
    """fw 0.5.6 の PLL を 8 kHz で再現した速度推定 [count/s]。最初の 1 秒は捨てる。"""
    kp = 2 * bw
    ki = 0.25 * kp * kp
    pos = C[0] + 0.5
    vel = 0.0
    j = 0
    t = T[0]
    v = []
    while t < T[-1]:
        while j + 1 < len(T) and T[j + 1] <= t:
            j += 1
        pos += FW_DT * vel
        d = C[j] - math.floor(pos)
        pos += FW_DT * kp * d
        vel += FW_DT * ki * d
        v.append(vel)
        t += FW_DT
    return v[int(1.0 / FW_DT) :]


def edges(T, S):
    return [(T[i], S[i - 1], S[i]) for i in range(1, len(S)) if S[i] != S[i - 1]]


def evaluate(T, S, R, sector_s):
    """参照系列 R に対するフィルタ出力 S の評価。"""
    e_ref, e_out = edges(T, R), edges(T, S)
    dur = T[-1] - T[0]
    lat = []
    j = 0
    for t, _a, b in e_ref:
        while j < len(T) and T[j] < t:
            j += 1
        k = j
        while k < len(T) and S[k] != b and T[k] - t < sector_s:
            k += 1
        if k < len(T) and S[k] == b:
            lat.append(T[k] - t)
    c_out, c_ref = counts(S), counts(R)
    err = [a - b for a, b in zip(c_out, c_ref)]
    wrong = sum(1 for a, b in zip(S, R) if a != b) / len(S)
    return {
        "extra_ps": (len(e_out) - len(e_ref)) / dur,
        "lat_med": statistics.median(lat) * 1e3 if lat else float("nan"),
        "lat_p99": sorted(lat)[int(0.99 * (len(lat) - 1))] * 1e3
        if lat
        else float("nan"),
        "wrong_pct": 100 * wrong,
        "drift": c_out[-1] - c_ref[-1],
        "maxerr": max(abs(e) for e in err),
        "counts": c_out,
    }


VARIANTS = [("V0 raw (fw now)", None), ("V1 adjacent", {"adjacent_only": True})]
VARIANTS += [(f"V2 debounce {d}ms", {"debounce_s": d / 1e3}) for d in (0.25, 0.5, 1.0)]
VARIANTS += [
    (f"V3 adj+deb {d}ms", {"debounce_s": d / 1e3, "adjacent_only": True})
    for d in (0.25, 0.5, 1.0)
]


def main() -> None:
    outdir = sys.argv[1]
    summary = json.load(open(f"{outdir}/summary.json"))
    result = {}
    for r in summary["runs"]:
        if r["current"] == 0:
            continue
        T, H = load(outdir, r["name"], r["steady_from"])
        R = reference(T, H)
        ideal = 6 * abs(r["vel"]) / (2 * math.pi)
        sector = math.pi / 3 / abs(r["vel"])
        deg_per_ms = abs(r["vel"]) * 180 / math.pi / 1000
        c_ref = counts(R)
        sd = lambda v: 100 * statistics.pstdev(v) / ideal  # noqa: E731
        sd_ref = sd(pll(T, c_ref, 50))
        print(
            f"\n== {r['name']}  ideal {ideal:.1f} cnt/s  "
            f"ref {(c_ref[-1] - c_ref[0]) / (T[-1] - T[0]):+.1f} cnt/s  "
            f"sector {sector * 1e3:.1f} ms  1 ms = {deg_per_ms:.1f} deg(e)  vel sd ref {sd_ref:.1f}%"
        )
        print(
            f"   {'filter':20s} {'extra/s':>7s} {'wrong%':>6s} {'lat med/p99 ms':>15s} "
            f"{'deg':>5s} {'drift':>5s} | vel sd% bw50 bw20 | bias% bw50"
        )
        res = {}
        for name, kw in VARIANTS:
            if kw is None:
                S = H
            else:
                f = HallFilter(**kw)
                S = [f.update(t, h) for t, h in zip(T, H)]
            m = evaluate(T, S, R, sector)
            v50, v20 = pll(T, m["counts"], 50), pll(T, m["counts"], 20)
            m.update(
                sd50=sd(v50),
                sd20=sd(v20),
                bias50=100 * (abs(statistics.mean(v50)) - ideal) / ideal,
            )
            del m["counts"]
            res[name] = m
            print(
                f"   {name:20s} {m['extra_ps']:7.1f} {m['wrong_pct']:6.2f} "
                f"{m['lat_med']:6.2f}/{m['lat_p99']:6.2f}   {m['lat_med'] * deg_per_ms:5.1f} "
                f"{m['drift']:5d} | {m['sd50']:9.1f} {m['sd20']:4.1f} | {m['bias50']:+6.1f}"
            )
        result[r["name"]] = res
    with open(f"{outdir}/filter_eval.json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
