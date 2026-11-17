#!/usr/bin/env python3
"""hall_record.py の記録を条件ごとに集計する。

ホスト側の生データからは、参照系列 (前後 1.5 ms の多数決) と食い違い、
元の値へ戻るまとまり (または 0/7) をノイズ 1 件として数える。補助データからは shadow_count の増え方と指令速度の比
(追従)、vel_estimate の平均とばらつきを出す。フィルタ付きファームでは、
定速区間の内部カウンタ (8 kHz の全標本) から棄却数と 0/7 の回数を出す。

Usage:  hall_analyze.py OUTDIR
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys

from hall_common import CNT, aux_rows, load, reference


def noise_events(T, H, R):
    """参照と食い違い、前後の生データが同じ値に戻るまとまり (または 0/7 を含む)。

    切り替わり位置の前後のずれ (補助データ読み取りの空白で生じる) は数えない。
    """
    bad = [i for i in range(len(H)) if H[i] != R[i]]
    groups = []
    for i in bad:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])
    events = []
    for g in groups:
        a, b = g[0] - 1, g[-1] + 1
        returns = a >= 0 and b < len(H) and H[a] == H[b]
        if returns or any(H[i] not in CNT for i in g):
            events.append(g)
    return events


def main() -> None:
    outdir = sys.argv[1]
    summary = json.load(open(f"{outdir}/summary.json"))
    print(
        f"fw {summary.get('fw_version', '?')} filter={summary.get('fw_hall_filter', False)} "
        f"direction={summary.get('encoder_direction', '?')} bw={summary['encoder_bandwidth']}"
    )
    print(
        f"{'run':18s} {'N':>3s} {'Hz':>5s} {'noise/s':>7s} {'ill/s':>6s} {'sync':>5s} "
        f"{'vel_est':>8s} {'sd%':>5s} {'fw rej/s':>8s} {'fw ill/s':>8s}  noise by true state"
    )
    out = {}
    for r in summary["runs"]:
        T, H = load(outdir, r["name"], r["steady_from"])
        dur = T[-1] - T[0]
        R = reference(T, H)
        ev = noise_events(T, H, R)
        ill = sum(1 for e in ev if any(H[i] not in CNT for i in e))
        by_state = dict(sorted(collections.Counter(R[e[0]] for e in ev).items()))
        aux = aux_rows(outdir, r["name"], r["steady_from"])
        sh = [(float(a["t"]), int(a["shadow"])) for a in aux]
        ideal = 6 * r["vel"] / (2 * math.pi)
        rate = (sh[-1][1] - sh[0][1]) / (sh[-1][0] - sh[0][0]) if len(sh) > 1 else 0.0
        sync = abs(rate / ideal) if ideal else float("nan")
        ve = [float(a["vel_est"]) for a in aux if a.get("vel_est")]
        ve_ideal = abs(r["vel"]) / (2 * math.pi * summary["pole_pairs"])
        ve_mean = statistics.mean(ve) if ve else float("nan")
        ve_sd = (
            100 * statistics.pstdev(ve) / ve_ideal if ve and ve_ideal else float("nan")
        )
        fc = r.get("fw_counters", {})
        s, e = fc.get("steady", {}), fc.get("end", {})
        if "rejected" in s and "rejected" in e and e["t"] > s["t"]:
            rej_ps = (e["rejected"] - s["rejected"]) / (e["t"] - s["t"])
            ill_ps = (e["illegal"] - s["illegal"]) / (e["t"] - s["t"])
        else:
            rej_ps = ill_ps = float("nan")
        n = r.get("debounce")
        out[r["name"]] = {
            "debounce": n,
            "hz": len(T) / dur,
            "noise_ps": len(ev) / dur,
            "illegal_ps": ill / dur,
            "sync": sync,
            "shadow_rate": rate,
            "vel_est_mean": ve_mean,
            "vel_est_sd_pct": ve_sd,
            "fw_rejected_ps": rej_ps,
            "fw_illegal_ps": ill_ps,
            "noise_by_state": by_state,
        }
        print(
            f"{r['name']:18s} {'-' if n is None else n:>3} {len(T) / dur:5.0f} "
            f"{len(ev) / dur:7.1f} {ill / dur:6.1f} {sync:5.2f} {ve_mean:+8.4f} {ve_sd:5.1f} "
            f"{rej_ps:8.1f} {ill_ps:8.1f}  {by_state}"
        )
    with open(f"{outdir}/analysis.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
