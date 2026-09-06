#!/usr/bin/env python3
"""トルクOFFのまま手で動かして可動範囲と対向符号を記録する（ダイレクトティーチ方式）。

通電してレンジ探索をすると、重力で垂れている関節を電動で振り回すことになるので、
全サーボをトルクOFFにしたまま人間が手で動かし、その軌跡から
  ・サーボごとの min / max / 可動幅 / 中点
  ・対向グループ内の真の符号と伝達比（相関から算出。自重の影響を受けない）
  ・実際に連動しているサーボの組み合わせ
を求める。lerobot の calibrate と同じ考え方。

  python teach_calibrate.py --seconds 120
"""

import argparse
import json
import math
import signal
import sys
import time
from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

R_TORQUE, R_POS = 40, 56
GROUPS = {"joint0": [4, 5, 6, 7], "joint1": [8, 9, 10, 11], "single": [12, 13, 15]}
MOVED_THRESHOLD = 30  # これ以上動いたサーボだけ「動かした」とみなす(step)


class Bus:
    def __init__(self, port, baud):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            sys.exit(f"ポートを開けません: {port}")
        self.ph.setBaudRate(baud)
        self.pk = PacketHandler(0)

    def pos(self, i):
        v, c, _ = self.pk.read2ByteTxRx(self.ph, i, R_POS)
        return v if c == COMM_SUCCESS else None

    def torque(self, i, on):
        for _ in range(3):
            if (
                self.pk.write1ByteTxRx(self.ph, i, R_TORQUE, 1 if on else 0)[0]
                == COMM_SUCCESS
            ):
                return True
        return False

    def close(self):
        self.ph.closePort()


def unwrap(prev_raw, raw, acc):
    """0/4095 をまたいでも連続な値にする。"""
    d = raw - prev_raw
    if d > 2048:
        d -= 4096
    elif d < -2048:
        d += 4096
    return acc + d


def fit(xs, ys):
    """ys = a*xs + b の傾き a と相関係数 r。"""
    n = len(xs)
    if n < 10:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--out", default="/home/hrt/aharobot-check/calibration.json")
    p.add_argument(
        "--verify",
        action="store_true",
        help="AhaRobot の JOINT_SERVO_SIGN 表を実測から生成し、整合を判定する",
    )
    a = p.parse_args()

    ids = sum(GROUPS.values(), [])
    bus = Bus(a.port, a.baud)

    stop = False
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

    print("全サーボのトルクを切ります…")
    for i in ids:
        bus.torque(i, False)

    start_raw, acc, track = {}, {}, {i: [] for i in ids}
    for i in ids:
        v = bus.pos(i)
        if v is None:
            print(f"  ID{i}: 位置が読めません（スキップ）")
            continue
        start_raw[i], acc[i] = v, 0

    live = sorted(start_raw)
    print(
        f"記録開始（{a.seconds:.0f}秒 / Ctrl-C で早期終了）  行の数字＝各サーボの到達レンジ(step)"
    )
    print("各関節をゆっくり可動範囲いっぱいまで手で往復させてください。")
    print(
        "順番の目安: joint0(ID4-7) → joint1(ID8-11) → 手首(ID12,13) → グリッパ(ID15)\n"
    )

    t0 = time.time()
    last_print = 0.0
    try:
        while not stop and time.time() - t0 < a.seconds:
            for i, v in ((i, bus.pos(i)) for i in live):
                if v is None:
                    continue
                acc[i] = unwrap(start_raw[i], v, acc[i])
                start_raw[i] = v
                track[i].append(acc[i])
            now = time.time() - t0
            if now - last_print > 1.0:
                last_print = now
                spans = "  ".join(
                    f"ID{i}:{(max(track[i]) - min(track[i])) if track[i] else 0:>4}"
                    for i in live
                )
                print(f"\r\033[K[{now:5.1f}s] {spans}", end="", flush=True)
    finally:
        print("\n\n記録終了。")

    # --- 集計 ---
    stats = {}
    for i in live:
        t = track[i]
        if not t:
            continue
        lo, hi = min(t), max(t)
        stats[i] = {
            "span": hi - lo,
            "min_rel": lo,
            "max_rel": hi,
            "mid_raw": (start_raw[i] - (acc[i] - (lo + hi) / 2)) % 4096,
            "samples": len(t),
        }

    if not stats:
        sys.exit(
            "サンプルが1つも取れませんでした。シリアル転送が生きているか確認してください。"
        )

    print(f"{'ID':>3} {'可動幅(step)':>12} {'deg':>8} {'サンプル':>7}")
    print("-" * 38)
    for i in live:
        s = stats[i]
        print(f"{i:>3} {s['span']:>12} {s['span']/4096*360:>8.1f} {s['samples']:>7}")

    # --- 全ペア相関から実際の連動関係を求める ---
    moved = [i for i in live if stats[i]["span"] >= MOVED_THRESHOLD]
    print(f"\n動いたサーボ: {moved}")
    n = min((len(track[i]) for i in moved), default=0)

    corr = {}
    if len(moved) >= 2:
        print("\n相関行列 r（|r|>0.9 で連動、符号が逆なら対向取付）")
        print("     " + "".join(f"{'ID'+str(j):>8}" for j in moved))
        for i in moved:
            row = []
            for j in moved:
                _, r = fit(track[i][:n], track[j][:n])
                corr[f"{i}-{j}"] = r
                row.append("    --  " if i == j else f"{r:+8.2f}")
            print(f"ID{i:<3}" + "".join(row))

    # |r|>0.9 で繋いでいき、連動している集合を作る
    linked, seen = [], set()
    for i in moved:
        if i in seen:
            continue
        cluster = [i]
        seen.add(i)
        for j in moved:
            if j not in seen and abs(corr.get(f"{i}-{j}") or 0) > 0.9:
                cluster.append(j)
                seen.add(j)
        linked.append(cluster)

    print("\n実際に連動しているグループ:")
    rel = {}
    for c in linked:
        if len(c) == 1:
            print(f"  {c}  … 単独（他のどれとも連動せず）")
            continue
        ref = max(c, key=lambda i: stats[i]["span"])
        same = [i for i in c if (fit(track[ref][:n], track[i][:n])[0] or 0) > 0]
        opp = [i for i in c if i not in same]
        print(f"  {c}  基準ID{ref}: 同相 {same} / 逆相 {opp}")
        rel[str(ref)] = {
            "members": c,
            "same": same,
            "opposite": opp,
            "slopes": {str(i): fit(track[ref][:n], track[i][:n])[0] for i in c},
        }

    # --- AhaRobot の符号表を実測から生成 ---
    if a.verify:
        AHA = {"joint0": [4, 5, 6, 7], "joint1": [8, 9, 10, 11]}
        print("\n" + "=" * 62)
        print("AhaRobot dualMotor.cpp 用 JOINT_SERVO_SIGN")
        lines, all_ok = [], True
        for gname, gids in AHA.items():
            present = [i for i in gids if i in stats]
            if len(present) < 4:
                print(
                    f"  [{gname}] ID {sorted(set(gids)-set(present))} が不在。判定不能"
                )
                all_ok = False
                lines.append(f"  +1, -1, +1, -1,   // {gname}: 判定不能")
                continue
            still = [i for i in present if stats[i]["span"] < MOVED_THRESHOLD]
            if still:
                print(
                    f"  [{gname}] ID{still} がほとんど動いていません。この関節を動かして再記録してください"
                )
                all_ok = False
                lines.append(f"  +1, -1, +1, -1,   // {gname}: 未測定")
                continue
            ref = gids[0]
            signs, weak = [], []
            for i in gids:
                r = corr.get(f"{ref}-{i}")
                if r is None or abs(r) < 0.9:
                    weak.append((i, r))
                    signs.append(+1)
                else:
                    signs.append(+1 if r > 0 else -1)
            if weak:
                print(
                    f"  [{gname}] ID{[i for i, _ in weak]} が ID{ref} と連動していません "
                    f"(r={[None if r is None else round(r,3) for _, r in weak]})。"
                )
                print(
                    "           台座が固定されていないと、この関節は分離して見えます。"
                )
                all_ok = False
            expect = [+1, -1, +1, -1]
            verdict = (
                "元コードと一致"
                if signs in (expect, [-x for x in expect])
                else "★元コードと不一致 → 表の差し替えが必要"
            )
            if not weak:
                print(f"  [{gname}] 実測符号 {signs}  ({verdict})")
            lines.append(
                "  "
                + ", ".join(f"{x:+d}" for x in signs)
                + f",   // {gname}: ID{gids[0]}-{gids[-1]}"
                + ("" if not weak else "  ※未確定")
            )
        print("\nconst int JOINT_SERVO_SIGN[4 * JOINT_NUM] = {")
        for line in lines:
            print(line)
        print("};")
        print("=" * 62)
        if not all_ok:
            print("★ 未確定の関節があります。トルクを入れる前に必ず解消してください。")

    json.dump(
        {str(k): v for k, v in track.items()},
        open(a.out.replace(".json", "_tracks.json"), "w"),
    )

    json.dump(
        {
            "stats": {str(k): v for k, v in stats.items()},
            "linked_groups": rel,
            "correlation": corr,
            "recorded_seconds": time.time() - t0,
        },
        open(a.out, "w"),
        indent=1,
        ensure_ascii=False,
    )
    print(f"\n保存: {a.out}")
    bus.close()
