#!/usr/bin/env python3
"""全サーボの単体動作確認 ＋ 対向グループの符号関係の推定。

ID4-11 は「4個で1関節」の対向駆動なので、1個だけトルクONにして微動させ、
トルクOFFの残り3個がどちら向きに連れられるかを読む。これで
 ・各モータが実際に回るか（単体動作確認）
 ・どの2個が反転取付か（キャリブレーションに必要な符号)
の両方が一度に取れる。トルク上限を絞っているので突っ張っても機構を痛めない。
"""

import json
import sys
import time
from scservo_sdk import SCS_MAKEWORD, PortHandler, PacketHandler, COMM_SUCCESS

R_TORQUE, R_GOAL_POS, R_GOAL_SPD, R_TORQUE_LIMIT = 40, 42, 46, 48
R_POS, R_LOAD = 56, 60

GROUPS = {"joint0": [4, 5, 6, 7], "joint1": [8, 9, 10, 11]}
SINGLES = [12, 13, 14, 15]

NUDGE = 20  # ±20 step = 1.8deg
TORQUE_LIMIT = 350  # 0-1000。35% に絞る
SPEED = 100


def signed(v, bits=15):
    return -(v & ((1 << bits) - 1)) if v & (1 << bits) else v


class Bus:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            sys.exit(f"ポートを開けません: {port}")
        self.ph.setBaudRate(baud)
        self.pk = PacketHandler(0)

    def r2(self, i, a, tries=5):
        for _ in range(tries):
            try:
                v, c, _ = self.pk.read2ByteTxRx(self.ph, i, a)
                if c == COMM_SUCCESS:
                    return v
            except (IndexError, TypeError):  # 応答が途中で切れると SDK が投げる
                pass
        return None

    def state(self, i, tries=4):
        """位置と負荷を 1 回で読む。別々に読むと取り違える (README の「落とし穴」)。"""
        for _ in range(tries):
            try:
                d, c, _ = self.pk.readTxRx(self.ph, i, R_POS, 6)
                if c == COMM_SUCCESS and len(d) == 6:
                    return SCS_MAKEWORD(d[0], d[1]), SCS_MAKEWORD(d[4], d[5])
            except (IndexError, TypeError):
                pass
        return None

    def w1(self, i, a, v):
        for _ in range(3):
            try:
                if self.pk.write1ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS:
                    return True
            except (IndexError, TypeError):
                pass
        return False

    def w2(self, i, a, v):
        for _ in range(3):
            try:
                if self.pk.write2ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS:
                    return True
            except (IndexError, TypeError):
                pass
        return False

    def close(self):
        self.ph.closePort()


def all_off(bus, ids):
    for i in ids:
        bus.w1(i, R_TORQUE, 0)


def nudge(bus, mid, peers):
    """mid だけトルクONで ±NUDGE 動かし、peers の追従量を返す。"""
    base = {i: bus.r2(i, R_POS) for i in [mid] + peers}
    if base[mid] is None:
        return None
    bus.w2(mid, R_TORQUE_LIMIT, TORQUE_LIMIT)
    bus.w2(mid, R_GOAL_SPD, SPEED)
    bus.w1(mid, R_TORQUE, 1)
    result = {"start": base[mid]}
    try:
        moved = {}
        for sign in (+1, -1):
            bus.w2(mid, R_GOAL_POS, max(0, min(4095, base[mid] + sign * NUDGE)))
            time.sleep(0.6)
            now = {i: bus.r2(i, R_POS) for i in [mid] + peers}
            st = bus.state(mid)
            load = abs(signed(st[1], 10)) if st else 0
            moved[sign] = {
                i: (None if now[i] is None or base[i] is None else now[i] - base[i])
                for i in now
            }
            moved[sign]["_load"] = load
        # 元位置へ
        bus.w2(mid, R_GOAL_POS, base[mid])
        time.sleep(0.6)
        result["end"] = bus.r2(mid, R_POS)
        result["moved"] = moved
    finally:
        bus.w1(mid, R_TORQUE, 0)
    return result


if __name__ == "__main__":
    bus = Bus()
    all_ids = sum(GROUPS.values(), []) + SINGLES
    report = {}
    try:
        all_off(bus, all_ids)
        for gname, ids in list(GROUPS.items()) + [("single", SINGLES)]:
            print(f"\n=== {gname}: {ids} ===")
            for mid in ids:
                peers = [i for i in ids if i != mid] if gname != "single" else []
                r = nudge(bus, mid, peers)
                if r is None:
                    print(f"  ID{mid}: 位置が読めず、スキップ")
                    continue
                p, m = r["moved"][+1], r["moved"][-1]
                own = f"+{p[mid]:>4} / {m[mid]:>4}"
                ok = (
                    "動いた"
                    if abs(p[mid] or 0) > 5 or abs(m[mid] or 0) > 5
                    else "★動かず"
                )
                peer_s = "  ".join(
                    f"ID{i}:{p[i]:+d}/{m[i]:+d}" for i in peers if p[i] is not None
                )
                print(
                    f"  ID{mid}: 自身 {own}  load={p['_load']},{m['_load']}  {ok}"
                    + (f"\n         連れられ → {peer_s}" if peer_s else "")
                )
                report[mid] = r
    finally:
        all_off(bus, all_ids)
        bus.close()
        print("\n全モータ トルクOFF")
    json.dump(
        report,
        open("/home/hrt/aharobot-check/nudge_report.json", "w"),
        indent=1,
        ensure_ascii=False,
    )
