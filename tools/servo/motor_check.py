#!/usr/bin/env python3
"""STS3215 モータ単位の動作確認（既定は読み取りのみ）。

python motor_check.py                     # 全モータの状態を読む
python motor_check.py --move 15 --delta 100   # 指定IDだけ ±delta ステップ動かして戻す
"""

import argparse
import sys
import time
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# --- STS3215 レジスタ ---
R_ID, R_BAUD, R_OFS, R_MODE = 5, 6, 31, 33
R_TORQUE, R_GOAL_POS, R_GOAL_SPD = 40, 42, 46
R_POS, R_SPEED, R_LOAD, R_VOLT, R_TEMP, R_MOVING, R_CURRENT = 56, 58, 60, 62, 63, 66, 69

# グリッパ(ID15)のステップ↔開き幅換算: AhaRobot の GRIPPER_GEAR_R = 0.027/2 [m]
GRIPPER_GEAR_R = 0.027 / 2
STEPS_PER_MM = 4096 / (2 * 3.141592653589793 * GRIPPER_GEAR_R) / 1000

BAUD_TABLE = {
    0: 1000000,
    1: 500000,
    2: 250000,
    3: 128000,
    4: 115200,
    5: 76800,
    6: 57600,
    7: 38400,
}


def signed(v, bits=15):
    """Feetech は最上位ビットを符号として使う。"""
    return -(v & ((1 << bits) - 1)) if v & (1 << bits) else v


class Bus:
    def __init__(self, port, baud):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            sys.exit(f"ポートを開けません: {port}")
        self.ph.setBaudRate(baud)
        self.pk = PacketHandler(0)  # SMS/STS

    # usbip 越しだと単発で取りこぼすことがあるので数回粘る
    def r1(self, i, a, tries=4):
        for _ in range(tries):
            v, c, e = self.pk.read1ByteTxRx(self.ph, i, a)
            if c == COMM_SUCCESS:
                return v
        return None

    def r2(self, i, a, tries=4):
        for _ in range(tries):
            v, c, e = self.pk.read2ByteTxRx(self.ph, i, a)
            if c == COMM_SUCCESS:
                return v
        return None

    def w1(self, i, a, v):
        return self.pk.write1ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def w2(self, i, a, v):
        return self.pk.write2ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def ping(self, i, tries=4):
        return any(self.pk.ping(self.ph, i)[1] == COMM_SUCCESS for _ in range(tries))

    def close(self):
        self.ph.closePort()


def read_all(bus, ids):
    print(
        f"{'ID':>3} {'pos':>6} {'deg':>7} {'speed':>6} {'load':>6} "
        f"{'V':>5} {'℃':>4} {'mA':>6} {'mode':>4} {'trq':>3} {'ofs':>5} {'baud':>7}"
    )
    print("-" * 78)
    ok = []
    for i in ids:
        if not bus.ping(i):
            print(f"{i:>3}  -- 応答なし --")
            continue
        pos = bus.r2(i, R_POS)
        if pos is None:
            print(f"{i:>3}  -- ping応答ありだが位置読み出し失敗 --")
            continue
        deg = pos / 4096 * 360
        spd = signed(bus.r2(i, R_SPEED) or 0)
        load = signed(bus.r2(i, R_LOAD) or 0, 10)
        volt = bus.r1(i, R_VOLT)
        temp = bus.r1(i, R_TEMP)
        mode_v, trq_v = bus.r1(i, R_MODE), bus.r1(i, R_TORQUE)
        cur = signed(bus.r2(i, R_CURRENT) or 0) * 6.5  # 単位 6.5mA
        mode = "?" if mode_v is None else mode_v
        trq = "?" if trq_v is None else trq_v
        ofs = signed(bus.r2(i, R_OFS) or 0, 11)
        bd = BAUD_TABLE.get(bus.r1(i, R_BAUD), "?")
        print(
            f"{i:>3} {pos:>6} {deg:>7.1f} {spd:>6} {load:>6} "
            f"{(volt or 0)/10:>5.1f} {temp if temp is not None else '?':>4} "
            f"{cur:>6.0f} {mode:>4} {trq:>3} {ofs:>5} {bd:>7}"
        )
        ok.append(i)
    return ok


def wiggle(bus, mid, delta, speed=200, step=50, load_limit=300, stall_limit=3):
    """1個だけ動かして元位置へ戻す。

    メカ端に突っ込むのを避けるため step ステップずつ進め、負荷が load_limit を
    超えるか、指令しても位置が動かない状態が stall_limit 回続いたら中断して戻す。
    """
    start = bus.r2(mid, R_POS)
    if start is None:
        sys.exit(f"ID{mid} の位置が読めません")
    print(
        f"\nID{mid}: 現在位置 {start} → ±{delta} ステップを往復"
        f"（グリッパなら約 ±{delta / STEPS_PER_MM:.1f}mm）"
    )
    bus.w1(mid, R_TORQUE, 1)
    bus.w2(mid, R_GOAL_SPD, speed)
    aborted = False

    def goto(target, label):
        """target まで step 刻みで寄せる。中断したら True。"""
        nonlocal aborted
        target = max(0, min(4095, target))
        peak = 0
        stalled = 0
        while True:
            now = bus.r2(mid, R_POS)
            if now is None:
                continue
            if abs(now - target) < 15:
                break
            nxt = now + max(-step, min(step, target - now))
            bus.w2(mid, R_GOAL_POS, nxt)
            time.sleep(0.12)
            after = bus.r2(mid, R_POS)
            load = abs(signed(bus.r2(mid, R_LOAD) or 0, 10))
            peak = max(peak, load)
            if load > load_limit:
                print(
                    f"  !! load={load} が上限 {load_limit} 超過（位置 {after}, "
                    f"開始から {abs((after or now) - start)}step = "
                    f"{abs((after or now) - start) / STEPS_PER_MM:.1f}mm）。メカ端の可能性ありで中断"
                )
                aborted = True
                return True
            if after is not None and abs(after - now) < 3:
                stalled += 1
                if stalled >= stall_limit:
                    print(
                        f"  !! 指令しても位置が動かない（位置 {after}, 開始から "
                        f"{abs(after - start) / STEPS_PER_MM:.1f}mm）。中断"
                    )
                    aborted = True
                    return True
            else:
                stalled = 0
        final = bus.r2(mid, R_POS)
        print(
            f"  {label:>6}: 目標 {target:>4} → 実測 {final if final is not None else '読めず':>6}  "
            f"誤差 {abs(final - target) if final is not None else -1:>4}  peak_load={peak:>4}"
        )
        return False

    try:
        d1 = "＋側" if delta > 0 else "−側"
        d2 = "−側" if delta > 0 else "＋側"
        for target, label in (
            (start + delta, d1),
            (start - delta, d2),
            (start, "復帰"),
        ):
            if goto(target, label) and label != "復帰":
                print("  → 中断地点から開始位置へ戻します")
                goto(start, "復帰")
                break
    finally:
        bus.w1(mid, R_TORQUE, 0)  # 確認後は必ずトルクを切る
        end = bus.r2(mid, R_POS)
        print(
            f"  トルクOFF（最終位置 {end}, 開始位置 {start}, ずれ {abs((end or start) - start)}）"
        )
        if aborted:
            print("  ※ 可動範囲の端に達したため、指定した幅の全域は動かしていません")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--ids", default="4-15")
    p.add_argument("--move", type=int, help="このIDだけ実際に動かす")
    p.add_argument(
        "--delta", type=int, default=100, help="動かす量(ステップ, 4096=1回転)"
    )
    p.add_argument(
        "--mm", type=float, help="動かす量をグリッパ開き幅[mm]で指定(--delta より優先)"
    )
    p.add_argument("--load-limit", type=int, default=300, help="この負荷を超えたら中断")
    a = p.parse_args()

    lo, _, hi = a.ids.partition("-")
    ids = range(int(lo), int(hi or lo) + 1)

    bus = Bus(a.port, a.baud)
    try:
        found = read_all(bus, ids)
        print(f"\n応答したモータ: {len(found)}個 {found}")
        missing = [i for i in ids if i not in found]
        if missing:
            print(f"欠番: {missing}")
        if a.move is not None:
            d = round(a.mm * STEPS_PER_MM) if a.mm else a.delta
            wiggle(bus, a.move, d, load_limit=a.load_limit)
    finally:
        bus.close()
