#!/usr/bin/env python3
"""キーボードで各関節を動かす。対向駆動の関節は 4 個まとめて動かす。

ファームを焼く前でも、制御基板をシリアル転送モードにしておけばホストから
直接動かせる (手順は README)。対向取付の符号は実測値
(docs/servo-bringup.md) をそのまま使う。符号が違うとサーボ同士が押し合うので、
機体を変えたら `teach_calibrate.py --verify` で測り直して JOINTS を更新すること。

安全のため
  - トルク上限を絞る (--torque-limit, 既定 500/1000)
  - 関節ごとに開始位置からの移動量を制限する (--max-offset)
  - 負荷が閾値を超えたらその関節をその向きへは進めない
  - 終了時とパニックキーで必ずトルクを切る

Usage:  keyboard_teleop.py [--port /dev/ttyUSB0] [--baud 115200]
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

R_ACC, R_GOAL_POS, R_GOAL_SPD, R_TORQUE, R_TORQUE_LIMIT = 41, 42, 46, 40, 48
R_POS, R_LOAD = 56, 60


class Joint:
    """1 つの関節。dual 構成では複数サーボを符号付きで同時に動かす。"""

    def __init__(self, name: str, servos: dict[int, int], keys: tuple[str, str]):
        self.name = name
        self.servos = servos  # {サーボID: 符号(+1/-1)}
        self.keys = keys
        self.offset = 0
        self.start: dict[int, int] = {}
        self.blocked = 0  # 負荷でこの向きへは進めない (+1/-1/0)


# 符号は実機の実測値 (|r| = 1.00)。docs/servo-bringup.md を参照。
JOINTS = [
    Joint("joint0", {4: +1, 5: -1, 6: -1, 7: +1}, ("w", "s")),
    Joint("joint1", {8: +1, 9: -1, 10: -1, 11: +1}, ("e", "d")),
    Joint("wrist12", {12: +1}, ("r", "f")),
    Joint("wrist13", {13: +1}, ("t", "g")),
    Joint("gripper", {15: +1}, ("y", "h")),
]

HELP = """
  w/s  joint0 (ID4-7 を 4 個同時)      r/f  wrist12
  e/d  joint1 (ID8-11 を 4 個同時)     t/g  wrist13
  y/h  gripper (ID15)
  [ ]  ステップ幅 -/+        space  その場で停止        0  トルクOFF
  ?    このヘルプ            q      終了 (トルクOFF)
"""


class Bus:
    def __init__(self, port: str, baud: int):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            sys.exit(f"ポートを開けません: {port}")
        self.ph.setBaudRate(baud)
        self.pk = PacketHandler(0)

    def r2(self, i: int, a: int, tries: int = 4) -> int | None:
        for _ in range(tries):
            v, c, _ = self.pk.read2ByteTxRx(self.ph, i, a)
            if c == COMM_SUCCESS:
                return v
        return None

    def w1(self, i: int, a: int, v: int) -> bool:
        return self.pk.write1ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def w2(self, i: int, a: int, v: int) -> bool:
        return self.pk.write2ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def close(self) -> None:
        self.ph.closePort()


def signed(v: int, bits: int = 10) -> int:
    """Feetech は最上位ビットを符号に使う (load は 10bit)。"""
    return -(v & ((1 << bits) - 1)) if v & (1 << bits) else v


def apply(bus: Bus, j: Joint) -> None:
    """関節の offset を各サーボの目標位置へ反映する。"""
    for sid, sign in j.servos.items():
        target = j.start[sid] + sign * j.offset
        bus.w2(sid, R_GOAL_POS, max(0, min(4095, target)))


def read_key(timeout: float) -> str | None:
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--step", type=int, default=20, help="1 キーあたりの移動量 (step)")
    p.add_argument("--speed", type=int, default=300, help="サーボの速度上限")
    p.add_argument("--acc", type=int, default=20)
    p.add_argument("--torque-limit", type=int, default=500, help="0-1000")
    p.add_argument(
        "--max-offset", type=int, default=700, help="開始位置からの最大移動量"
    )
    p.add_argument("--load-stop", type=int, default=450, help="この負荷で進行を止める")
    a = p.parse_args()

    bus = Bus(a.port, a.baud)
    all_ids = [sid for j in JOINTS for sid in j.servos]

    # 現在位置を目標にしてからトルクを入れる。順序を逆にすると跳ねる。
    print("現在位置を読み出し中…")
    for j in JOINTS:
        for sid in j.servos:
            pos = bus.r2(sid, R_POS)
            if pos is None:
                bus.close()
                sys.exit(
                    f"ID{sid} の位置が読めません。シリアル転送が有効か確認してください。"
                )
            j.start[sid] = pos
    for sid in all_ids:
        bus.w1(sid, R_ACC, a.acc)
        bus.w2(sid, R_GOAL_SPD, a.speed)
        bus.w2(sid, R_TORQUE_LIMIT, a.torque_limit)
    for j in JOINTS:
        apply(bus, j)
    for sid in all_ids:
        bus.w1(sid, R_TORQUE, 1)

    step = a.step
    torque_on = True
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    last_poll = 0.0
    loads: dict[int, int] = {}
    msg = "トルクON"

    print(HELP)
    try:
        tty.setcbreak(fd)
        while True:
            k = read_key(0.05)
            if k == "q":
                break
            elif k == "?":
                print(HELP)
            elif k == "[":
                step = max(1, step - 5)
                msg = f"ステップ {step}"
            elif k == "]":
                step = min(200, step + 5)
                msg = f"ステップ {step}"
            elif k == " ":
                for j in JOINTS:  # いまの位置を目標に固定する
                    apply(bus, j)
                msg = "停止"
            elif k == "0":
                for sid in all_ids:
                    bus.w1(sid, R_TORQUE, 0)
                torque_on = False
                msg = "トルクOFF (何かキーを押すと再投入)"
            elif k is not None:
                for j in JOINTS:
                    if k not in j.keys:
                        continue
                    if not torque_on:  # OFF 中の入力は現在位置から再開する
                        for sid in j.servos:
                            j.start[sid] = bus.r2(sid, R_POS) or j.start[sid]
                        for jj in JOINTS:
                            jj.offset = 0
                            for sid in jj.servos:
                                jj.start[sid] = bus.r2(sid, R_POS) or jj.start[sid]
                            apply(bus, jj)
                        for sid in all_ids:
                            bus.w1(sid, R_TORQUE, 1)
                        torque_on = True
                    d = step if k == j.keys[0] else -step
                    if j.blocked and (d > 0) == (j.blocked > 0):
                        msg = f"{j.name}: 負荷が高いのでこの向きへは進みません"
                        break
                    nxt = max(-a.max_offset, min(a.max_offset, j.offset + d))
                    if nxt == j.offset:
                        msg = f"{j.name}: 移動量の上限 ±{a.max_offset}"
                        break
                    j.offset = nxt
                    j.blocked = 0
                    apply(bus, j)
                    msg = f"{j.name} offset={j.offset:+d}"
                    break

            now = time.time()
            if now - last_poll > 0.35:
                last_poll = now
                for j in JOINTS:  # 各関節から 1 個だけ負荷を見る (全部読むと遅い)
                    sid = next(iter(j.servos))
                    load = bus.r2(sid, R_LOAD)
                    if load is None:
                        continue
                    loads[sid] = signed(load)
                    if abs(loads[sid]) > a.load_stop:
                        j.blocked = 1 if j.offset >= 0 else -1
                        j.offset -= 1 if j.offset >= 0 else -1
                        apply(bus, j)
                        msg = f"★{j.name}: load={loads[sid]} で停止"
                state = "  ".join(
                    f"{j.name}:{j.offset:+5d}" + ("!" if j.blocked else " ")
                    for j in JOINTS
                )
                sys.stdout.write(f"\r\033[K{state}  step={step:<3} | {msg}")
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        for sid in all_ids:
            bus.w1(sid, R_TORQUE, 0)
        bus.close()
        print("\n全サーボ トルクOFF。終了しました。")


if __name__ == "__main__":
    main()
