#!/usr/bin/env python3
"""サーボの中点を取り直す (EEPROM 書き込み)。

STS3215 の位置は 0-4095 の絶対値で、原点 (0/4095 の境目) に止まっているサーボが
あると読み値が飛び、位置指令も範囲外を出せなくなって関節が片側に動かせない。
現在位置を 2048 (中点) として登録し直すことで解消する。

レジスタ 40 (トルクイネーブル) に 128 を書くと、サーボが現在位置を中点として
オフセットレジスタを書き換える (Feetech SDK の CalibrationOfs 相当)。AhaRobot の
`doInitJoint` も同じ手順を踏む。**EEPROM 書き込みなので電源を切っても残る。**

  python3 rezero.py                 # 何が起きるか表示するだけ
  python3 rezero.py --apply         # 実行
  python3 rezero.py --apply --ids 8,9,10,11
"""

from __future__ import annotations

import argparse
import sys
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

R_TORQUE, R_OFS, R_POS = 40, 31, 56
CALIBRATE_MID = 128  # レジスタ 40 に書くと現在位置を中点にする
DEFAULT_IDS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
ORIGIN_MARGIN = 150


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--ids", help="対象IDをカンマ区切りで指定 (既定は全部)")
    p.add_argument("--apply", action="store_true", help="実際に EEPROM を書き換える")
    a = p.parse_args()

    ids = [int(x) for x in a.ids.split(",")] if a.ids else DEFAULT_IDS

    ph = PortHandler(a.port)
    if not ph.openPort():
        sys.exit(f"ポートを開けません: {a.port}")
    ph.setBaudRate(a.baud)
    pk = PacketHandler(0)

    def r2(i: int, addr: int) -> int | None:
        for _ in range(4):
            v, c, _ = pk.read2ByteTxRx(ph, i, addr)
            if c == COMM_SUCCESS:
                return v
        return None

    def w1(i: int, addr: int, v: int) -> bool:
        for _ in range(3):
            if pk.write1ByteTxRx(ph, i, addr, v)[0] == COMM_SUCCESS:
                return True
        return False

    print("全サーボのトルクを切ります (校正中に動かないように)")
    for i in ids:
        w1(i, R_TORQUE, 0)
    time.sleep(0.3)

    before = {i: r2(i, R_POS) for i in ids}
    ofs_before = {i: r2(i, R_OFS) for i in ids}
    missing = [i for i in ids if before[i] is None]
    if missing:
        ph.closePort()
        sys.exit(f"ID{missing} が読めません。シリアル転送が有効か確認してください。")

    print(f"\n{'ID':>3} {'現在位置':>8} {'原点まで':>8} {'現在ofs':>8}   状態")
    for i in ids:
        room = min(before[i], 4095 - before[i])
        state = "★原点に近い" if room < ORIGIN_MARGIN else ""
        print(f"{i:>3} {before[i]:>8} {room:>8} {ofs_before[i]:>8}   {state}")

    if not a.apply:
        print("\n--apply を付けると、上の現在位置がそれぞれ 2048 になるよう")
        print("オフセットを書き換えます (EEPROM。電源を切っても残ります)。")
        ph.closePort()
        return

    print("\n中点を書き込み中…")
    failed = []
    for i in ids:
        if not w1(i, R_TORQUE, CALIBRATE_MID):
            failed.append(i)
        time.sleep(0.05)
    time.sleep(0.5)

    after = {i: r2(i, R_POS) for i in ids}
    print(f"\n{'ID':>3} {'変更前':>8} {'変更後':>8}   判定")
    ok = True
    for i in ids:
        good = after[i] is not None and abs(after[i] - 2048) < 30
        ok = ok and good
        print(f"{i:>3} {before[i]:>8} {str(after[i]):>8}   {'OK' if good else '★失敗'}")
    if failed:
        print(f"書き込みに失敗した ID: {failed}")

    for i in ids:
        w1(i, R_TORQUE, 0)
    ph.closePort()
    print(
        "\n全サーボ トルクOFF。"
        + ("完了しました。" if ok else "失敗したサーボがあります。")
    )


if __name__ == "__main__":
    main()
