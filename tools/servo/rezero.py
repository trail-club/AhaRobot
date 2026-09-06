#!/usr/bin/env python3
"""サーボの中点を取り直す (EEPROM 書き込み)。

STS3215 の位置は 0-4095 の絶対値で、原点 (0/4095 の境目) に止まっているサーボが
あると読み値が飛び、位置指令も範囲外を出せなくなって関節が片側に動かせない。
現在位置を 2048 (中点) として登録し直すことで解消する。

レジスタ 40 (トルクイネーブル) に 128 を書くと、サーボが現在位置を中点として
オフセットレジスタを書き換える (Feetech SDK の CalibrationOfs 相当)。AhaRobot の
`doInitJoint` も同じ手順を踏む。**EEPROM 書き込みなので電源を切っても残る。**

もう一つの用途が中心合わせ。teach_calibrate.py で測った可動域の中心が 2048 に
来るようオフセットをずらす。可動域が原点をまたぐサーボはこれをしないと片側に
動かせない。オフセットを +1 すると読み値が -1 されるので、
  新オフセット = 旧オフセット + (実測中心 - 2048)

  python3 rezero.py                 # 何が起きるか表示するだけ
  python3 rezero.py --apply         # 現在位置を中点にする
  python3 rezero.py --center calibration.json          # 中心合わせの計画を表示
  python3 rezero.py --center calibration.json --apply  # 実行
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

R_TORQUE, R_OFS, R_POS, R_LOCK = 40, 31, 56, 55
CALIBRATE_MID = 128  # レジスタ 40 に書くと現在位置を中点にする
DEFAULT_IDS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ORIGIN_MARGIN = 150


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--ids", help="対象IDをカンマ区切りで指定 (既定は全部)")
    p.add_argument("--apply", action="store_true", help="実際に EEPROM を書き換える")
    p.add_argument(
        "--center", help="teach_calibrate.py の JSON。実測中心を 2048 に合わせる"
    )
    p.add_argument(
        "--min-shift",
        type=int,
        default=50,
        help="このステップ数未満のずれは書き換えない",
    )
    a = p.parse_args()

    ids = [int(x) for x in a.ids.split(",")] if a.ids else DEFAULT_IDS

    ph = PortHandler(a.port)
    if not ph.openPort():
        sys.exit(f"ポートを開けません: {a.port}")
    ph.setBaudRate(a.baud)
    pk = PacketHandler(0)

    # SDK は応答が途中で切れると IndexError を投げるので、握って再試行する
    def r2(i: int, addr: int) -> int | None:
        for _ in range(5):
            try:
                v, c, _ = pk.read2ByteTxRx(ph, i, addr)
                if c == COMM_SUCCESS:
                    return v
            except (IndexError, TypeError):
                pass
        return None

    def w1(i: int, addr: int, v: int) -> bool:
        for _ in range(3):
            try:
                if pk.write1ByteTxRx(ph, i, addr, v)[0] == COMM_SUCCESS:
                    return True
            except (IndexError, TypeError):
                pass
        return False

    def w2(i: int, addr: int, v: int) -> bool:
        for _ in range(3):
            try:
                if pk.write2ByteTxRx(ph, i, addr, v)[0] == COMM_SUCCESS:
                    return True
            except (IndexError, TypeError):
                pass
        return False

    def dec(o: int) -> int:
        """オフセットは符号ビット (2048) + 大きさ 11bit で格納されている。"""
        return -(o - 2048) if o > 2048 else o

    def enc(o: int) -> int:
        # 大きさは 11bit しかないので [-2048, 2047] に畳んでから符号ビットを付ける
        o = (o + 2048) % 4096 - 2048
        return o if o >= 0 else (-o) + 2048

    def write_offset(i: int, value: int) -> bool:
        """EEPROM を解錠して書き、施錠し直す。"""
        w1(i, R_LOCK, 0)
        time.sleep(0.05)
        ok = w2(i, R_OFS, enc(value))
        time.sleep(0.05)
        w1(i, R_LOCK, 1)
        time.sleep(0.1)
        return ok

    def do_center(path: str) -> None:
        stats = json.load(open(path))["stats"]
        for i in ids:
            w1(i, R_TORQUE, 0)
        time.sleep(0.3)
        print(
            f"\n{'ID':>3} {'実測中心':>9} {'ずれ':>7} {'旧ofs':>7} {'新ofs':>7}   処理"
        )
        plan = {}
        for i in ids:
            st = stats.get(str(i))
            if st is None:
                print(f"{i:>3}   測定データがありません")
                continue
            cur_ofs = r2(i, R_OFS)
            if cur_ofs is None:
                print(f"{i:>3}   ★オフセットが読めません。再実行してください")
                continue
            center = round(st["mid_raw"])
            shift = center - 2048
            new = (dec(cur_ofs) + shift + 2048) % 4096 - 2048
            skip = abs(shift) < a.min_shift
            print(
                f"{i:>3} {center:>9} {shift:>+7} {dec(cur_ofs):>7} {new:>7}   "
                + ("ずれが小さいので変更しない" if skip else "書き換える")
            )
            if not skip:
                plan[i] = new
        if not a.apply:
            print("\n--apply を付けると上の書き換えを実行します。")
            return
        print("\n書き込み中…")
        for i, new in plan.items():
            write_offset(i, new)
        time.sleep(0.3)
        print(f"\n{'ID':>3} {'新ofs':>7} {'現在位置':>9}")
        for i in plan:
            print(f"{i:>3} {dec(r2(i, R_OFS) or 0):>7} {str(r2(i, R_POS)):>9}")
        for i in ids:
            w1(i, R_TORQUE, 0)

    if a.center:
        do_center(a.center)
        ph.closePort()
        return

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
