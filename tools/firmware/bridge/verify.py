#!/usr/bin/env python3
"""ブリッジファーム焼き込み後の動作確認。

STS ping (broadcast) を投げ、応答があった ID を列挙する。サーボが繋がっていない
場合は「無応答」で正常だが、少なくともブリッジがバスに書き込めていること
(TX が反射エコーで返ってくるなど) は確認できる。
"""

import argparse
import time

import serial

BROADCAST = 0xFE


def build_ping(id_: int) -> bytes:
    # header, id, len(2), instr=0x01 (PING), checksum
    body = [id_, 0x02, 0x01]
    chk = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, chk])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/cu.usbserial-0001")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--ids", default="4-15")
    args = p.parse_args()

    lo, hi = (int(x) for x in args.ids.split("-"))

    s = serial.Serial()
    s.port = args.port
    s.baudrate = args.baud
    s.dtr = False
    s.rts = False
    s.timeout = 0.05
    s.open()
    time.sleep(0.5)  # let bridge boot after DTR/RTS reset
    s.reset_input_buffer()

    found = []
    for i in range(lo, hi + 1):
        s.write(build_ping(i))
        s.flush()
        time.sleep(0.02)
        r = s.read(64)
        if r and 0xFF in r:
            print(f"  ID {i:3d}: reply {r.hex()}")
            found.append(i)
    print(f"\nfound: {found}")
    s.close()


if __name__ == "__main__":
    main()
