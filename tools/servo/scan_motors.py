#!/usr/bin/env python3
"""Feetech STS3215 バススキャン: /dev/ttyUSB* 上の全サーボIDを列挙する。
使い方: python scan_motors.py [ポート] [ボーレート...]
"""

import sys
import glob
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

port = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (sorted(glob.glob("/dev/ttyUSB*")) or ["/dev/ttyUSB0"])[0]
)
bauds = [int(b) for b in sys.argv[2:]] or [1_000_000, 500_000, 250_000, 115_200, 57_600]

ph = PortHandler(port)
if not ph.openPort():
    sys.exit(f"ポートを開けません: {port}")
pk = PacketHandler(0)  # STS/SMS protocol

for baud in bauds:
    ph.setBaudRate(baud)
    found = []
    for mid in range(0, 253):
        model, comm, err = pk.ping(ph, mid)
        if comm == COMM_SUCCESS:
            found.append((mid, model))
    print(
        f"[{baud:>9} bps] {len(found)} 個: "
        + (", ".join(f"ID{m}(model {md})" for m, md in found) if found else "なし")
    )
ph.closePort()
