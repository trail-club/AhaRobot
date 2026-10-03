#!/usr/bin/env python3
"""STS3215 state snapshot / position monitor (via Serial Forwarding of ESP32 stock firmware).

Usage:
  uv run servo_snapshot.py dump                 # Dump EEPROM area (0x00-0x27) of ID4-15 to JSON
  uv run servo_snapshot.py dump --ids 15-15     # Dump only a specific ID
  uv run servo_snapshot.py watch 15             # Continuously print current position of ID15 (manually move the gripper to observe direction)

Read-only; does not write any register.
"""

import argparse
import datetime
import json
import sys
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_ID, ADDR_BAUD = 0x05, 0x06
ADDR_MIN_ANGLE, ADDR_MAX_ANGLE = 0x09, 0x0B
ADDR_OFS, ADDR_MODE = 0x1F, 0x21
ADDR_PRESENT_POS = 0x38
EEPROM_LEN = 0x28


def sign_mag(v: int) -> int:
    """STS offset is sign bit (bit11) + 11-bit magnitude."""
    mag = v & 0x7FF
    return -mag if v & 0x800 else mag


def parse_ids(s: str):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--baud", type=int, default=115200)
sub = ap.add_subparsers(dest="cmd", required=True)
p_dump = sub.add_parser("dump")
p_dump.add_argument("--ids", default="4-15")
p_dump.add_argument("-o", "--out", default=None)
p_watch = sub.add_parser("watch")
p_watch.add_argument("id", type=int)
p_watch.add_argument("--sec", type=float, default=15.0)
args = ap.parse_args()

ph = PortHandler(args.port)
if not ph.openPort():
    sys.exit(f"Failed to open port: {args.port}")
ph.setBaudRate(args.baud)
pk = PacketHandler(0)

if args.cmd == "dump":
    result = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
              "port": args.port, "baud": args.baud, "servos": {}}
    for sid in parse_ids(args.ids):
        # Reading 40 bytes at once via ESP32 forwarding causes "Incorrect status packet",
        # so read byte-by-byte (each is a short packet, stable like motor_check.py).
        data, fail = [], None
        for addr in range(EEPROM_LEN):
            v, comm, err = pk.read1ByteTxRx(ph, sid, addr)
            if comm != COMM_SUCCESS:
                v, comm, err = pk.read1ByteTxRx(ph, sid, addr)  # retry once
            if comm != COMM_SUCCESS:
                fail = f"addr 0x{addr:02X}: {pk.getTxRxResult(comm)}"
                break
            data.append(v)
        if fail:
            print(f"ID{sid:>2}: read failed ({fail})")
            result["servos"][sid] = {"error": fail}
            continue
        word = lambda a: data[a] | (data[a + 1] << 8)
        pos, comm2, _ = pk.read2ByteTxRx(ph, sid, ADDR_PRESENT_POS)
        info = {
            "eeprom_raw": list(data),
            "id": data[ADDR_ID],
            "baud_code": data[ADDR_BAUD],
            "min_angle": word(ADDR_MIN_ANGLE),
            "max_angle": word(ADDR_MAX_ANGLE),
            "offset_raw": word(ADDR_OFS),
            "offset": sign_mag(word(ADDR_OFS)),
            "mode": data[ADDR_MODE],
            "present_pos": pos if comm2 == COMM_SUCCESS else None,
        }
        result["servos"][sid] = info
        print(f"ID{sid:>2}: mode={info['mode']} offset={info['offset']:>5} "
              f"(raw {info['offset_raw']}) limits=[{info['min_angle']},{info['max_angle']}] "
              f"pos={info['present_pos']}")
    out = args.out or f"servo_eeprom_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Saved: {out}")

elif args.cmd == "watch":
    print(f"Watching position of ID{args.id} for {args.sec:.0f} seconds. Move it manually with torque disabled. Ctrl+C to stop.")
    t_end = time.time() + args.sec
    lo, hi = 4096, -1
    try:
        while time.time() < t_end:
            pos, comm, _ = pk.read2ByteTxRx(ph, args.id, ADDR_PRESENT_POS)
            if comm == COMM_SUCCESS:
                lo, hi = min(lo, pos), max(hi, pos)
                print(f"\rpos={pos:>4}  min={lo:>4}  max={hi:>4}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()

ph.closePort()
