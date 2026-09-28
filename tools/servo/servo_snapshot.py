#!/usr/bin/env python3
"""STS3215 状态快照 / 位置监视（经由 ESP32 原厂固件的 Serial Forwarding）。

用法:
  uv run servo_snapshot.py dump                 # 导出 ID4-15 的 EEPROM 区 (0x00-0x27) 到 JSON
  uv run servo_snapshot.py dump --ids 15-15     # 只导出某个 ID
  uv run servo_snapshot.py watch 15             # 连续打印 ID15 的当前位置（手动拨动夹爪观察方向）

只读，不写任何寄存器。
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
    """STS 的 offset 是 符号位(bit11) + 11bit 大小。"""
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
    sys.exit(f"无法打开端口: {args.port}")
ph.setBaudRate(args.baud)
pk = PacketHandler(0)

if args.cmd == "dump":
    result = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
              "port": args.port, "baud": args.baud, "servos": {}}
    for sid in parse_ids(args.ids):
        # 经 ESP32 转发时一次读 40 字节会出现 "Incorrect status packet"，
        # 所以逐字节读取（每次都是短包，和 motor_check.py 一样稳定）
        data, fail = [], None
        for addr in range(EEPROM_LEN):
            v, comm, err = pk.read1ByteTxRx(ph, sid, addr)
            if comm != COMM_SUCCESS:
                v, comm, err = pk.read1ByteTxRx(ph, sid, addr)  # 重试一次
            if comm != COMM_SUCCESS:
                fail = f"addr 0x{addr:02X}: {pk.getTxRxResult(comm)}"
                break
            data.append(v)
        if fail:
            print(f"ID{sid:>2}: 读取失败 ({fail})")
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
    print(f"已保存: {out}")

elif args.cmd == "watch":
    print(f"监视 ID{args.id} 的位置 {args.sec:.0f} 秒，请在力矩关闭状态下手动拨动。Ctrl+C 结束。")
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