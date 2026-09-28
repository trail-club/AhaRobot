#!/usr/bin/env python3
"""从 ESP32 整片 Flash 备份中解析分区表 (默认偏移 0x8000)。
用法: python3 parse_partitions.py waveshare_stock_full_4MB.bin [分区表偏移]
"""

import struct
import sys

TYPES = {0x00: "app", 0x01: "data"}
APP_SUB = {0x00: "factory", 0x10: "ota_0", 0x11: "ota_1", 0x20: "test"}
DATA_SUB = {0x00: "ota", 0x01: "phy", 0x02: "nvs", 0x03: "coredump",
            0x04: "nvs_keys", 0x80: "esphttpd", 0x81: "fat", 0x82: "spiffs",
            0x83: "littlefs"}

path = sys.argv[1]
base = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x8000

with open(path, "rb") as f:
    f.seek(base)
    table = f.read(0xC00)

print(f"{'label':<12} {'type':<5} {'subtype':<9} {'offset':>10} {'size':>10}  size(KB)")
for i in range(0, len(table), 32):
    entry = table[i:i + 32]
    magic = entry[:2]
    if magic == b"\xEB\xEB":
        print("(MD5 校验条目)")
        continue
    if magic != b"\xAA\x50":
        break
    ptype, sub, off, size = struct.unpack_from("<BBII", entry, 2)
    label = entry[12:28].split(b"\x00")[0].decode(errors="replace")
    sub_name = (APP_SUB if ptype == 0 else DATA_SUB).get(sub, hex(sub))
    print(f"{label:<12} {TYPES.get(ptype, hex(ptype)):<5} {sub_name:<9} "
          f"{off:>#10x} {size:>#10x}  {size // 1024}")
