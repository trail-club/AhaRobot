#!/usr/bin/env bash
# ODrive v3.6 (MKS) へファームウェアを書き込む。
#
# odrivetool dfu は MKS 基板の OTP にハードウェア版数がないため使えない。
# 書き込み前に設定を JSON へ退避し、DFU へ切り替えて dfu-util で書き込み、
# 再接続後に版数と主要な設定値を退避した JSON と比べる。
# 設定が変わっていたら、表示される restore-config を手で実行する。
#
# Usage:  flash_fw.sh FIRMWARE.bin [BACKUP_DIR]
#   ODRIVE_PY に odrive==0.5.4 を入れた python を指定する (既定 ~/odrive-venv/bin/python)
set -euo pipefail

BIN=${1:?firmware .bin を指定する}
BACKUP_DIR=${2:-$HOME/odrive-fw/backups}
PY=${ODRIVE_PY:-$HOME/odrive-venv/bin/python}
ODRIVETOOL=$(dirname "$PY")/odrivetool
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$BACKUP_DIR/config_before_flash_$STAMP.json"

test -f "$BIN"
mkdir -p "$BACKUP_DIR"

echo "== 設定を退避: $BACKUP"
"$ODRIVETOOL" backup-config "$BACKUP"

echo "== DFU モードへ切り替え"
"$PY" - <<'EOF' || true
import odrive
d = odrive.find_any(timeout=15)
try:
    d.enter_dfu_mode()
except Exception:
    pass  # 切断されるのが正常
EOF

for _ in $(seq 30); do
    lsusb -d 0483:df11 >/dev/null && break
    sleep 1
done
lsusb -d 0483:df11 >/dev/null || { echo "DFU デバイスが見つからない"; exit 1; }

echo "== 書き込み: $BIN"
dfu-util -a 0 -s 0x08000000:leave -D "$BIN"

echo "== 再接続と確認"
sleep 5
"$PY" - "$BACKUP" <<'EOF'
import json, sys
import odrive

backup = json.load(open(sys.argv[1]))
d = odrive.find_any(timeout=30)
print(f"fw {d.fw_version_major}.{d.fw_version_minor}.{d.fw_version_revision}"
      f" unreleased={d.fw_version_unreleased}")
print("hall filter attrs:", hasattr(d.axis0.encoder, "hall_debounce_cycles"))
keys = [
    ("motor", "config", "phase_resistance"), ("motor", "config", "phase_inductance"),
    ("motor", "config", "pre_calibrated"), ("encoder", "config", "mode"),
    ("encoder", "config", "pre_calibrated"), ("encoder", "config", "direction"),
    ("encoder", "config", "phase_offset"), ("encoder", "config", "hall_polarity"),
    ("config", "can", "node_id"),
]
ok = True
for ax in ("axis0", "axis1"):
    for path in keys:
        want = backup[ax]
        for k in path:
            want = want.get(k) if isinstance(want, dict) else None
        obj = getattr(d, ax)
        for k in path:
            obj = getattr(obj, k, None)
        same = want is None or (abs(obj - want) < 1e-6 if isinstance(want, float) else obj == want)
        ok &= same
        print(f"{ax}.{'.'.join(path)}: now={obj} backup={want} {'OK' if same else 'DIFF'}")
print("設定は保持されている" if ok else
      f"設定が変わった。復元: odrivetool restore-config {sys.argv[1]} の後に save_configuration()")
EOF
