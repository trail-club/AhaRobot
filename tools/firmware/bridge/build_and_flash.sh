#!/usr/bin/env bash
# arduino-cli を使って uart_bridge をコンパイル & 焼き込む。
# 純正ファームを差し替えるので、事前にバックアップを取っておくこと
# (tools/firmware/backup/stock-*.bin)。
#
# 依存: arduino-cli (brew install arduino-cli)、esptool (uv pip install esptool)。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SKETCH_DIR="$SCRIPT_DIR/uart_bridge"
BUILD_DIR="$SCRIPT_DIR/build"
FQBN="esp32:esp32:esp32"
PORT="${PORT:-/dev/cu.usbserial-0001}"
UPLOAD_BAUD="${UPLOAD_BAUD:-460800}"

if ! command -v arduino-cli >/dev/null; then
  echo "arduino-cli が見つからない。brew install arduino-cli を実行してください。" >&2
  exit 1
fi

echo "==> ESP32 core 導入確認"
arduino-cli core list 2>/dev/null | grep -q "^esp32:esp32" || {
  arduino-cli config init --overwrite >/dev/null 2>&1 || true
  arduino-cli config add board_manager.additional_urls \
    https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  arduino-cli core update-index
  arduino-cli core install esp32:esp32
}

echo "==> コンパイル ($FQBN)"
arduino-cli compile \
  --fqbn "$FQBN" \
  --build-path "$BUILD_DIR" \
  "$SKETCH_DIR"

BIN="$BUILD_DIR/uart_bridge.ino.bin"
if [[ ! -f "$BIN" ]]; then
  echo "ビルド成果物が見つからない: $BIN" >&2
  exit 1
fi

echo "==> 焼き込み ($PORT, upload baud $UPLOAD_BAUD)"
arduino-cli upload \
  --fqbn "$FQBN" \
  --port "$PORT" \
  --input-dir "$BUILD_DIR" \
  --upload-property upload.speed="$UPLOAD_BAUD"

echo
echo "焼き込み完了。GPIO2 の LED が 1 秒周期で点滅すればブリッジ動作中。"
echo "以降のツールは 921600bps で開いてください。例:"
echo "  python3 tools/servo/scan_motors.py /dev/cu.usbserial-0001 921600"
echo "  python3 tools/servo/motor_check.py --baud 921600"
