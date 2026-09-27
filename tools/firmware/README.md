# サーボ制御基板のファーム

Waveshare "Servo Driver with ESP32" (ESP32-D0WD-V3) 用。

## 構成

- `backup/stock-*.bin` — 出荷時 (Waveshare 純正デモ) の 4MB フラッシュ dump。
  `sha256` を並置。以降のファームで問題があればここに戻せる
- `bridge/` — USB (UART0) ↔ サーボバス (UART1, GPIO18/19) を 1Mbps で素通しする
  だけの Arduino スケッチ。純正デモは Wi-Fi + Web UI (`Start Serial Forwarding`)
  経由でしか透過モードに入れず、この個体は Wi-Fi が起動しなかったため差し替えた

## ブリッジを焼く

```bash
brew install arduino-cli           # 未導入なら
cd tools/firmware/bridge
PORT=/dev/cu.usbserial-0001 bash build_and_flash.sh
```

初回は ESP32 コア (~1GB) がダウンロードされる。焼き込み後、基板の GPIO2 LED が
1 秒周期で点滅すればブリッジ動作中。

## 動作確認

```bash
source /tmp/esp-diag/bin/activate   # または pyserial が入った venv
python3 tools/firmware/bridge/verify.py --port /dev/cu.usbserial-0001
```

サーボバスが繋がっていれば応答した ID が並ぶ。全 ID 無応答でも、ブリッジ自体が
書き込みできていれば bytes を投げた事実は残る (バスの物理接続や電源を確認)。

## 純正ファームに戻す

```bash
source /tmp/esp-diag/bin/activate
esptool --port /dev/cu.usbserial-0001 --baud 460800 \
    write-flash 0x0 tools/firmware/backup/stock-waveshare-esp32-*.bin
```

## 以降のツール利用

ブリッジは USB 921600bps / バス 1Mbps のレート変換型なので、`tools/servo/*.py` は
921600 を明示指定する。`scan_motors.py` だけ引数が位置指定 (port bauds...)、
その他は `--baud` オプション。

```bash
python3 tools/servo/scan_motors.py /dev/cu.usbserial-0001 921600
python3 tools/servo/motor_check.py --baud 921600
python3 tools/servo/rezero.py --baud 921600
python3 tools/servo/teach_calibrate.py --baud 921600
python3 tools/servo/keyboard_teleop.py --baud 921600
```

## macOS 上での注意

CP2102 のドライバは tty オープン時に DTR/RTS をアサートする。Waveshare 基板の
自動リセット回路経由で ESP32 が EN=Low で保持されるので、ツール側で
`dtr=False, rts=False` を指定して開く必要がある。`bridge/verify.py` は既にそう
なっているが、他のツールが対応していない場合は同様の修正が必要。
