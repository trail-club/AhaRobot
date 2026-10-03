# 制御基板のファームウェアツール

Waveshare Servo Driver with ESP32（ESP32-D0WD-V3）用。
実機での導入経緯と観測結果は [ファームウェア・通信の検証記録](../../docs/context/firmware.md)を参照。

| パス | 内容 |
| --- | --- |
| `bridge/` | USB（UART0）とサーボバス（UART1、GPIO18 / 19）の透過ブリッジ |
| `backup/stock-*.bin` | 純正デモの4 MBフラッシュdump。SHA256を併記 |

## ビルド・書き込み

ホストにarduino-cliとesptoolを用意する。既存のファームを書き換えるため、バックアップを確認する。

```bash
cd tools/firmware/bridge
PORT=/dev/cu.usbserial-0001 bash build_and_flash.sh
```

GPIO2のLEDが1秒周期で点滅すればブリッジ動作中。USBは921600 bps、サーボバスは1 Mbps。

```bash
# リポジトリ直下
python3 tools/firmware/bridge/verify.py --port /dev/cu.usbserial-0001
```

サーボIDの応答まで確認する。全ID無応答なら電源・配線・サーボ側のボーレートを確認する。
サーボツールの使用方法は [README](../servo/README.md)を参照。

## 純正ファームへ戻す

```bash
esptool --port /dev/cu.usbserial-0001 --baud 460800 \
  write-flash 0x0 tools/firmware/backup/stock-waveshare-esp32-*.bin
```

純正デモの透過通信はWeb UIの `Start Serial Forwarding` を有効にして115200 bpsで接続する。
macOSではDTR / RTSによりESP32がリセット状態になる場合がある。
`bridge/verify.py` はポートを開く前に両方をFalseへ設定している。
