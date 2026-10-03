# 制御基板のファームウェアツール

Waveshare Servo Driver with ESP32（ESP32-D0WD-V3）用。
実機での導入経緯と観測結果は [ファームウェア・通信の検証記録](../../docs/context/firmware.md)を参照。

| パス | 内容 |
| --- | --- |
| `bridge/` | USB（UART0）とサーボバス（UART1、GPIO18 / 19）の透過ブリッジ |
| `backup/stock-*.bin` | 純正デモの4 MBフラッシュdump。SHA256を併記 |
| `backup/backup_0927/` | AstraArmController導入前のフラッシュ・サーボ設定。[保存内容](#バックアップ) |
| `esp32_backup/` | 9月27日のフラッシュdumpの複製と `parse_partitions.py`（パーティション表の表示） |

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

## バックアップ

`backup/backup_0927/` には以下を保存している。取得対象・条件・観測結果は
[ファームウェアの検証記録](../../docs/context/firmware.md#2026-09-27--astraarmcontroller導入前のバックアップと書き込み)、
サーボ設定の解釈は [モーターの検証記録](../../docs/context/motor.md#初期化前のサーボ設定とグリッパの解釈)を参照。

| ファイル | 内容 |
| --- | --- |
| `Aharobot_esp32_backup.bin` | 純正デモのフラッシュ全域（4 MB） |
| `Aharobot_esp32_backup.sha256` | 上記のSHA256 |
| `servo_eeprom_before_init_20260927.json` | ID4–15のEEPROM（0x00–0x27）と主要値 |
| `servo_state_before_init_20260927.txt` | `motor_check.py` の状態出力 |

フラッシュの取得・照合はホストで行う。ポートと保存先は使用環境に合わせる。
`--after no-reset` で取得後の再起動を避け、純正ファームがNVSを更新する前に照合する。

```bash
esptool --port /dev/ttyUSB0 --baud 460800 --after no-reset \
  read-flash 0x0 0x400000 firmware-backup.bin
esptool --port /dev/ttyUSB0 --baud 460800 --after no-reset \
  verify-flash 0x0 firmware-backup.bin
python3 tools/firmware/esp32_backup/parse_partitions.py firmware-backup.bin
```

サーボEEPROMの読み取りは [servo_snapshot.py](../servo/README.md#astraarmcontroller用の検証スクリプト)を使う。

## 純正ファームへ戻す

復元対象のバックアップを選び、同じディレクトリのSHA256を照合してから書き込む。
次はリポジトリ直下から `backup_0927` を復元する例。macOSでの照合は `shasum -a 256 -c`、Linuxでは `sha256sum -c` を使う。

```bash
cd tools/firmware/backup/backup_0927
shasum -a 256 -c Aharobot_esp32_backup.sha256
esptool --port /dev/cu.usbserial-0001 --baud 460800 \
  write-flash 0x0 Aharobot_esp32_backup.bin
```

ESP32のフラッシュを復元してもサーボ側のEEPROMは戻らない。
初期化で変更した動作モード・オフセットは、保存済みJSONと機体の状態を確認して個別に復元する。

純正デモの透過通信はWeb UIの `Start Serial Forwarding` を有効にして115200 bpsで接続する。
macOSではDTR / RTSによりESP32がリセット状態になる場合がある。
`bridge/verify.py` はポートを開く前に両方をFalseへ設定している。
