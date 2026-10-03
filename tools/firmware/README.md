# 制御基板のファームウェアツール

Waveshare Servo Driver with ESP32（ESP32-D0WD-V3）用。左右の制御基板は別個体のため、
バックアップを取り違えないよう `left_arm/` と `right_arm/` に分けて扱う。実機での
導入経緯と観測結果は [ファームウェア・通信の検証記録](../../docs/context/firmware.md)、
サーボ側の記録は [モーターの検証記録](../../docs/context/motor.md) を参照。

## 構成

| パス | 内容 |
| --- | --- |
| `left_arm/` | 左腕側の純正デモバックアップ、サーボ状態、パーティション解析スクリプト |
| `right_arm/` | 右腕側の純正デモバックアップ、サーボ状態 |
| `right_arm/bridge/` | USB（UART0）とサーボバス（UART1、GPIO18/19）の透過ブリッジ |

各腕のディレクトリは以下の構成で揃えている。`servo_state_before_init_*` の日付は
サーボ初期化（`init_arm.py` / `setupTorque(128)`）実行前の採取日。

```
<arm>/
├── Aharobot_esp32_backup.bin           ← 純正デモの4 MBフラッシュdump
├── Aharobot_esp32_backup.sha256
├── servo_eeprom_before_init_<date>.json
└── servo_state_before_init_<date>.txt  ← motor_check.py の出力
```

`parse_partitions.py` は左右で同一のため `left_arm/` にのみ置いている。
右腕は採取時点でAstraArmControllerが焼かれていたため、サーボEEPROMの吸い出しに
`bridge/` を一時書き込みして採取・書き戻しを行っている
（手順は後述、観測条件は [検証記録](../../docs/context/firmware.md#2026-10-03--右腕側esp32の-init_armpy-実行前バックアップ)）。

## UARTブリッジのビルド・書き込み

ホストにarduino-cliとesptoolを用意する。既存のファームを書き換えるため、バックアップを確認する。
現時点では右腕側で検証済み。左腕側に適用する際は同じ手順で `left_arm/` 配下にバックアップを残す。

```bash
cd tools/firmware/right_arm/bridge
PORT=/dev/cu.usbserial-0001 bash build_and_flash.sh
```

GPIO2のLEDが1秒周期で点滅すればブリッジ動作中。USBは921600 bps、サーボバスは1 Mbps。

```bash
# リポジトリ直下
python3 tools/firmware/right_arm/bridge/verify.py --port /dev/cu.usbserial-0001
```

サーボIDの応答まで確認する。全ID無応答なら電源・配線・サーボ側のボーレートを確認する。
サーボツールの使用方法は [README](../servo/README.md) を参照。

## バックアップの取得と照合

フラッシュの取得・照合はホストで行う。ポートと保存先は使用環境に合わせる。
`--after no-reset` で取得後の再起動を避け、純正ファームがNVSを更新する前に照合する。

```bash
esptool --port /dev/ttyUSB0 --baud 460800 --after no-reset \
  read-flash 0x0 0x400000 firmware-backup.bin
esptool --port /dev/ttyUSB0 --baud 460800 --after no-reset \
  verify-flash 0x0 firmware-backup.bin
python3 tools/firmware/left_arm/parse_partitions.py firmware-backup.bin
```

サーボEEPROMの読み取りは
[servo_snapshot.py](../servo/README.md#astraarmcontroller用の検証スクリプト) を使う。
AstraArmControllerが書き込まれた基板からサーボEEPROMを吸い出す場合は、下記
「AstraArmControllerが焼かれた状態からサーボEEPROMを吸い出す手順」を参照。

## 純正ファームへ戻す

復元対象のバックアップを選び、同じディレクトリのSHA256を照合してから書き込む。
左腕を復元する例。macOSでの照合は `shasum -a 256 -c`、Linuxでは `sha256sum -c` を使う。

```bash
cd tools/firmware/left_arm
shasum -a 256 -c Aharobot_esp32_backup.sha256
esptool --port /dev/cu.usbserial-0001 --baud 460800 \
  write-flash 0x0 Aharobot_esp32_backup.bin
```

ESP32のフラッシュを復元してもサーボ側のEEPROMは戻らない。
初期化で変更した動作モード・オフセットは、保存済みJSONと機体の状態を確認して個別に復元する。

純正デモの透過通信はWeb UIの `Start Serial Forwarding` を有効にして115200 bpsで接続する。
macOSではDTR / RTSによりESP32がリセット状態になる場合がある。
`bridge/verify.py` はポートを開く前に両方をFalseへ設定している。

## AstraArmControllerが焼かれた状態からサーボEEPROMを吸い出す手順

AstraArmControllerのUART0（921600 bps）は独自プロトコル専用でサーボバスへの
素通し窓口を持たないため、`servo_snapshot.py dump` を直接走らせても応答しない。
一時的に `bridge/` を焼き、採取後に書き戻す。LittleFSの `/config.txt`（キャリブ値）
を失わないよう、先にFlash全域をdumpする。macOS CP2102でのボーレートの観測結果は
[ファームウェアの検証記録](../../docs/context/firmware.md#2026-10-03--右腕側esp32の-init_armpy-実行前バックアップ)を参照。

```bash
# 1. 現状 (AstraArmController + /config.txt) を 4MB フル dump
mkdir -p tools/firmware/right_arm/pre_init_$(date +%Y%m%d)
cd tools/firmware/right_arm/pre_init_$(date +%Y%m%d)
uvx esptool --port /dev/cu.usbserial-0001 -b 230400 --after no-reset \
    read-flash 0x0 0x400000 Aharobot_esp32_backup.bin
shasum -a 256 Aharobot_esp32_backup.bin > Aharobot_esp32_backup.sha256
uvx esptool --port /dev/cu.usbserial-0001 -b 230400 --after no-reset \
    verify-flash 0x0 Aharobot_esp32_backup.bin        # → digest matched

# 2. bridge を焼く (AstraArmController を一時上書き)
cd $REPO/tools/firmware/right_arm/bridge
PORT=/dev/cu.usbserial-0001 bash build_and_flash.sh

# 3. ブリッジ経由でサーボ状態を dump (bridge は USB 921600bps)
cd $REPO
python3 tools/firmware/right_arm/bridge/verify.py --port /dev/cu.usbserial-0001  # ID 応答確認
python3 tools/servo/servo_snapshot.py --baud 921600 --port /dev/cu.usbserial-0001 dump \
    -o tools/firmware/right_arm/pre_init_$(date +%Y%m%d)/servo_eeprom_before_init.json
python3 tools/servo/motor_check.py --baud 921600 --port /dev/cu.usbserial-0001 \
    > tools/firmware/right_arm/pre_init_$(date +%Y%m%d)/servo_state_before_init.txt

# 4. AstraArmController を書き戻す (config.txt もフル dump に含まれるので復元)
uvx esptool --port /dev/cu.usbserial-0001 -b 230400 \
    write-flash 0x0 tools/firmware/right_arm/pre_init_*/Aharobot_esp32_backup.bin
uvx esptool --port /dev/cu.usbserial-0001 -b 230400 --after no-reset \
    verify-flash 0x0 tools/firmware/right_arm/pre_init_*/Aharobot_esp32_backup.bin
```
