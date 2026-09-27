# ESP32 制御基板バックアップ (2026-09-27)

AstraArmController を書き込む**前**の状態を保存したもの。
純正ファームへの復元、および `do_init` (setupTorque(128)) 実行前のサーボ設定の参照用。

## 対象

| 項目 | 値 |
| --- | --- |
| 基板 | Waveshare "Servo Driver with ESP32" |
| チップ | ESP32-D0WD-V3 (revision v3.1), 40MHz crystal |
| MAC | `28:05:a5:c4:f7:44` |
| Flash | 4MB (manufacturer 0x46, device 0x4016), 3.3V |
| 中身 | Waveshare 純正デモファーム (AhaRobot のファームではない) |

## ファイル

| ファイル | 内容 |
| --- | --- |
| `Aharobot_esp32_backup.bin` | Flash 全域 4MB (0x0–0x3FFFFF) のダンプ |
| `Aharobot_esp32_backup.sha256` | 上記の SHA-256 |
| `servo_eeprom_before_init_20260927.json` | ID4–15 の EEPROM 領域 (0x00–0x27) の生データと主要値 |
| `servo_state_before_init_20260927.txt` | `tools/servo/motor_check.py` の出力 |

## 取得方法

- 環境: WSL2 + usbipd-win (CP210x を WSL に attach), esptool v5.4.0 (`uvx esptool`)
- **ボーレートは 460800**。921600 では usbipd 経由で読み出し中にバイト欠落が起きた
  (`Corrupt data, expected 0x1000 bytes but received 0xfdd bytes`)

```bash
uvx esptool --port /dev/ttyUSB0 -b 460800 --after no-reset read-flash 0x0 0x400000 Aharobot_esp32_backup.bin
uvx esptool --port /dev/ttyUSB0 -b 460800 verify-flash 0x0 Aharobot_esp32_backup.bin   # → digest matched
```

### 注意: 純正ファームは起動のたびに NVS を書き換える

読み出し後にリセットして純正ファームが一度起動すると、NVS の 1 セクタ (`0xB000`) が
変わり、`verify-flash` が失敗する。2 回ダンプを取って比較し、差分がこの 1 セクタだけで
あることを確認済み (bootloader / パーティションテーブル / app は一致)。
上記のように `--after no-reset` で読み出し、ファームを起動させずに verify すること。

## パーティションテーブル (純正ファーム)

Arduino-ESP32 1.0.x 世代のデフォルト構成。

| label | type | subtype | offset | size |
| --- | --- | --- | --- | --- |
| nvs | data | nvs | 0x9000 | 0x5000 |
| otadata | data | ota | 0xE000 | 0x2000 |
| app0 | app | ota_0 | 0x10000 | 0x140000 |
| app1 | app | ota_1 | 0x150000 | 0x140000 |
| eeprom | data | 0x99 | 0x290000 | 0x1000 |
| spiffs | data | spiffs | 0x291000 | 0x16F000 |

- 起動スロットは app0 (otadata seq=1)。app1 は空
- eeprom パーティションは全て 0xFF (未使用)

AstraArmController (espressif32 7.1.3 / Arduino-ESP32 2.0.17) はデフォルトの
パーティション表を使い、設定を LittleFS の `/config.txt` に保存する
(`LittleFS.begin(true)` なのでマウント失敗時は自動フォーマット)。
配置が異なるため、書き込み前に `erase-flash` で全消去する運用にした。

## 復元

```bash
sha256sum -c Aharobot_esp32_backup.sha256
uvx esptool --port /dev/ttyUSB0 -b 460800 write-flash 0x0 Aharobot_esp32_backup.bin
```

**サーボ側の EEPROM は Flash の復元では戻らない。** `do_init` で書き換わった
動作モード (ID4–11) や ID15 のオフセットは、下記の JSON を参照して個別に書き戻すこと。

## サーボ設定 (do_init 実行前)

全数 mode=0 (位置モード)、角度リミット [0, 4095]、ボーレート 1Mbps。

| ID | offset | ID | offset | ID | offset |
| --- | --- | --- | --- | --- | --- |
| 4 | -1042 | 8 | 1830 | 12 | -1636 |
| 5 | -1762 | 9 | -1900 | 13 | -1660 |
| 6 | -1928 | 10 | **2047** | 14 | -627 |
| 7 | -1836 | 11 | -1685 | 15 | **-1000** |

- ID10 の offset はちょうど正側の上限値 (2047)。`do_init` は触らないが、経緯は未確認
- ID15 (グリッパ) はトルク OFF で手動計測して **全閉 ≈ 3500 / 全開 ≈ 656**。
  開く方向で読み値が減り、原点をまたがない
- `doInitJoint(15, 1448)` は「実行時点のグリッパ位置を読み値 2048+1448 = 3496 にする」
  処理。全閉で実行すればオフセットの変化は数 step に収まる。**全閉以外で実行すると
  可動域が原点をまたぐ可能性がある** (全閉姿勢が upstream の前提かは未確認)
