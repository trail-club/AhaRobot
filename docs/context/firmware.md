# 制御基板のファームウェア・通信検証記録

実機の制御基板の通信調査、バックアップと書き込みの確認を日付順に記録する。
以下は当時の検証条件での結果であり、現在の仕様や動作保証とは区別する。
操作手順は [ファームウェアツール](../../tools/firmware/README.md)、サーボの測定・操作時の問題は
[モーターの測定・検証記録](motor.md)を参照。

## 2026-09-27 — 純正制御基板で通信できない問題の調査

### 背景・条件

購入直後のWaveshare Servo Driver with ESP32で、Wi-Fi APとOLEDが起動せず、
Web UIから透過通信を有効にできなかった。サーボスキャンツールからも通信できなかった。
この確認ではID4–13と15の11個が対象で、ID14は未搭載だった。
9月6日の12個構成と異なり、同じ個体・組み付け状態と確認できる識別情報は残っていない。

### 切り分けと試した条件

CPUは動作しており、搭載ファームも純正デモだった。
当時の調査では、サーボが1個も見つからない場合にスキャンでループし、
OLED / Wi-Fiの初期化まで到達しない挙動と判断した。
UARTコマンドから透過モードへ移れないかも試したが、試した書式では受け付けられなかった。
具体的なコマンド列と起動ログは保存されておらず、純正デモ全般の仕様とは断定しない。

純正4 MBフラッシュを [backup/](../../tools/firmware/backup/) に保存した上で、
Wi-Fiを使わないUARTブリッジによる通信を試した。
USBとサーボバスの両側を1 Mbpsにした条件では、macOSのCP2102経由で応答が0バイトだった。
USB側921600 bps、サーボバス側1 Mbpsの条件に変更すると応答が得られた。
また、ポートを開く際のDTR / RTSによりESP32がEN=Lowのリセット状態に保持される現象があり、
ポートを開く前に両方をFalseにする必要があった。

### 決定事項・得られた結果・残った制限

Wi-Fi起動やWeb UIに依存せずサーボバスの確認を進めるため、独自UARTブリッジを使うことにした。
通信応答が得られたUSB側921600 bps / バス側1 Mbpsの組み合わせを採用した。

`verify.py` でID4–13と15の11個が921600 bpsで応答することを確認した。
この結果はUSBからサーボバスまでのping応答の確認であり、
関節動作やROS制御、純正デモのWi-Fi問題の解決を示すものではない。
9月6日の純正デモ＋115200 bpsでの通信結果と混同しない。

## 2026-09-27 — AstraArmController導入前のバックアップと書き込み

### 対象・条件

[backup_0927/](../../tools/firmware/backup/backup_0927/) に、AstraArmController導入前の
Waveshare Servo Driver with ESP32の純正デモを保存した。
記録上のチップはESP32-D0WD-V3（revision v3.1、40 MHz）、MACは `28:05:a5:c4:f7:44`、
フラッシュは4 MB（manufacturer 0x46 / device 0x4016、3.3 V）。
取得環境はWSL2 + usbipd-winのCP210x、esptool v5.4.0だった。
サーボ設定はID4–15の12個が対象で、前節の11個構成とは区別する。

### 読み出し中の欠落と、再起動後の照合失敗

921600 bpsでフラッシュを読むと `Corrupt data, expected 0x1000 bytes but received 0xfdd bytes`
となったため、460800 bpsを使用した。再起動させずに読み出した直後の照合ではdigestが一致した。

純正ファームを一度起動した後は、NVSの `0xB000` の1セクタが変わり、照合が失敗した。
当時の2回のdump比較では差分はそのセクタだけで、bootloader・パーティション表・appは一致した。
この結果から、取得後に再起動を避けて照合する運用にした。
比較ログと2つのdumpの対応付けは残っていないため、純正ファーム全般の挙動とは断定しない。

### 保存されたフラッシュの構成とファーム書き込み

`parse_partitions.py` で確認できるパーティション表は以下のとおり。

| label | type | subtype | offset | size |
| --- | --- | --- | --- | --- |
| nvs | data | nvs | 0x9000 | 0x5000 |
| otadata | data | ota | 0xE000 | 0x2000 |
| app0 | app | ota_0 | 0x10000 | 0x140000 |
| app1 | app | ota_1 | 0x150000 | 0x140000 |
| eeprom | data | 0x99 | 0x290000 | 0x1000 |
| spiffs | data | spiffs | 0x291000 | 0x16F000 |

旧資料ではArduino-ESP32 1.0.x世代の既定構成、起動スロットはapp0（otadata seq=1）、
app1は空、eepromは全て0xFFと記録されている。純正ファームのビルド情報は残っていない。

AstraArmControllerはLittleFSの `/config.txt` に設定を保存し、`LittleFS.begin(true)` は
マウント失敗時にフォーマットする。[設定保存の実装](../../upstream/AstraFirmwares/AstraArmController/src/config.cpp)を参照。
当時は純正ファームと配置が異なるため、`erase-flash` 後にパッチ適用済みファームを書き込む運用にした。
espressif32 7.1.3 / Arduino-ESP32 2.0.17、460800 bpsで書き込み、起動を確認したと記録されている。
関節動作の確認範囲は [モーターの検証記録](motor.md#astraarmcontrollerの初期化と閉ループ試験)を参照。

`esp32_backup/Aharobot_esp32_backup.bin` はこのバックアップとSHA256が一致する複製。
同日の `stock-waveshare-esp32-2026-09-27.bin` はSHA256が異なるため、同じdumpとして扱わない。
両者が同じ基板から取得されたと確認できる対応付けは残っていない。
保存ファイルと取得・復元手順は [ファームウェアツール](../../tools/firmware/README.md#バックアップ)を参照。
