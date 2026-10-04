# サーボツール

STS3215の確認・校正・手動操作用。機体固有のID・符号・可動域と通信の観測結果は
[モーターの測定・検証記録](../../docs/context/motor.md)に日付順でまとめる。

## 接続

pyserialとFeetech SDKを使用する。制御基板の透過モードとボーレートは
[ファームウェアツール](../firmware/README.md)を参照。
`--baud` を持つ直接操作ツールの既定値は115200 bpsなので、独自ブリッジでは921600を指定する。
`scan_motors.py` は位置引数で指定する。`nudge_check.py` はport / baudのCLI指定がなく、
`Bus` の既定値（`/dev/ttyUSB0` / 115200 bps）を使用するため、異なる接続ではコードの設定変更が必要。

```bash
# リポジトリ直下。ポートは使用環境に合わせる
python3 tools/servo/scan_motors.py /dev/ttyUSB0 921600
python3 tools/servo/motor_check.py --port /dev/ttyUSB0 --baud 921600
```

## ツール

| スクリプト | 用途 |
| --- | --- |
| `scan_motors.py` | ID・ボーレートのスキャン。port / baudは位置引数 |
| `motor_check.py` | 状態確認。`--move` は指定サーボを動かす |
| `nudge_check.py` | 微動による連動確認 |
| `teach_calibrate.py` | トルクOFFで手動測定し、`--verify` で対向符号を確認 |
| `rezero.py` | 中点校正。`--apply` 指定時だけEEPROMへ書き込む |
| `keyboard_teleop.py` | 関節単位の手動操作 |
| `servo_snapshot.py` | サーボEEPROMの読み取りと位置モニタ。純正ファームの透過通信用 |
| `init_arm.py` | AstraArmControllerの原点初期化。サーボEEPROM・ESP32 LittleFSへ書き込む |
| `check_zero.py` | PIDを設定せずトルクを入れる静置テスト。終了時にトルクOFF |
| `motion_test.py` | joint0 / joint1の往復試験。ソフトウェアリミット・誤差監視付き |

ID4–11は対向駆動のため、通常操作は関節単位で行う。
`nudge_check.py` はトルク・速度・移動量を制限した単体の連動確認用。
機体を変更したら符号と可動域を再測定する。
台座を固定し、機械端やエンコーダ原点をまたぐ位置指令を避ける。

## 校正・操作

```bash
python3 tools/servo/rezero.py --baud 921600
python3 tools/servo/teach_calibrate.py --baud 921600 --seconds 120 --verify --out calibration.json
python3 tools/servo/rezero.py --baud 921600 --center calibration.json
# 表示された計画を確認して書き込む場合は --apply を追加
python3 tools/servo/keyboard_teleop.py --baud 921600
```

原点付近で測定できない場合は、トルクOFFで中央寄りへ動かしてから中点を確認する。
teleopは `?` でヘルプ、spaceで停止、`0` でトルクOFF、`q` で終了。
可動域・トルク・目標先行量を制限し、位置の異常な跳びでは該当関節のトルクを切る。
測定データは [data/](data/) に保存されている。

`teach_calibrate.py` と `nudge_check.py` は、実行ディレクトリの `calibration.json` / `nudge_report.json`
へ保存する。保存先は `--out` で指定できる。`teach_calibrate.py` は同じ場所へ `*_tracks.json` も保存する。

## AstraArmController用の検証スクリプト

`init_arm.py` / `check_zero.py` / `motion_test.py` はAstraArmControllerファーム用で、
サーボバスを直接操作する透過ブリッジ用ツールとは接続条件が異なる。
現在は `~/aharobot/AhaRobot/upstream/astra_controller` から `ArmController` を読み込むため、
配置が異なる場合は各スクリプトの `REPO` を使用環境に合わせる。
ポートは位置引数で指定し、ボーレートは `ArmController` の設定を使用する。numpyとpyserialが必要。

```bash
# 純正ファームの透過通信で設定を保存（読み取りのみ）
python3 tools/servo/servo_snapshot.py --port /dev/ttyUSB0 --baud 115200 dump --out servo_eeprom.json
python3 tools/servo/servo_snapshot.py --port /dev/ttyUSB0 --baud 115200 watch 15

# AstraArmControllerファームで実行。確認入力後に書き込み・トルク投入を行う
python3 tools/servo/init_arm.py /dev/ttyUSB0
python3 tools/servo/check_zero.py /dev/ttyUSB0 10
python3 tools/servo/motion_test.py 0 10 8 2 /dev/ttyUSB0
```

`init_arm.py` はグリッパ全閉（開き幅0 mm）の姿勢で実行する。
以前の確認文は開き幅60 mmを指示していたため、その指示で初期化した右腕側は再実行が必要。
根拠は [初期化前のサーボ設定とグリッパの解釈](../../docs/context/motor.md#初期化前のサーボ設定とグリッパの解釈)を参照。
`check_zero.py` / `motion_test.py` のCSVは実行ディレクトリへ保存される。
保存済みの初期化前設定と実機試験の確認範囲は [検証記録](../../docs/context/motor.md#astraarmcontrollerの初期化と閉ループ試験)を参照。


### URDFゼロ姿勢・正方向の確認用スクリプト（2026-10-04）
[腕ごとの符号・可動域](../../docs/context/motor.md)を入れた `ArmController` を前提とする。
`<side>` は `left` / `right`。

```bash
# URDFを表示し、指定した関節を+方向に回した姿勢を見せる（要 libgl1 libglu1-mesa）
uv run --with yourdfpy --with "pyglet<2" tools/servo/view_urdf.py --arm l --joints 2,3,4,5,6

# PIDなしでトルクを入れ、手で+方向に押して値が増えるかを見る（joint0 / joint1）
svpy tools/servo/check_zero.py <port> 30 <side>

# wristを1関節ずつ+30°指令して戻し、グリッパを15 mm開いて戻す
svpy tools/servo/wrist_dir_test.py <port> <side> 30 15
```

`wrist_dir_test.py` はjoint0 / joint1にPIDを入れないため、腕は支えるか下ろした状態で実行する。
終了・Ctrl+C・例外のいずれでもトルクをOFFにする。