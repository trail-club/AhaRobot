# サーボツール

STS3215の確認・校正・手動操作用。機体固有のID・符号・可動域と通信の観測結果は
[モーターの測定・検証記録](../../docs/context/motor.md)に日付順でまとめる。

## 接続

pyserialとFeetech SDKを使用する。制御基板の透過モードとボーレートは
[ファームウェアツール](../firmware/README.md)を参照。
各スクリプトの既定値は115200 bpsなので、独自ブリッジでは明示指定する。

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

ID4–11は対向駆動のため単体で動かさない。機体を変更したら符号と可動域を再測定する。
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
