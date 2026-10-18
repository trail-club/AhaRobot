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

ツールは開発コンテナ内で実行する。macOSのDocker DesktopではUSBシリアルがコンテナから見えないため、
実機試験はLinuxホストで行い、Macは軌道の確認・テスト・記録の解析に使う。

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
| `generate_trajectory.py` | 実機に接続せず、台形速度の目標角度・速度をCSVとグラフに出力 |
| `record_joint_states.py` | ROSから回転関節の角度を読み取り、CSVに保存。受信途絶を検知 |
| `plot_joint_states.py` | 記録済みCSVから関節角度のグラフを作成 |
| `send_trajectory.py` | 1関節を台形速度で動かし、状態と追従誤差を監視・記録 |
| `trajectory_guard.py` | `send_trajectory.py` の監視処理（ROS非依存のモジュール） |
| `servo_common.py` | 入力検査・時刻変換・日本語グラフ設定の共通モジュール |

ID4–11は対向駆動のため、通常操作は関節単位で行う。
`nudge_check.py` はトルク・速度・移動量を制限した単体の連動確認用。
機体を変更したら符号と可動域を再測定する。
台座を固定し、機械端やエンコーダ原点をまたぐ位置指令を避ける。

## 関節の軌道試験

腕関節の速度・加速度制限の測定用。開発コンテナの `/app` から実行する。
数値は例で、実機で確認した値ではない。オプションの詳細は各スクリプトの `--help` を参照。

```bash
# 台形速度の目標を計算し、CSVとグラフを出力（実機接続なし）
python3 tools/servo/generate_trajectory.py --start-deg 0 --end-deg 20 \
  --max-velocity-deg-s 10 --acceleration-deg-s2 20 --output-dir tools/servo/output/ramp

# 関節角度の記録とグラフ化（動作指令なし）
python3 tools/servo/record_joint_states.py --joints joint_r2 joint_r3 \
  --duration 30 --output tools/servo/output/recording.csv
python3 tools/servo/plot_joint_states.py tools/servo/output/recording.csv \
  --output tools/servo/output/recording.png

# 1関節を動かし、状態と追従誤差を監視・記録（実機を動かす）
source /app/overlay_ws/install/setup.bash
python3 tools/servo/send_trajectory.py \
  --arm right --joint joint_r2 --hold-deg 0 0 0 0 0 --end-deg 20 \
  --max-velocity-deg-s 10 --acceleration-deg-s2 20 \
  --min-deg -90 -90 -90 -90 -90 --max-deg 90 90 90 90 90 --max-error-deg 2 \
  --topic /right/arm/joint_command --state-topic /joint_states \
  --output tools/servo/output/commands.csv
```

`record_joint_states.py` は既存のCSVを上書きせず、関節ごとに `--timeout` 秒（既定2秒）
新しい値が届かなければ終了コード2で終了する。

`send_trajectory.py` は片腕の関節2〜6のうち1関節を動かし、残りを `--hold-deg` の角度で保持する。
既定のtopicは模擬試験用のため、実機では指令先と状態のtopicを指定し、`arm_node` と同じPCで実行する。
指令topicに `ik_node`・`teleop_web_node` などの送信元があると開始しないため、これらは止めておく。
開始姿勢のずれ、状態の途絶・不正値、追従誤差の超過（遅れの補正なし）、送信遅延、受信先の切断、
Ctrl+Cで中断し、最新の実測位置を保持する指令を1回送るが、停止は保証しない。
終点で全関節の静止を確認すると終了コード0で終了する。異常は2、手動中断は130。
実機ではまだ動かしていないため、機体のそばで電源を切れる状態にし、小さい角度・遅い速度から試す。

`--output commands.csv` の場合、送った指令を `commands.csv`、実測値と追従誤差を
`commands.feedback.csv`、設定と終了理由を `commands.summary.json` へ保存する。
自動テストは `make test-servo` で実行する（`make test` にも含まれる）。
実機を動かす `motion_test.py` を読み込まないよう、テストはモジュール名を指定して実行する。

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
