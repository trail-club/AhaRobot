# aha_servo

Feetech STSサーボ（STS3215）のバスをROSから動かすパッケージ。
`servo_trajectory_bridge.py` はros2_controlのJointTrajectoryControllerと同じtopic / actionを出すため、
シミュレーション用の操作パネル・`head_look_at.py`・`teleop_head.py` で実機のサーボを動かせる。
実機の頭部を動かすlaunchと接続は [aha_perception](../aha_perception/README.md#実機の頭部とカメラ)、
校正値の測定記録は [モーターの検証記録](../../../docs/context/motor.md)を参照。

| ファイル | 内容 |
| --- | --- |
| `aha_servo/sts.py` | STSプロトコルと標準ライブラリだけのシリアル通信（ROS非依存） |
| `aha_servo/joint_map.py` | 関節とサーボの対応（id、zero、sign、可動域）と設定yamlの読み書き（ROS非依存） |
| `scripts/servo_trajectory_bridge.py` | 1バスN関節のブリッジnode |
| `scripts/servo_calibrate.py` | zero / signの対話式キャリブレーション |
| `config/head.yaml` | 頭部（pan ID12 / tilt ID13、`/dev/ttyUSB0`、115200 bps） |

## ノードのインターフェース

設定yamlの `controller_name` ごとに以下を出す。

| 名前 | 型 | 内容 |
| --- | --- | --- |
| `/<controller>/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | 購読。最後の点だけ使い、空なら現在位置（その場で読んだ値）で停止 |
| `/<controller>/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | 点を順に実行。cancel・期限切れの中断で現在位置（その場で読んだ値）を保持 |
| `/<controller>/joint_states` | `sensor_msgs/msg/JointState` | この関節群だけ。`joint_state_publisher` の `source_list` で `/joint_states` に合流させる |

- 起動時に全IDへpingし、応答がなければ終了コード1で終了する。目標を現在位置にしてからトルクを入れる。
- 指令はyamlの `min` / `max` で丸めてwarningを出す。速度は `|変位| / time_from_start`、上限は `max_speed`。
- `header.stamp` は無視し、受信時に開始する。wall clockで動く。
- actionは目標の全関節が `state_timeout`（既定0.3 s）以内に読めているときだけ成功にする。読めない関節があれば待ち、期限（最後の点から `goal_timeout`、goalに `goal_time_tolerance` があればその値）で中断して位置を保持する。
- `state_timeout` 以内に読めていない関節には指令を送らない（warning）。古い位置から速度を計算したり、古い位置へ戻したりしないため。停止（空の軌道・cancel・中断）でその場で読めない関節は直前の目標のまま動く。
- 通信エラーは間引いてログに出して続行し、2秒続くとエラーを出す。`joint_states` にはその周期に読めた関節だけを入れる。
- SIGINT / SIGTERMでトルクを切り、ポートを閉じる。

## 設定yaml

ROSのパラメータファイル。形式は `aha_servo/joint_map.py` のdocstringを参照。
関節角は `rad = sign * (steps - zero) / steps_per_rev * 2π`。
読み込み時に範囲を検査し、範囲外ならエラーで止まる（`id` 0..253、`zero` 0..`steps_per_rev`-1、`torque_limit` 0..1000、`acc` 0..254、`baud` は `aha_servo/sts.py` の `BAUDS`、レート・速度・timeout・toleranceは正）。

## キャリブレーション

`servo_calibrate.py` はROS非依存（pyyamlのみ）で、ホストのリポジトリ直下からも実行できる。
コンテナでシリアルデバイスを開けない場合はホストで実行する。

```bash
uv run --no-project --with pyyaml python overlay_ws/src/aha_servo/scripts/servo_calibrate.py
ros2 run aha_servo servo_calibrate.py  # コンテナ内
```

トルクを切った後、表示に従って手で動かしてEnterを押す。頭部では正面・水平 → 右を向く → 下を向く。
zeroとsignを求め、最後の確認で `y` を入力するとyamlへ書き込む（空Enterは聞き直す）。
`n` では書き込むキーと値を表示する。同じポートを使う `servo_trajectory_bridge.py` は止めておく。

- `--config` を省略すると、ament indexの `share/aha_servo/config/head.yaml`、なければスクリプト隣のソースツリーの `config/head.yaml` を使う。
- `--port` / `--baud` でyamlの値を上書きする。

## 関節群の追加

1. `config/<group>.yaml` を作る（`controller_name`、`port`、`baud`、`joints` と関節ごとの `id` / `min` / `max` / `positive`）。1 node = 1シリアルバス。
2. `servo_calibrate.py --config <path>` でzero / signを書き込む。
3. launchに `servo_trajectory_bridge.py` をyamlとnode名を変えて追加し、`joint_state_publisher` の `source_list` に `/<controller>/joint_states` を加える（例: `aha_perception/launch/real_head_camera.launch.py`）。

1関節を複数サーボで動かす対向駆動には対応していない。

## テスト

pty上で2サーボを模擬する偽のSTSバスに対し、プロトコル（送信エコーを含む）・キャリブレーション・nodeのtopic / action /
丸め / cancel / 割り込み / 通信断 / 終了時のトルクOFFを確認する。実機は不要。

```bash
cd /app/overlay_ws/src/aha_servo
source /opt/ros/jazzy/setup.bash
python3 -m pytest test -v
```

nodeは `ROS_DOMAIN_ID=97`（`AHA_TEST_DOMAIN_ID` で変更可）で起動する。
`colcon test --packages-select aha_servo` でも同じテストを実行する。
