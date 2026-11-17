# aha_base

実機の台車（HB-139 ×2、Makerbase ODrive v3.6-56V）をUSBで動かすパッケージ。
`base_node.py` はシミュレーションの `diff_drive_controller` と同じtopicを出すため、
`teleop_base.sh` やNav2の設定をそのまま実機に使える。
ODriveの校正・ゲイン調整の手順は [ODriveツール](../../../tools/odrive/README.md)、
測定記録は [台車駆動の測定・検証記録](../../../docs/context/odrive.md)を参照。

| ファイル | 内容 |
| --- | --- |
| `aha_base/diff_drive.py` | 差動二輪の運動学・オドメトリ・目標位置の積算（ROS非依存） |
| `aha_base/odrive_usb.py` | `odrive` パッケージによるODriveの接続・閉ループ開始・停止（ROS非依存） |
| `scripts/base_node.py` | 台車node |
| `config/base.yaml` | パラメータ |

## 起動

ODriveをUSBで接続し、開発コンテナ内で起動する。キー操作は別の対話シェルで `teleop_base.sh` を使う。

```bash
ros2 launch aha_bringup base.launch.py
ros2 run aha_bringup teleop_base.sh  # 別シェル
```

`params_file:=<path>` で別のパラメータファイルを使う。シミュレーションと同じ名前を使うため、
同じ `ROS_DOMAIN_ID` で `sim.launch.py` と同時に起動しない。

`odrive==0.5.4` を依存なしで入れる（[Dockerfile](../../../docker/Dockerfile)）。
イメージを作り直す前のコンテナでは `pip3 install --user --break-system-packages --no-deps odrive==0.5.4` を実行する。
依存を入れるとシステムのnumpyが置き換わる。ODriveを開けるのは1プロセスだけなので、
`odrivetool` や `tools/odrive/` のスクリプトとは同時に使わない。

## ノードのインターフェース

| 名前 | 型 | 内容 |
| --- | --- | --- |
| `/diff_drive_controller/cmd_vel` | `geometry_msgs/msg/TwistStamped` | 購読（`stamped:=false` で `Twist`）。`linear.x` と `angular.z` だけ使う |
| `/diff_drive_controller/odom` | `nav_msgs/msg/Odometry` | `rate_hz` で配信。姿勢は車輪の位置、速度は `vel_estimate` から求める。共分散は0 |
| `/tf` | `odom` → `base_footprint` | `publish_tf:=false` で止める |

- 起動時にODriveを探し、`connect_timeout` 秒で見つからなければ終了コード1で終了する。
- エラーを消した後、両軸がIDLEで、モーターが校正済み・エンコーダが準備済みであることを確かめる。満たさなければ終了コード1。校正は行わない。
- 制御はupstreamの `astra_controller` の `base_controller.py`（CAN）と同じ方式。位置制御＋台形軌道で、
  周期ごとに目標位置を速度 × 経過時間だけ進め、現在位置から `max_lead_turns` 以内に収める。
  目標位置の初期値は現在位置で、閉ループに入るときに動かない。
- ゲイン・Kt・電流制限はODriveに保存された値を使う。`pos_gain` / `vel_gain` / `vel_integrator_gain` / `current_lim` を0以上にした場合だけRAMで上書きする。
- 台形軌道の速度上限 `traj_vel_limit` はODriveの `controller.config.vel_limit` を超えないよう切り詰める。
  左右の車輪速度は、どちらも `traj_vel_limit` を超えないよう同じ比率で縮める（曲率を保つ）。
- `cmd_timeout` 秒より古い指令は0として扱う。
- ODriveのwatchdog（`watchdog_timeout`、0で無効）を使い、nodeが止まると軸がIDLEになる。
  このときODriveには `WATCHDOG_TIMER_EXPIRED` のエラーが残る。
- どちらかの軸が閉ループから外れるか、USB通信に失敗したら、両軸をIDLEにして終了コード1で終了する。
- SIGINT / SIGTERMで両軸をIDLEにし、RAMで変えた値を元に戻して終了する。設定は保存しない。

## パラメータ

既定値と意味は `config/base.yaml` と `scripts/base_node.py` の `PARAMS` を参照。

| 名前 | 既定 | 内容 |
| --- | --- | --- |
| `serial_number` | `""` | 16進（`0x` は省略可）。空なら最初に見つかったODrive |
| `left_axis` / `right_axis` | 0 / 1 | 左右の車輪の軸 |
| `left_sign` / `right_sign` | −1 / 1 | 前進でのモーターの回転方向 |
| `wheel_diameter` / `wheel_separation` | 0.139 / 0.44 m | upstreamの `WHEEL_D` / `WHEEL_BASE` |
| `traj_vel_limit` | 2.0 turn/s | 車輪の最大速度（0.87 m/s） |
| `traj_accel_limit` / `traj_decel_limit` | 1.0 turn/s² | upstreamと同じ |

URDF（`aha_description`）の車輪半径0.042 m・車輪間隔0.40 mは、ここの値と一致していない。

## upstreamのCAN版との違い

`ros2 run astra_controller base_node` はCAN（`odrive_can`）で同じODriveを動かすが、
topic（`cmd_vel` / `odom`、`Twist`）とframe（`base_link`）がシミュレーションと異なる。
upstream版は起動時に `pos_gain` 20・`vel_gain` 0.1を送り、台形軌道の速度上限は20 turn/sで、
`cmd_vel` を受けるたびに受信間隔によらず目標位置を0.1秒分進める。watchdogは使わない。

## テスト

`test/fake_odrive/` の偽の `odrive` パッケージ（2軸のODriveを模擬する）に対し、運動学・オドメトリ、
閉ループ開始・停止とRAM値の復元、nodeのtopic・速度上限・指令の途絶・異常時の終了を確認する。実機は不要。

```bash
cd /app/overlay_ws/src/aha_base
source /opt/ros/jazzy/setup.bash
python3 -m pytest test -v
```

nodeは `ROS_DOMAIN_ID=97`（`AHA_TEST_DOMAIN_ID` で変更可）で起動する。
`colcon test --packages-select aha_base` でも同じテストを実行する。
