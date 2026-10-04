# aha_perception

頭部RGB-Dカメラ（D435i）の点群生成、RViz表示（シミュレーション / 実機）、頭部の操作UI。
topic・frameは [ROSインターフェース](../../../docs/interfaces.md#頭部カメラ)、
macOSのカメラ経路は [カメラツール](../../../tools/perception/macos/README.md)を参照。
D435iの頭部ブラケットの設計・印刷データは [hardware/head_cam_mount_d435i](hardware/head_cam_mount_d435i/README.md)（GPL-3.0 + 非商用、installしない）。

## launch

| launch | 内容 |
| --- | --- |
| `perception.launch.py` | `sim.launch.py use_perception:=true` から起動。Gazeboカメラのbridge（`config/sim_camera_bridge.yaml`）、`pointcloud.launch.py`、`sim_view.launch.py` |
| `pointcloud.launch.py` | depth_image_procで `/camera/depth/points` を生成。シミュレーションと実機で共通 |
| `sim_view.launch.py` | RViz（`rviz/sim_view.rviz`）、操作パネル、`head_look_at.py` |
| `camera_view.launch.py` | 実機カメラだけの表示。`pointcloud.launch.py` とRViz（`rviz/camera.rviz`） |
| `real_head_camera.launch.py` | 実機の頭部（と任意でカメラ）をシミュレーションと同じUIで動かす |

## ノード

| node | 内容 |
| --- | --- |
| `head_look_at.py` | `/aha/perception/look_at` の点へ頭を向ける。可動範囲外は限界に丸めてwarningを出す。注視点を `/aha/perception/look_at/markers` に表示（丸めた場合は赤） |
| `sim_control_panel.py` | Qtの操作パネル。台車のジョイスティック（W/A/S/Dキー、停止ボタン）と頭のpan / tiltスライダー・プリセット |
| `teleop_head.py` | 端末から頭を動かす。矢印キーで5°ずつ、spaceで正面、qで終了 |

頭の可動範囲・移動速度・プリセットは `config/head.yaml` で設定し、3つのノードが
`aha_perception.head_config` 経由で読む。符号は +pan = ロボットの右、+tilt = 下。
可動範囲は設定値で、実機の頭の可動域は測定していない。

操作パネルは `/diff_drive_controller/cmd_vel` に10 Hzで送信し、離すと0を1回送る。
`header.stamp` は0で送り、diff_drive_controllerが受信時の自分の時刻（sim時刻）を入れる。
送信が止まると `cmd_vel_timeout`（0.5 s）で台車が止まる。
最大速度は0.4 m/s / 1.5 rad/s（パラメータ `max_linear` / `max_angular`）で、
URDFの車輪関節の上限（10 rad/s × 半径0.042 m ≈ 0.42 m/s）より低くしている。
`head_look_at.py` と `sim_control_panel.py` はwall clockで動く。
`head_look_at.py` はシミュレーションの再起動・リセットを検知するとTFバッファと注視点を消す。

## シミュレーション

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true
```

`perception.launch.py` の引数はコマンドラインで渡す。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `rviz` | `true` | `false` でbridgeと点群だけ起動 |
| `fixed_frame` | `odom` | RVizのfixed frame |
| `control_panel` | `true` | 操作パネルの起動 |
| `use_sim_time` | `true` | bridge・点群node・RVizでシミュレーション時刻を使う |

Gazeboを止めずに表示だけ再起動する場合は `rviz:=false` で起動し、別ターミナルで以下を実行する。

```bash
ros2 launch aha_perception sim_view.launch.py
```

RVizのPublish Pointツール（`/clicked_point`）は `/aha/perception/look_at` へremapしてあり、
クリックした点に頭が向く。他のノードからは同じtopicへ送る。

```bash
ros2 topic pub --once /aha/perception/look_at geometry_msgs/msg/PointStamped \
  "{header: {frame_id: base_link}, point: {x: 1.0, y: 0.3, z: 0.5}}"
ros2 run aha_perception teleop_head.py  # キーボード操作（対話シェルで）
```

地図と重ねる場合は `use_nav:=true fixed_frame:=map` を付ける。
RVizの設定には `/map` / `/scan` の表示とslam_toolboxパネルが含まれ、これらのtopicがあれば表示する。

## 実機の頭部とカメラ

実機の頭部（STS3215 × 2、pan ID12 / tilt ID13）を [aha_servo](../aha_servo/README.md) で動かし、
シミュレーションと同じ操作パネル・Publish Point・`teleop_head.py` を使う。

### 接続

- 頭部のサーボをWaveshare Servo Driver with ESP32につなぎ、12 Vを入れる。
- 基板を透過通信（`Start Serial Forwarding`）にする。電源を切るたびに再設定する。
  [ファームウェアツール](../../../tools/firmware/README.md)を参照。
- USB接続で `/dev/ttyUSB0`、115200 bps。透過ブリッジ（`tools/firmware/right_arm/bridge`）では `baud:=921600`。
- デバイスは `root:dialout` の660で、挿し直し・再起動のたびに作り直される。
  コンテナのユーザーが開けない場合は、ホストで `sudo chmod 666 /dev/ttyUSB0` を実行する。

初回やサーボを付け直したときは [キャリブレーション](../aha_servo/README.md#キャリブレーション)を行う。

### 起動

同じコンテナでシミュレーションが動いている場合は、別の `ROS_DOMAIN_ID` で起動する。

```bash
ROS_DOMAIN_ID=7 ros2 launch aha_perception real_head_camera.launch.py
ROS_DOMAIN_ID=7 ros2 run aha_perception teleop_head.py  # 別シェルからキーボード操作
```

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `config` | `aha_servo/config/head.yaml` | サーボバスの設定 |
| `port` / `baud` | 空（configの値） | configの上書き |
| `rviz` | `true` | `sim_view.launch.py` をfixed frame `base_link` で起動。台車の操作は無効 |
| `camera` | `false` | rosbridge（127.0.0.1:9090）と `pointcloud.launch.py` を起動 |

TFはrobot_state_publisher（`sim:=false`）とjoint_state_publisherが出す。
`/joint_states` の頭は `/head_controller/joint_states`、その他の関節は0。
Ctrl-Cでトルクを切る。サーボが見つからない場合、ブリッジはエラーで終了する。

`camera:=true` では、D435iの映像を [stream_realsense.py](../../../tools/perception/macos/README.md#rosbridgeへの直接送信)
で既定の `ws://127.0.0.1:9090` へ送ると、RVizに点群が表示される。
