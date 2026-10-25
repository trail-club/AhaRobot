# aha_perception

頭部RGB-Dカメラ（D435i）の点群生成、RViz表示（シミュレーション / 実機）、頭部の操作UI、
シミュレーションの左右の手首RGBカメラのbridge、SAM 3.1のセグメンテーション検証、
SAM 3.1による人の認識（[person_perception](#人の認識)）。
topic・frameは [ROSインターフェース](../../../docs/interfaces.md#頭部カメラ)を参照。
D435iの頭部ブラケットの設計・印刷データは [hardware/head_cam_mount_d435i](hardware/head_cam_mount_d435i/README.md)（GPL-3.0 + 非商用、installしない）。

- [Macカメラツール](../../../tools/perception/macos/pointcloud/README.md): RGB-DをROSへ配信し、RVizで点群を表示する。
- [SAM対話検証ツール](../../../tools/perception/macos/sam31/README.md): Macで撮影し、DGX Sparkで推論して結果を保存する。

`aha_perception.sam31` が単一フレームのモデル推論・マスク処理・深度集計を行い、
`aha_perception.sam31_worker` が対話検証用の常駐ワーカーを提供する。
SAM検証はROS topicを使わず、カメラ表示launchとは独立して動作する。

## launch

| launch | 内容 |
| --- | --- |
| `perception.launch.py` | `sim.launch.py use_perception:=true` から起動。`camera_bridge.launch.py`、`pointcloud.launch.py`、`sim_view.launch.py` |
| `camera_bridge.launch.py` | 頭部RGB-D・左右の手首RGBカメラを `config/sim_camera_bridge.yaml` でbridge。GUI・点群処理なしでタスク評価にも使用 |
| `pointcloud.launch.py` | depth_image_procで `/camera/depth/points` を生成。シミュレーションと実機で共通 |
| `sim_view.launch.py` | RViz（`rviz/sim_view.rviz`）、操作パネル、`head_look_at.py` |
| `camera_view.launch.py` | 実機カメラだけの表示。`pointcloud.launch.py` とRViz（`rviz/camera.rviz`） |
| `real_head_camera.launch.py` | 実機の頭部（と任意でカメラ）をシミュレーションと同じUIで動かす |
| `people.launch.py` | `person_perception.py`。頭部カメラの人をDGX SparkのSAM 3.1で認識する（[人の認識](#人の認識)） |

## ノード

| node | 内容 |
| --- | --- |
| `person_perception.py` | 頭部カメラの人の3D位置・追跡ID・覚えた名前を配信し、外見の特徴を答える（[人の認識](#人の認識)） |
| `head_look_at.py` | `/aha/perception/look_at` の点へ頭を向ける。可動範囲外は限界に丸めてwarningを出す。注視点を `/aha/perception/look_at/markers` に表示（丸めた場合は赤） |
| `sim_control_panel.py` | Qtの操作パネル。台車のジョイスティック（W/A/S/Dキー、停止ボタン）と頭のpan / tiltスライダー・プリセット。`arm_controls:=true` で左右のグリッパと昇降の欄を追加 |
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
`head_look_at.py` は最新のTFを待たずに使い、TFがまだない注視点はwarningを出して捨てる。
シミュレーションの再起動・リセットを検知するとTFバッファと注視点を消す。
操作パネルの頭のスライダーは、パネルから送った目標に `/joint_states` が2°以内に近づくまで（最長で移動時間の4倍 + 1 s）目標値を表示し、その後は `/joint_states` に追従する。

`arm_controls:=true`（[aha_manipulationの操作画面](../aha_manipulation/README.md#rviz操作画面)が指定）では右手・左手の欄を追加する。
グリッパは開く・閉じるボタンと指1本の開き（0–60 mm）のスライダーで、
`/right_gripper_controller/commands`（左手は `/left_gripper_controller/commands`）に
`[joint_r7r, joint_r7l]`（左手は `[joint_l7r, joint_l7l]`）= `[開き, -開き]` を送る。
指は物体で止まるため、グリッパのスライダーは送った値を表示し、`/joint_states` に追従しない。
昇降のスライダー（cm）はその手の昇降（`joint_r1` / `joint_l1`）だけの軌道を `/lift_controller/joint_trajectory` に送り、反対側の昇降は位置を保持する。
移動時間は距離 / `lift_speed`（既定0.1 m/s、最短0.5 s）、範囲は `lift_limits`（既定0–1.2 m）。
スライダーは離したとき・溝をクリックしたときに1回送り、頭と同じ方法で1 cm以内に近づいた後は `/joint_states` に追従する。
左右の昇降とMoveItの軌道実行は同じ `lift_controller` を使う。昇降を送ると実行中の軌道を置き換えるため、動作中の昇降は止まる。
パネルの片方の昇降を送ったとき、移動中だった反対側のスライダーは目標の表示をやめ、すぐに `/joint_states` に追従する。

## シミュレーション

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true
```

手首カメラは既存の `link_l6` / `link_r6` メッシュに含まれるカメラのレンズ位置に置く。
[Astra.STEP](../../../upstream/Astra_Hardwares/Astra/Astra.STEP) の `Camera` 部品の配置に基づき、
カメラ原点は手首座標で `(0.026, 0, 0.101)` m、pitchは−60°、レンズ前面はカメラの前方16 mm。
腕を前方に伸ばした姿勢ではグリッパの上側から前方・下向き30°を見て、
水平視野角90°、clip範囲0.02–3 mのRGBを配信する。
取付位置と光学特性は実機で校正していない。既存メッシュのカメラを使い、別のvisual・質量・collisionは追加しない。
手首カメラの画像はRVizのImage表示で `/camera/left_wrist/image_raw` または
`/camera/right_wrist/image_raw` を選択して確認できる。
`cameras:=false` はカメラセンサを除くため、画像を使う場合は付けない。

GUIなしで画像だけ配信する場合は以下を使う。Ogre2/EGLによる描画が必要で、GPUを推奨する。
ソフトウェア描画ではシミュレーションが遅くなる場合がある。

```bash
ros2 launch aha_bringup sim.launch.py headless:=true use_perception:=true rviz:=false
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

`camera:=true` では、D435iの映像を [stream_realsense.py](../../../tools/perception/macos/pointcloud/README.md#rosbridgeへの直接送信)
で既定の `ws://127.0.0.1:9090` へ送ると、RVizに点群が表示される。

## 人の認識

`person_perception.py` は頭部カメラの色画像をDGX SparkのSAM 3.1へ送り、Depthから求めた人の3D位置・追跡ID・覚えた名前を配信する。
覚えた人との照合と外見の特徴は、[ローカルLLM](../../../tools/local-llm/README.md) のQwen3.8が人の切り出し画像を見て答える。
topic・serviceは [ROSインターフェース](../../../docs/interfaces.md#人の認識)、追跡・名前付けの設定は `config/people.yaml` を参照。

```bash
ros2 launch aha_perception people.launch.py ssh_user:=<Unixユーザー名>
```

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `target_frame` | `map` | 配信する座標系。TFのない場合（Macのカメラだけ）は空にしてカメラの座標系にする |
| `ssh_host` / `ssh_port` / `ssh_user` | `10.99.0.1` / `2222` / 空 | DGXの接続先。SSH configの別名を使う場合は `ssh_port:=0` |
| `identity_file` | 空 | SSHの秘密鍵。空なら既定の鍵かssh-agent |
| `use_sim_time` | `false` | シミュレーションでは `true` |

DGXへのSSH接続の準備は [SAM対話検証ツール](../../../tools/perception/macos/sam31/README.md#準備)と同じ。
起動時にDGXでイメージを作ってコンテナを起動し（初回は数分）、Ctrl-Cで削除する。
Mac用の知覚コンテナ（[Macカメラツール](../../../tools/perception/macos/pointcloud/README.md)）はMacのssh-agentを使うため、Mac側で `ssh-add -l` に鍵が出ることを確認する。
シミュレーションのタスクからは [aha_tasks](../aha_tasks/README.md) の `people:=true` で起動する（人のモデルは置いていない）。
