# aha_perception

頭部RGB-Dカメラ（D435i）の点群生成、RViz表示（シミュレーション / 実機）、頭部の操作UI、平面検出と置き場所の候補、
シミュレーションの左右の手首RGBカメラのbridgeと物体検出（真の姿勢から）、SAM 3.1のセグメンテーション検証。
topic・frame・serviceは [ROSインターフェース](../../../docs/interfaces.md#頭部カメラ)を参照。
D435iの頭部ブラケットの設計・印刷データは [hardware/head_cam_mount_d435i](hardware/head_cam_mount_d435i/README.md)（GPL-3.0 + 非商用、installしない）。

- [Macカメラツール](../../../tools/perception/macos/pointcloud/README.md): RGB-DをROSへ配信し、RVizで点群を表示する。
- [SAM対話検証ツール](../../../tools/perception/macos/sam31/README.md): Macで撮影し、DGX Sparkで推論して結果を保存する。

`aha_perception.sam31` が単一フレームのモデル推論・マスク処理・深度集計を行い、
`aha_perception.sam31_worker` が対話検証用の常駐ワーカーを提供する。
SAM検証はROS topicを使わず、カメラ表示launchとは独立して動作する。

## launch

| launch | 内容 |
| --- | --- |
| `perception.launch.py` | `sim.launch.py use_perception:=true` から起動。`camera_bridge.launch.py`、`depth_noise` 指定時は `sim_depth_noise.py`、`pointcloud.launch.py`、`sim_view.launch.py`、`planes:=true` で `plane_detection.launch.py` |
| `camera_bridge.launch.py` | 頭部RGB-D・左右の手首RGBカメラを `config/sim_camera_bridge.yaml` でbridge。GUI・点群処理なしでタスク評価にも使用。`depth_topic` で頭部Depthの配信先を変える |
| `pointcloud.launch.py` | depth_image_procで `/camera/depth/points` を生成。シミュレーションと実機で共通 |
| `sim_view.launch.py` | RViz（`rviz/sim_view.rviz`）、操作パネル、`head_look_at.py` |
| `camera_view.launch.py` | 実機カメラだけの表示。`pointcloud.launch.py` とRViz（`rviz/camera.rviz`）。`camera_source:=realsense` で `realsense.launch.py` も起動 |
| `realsense.launch.py` | RealSense ROSドライバでD435iを配信する（Linux、[RealSenseドライバ](#realsenseドライバ)） |
| `real_head_camera.launch.py` | 実機の頭部（と任意でカメラ・平面検出）をシミュレーションと同じUIで動かす |
| `plane_detection.launch.py` | `plane_detection_node.py`。シミュレーションと実機で共通 |

## ノード

| node | 内容 |
| --- | --- |
| `head_look_at.py` | `/aha/perception/look_at` の点へ頭を向ける。可動範囲外は限界に丸めてwarningを出す。注視点を `/aha/perception/look_at/markers` に表示（丸めた場合は赤） |
| `sim_control_panel.py` | Qtの操作パネル。台車のジョイスティック（W/A/S/Dキー、停止ボタン）と頭のpan / tiltスライダー・プリセット。`arm_controls:=true` で左右のグリッパと昇降の欄を追加 |
| `teleop_head.py` | 端末から頭を動かす。矢印キーで5°ずつ、spaceで正面、qで終了 |
| `manual_head.py` | サーボなしで、指令した角度を `/head_controller/joint_states` に出す（`head:=manual`） |
| `imu_head.py` | D435iのIMUで求めたtiltを頭の状態にする（`head:=imu`）か、サーボのtiltと比べる（`imu_check:=true`）（[IMUによる頭部のtilt](#imuによる頭部のtilt)） |
| `plane_detection_node.py` | Depthから床・机・棚・壁の平面を検出して面をIDで追跡し、置き場所の候補をserviceで返す（[平面検出のノード](#平面検出のノード)） |
| `sim_depth_noise.py` | シミュレーションのDepthに実機D435i相当のノイズを加える（[Depthノイズ](#depthノイズ)） |
| `sim_object_detector.py` | シミュレーション専用。Gazeboの真の姿勢から頭部カメラに見える物体の検出を出す（[シミュレーションの物体検出](#シミュレーションの物体検出)） |

頭の可動範囲・移動速度・プリセットは `config/head.yaml` で設定する（`aha_perception.head_config`）。
符号は +pan = ロボットの右、+tilt = 下。可動範囲は設定値で、実機の頭の可動域は測定していない。
操作パネルは `/diff_drive_controller/cmd_vel` に10 Hzで送り、離すと0を1回送る。送信が止まると `cmd_vel_timeout`（0.5 s）で台車が止まる。
最大速度は0.4 m/s / 1.5 rad/s（パラメータ `max_linear` / `max_angular`、車輪の上限は約0.42 m/s）。
`head_look_at.py`・`sim_control_panel.py`・`plane_detection_node.py`・`sim_depth_noise.py`・`sim_object_detector.py` はwall clockで動く。
`head_look_at.py`・`plane_detection_node.py` はシミュレーションの再起動・リセットを検知するとTFバッファを消す。

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

GUIなしで画像だけ配信する場合は以下を使う。Ogre2/EGLの描画が必要で、ソフトウェア描画では遅くなる場合がある（GPUを推奨）。

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
| `planes` | `false` | 平面検出（`plane_detection.launch.py`）の起動。`max_rate:=` も渡せる |
| `furniture_config` | 空 | 平面検出に渡す[家具の設定](#家具の設定)のYAML |
| `depth_noise` | `none` | Depthに加えるノイズ（`none` / `d435i_filtered` / `d435i_raw`、[Depthノイズ](#depthノイズ)） |
| `objects` / `objects_config` | `false` / 空 | [シミュレーションの物体検出](#シミュレーションの物体検出)の起動と物体リスト（空はパッケージの `config/objects/robocup_objects.yaml`） |

Gazeboを止めずに表示だけ再起動する場合は `rviz:=false` で起動し、別ターミナルで `ros2 launch aha_perception sim_view.launch.py` を実行する。
RVizのPublish Pointツール（`/clicked_point`）は `/aha/perception/look_at` へremapしてあり、
クリックした点に頭が向く。他のノードからは同じtopicへ送る。

```bash
ros2 topic pub --once /aha/perception/look_at geometry_msgs/msg/PointStamped \
  "{header: {frame_id: base_link}, point: {x: 1.0, y: 0.3, z: 0.5}}"
ros2 run aha_perception teleop_head.py  # キーボード操作（対話シェルで）
```

地図と重ねる場合は `use_nav:=true fixed_frame:=map` を付ける。
RVizの設定には `/map` / `/scan` の表示とslam_toolboxパネルが含まれ、これらのtopicがあれば表示する。

### Depthノイズ

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true depth_noise:=d435i_filtered
```

`sim_depth_noise.py` がbridgeのDepthにノイズを加え、点群・平面検出・RVizはノイズを加えたDepthを使う（topicは [ROSインターフェース](../../../docs/interfaces.md#頭部カメラ)）。
プリセットは実機D435iの静止フレームに合わせた値（[較正](../../../tools/perception/plane_bench/README.md#ノイズの較正)）。
Gazebo側はDepthにノイズを設定していない（`aha_description/urdf/sensors.xacro` のノイズはRGBの `camera_rgb_noise_stddev`）。
ノードのパラメータは `preset`（launchの `depth_noise`）・`seed`（既定0）・入出力topic。

### シミュレーションの物体検出

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true objects:=true
```

ROSの物体検出器がないため、`sim_object_detector.py` がGazeboのモデルの真の姿勢から `/aha/perception/objects`
（`aha_msgs/DetectedObjectArray`、[ROSインターフェース](../../../docs/interfaces.md#物体検出シミュレーションのみ)）を出す。
対象は名前が物体リスト（[config/objects/robocup_objects.yaml](config/objects/robocup_objects.yaml)、`aha_perception.object_list`）の
パターン（既定はラベルの空白を `_` にした名前と `<それ>_*`、例：`tuna_can_2` は `tuna can`）に合うモデルだけ。
物体の中心か上面の中心が画像に入り、そこのDepthがその点より手前（中心は物体の最大寸法の半分 + `depth_tolerance`）でなければ見えているとする。
姿勢はロボットのモデル（原点が `base_footprint`）からの相対で、Depthの時刻のTFで `output_frame` に直す（カメラの検出器と同じくodomのずれを含む）。
`class_name` はラベル、`track_id` はモデルごとの番号、`size` はリストの大きさ、`confidence` は1。
GZ_PARTITIONはシミュレーションと同じにする（launchから起動すれば同じ）。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `objects_config` | 空 | 物体リスト（空はパッケージのもの） |
| `output_frame` / `base_frame` / `robot_model` | `odom` / `base_footprint` / `aha_robot` | 出力のframe（TFがなければ `base_frame`）、ロボットのGazeboのモデル名 |
| `rate` | `2.0` | 出力の上限頻度（Hz、wall clock、最新のDepthで判定） |
| `depth_tolerance` / `image_margin` | `0.03` / `4.0` | 見えているとするDepthの余裕（m）、画像の縁の除外（px） |
| `pose_noise` | `0.01` | 位置に加える正規分布のσ（m、軸・検出ごと）。共分散にも入れる |
| `miss_rate` / `mislabel_rate` / `seed` | `0.0` / `0.0` / `0` | 見落とす割合、別のラベルにする割合。`seed` とモデル名ごとに決め、`seed` を変えるまで同じ物体が見落とされる・間違われる |

パラメータは実行中に `ros2 param set` で変えられる。物体の向き・遮蔽の割合・検出器の誤差特性は再現しない。

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
ROS_DOMAIN_ID=7 ros2 launch aha_perception real_head_camera.launch.py camera:=true camera_source:=realsense planes:=true
ROS_DOMAIN_ID=7 ros2 launch aha_perception real_head_camera.launch.py head:=imu camera:=true camera_source:=realsense planes:=true
```

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `config` | `aha_servo/config/head.yaml` | サーボバスの設定 |
| `port` / `baud` | 空（configの値） | configの上書き |
| `head` | `servo` | `servo`：サーボを動かす。`manual`：サーボなしで `manual_head.py` が指令した角度を頭の状態にする。`imu`：サーボなしで `imu_head.py` がIMUのtiltと指令したpanを頭の状態にする（`camera_source:=realsense` が必要） |
| `imu_check` | `false` | `head:=servo` で、サーボとIMUのtiltが3°より違うとwarningを出す（`camera_source:=realsense` が必要） |
| `rviz` | `true` | `sim_view.launch.py` をfixed frame `base_link` で起動。台車の操作は無効 |
| `camera` | `false` | カメラの配信元と `pointcloud.launch.py` を起動 |
| `camera_source` | `rosbridge` | `rosbridge`：rosbridge（127.0.0.1:9090）を起動し、`stream_realsense.py` から受ける。`realsense`：`realsense.launch.py` を起動 |
| `color_profile` / `depth_profile` | `640,480,15` | `camera_source:=realsense` の解像度とfps（[RealSenseドライバ](#realsenseドライバ)） |
| `planes` | `false` | 平面検出（`plane_detection.launch.py`）を起動。`camera:=true` と併用する。`max_rate:=` も渡せる |

TFはrobot_state_publisher（`sim:=false`）とjoint_state_publisherが出し、`/joint_states` の頭以外の関節は0。
Ctrl-Cでトルクを切る。サーボが見つからない場合、ブリッジはエラーで終了する。
`camera_source:=rosbridge` では [stream_realsense.py](../../../tools/perception/macos/pointcloud/README.md#rosbridgeへの直接送信) で映像を `ws://127.0.0.1:9090` へ送る。
Linuxでは `camera_source:=realsense` で直接配信できる（rosbridgeはDepthのJSON変換で遅れる。[測定](../../../docs/context/camera.md)）。

### RealSenseドライバ

`realsense.launch.py` は `realsense2_camera_node`（node `/camera`）を起動し、[頭部カメラ](../../../docs/interfaces.md#頭部カメラ)のtopicで配信する。
Depthはカラーへ位置合わせし、その前に `stream_realsense.py` と同じ後処理（disparity変換 → spatial → temporal → depthへ戻す、SDKの既定値）をかける。
カメラのTFはURDFが出す（`publish_tf` は無効）。IMUはaccelをgyroの時刻へ線形補間して出す。
`real_head_camera.launch.py` は `head:=imu` か `imu_check:=true` のときだけIMUを有効にする。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `color_profile` / `depth_profile` | `640,480,15` | カラー（位置合わせしたDepthも同じ大きさ）/ 位置合わせ前のDepthの幅,高さ,fps |
| `filters` | `true` | Depthの後処理 |
| `imu` | `true` | accel・gyro |
| `serial_no` | 空 | 使う機体のシリアル番号（空は最初の機体） |

USB 2.0ではカラー848×480がなく、Depth 848×480は10 / 6 fpsだけ（既定はUSB 2.0で選べる組み合わせ）。USB 3では両方を848×480（15 / 30 fps）にできる。
コンテナのユーザーがIMU（HIDのiioデバイス）を開くには、ホストにlibrealsenseのudevルールを入れる（NixOSでは `services.udev.packages = [ pkgs.librealsense ];`、ルールなしでの経緯は [記録](../../../docs/context/camera.md)）。

### IMUによる頭部のtilt

`imu_head.py` は `/camera/imu` のaccelとgyroを相補フィルタで重力の向きにし、カメラのtilt + `tilt_offset` を `joint_head_tilt` とする。
台車が水平であることを仮定する。rollは頭の関節で表せないため、|roll| が `roll_tolerance_deg` を超えるとwarningを出す。
`mode:=state`（`head:=imu`）はtiltとpan（指令した軌道）を30 Hzで出す。
`mode:=check`（`imu_check:=true`）は頭とIMUが静止しているときにサーボのtiltと比べる。
計算は [imu_tilt.py](aha_perception/imu_tilt.py) と [imu_head.py](scripts/imu_head.py) を参照。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `mode` | `state` | `state` / `check` |
| `imu_topic` | `/camera/imu` | IMU（accel・gyro） |
| `tilt_offset` | `0.0` | tiltに加える値（rad）。取付の誤差の補正 |
| `time_constant` | `0.5` | accelによる補正の時定数（s） |
| `gyro_threshold` / `accel_tolerance` | `0.05` / `0.8` | 静止の判定（rad/s）/ accelで補正するaccelのノルムと重力の差の上限（m/s²） |
| `roll_tolerance_deg` / `check_tolerance_deg` | `3.0` / `3.0` | warningの閾値（°） |

## 平面検出と置き場所の候補

### 平面検出のノード

```bash
ros2 launch aha_perception plane_detection.launch.py
```

`plane_detection_node.py` はDepthとCameraInfoから点群を作り、画像時刻のTFで `target_frame` へ変換して平面を検出する。
水平面と壁を出力し、最新の検出から置き場所の候補をserviceで返す（topic・service・型は [ROSインターフェース](../../../docs/interfaces.md#平面検出)）。
処理が追いつかないフレームは捨てる。`rviz/sim_view.rviz` の `Planes` グループに平面と置き場所の候補の表示がある。
launch引数は `method`・`target_frame`・`max_rate`・`furniture_config`。その他はノードのパラメータで変える（入力は `depth_topic`・`camera_info_topic`、既定はインターフェースの入力topic）。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `method` | `ransac_gravity_refined` | `DETECTORS` の名前 |
| `target_frame` | `base_footprint` | 出力frame。原点が床、z軸が上のframeにする |
| `max_rate` | `5.0` | 検出の上限頻度（Hz、wall clock）。0で上限なし |
| `stride` | `2` | 検出器に渡す画素の間引き（行・列）。1で全画素（遅い） |
| `detect_vertical` | `true` | 垂直面も検出し、壁を出力する |
| `min_depth` / `max_depth` | `0.3` / `3.0` | 使う深度の範囲（m） |
| `merge` | `true` | 同一平面の断片の統合（params の `merge`、ない手法では無視） |
| `noise_subpixel` | `0.12` | 検出器のノイズモデル σ(Z) の subpixel（0より大きい値） |
| `max_points_per_plane` | `5000` | 出力点群の平面あたりの最大点数（0で全点） |
| `max_detection_age` | `3.0` | 置き場所のserviceが使う検出の古さの上限（s、wall clock）。0以下で無制限 |
| `tracking_frame` | `odom` | 面（[面の追跡](#面の追跡)）を追跡するframe。z軸が上で、台車が動いても面が動かないframeにする |
| `surface_confirm_hits` / `surface_max_misses` / `surface_max_unseen` | `2` / `5` / `120.0` | 面にIDを付けるまでの観測回数 / 視野内なのに観測されない検出が続くと消す回数 / 観測されないまま消すまでの時間（s、画像時刻） |
| `furniture_config` | 空 | [家具の設定](#家具の設定)のYAMLのパス。空なら家具を使わない |
| `furniture_margin` / `furniture_height_tolerance` | `0.10` / `0.05` | 面を家具の段に対応させる外形の広げ幅 / 段の高さとの差の上限（m） |
| `furniture_clip_margin` | `0.02` | 家具の段に対応した平面の凸包を切り取る外形の広げ幅（m、負なら切り取らない） |

`merge`・`noise_subpixel` の既定は較正したD435iのノイズで選んだ値で、ライブラリの既定（`merge=False`・0.08）とは異なる（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)）。

### 面の追跡

`/aha/perception/planes` の `Plane.id` は検出ごとの番号で、検出をまたいで同じ面を指さない。
同じ面を後で指定するには、検出のたびに `tracking_frame` で平面を対応付けた面（`/aha/perception/surfaces` の `Surface`、
[ROSインターフェース](../../../docs/interfaces.md#平面検出)）の `id` を使う。面のIDはマーカーの文字（`S7 table 0.72 m`）にも出る。

- 同じ高さの机でも、凸包が5 cm以上離れていれば別の面になる。床の高さの水平面は1つの面にまとめる。
- 同じ面の一部だけが見えてもIDは変わらない。凸包は見えていない部分（視野外・遮蔽）を前の値のまま残す。
- `surface_confirm_hits` 回観測するとIDが付く（既定で2回目の検出）。視野内なのに `surface_max_misses` 回続けて観測されないか、
  `surface_max_unseen` 秒観測されない面は消す。
- `level` は同じ所に重なった水平面（棚板など）の下からの順番（1から、重ならなければ0）。
- `tracking_frame` のTFがない間は `target_frame` で追跡する（台車が動くと面も動く）。TFが来た時点と、シミュレーションの再起動で面を消す。

対応付けの方法は [surfaces.py](aha_perception/surfaces.py)、合成シーンでの評価は [tracking.md](../../../tools/perception/plane_bench/results/tracking.md) を参照。

### 置き場所の候補

`object_size`（底面 x × y、高さ z、m）の物体を置ける机・棚板上の点を返す。要求・応答の各フィールドの意味と `status` は
[ROSインターフェース](../../../docs/interfaces.md#平面検出)、方式は [place.py](aha_perception/place.py) を参照。
障害物の箱にする点の高さは、上に検出した棚板（支持面の3割以上を覆う面）か、家具の段に対応した面では設定の上の棚板の下面から、
`cap_margin`（0.05 m）下まで（上の棚板が断片的にしか見えず検出されないと、その前の縁が棚の入口の箱になった）。

```bash
ros2 service call /aha/perception/get_place_candidates aha_msgs/srv/GetPlaceCandidates \
  "{object_size: {x: 0.08, y: 0.08, z: 0.15}, surface_labels: [table], max_candidates: 3,
    near: {header: {frame_id: base_link}, point: {x: 1.0, y: 0.2, z: 0.75}}}"
ros2 service call /aha/perception/get_place_candidates aha_msgs/srv/GetPlaceCandidates \
  "{object_size: {x: 0.06, y: 0.06, z: 0.12}, restrict_height: true, surface_height: 0.61}"
```

候補と障害物の箱は `/aha/perception/place/markers` に表示する。
前の検出で見た面に置くには、その面のID（[面の追跡](#面の追跡)）を `surface_ids` に渡す。
持っている物体とロボットの腕は `exclude_regions` に渡すと、障害物にならない。
似た物体の隣に置く場合は、その物体の位置を `near` に渡す（面の上の物体の種類は扱わない）。

### 家具の設定

事前に測った家具の外形と段の高さをYAMLに書き、`furniture_config` で渡すと、追跡した水平面を家具の段に対応させ、
`get_place_candidates` で家具と段を名前で指定できる。

```yaml
frame: map            # 下のposeのframe（z上向き、床がz = 0）
furniture:
  - name: cabinet
    category: shelf   # 自由な文字列
    pose: {x: 1.2, y: -0.4, yaw: 1.57}   # 外形の中心、局所x軸（奥行き）の向き
    size: {x: 0.35, y: 0.80}             # 奥行き x 幅（m）
    levels: [0.05, 0.42, 0.78, 1.12]     # 各段の上面の高さ（m）、下から
    board_thickness: 0.02                # 省略時0.02
```

- 段の番号は設定の一番下の段が1。段の上の空きは次の段の高さ − `board_thickness` − その段の高さ（一番上の段は制限なし）。
- 面の重心を設定のframeへ変換し、`furniture_margin` だけ広げた外形の中にあり、段の高さとの差が `furniture_height_tolerance` 未満なら、最も近い段に対応させる。
  複数の家具の外形に入る場合は外形の中心が最も近い家具にする。床・壁のラベルの面は対応させない（床の高さの段は対応しない）。
- 対応した面のその検出の平面は、凸包を家具の外形を `furniture_clip_margin` 広げた範囲で切り取る（Depthの縁のにじみで棚板の凸包が縁より手前・横へ数cm伸び、
  planning sceneの上の棚板が腕の通り道に掛かった）。`/aha/perception/planes` と置き場所の候補は切り取った凸包を、
  面の追跡と障害物の箱は検出したままの凸包を使う（設定やTFの誤差で切り落とした縁の上の物体も箱にする）。
  切り取ると何も残らない平面（外形の外にある、誤った対応）は、その検出では対応させない（警告を出す）。
- 上に棚板がある段に対応した面では、段の上の空きから面の高さの誤差（面の高さ − 段の高さ）を引いた高さを、
  上の棚板が検出されていなくても障害物の箱の帯の上限に使い、それより高い物体の候補を出さない（家具を指定しない要求でも）。
- 名前の重複、大きさ0以下、段の高さが増えない設定は起動時のエラーになる。

```bash
ros2 service call /aha/perception/get_place_candidates aha_msgs/srv/GetPlaceCandidates \
  "{object_size: {x: 0.06, y: 0.06, z: 0.12}, furniture: shelf, furniture_level: 3, clearance: 0.05}"
```

シミュレーションのworldの家具は [config/furniture/rcjo2026_arena.yaml](config/furniture/rcjo2026_arena.yaml)（frame `world`、形状はworldのSDFから）。
`world` のTFは `sim.launch.py world_tf:=true` が出す。

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true world_tf:=true \
  furniture_config:=$(ros2 pkg prefix aha_perception)/share/aha_perception/config/furniture/rcjo2026_arena.yaml
```

### ライブラリ

ROS非依存（numpy / scipy のみ）で、ノードと [平面検出ベンチマーク](../../../tools/perception/plane_bench/README.md) が使う。

- [`aha_perception.planes`](aha_perception/planes/__init__.py)：organized point cloudから平面を検出する。`detect_planes(cloud, method, options, params)`、手法は `DETECTORS`（8種）、ノードの出力の選択は `select_output_planes`。
- [`aha_perception.place`](aha_perception/place.py)：`find_place_candidates` が検出した平面と点群から置き場所の候補を、`find_place_obstacles` が支持面の上の物体の箱を返す。格子・余裕・箱の既定は `PlaceOptions`。
- [`aha_perception.surfaces`](aha_perception/surfaces.py)：`SurfaceTracker.update` が検出ごとの平面を追跡し、面のIDを返す。閾値の既定は `TrackerOptions`。
- [`aha_perception.furniture`](aha_perception/furniture.py)：家具の設定の読み込み（`load_config`）と面の段への対応（`associate_surfaces`）。PyYAMLも使う。
- [`aha_perception.depth_noise`](aha_perception/depth_noise.py)：D435i相当のDepthノイズ（`NOISE_PRESETS`）。`sim_depth_noise.py` とベンチマークで共通。
- [`aha_perception.imu_tilt`](aha_perception/imu_tilt.py)：IMUの重力の向きの相補フィルタとtilt / roll。
- [`aha_perception.object_list`](aha_perception/object_list.py)：物体リスト（ラベル・カテゴリ・大きさ・Gazeboのモデル名）の読み込み。PyYAMLも使う。
- [`aha_perception.sim_object_detection`](aha_perception/sim_object_detection.py)：`sim_object_detector.py` の見えているかの判定と見落とし・ラベルの誤り。

### 制限

- `target_frame` の原点が床でz軸が上、台車が水平であることを仮定する（高さ・ラベル・IMUのtilt）。
- 出力は水平面と壁だけ。斜めの面、小さい垂直面、下を向く水平面（カメラより上の棚板の裏など）は出さない。
- 置き場所は見えている支持面の上だけで、物体の陰など見えていない所は候補にしない。`approach_from` がなければ物体の向きは考慮せず底面の外接円で、
  あれば置く向きの長方形で判定する（向きは5°刻みで、刻みの中のどの向きでも収まるよう長方形を広げる）。`near` を渡すと候補の余裕は必要最小限のことがある。
- 障害物の箱は見えている点だけから作る。接した物体は1個の箱になり、奥側が隠れた物体は見えている範囲だけになる。Depthの縁のにじみで底面は視線方向に長くなりやすい。箱の既定値は合成シーンの調整の対象外。
- 厚さ約2 cm未満の物体（本など）は障害物にならず、厚さ1 cmでは平面のインライアに入ってその上が候補になる。
  棚の一番下の段は上の棚板に遮られ、手前の一部しか見えない（シミュレーションで確認）。
- 障害物の帯の下端は max(`obstacle_min_height`, `obstacle_k` × σ(Z)) で、既定のノイズでは3 m先で約5 cmになり、遠い低い物体（皿・ふた）は候補も箱も避けない。
- 机（`table`）は高さ0.4–1.0 m・面積0.05 m²以上（`ClassifyOptions`）。範囲外の水平面（低い机、カウンター、小さい台）は棚板でなければ `other` になり、既定の支持面に入らない。
- 面の追跡は `tracking_frame` の位置の誤差（odomのずれ）をそのまま受ける。戻ってきた面が前の凸包と重ならないほどずれると別のIDになる。
- 家具の段への対応は設定の寸法と `tracking_frame` → 設定のframeのTFの誤差を受ける。重心のずれが `furniture_margin`、高さの差が `furniture_height_tolerance` を超えると対応しない。
- 既定値は合成シーンと較正したノイズで選んだ（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)、[place.md](../../../tools/perception/plane_bench/results/place.md)）。実機ではGTを使った評価をしていない。

## テスト

`colcon test --packages-select aha_perception`。ROSに依存しないテストはホストでも実行できる。

```bash
cd overlay_ws/src/aha_perception
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 --with pyyaml --with pytest \
  python -m pytest test/test_planes_*.py test/test_plane_detection_io.py test/test_depth_noise.py test/test_place.py \
  test/test_surfaces.py test/test_furniture.py test/test_imu_tilt.py test/test_object_list.py test/test_sim_object_detection.py
```

`test/test_plane_detection_node.py`（ノードのメッセージ変換）と `test/test_plane_detection_service.py`（置き場所のserviceの分岐）はrclpyとROSのメッセージが必要で、colconでだけ実行する。
