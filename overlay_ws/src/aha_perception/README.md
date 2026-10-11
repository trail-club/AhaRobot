# aha_perception

頭部RGB-Dカメラ（D435i）の点群生成、RViz表示（シミュレーション / 実機）、頭部の操作UI、平面検出と置き場所の候補、
シミュレーションの左右の手首RGBカメラのbridge、SAM 3.1のセグメンテーション検証。
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
| `imu_head.py` | D435iのIMUでtiltを求め、`/head_controller/joint_states` に出す（`head:=imu`）か、サーボのtiltと比べる（`imu_check:=true`）（[IMUによる頭部のtilt](#imuによる頭部のtilt)） |
| `plane_detection_node.py` | 頭部のDepthから床・机・棚・壁の平面を検出して出力し、置き場所の候補をserviceで返す（[平面検出](#平面検出のノード)） |
| `sim_depth_noise.py` | シミュレーションのDepthに実機D435i相当のノイズを加える（[Depthノイズ](#depthノイズ)） |

頭の可動範囲・移動速度・プリセットは `config/head.yaml` で設定し、頭を扱うノードが
`aha_perception.head_config` 経由で読む。符号は +pan = ロボットの右、+tilt = 下。
可動範囲は設定値で、実機の頭の可動域は測定していない。

操作パネルは `/diff_drive_controller/cmd_vel` に10 Hzで送信し、離すと0を1回送る。
`header.stamp` は0で送り、diff_drive_controllerが受信時の自分の時刻（sim時刻）を入れる。
送信が止まると `cmd_vel_timeout`（0.5 s）で台車が止まる。
最大速度は0.4 m/s / 1.5 rad/s（パラメータ `max_linear` / `max_angular`）で、
URDFの車輪関節の上限（10 rad/s × 半径0.042 m ≈ 0.42 m/s）より低くしている。
`head_look_at.py`・`sim_control_panel.py`・`plane_detection_node.py`・`sim_depth_noise.py` はwall clockで動く。
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
| `planes` | `false` | 平面検出（`plane_detection.launch.py`）の起動 |
| `depth_noise` | `none` | Depthに加えるノイズ（`none` / `d435i_filtered` / `d435i_raw`、[Depthノイズ](#depthノイズ)） |

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

### Depthノイズ

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true depth_noise:=d435i_filtered
```

`depth_noise` を指定すると、bridgeはDepthを `/camera/depth_registered/image_rect_clean` に出し、
`sim_depth_noise.py` がノイズを加えて `/camera/depth_registered/image_rect` に出す。
点群・平面検出・RVizはノイズを加えたDepthを使う。

- ノイズは平面検出ベンチマークと共通の `aha_perception.depth_noise` で、プリセットは実機D435iの静止フレームに合わせた値
  （`d435i_filtered` / `d435i_raw`、[較正](../../../tools/perception/plane_bench/README.md#ノイズの較正)）。
- 静的な誤差場と穴は最初のフレームで引き、画像座標に固定する（画像サイズが変わると引き直す）。
  時間方向のノイズ、無効画素の周りのフリッカー、孤立した欠損、flying pixel、遮蔽帯（前景の左側）はフレームごとに計算する。
- f はCameraInfoの fx。シミュレーションのカメラは実機のカラーと同じ fx（約603、水平画角55.9°）にしてあり、
  m単位の σ(Z) も実機に合う。
- 6 mより遠い画素は無効になる。入力の無効画素（NaN・inf・0以下）は無効のまま。
- Gazebo自体のDepthノイズ（ガウシアン σ 5 mm、`aha_description/urdf/sensors.xacro`）は残り、その上に加わる。
- 最新のフレームだけを別スレッドで処理し、処理中に届いたフレームは捨てる。10 sごとに処理時間と捨てた数をログに出す。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `preset` | `d435i_filtered` | `NOISE_PRESETS` の名前（launchの `depth_noise`） |
| `input_topic` | `/camera/depth_registered/image_rect_clean` | 入力のDepth（32FC1のm、または16UC1のmm） |
| `output_topic` | `/camera/depth_registered/image_rect` | 出力のDepth。header・エンコーディングは入力と同じ（16ビットは16UC1） |
| `camera_info_topic` | `/camera/color/camera_info` | fxの取得 |
| `seed` | `0` | 乱数のシード |

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
| `head` | `servo` | `servo`：サーボを動かす。`manual`：サーボなしで `manual_head.py` が指令した角度を頭の状態にする。`imu`：サーボなしで `imu_head.py` がIMUのtiltと指令したpanを頭の状態にする（`camera_source:=realsense` が必要） |
| `imu_check` | `false` | `head:=servo` で、サーボとIMUのtiltが3°より違うとwarningを出す（`camera_source:=realsense` が必要） |
| `rviz` | `true` | `sim_view.launch.py` をfixed frame `base_link` で起動。台車の操作は無効 |
| `camera` | `false` | カメラの配信元と `pointcloud.launch.py` を起動 |
| `camera_source` | `rosbridge` | `rosbridge`：rosbridge（127.0.0.1:9090）を起動し、`stream_realsense.py` から受ける。`realsense`：`realsense.launch.py` を起動 |
| `color_profile` / `depth_profile` | `640,480,15` | `camera_source:=realsense` の解像度とfps（[RealSenseドライバ](#realsenseドライバ)） |
| `planes` | `false` | 平面検出（`plane_detection.launch.py`）を起動。`camera:=true` と併用する |

TFはrobot_state_publisher（`sim:=false`）とjoint_state_publisherが出す。
`/joint_states` の頭は `/head_controller/joint_states`、その他の関節は0。
Ctrl-Cでトルクを切る。サーボが見つからない場合、ブリッジはエラーで終了する。

`camera:=true camera_source:=rosbridge` では、D435iの映像を [stream_realsense.py](../../../tools/perception/macos/pointcloud/README.md#rosbridgeへの直接送信)
で既定の `ws://127.0.0.1:9090` へ送ると、RVizに点群が表示される。
Linuxでは `camera_source:=realsense` でRealSense ROSドライバから直接配信できる（rosbridgeはDepthのJSON変換で遅れる。[測定](../../../docs/context/camera.md)）。

```bash
ROS_DOMAIN_ID=7 ros2 launch aha_perception real_head_camera.launch.py camera:=true camera_source:=realsense planes:=true
ROS_DOMAIN_ID=7 ros2 launch aha_perception real_head_camera.launch.py head:=imu camera:=true camera_source:=realsense planes:=true
```

### RealSenseドライバ

`realsense.launch.py` は `realsense2_camera_node`（node `/camera`）を起動し、
[頭部カメラ](../../../docs/interfaces.md#頭部カメラ)のtopicで配信する。

- カラーはrgb8、Depthは16UC1（mm）でカラーへ位置合わせし、`/camera/depth_registered/image_rect` に出す。
- 位置合わせの前に `stream_realsense.py` と同じ後処理（disparity変換 → spatial → temporal → depthへ戻す、SDKの既定値）をかける（`filters:=false` で無効）。
- 画像は `camera_color_optical_frame`、IMUは `camera_imu_optical_frame`。`publish_tf` は無効で、カメラのTFはURDFが出す。
- 赤外線は配信しない。IMU（`imu:=true`、既定）はaccelをgyroの時刻へ線形補間して `/camera/imu` に出す。
  `real_head_camera.launch.py` は `head:=imu` か `imu_check:=true` のときだけIMUを有効にする。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `color_profile` | `640,480,15` | カラーの幅,高さ,fps。位置合わせしたDepthもこの大きさ |
| `depth_profile` | `640,480,15` | 位置合わせ前のDepthの幅,高さ,fps |
| `filters` | `true` | Depthの後処理 |
| `imu` | `true` | accel・gyro |
| `serial_no` | 空 | 使う機体のシリアル番号（空は最初の機体） |

USB 2.0接続では、D435iはカラー848×480を持たず、Depth 848×480は10 / 6 fpsだけ、カラー1280×720は15 fpsまでになる。
既定の640×480・15 fpsはUSB 2.0で選べる組み合わせ。USB 3では両方を848×480（15 / 30 fps）にできる。

コンテナのユーザー（uid 1000）は、映像（`/dev/video*`）はホストのログインユーザーのACLで開けるが、
IMU（HIDのiioデバイス）は開けない。ホストにlibrealsenseのudevルールを入れる（NixOSでは `services.udev.packages = [ pkgs.librealsense ];`）。
ルールがない場合は、挿すたびにホストで次を実行する。

```bash
for d in /sys/bus/iio/devices/iio:device*; do sudo chmod -R a+rw "$(dirname "$(dirname "$(readlink -f "$d")")")"; done
sudo chmod a+rw /dev/iio:device*
```

### IMUによる頭部のtilt

`imu_head.py`（`aha_perception.imu_tilt`）は `/camera/imu` のaccelとgyroから重力の向き（上向きの単位ベクトル）を求め、頭のtiltにする。

- 相補フィルタ：サンプルごとに推定をgyroで回転させ（du/dt = u × ω）、accel/|accel| へ係数 1 − exp(−dt/`time_constant`) で近づける。accelのノルムと重力の差が `accel_tolerance` 以上のとき（加速中）はgyroだけで追従する。動いている間もtiltは連続に変わる。
- 最初のサンプルと、0.5 sを超えて途切れた後はaccelから初期化する。
- accelは `camera_color_optical_frame` と同じ向きとして扱う（librealsenseはD435iのIMUをDepthの光学座標系の向きで出す。Depthとカラーの間の回転は無視）。
  水平に置くとaccelは約 (0, −9.8, 0)。カメラのtilt = atan2(−a_z, √(a_x² + a_y²))（+ = 下）、roll = atan2(−a_x, −a_y)（+ = 画像の右が下）。
- URDFでは頭の関節が0のとき光軸が水平で、tiltはpanによらないため、`joint_head_tilt` = カメラのtilt + `tilt_offset`。台車が水平であることを仮定する。
- rollは頭の関節で表せないため、|roll| が3°を超えると5 sごとにwarningを出す。
- `mode:=state`（`head:=imu`）：tiltはIMU、panは指令した軌道（`manual_head.py` と同じ）で30 Hzで出す。最初のIMUが届くまでは出さない。
- `mode:=check`（`imu_check:=true`）：サーボのtiltが1 s以上変わらず、IMUが静止している（gyroのノルムが `gyro_threshold` 未満、accelのノルムが重力に近い）ときに比べ、3°を超えて違うとwarningを出す。
- IMUは200 Hzで届くため、IMUのコールバックはフィルタの更新だけにし、`rclpy.experimental.EventsExecutor` で動かす（既定のexecutorより1メッセージあたりのCPUが約1/3）。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `mode` | `state` | `state` / `check` |
| `imu_topic` | `/camera/imu` | IMU（accel・gyro） |
| `tilt_offset` | `0.0` | tiltに加える値（rad）。取付の誤差の補正 |
| `time_constant` | `0.5` | accelによる補正の時定数（s） |
| `gyro_threshold` / `accel_tolerance` | `0.05` / `0.8` | 静止の判定（rad/s）/ accelで補正するaccelのノルムと重力の差の上限（m/s²） |
| `roll_tolerance_deg` / `check_tolerance_deg` | `3.0` / `3.0` | warningの閾値（°） |

## 平面検出 (`aha_perception.planes`)

ROSに依存しない平面検出の候補実装（numpy / scipy のみ）。
入力は organized point cloud `(H, W, 3)`（カメラ光学座標系、無効画素はNaN）、
出力は `DetectionOptions.camera_to_base` で変換した座標系（床が z=0、`up_axis` が上）の `Plane` のリスト。

```python
from aha_perception.planes import DETECTORS, DetectionOptions, detect_planes

options = DetectionOptions(camera_to_base=T_base_camera)  # 4x4
planes = detect_planes(cloud, "ransac_gravity", options)
```

`Plane` は単位法線 `normal`・`offset`（`normal . p + offset = 0`）・入力画素の flat index `inliers`・
重心・平面内の凸包と面積・RMS・ラベル（floor / table / shelf / wall / other）を持つ。
`options.detect_vertical=False` で水平面のみを出力する。
`select_output_planes(planes, options)` はラベル付け済みの結果から水平面と壁（`wall_min_area` 0.3 m² 以上・
高さ方向の広がり `wall_min_extent` 0.3 m 以上の垂直面）だけを順序を保って残し、斜めの面と小さい垂直面を捨てる。
`detect_planes` 自体の出力は変えない。
法線（カメラ側）が下を向く水平面（カメラより上の棚板・机の裏面、天井）も捨てる（`ClassifyOptions.drop_downward`、既定 `True`）
（棚での比較は [shelf.md](../../../tools/perception/plane_bench/results/shelf.md)）。

| 名前 | 方式 |
| --- | --- |
| `ransac_sequential` | voxel化した点群で固定閾値のRANSACを繰り返し、平面ごとに連結成分へ分割 |
| `ransac_gravity` | 法線で水平・垂直候補に分け、向きを拘束したRANSAC（閾値 k·σ(Z)）と連結成分。法線はvoxelのkNN（既定）か `normals="organized"` で画像法線 |
| `ransac_gravity_refined` | `ransac_gravity` の平面に全解像度の画素を残差比 \|n·p+d\|/(k·σ(Z)) 最小で再割当（インライアと連結な部分のみ）、小片除去、平面上の穴埋め、再フィット |
| `ransac_gravity_cape_hybrid` | `ransac_gravity` の平面をシードに、セル単位の領域成長と画素単位の境界補正で領域を広げる |
| `organized_region_growing` | 画像セルのPCA平面性判定、隣接セルの領域成長と同一平面の統合、画素単位の境界補正 |
| `normal_clustering` | 積分画像法線を球面上でクラスタリングし、平面オフセットと画像上の連結成分で分割 |
| `peac_ahc` | PEAC（Feng et al., ICRA 2014）の論文に基づく独自実装。ノード格子の凝集型階層クラスタリング（min-heap、T_MSE(Z)）と画素単位の補正 |
| `cape` | CAPE（Proença & Gao, IROS 2018）の平面部分の再実装。セル平面性判定、法線ヒストグラムからのシード、セル領域成長、統合、境界の画素補正 |

前処理（深度範囲・間引き・voxel化・座標変換）、法線推定、セル格子とモーメント和（`cells.py`）、
画素の平面への割当（`refine.py`）、ラベル付け、
ラベル画像から平面への変換（再フィット・k·σ(Z)での画素選別・面積フィルタ）は全候補で共通。
深度ノイズは `DepthNoiseModel`（σ(Z) = Z²·subpixel / (f·baseline)）で与える。

`ransac_gravity` と `ransac_gravity_refined` は params の `merge=True`（既定 `False`）で同一平面の断片を統合する（`merge.py` の `merge_coplanar`）。
法線の差が `merge_angle_deg`（既定10°）未満、平均法線方向の重心の差が `merge_k`·σ(Z)（既定4、Zは2平面の平均深度の大きい方）未満で、
画像上のインライアが `merge_gap_px`（既定3 px）以内で接する組を、重心の差が小さい順に統合する。
両方が水平（または垂直）な組は、和集合に当てはめた平面も水平（垂直）な場合だけ統合する。
`merge_hull_gap` を0以上にすると、平面上に投影した凸包の距離がその値（m）以下の組も統合する（既定 −1 で無効）。
統合した平面だけを `planes_from_labels` で再フィットし、全平面のラベルを付け直す。`ransac_gravity_refined` は画素の再割当の前後で統合する。

```python
from aha_perception.planes import DETECTORS, DepthNoiseModel

params = DETECTORS["ransac_gravity_refined"].params_type(merge=True)
options = DetectionOptions(camera_to_base=T_base_camera, noise=DepthNoiseModel(focal=fx))
planes = detect_planes(cloud, "ransac_gravity_refined", options, params)
```

候補の比較手順は [平面検出ベンチマーク](../../../tools/perception/plane_bench/README.md) を参照。

### 平面検出のノード

`plane_detection_node.py` はDepth（32FC1のm、または16UC1のmm）とCameraInfoから
organized point cloudを作り、画像時刻のTFで `target_frame` へ変換して `detect_planes` を実行する。
出力は `select_output_planes` で選んだ水平面と壁だけで、斜めの面と小さい垂直面は出さない。
起動時のログに手法・`merge`・`noise_subpixel` を出す。
ログの要約は `3 planes (floor 1, table 1, wall 1) of 14 detected` のように出力数と検出数を示す。
`DepthNoiseModel` の f にはCameraInfoの fx、subpixelにはパラメータ `noise_subpixel` を使い、baselineは既定値。
params はパラメータ `merge` 以外を既定値にする（`merge` のない手法では無視してwarningを出す）。
ノードの既定（`merge` true・`noise_subpixel` 0.12）は較正したD435iのノイズで選んだ値で、
ライブラリの既定（`merge=False`・subpixel 0.08）とは異なる（[比較](../../../tools/perception/plane_bench/results/tuning.md)）。
画像時刻のTFが0.2 sで得られない場合は最新のTFを使いwarningを出す。
最新のフレームだけを別スレッドで処理し、処理中に届いたフレームは捨てる。
`/tf` の時刻が戻るとTFバッファを消す（`head_look_at.py` と共通の `aha_perception.tf_jump`）。
`stride` N では画像のN行・N列ごとの画素を検出器に渡す（画素座標は全画像のまま）。
RANSAC段の間引き（params の `stride`）は `params_for_input_stride` で 1/N（1以上）にし、全画像に対する作業解像度を保つ。
置き場所は `PlaceOptions.for_input_stride` で `obstacle_min_points` を 1/N²（1以上）、`support_closing` を +N−1 セルにする。
`min_inliers` などの画素数の閾値は変えない（1/N² にすると合成シーンで断片が増え、PQが stride 1 の値まで下がった）。
1フレームの処理時間（実機の640×480、開発コンテナ、中央値）は stride 2 で約0.27 s、stride 1 で約0.62 s（他のシミュレーションが動いている負荷あり）。

```bash
ros2 launch aha_perception plane_detection.launch.py
```

launch引数は `method`・`target_frame`・`max_rate`。その他はノードのパラメータで変える。

| パラメータ | 既定 | 内容 |
| --- | --- | --- |
| `method` | `ransac_gravity_refined` | `DETECTORS` の名前 |
| `target_frame` | `base_footprint` | 出力frame。原点が床、z軸が上のframeにする |
| `depth_topic` | `/camera/depth_registered/image_rect` | RGBへ位置合わせしたDepth |
| `camera_info_topic` | `/camera/color/camera_info` | Depthと同じ解像度のCameraInfo |
| `max_rate` | `5.0` | 検出の上限頻度（Hz、wall clock）。0で上限なし。処理が追いつかないフレームは捨てる。`real_head_camera.launch.py` / `perception.launch.py` にも `max_rate:=` で渡せる |
| `detect_vertical` | `true` | 垂直面も検出し、壁を出力する |
| `min_depth` / `max_depth` | `0.3` / `3.0` | 使う深度の範囲（m） |
| `max_points_per_plane` | `5000` | 出力点群の平面あたりの最大点数（0で全点） |
| `stride` | `2` | Depthの間引き（行・列）。2は1より検出が1.5〜2.4倍速い。合成シーンではPQが高く、水平面のrecall・置き場所のvalidity・availabilityの差は0.01以内（[ベンチマーク](../../../tools/perception/plane_bench/README.md#設定の比較)の `--node-stride`） |
| `merge` | `true` | 同一平面の断片の統合（params の `merge`）。較正ノイズでの断片化を抑える（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)） |
| `noise_subpixel` | `0.12` | 検出器の `DepthNoiseModel` の subpixel（0より大きい値）。較正ノイズでの利得と既定ノイズでの低下の釣り合い（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)） |
| `max_detection_age` | `3.0` | 置き場所のserviceが使う検出の古さの上限（s、wall clock）。0以下で無制限 |

出力は `target_frame`、Depthの時刻で配信する（topic・型は [ROSインターフェース](../../../docs/interfaces.md#平面検出)）。
`/aha/perception/planes` は出力する平面（`aha_msgs/PlaneArray`、`id` は配列の番号）。
`/aha/perception/planes/cloud` は平面のインライア点を平面ごとに色分けする。
`/aha/perception/planes/markers` はDELETEALLの後、平面ごとに凸包（ns `hull`）と文字（ns `label`）を出す。
文字は水平面が `table 0.72 m` のように床からの高さ、垂直面が `wall 2.3 m2` のように面積。
ラベルの色は床が灰、机が緑、棚が橙、壁が青、その他が紫。
`rviz/sim_view.rviz` の `Planes` グループに両topicと置き場所の候補（`Place candidates`）の表示がある。

ノードは最新の検出（平面・点群・ノイズモデル・時刻）を保持し、service `/aha/perception/get_place_candidates`
（`aha_msgs/GetPlaceCandidates`）で [置き場所の候補](#置き場所の候補) を返す。
検出がない場合と `max_detection_age` より古い場合は `success` false。`near` は最新のTFで `target_frame` へ変換する。
呼び出しごとに `/aha/perception/place/markers` へDELETEALLの後、候補ごとに clearance を半径とする半透明の円盤（ns `clearance`）、
中心の球（ns `center`、1位は緑）、順位の文字（ns `rank`）を出す。

```bash
ros2 service call /aha/perception/get_place_candidates aha_msgs/srv/GetPlaceCandidates \
  "{object_size: {x: 0.08, y: 0.08, z: 0.15}}"
ros2 service call /aha/perception/get_place_candidates aha_msgs/srv/GetPlaceCandidates \
  "{object_size: {x: 0.08, y: 0.08, z: 0.15}, surface_labels: [table], max_candidates: 3,
    near: {header: {frame_id: base_link}, point: {x: 1.0, y: 0.2, z: 0.75}}}"
```

### 置き場所の候補

`aha_perception.place.find_place_candidates` は検出した平面と点群から、底面 x × y・高さ z の物体を置ける面上の点を返す（ROS非依存、numpy / scipy のみ）。

```python
from aha_perception.place import base_cloud, find_place_candidates

points, depth = base_cloud(cloud, T_base_camera)  # 検出に渡した点群（カメラ座標系）
result = find_place_candidates(planes, points, depth, (0.08, 0.08, 0.15), options.noise)
for c in result.candidates:  # position, clearance, label, plane_index, score
    ...
```

- 支持面は上を向く水平面のうちラベルが `labels`（既定 table / shelf、floor / other も指定可）のもの。
  平面ごとに `plane.axes`・`plane.centroid` の平面座標で格子（`resolution` 1 cm、凸包の外接矩形 + 5 cm）を作る。
- 支持：平面のインライアの画素がある格子（closing 2セル）と凸包の積。インライアのない格子（物体の陰など見えていない所）は置けない。
- 障害物：有効な点のうち、平面からの高さが max(`obstacle_min_height`, `obstacle_k`·σ(Z)) 以上、物体の高さ + `height_margin` 以下の点が
  `obstacle_min_points` 個以上ある格子。`speckle_cells` 未満の連結成分は除く。σ(Z) は検出器の `DepthNoiseModel`。
- 上の面：ほかの水平面（物体の上面・上の棚板も含む）の重心が `overhead_min_gap` 以上、物体の高さ + `height_margin` 以下だけ上にあれば、
  その凸包の範囲は障害物として扱う。
- 置ける中心：支持の縁（見えていない所を含む）から r + `edge_margin`、障害物から r + `obstacle_margin` 以上離れた格子。
  r は底面の外接円の半径（向きによらない）。clearance は中心から最も近い縁・障害物までの距離。
- 候補：既定は clearance の局所最大を大きい順（同じ clearance は支持面の中心、平面間では `target_frame` の原点に近い順）。
  `near` を渡すと置ける格子を `near` に近い順に並べる（clearance は必要最小限のことがある）。
  平面ごとに max(r, `min_spacing`) 以上離して、合計 `max_candidates` 個まで返す。`score` は clearance、`near` ありでは −距離。
- `result.grids` に平面ごとの格子（support / observed / obstacle / overhead / placeable / clearance）を返す。
- 640×480の1フレームで平均34 ms・最大59 ms（支持面1枚で21 ms、5枚で47 ms。1スレッド、合成シーン）。

`PlaceOptions` の既定（合成シーンで調整、[place.md](../../../tools/perception/plane_bench/results/place.md)）：

| 項目 | 既定 | 内容 |
| --- | --- | --- |
| `resolution` / `grid_margin` | 0.01 / 0.05 m | 格子の大きさと凸包の外側の余白 |
| `support_closing` | 2 | インライアの格子のclosingの半径（セル） |
| `obstacle_min_height` / `obstacle_k` | 0.02 m / 1.5 | 障害物とみなす高さの下限 max(下限, k·σ(Z))。k 3 では遠くの低い物体（1.5 m先の高さ5–6 cmの箱）を見落とす |
| `height_margin` | 0.03 m | 障害物の高さの上限 = 物体の高さ + これ |
| `obstacle_min_points` / `speckle_cells` | 2 / 3 | 障害物の格子の点数の下限、除く連結成分の大きさ |
| `edge_margin` / `obstacle_margin` | 0.02 / 0.02 m | 縁・障害物までの余裕（r に加える） |
| `overhead_min_gap` | 0.03 m | 上の面とみなす高さの差の下限 |
| `min_spacing` / `max_candidates` | 0.05 m / 10 | 候補の間隔の下限、候補数 |

単体テストは `colcon test --packages-select aha_perception` で実行する。
ホストでは numpy / scipy / pytest のある Python 3.12 で次を実行する。

```bash
cd overlay_ws/src/aha_perception
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with pytest python -m pytest test/test_planes_*.py test/test_plane_detection_io.py \
  test/test_depth_noise.py test/test_place.py test/test_imu_tilt.py
```
