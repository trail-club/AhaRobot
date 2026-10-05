# aha_perception

頭部RGB-Dカメラ（D435i）の点群生成、RViz表示（シミュレーション / 実機）、頭部の操作UI、平面検出、
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
| `perception.launch.py` | `sim.launch.py use_perception:=true` から起動。`camera_bridge.launch.py`、`pointcloud.launch.py`、`sim_view.launch.py`、`planes:=true` で `plane_detection.launch.py` |
| `camera_bridge.launch.py` | 頭部RGB-D・左右の手首RGBカメラを `config/sim_camera_bridge.yaml` でbridge。GUI・点群処理なしでタスク評価にも使用 |
| `pointcloud.launch.py` | depth_image_procで `/camera/depth/points` を生成。シミュレーションと実機で共通 |
| `sim_view.launch.py` | RViz（`rviz/sim_view.rviz`）、操作パネル、`head_look_at.py` |
| `camera_view.launch.py` | 実機カメラだけの表示。`pointcloud.launch.py` とRViz（`rviz/camera.rviz`） |
| `real_head_camera.launch.py` | 実機の頭部（と任意でカメラ・平面検出）をシミュレーションと同じUIで動かす |
| `plane_detection.launch.py` | `plane_detection_node.py`。シミュレーションと実機で共通 |

## ノード

| node | 内容 |
| --- | --- |
| `head_look_at.py` | `/aha/perception/look_at` の点へ頭を向ける。可動範囲外は限界に丸めてwarningを出す。注視点を `/aha/perception/look_at/markers` に表示（丸めた場合は赤） |
| `sim_control_panel.py` | Qtの操作パネル。台車のジョイスティック（W/A/S/Dキー、停止ボタン）と頭のpan / tiltスライダー・プリセット。`arm_controls:=true` で左右のグリッパと昇降の欄を追加 |
| `teleop_head.py` | 端末から頭を動かす。矢印キーで5°ずつ、spaceで正面、qで終了 |
| `plane_detection_node.py` | 頭部のDepthから床・机・棚・壁の平面を検出し、点群とMarkerArrayで出力する（[平面検出](#平面検出のノード)） |

頭の可動範囲・移動速度・プリセットは `config/head.yaml` で設定し、3つのノードが
`aha_perception.head_config` 経由で読む。符号は +pan = ロボットの右、+tilt = 下。
可動範囲は設定値で、実機の頭の可動域は測定していない。

操作パネルは `/diff_drive_controller/cmd_vel` に10 Hzで送信し、離すと0を1回送る。
`header.stamp` は0で送り、diff_drive_controllerが受信時の自分の時刻（sim時刻）を入れる。
送信が止まると `cmd_vel_timeout`（0.5 s）で台車が止まる。
最大速度は0.4 m/s / 1.5 rad/s（パラメータ `max_linear` / `max_angular`）で、
URDFの車輪関節の上限（10 rad/s × 半径0.042 m ≈ 0.42 m/s）より低くしている。
`head_look_at.py`・`sim_control_panel.py`・`plane_detection_node.py` はwall clockで動く。
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
| `planes` | `false` | 平面検出（`plane_detection.launch.py`）を起動。`camera:=true` と併用する |

TFはrobot_state_publisher（`sim:=false`）とjoint_state_publisherが出す。
`/joint_states` の頭は `/head_controller/joint_states`、その他の関節は0。
Ctrl-Cでトルクを切る。サーボが見つからない場合、ブリッジはエラーで終了する。

`camera:=true` では、D435iの映像を [stream_realsense.py](../../../tools/perception/macos/pointcloud/README.md#rosbridgeへの直接送信)
で既定の `ws://127.0.0.1:9090` へ送ると、RVizに点群が表示される。

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
| `max_rate` | `1.0` | 検出の上限頻度（Hz、wall clock）。0で上限なし |
| `detect_vertical` | `true` | 垂直面も検出し、壁を出力する |
| `min_depth` / `max_depth` | `0.3` / `3.0` | 使う深度の範囲（m） |
| `max_points_per_plane` | `5000` | 出力点群の平面あたりの最大点数（0で全点） |
| `stride` | `1` | Depthの間引き（行・列） |
| `merge` | `true` | 同一平面の断片の統合（params の `merge`）。較正ノイズでの断片化を抑える（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)） |
| `noise_subpixel` | `0.12` | 検出器の `DepthNoiseModel` の subpixel（0より大きい値）。較正ノイズでの利得と既定ノイズでの低下の釣り合い（[tuning.md](../../../tools/perception/plane_bench/results/tuning.md)） |

出力は `target_frame`、Depthの時刻で配信する（topic・型は [ROSインターフェース](../../../docs/interfaces.md#平面検出)）。
`/aha/perception/planes/cloud` は平面のインライア点を平面ごとに色分けする。
`/aha/perception/planes/markers` はDELETEALLの後、平面ごとに凸包（ns `hull`）と文字（ns `label`）を出す。
文字は水平面が `table 0.72 m` のように床からの高さ、垂直面が `wall 2.3 m2` のように面積。
ラベルの色は床が灰、机が緑、棚が橙、壁が青、その他が紫。
`rviz/sim_view.rviz` の `Planes` グループに両topicの表示がある。

単体テストは `colcon test --packages-select aha_perception` で実行する。
ホストでは numpy / scipy / pytest のある Python 3.12 で次を実行する。

```bash
cd overlay_ws/src/aha_perception
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with pytest python -m pytest test/test_planes_*.py
```
