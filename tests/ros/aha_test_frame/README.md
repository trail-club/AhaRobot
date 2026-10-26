# aha_test_frame

試験用。台車と昇降のない実機（机の上の30 mm系アルミフレームの骨組みに両腕と頭を載せた構成）を動かすパッケージ。
骨組みのモデル、起動、カメラと腕の取り付けの校正の道具をまとめる。ロボットの本来の構成（台車・昇降あり）では使わない。
寸法と校正の記録は [骨組み](../../../docs/context/frame.md)、置く動作は [aha_test_manipulation](../aha_test_manipulation/README.md#実機骨組み) を参照。

## 起動

```bash
ros2 launch aha_test_frame real.launch.py
ros2 launch aha_test_frame real.launch.py arms:=false head:=manual    # サーボの基板なし
```

robot_state_publisher（骨組みのモデル）、静的TF `odom` → `base_footprint`（骨組みは動かない）、
`/head_controller/joint_states` と腕の `/<side>/arm/joint_states` を `/joint_states` に合流させる `joint_state_publisher`、
[`arms.launch.py`](../../../overlay_ws/src/aha_bringup/launch/arms.launch.py)（`arm_node`）、腕の軌道ブリッジ、頭のサーボ、D435i、点群、平面検出、RVizを起動する。
`base_footprint` は床、`base_link` はカメラの柱の下の天板の上。
状態のない関節（`use_left:=false` の左腕、`head:=none` の頭）は0として出るので、MoveItはその腕がゼロ姿勢にあるものとして衝突を判定する。使わない腕は実機でもそこから離して置く。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `frame_config` | `aha_test_frame/config/frame_table_202610.yaml` | 骨組みのモデル（下記） |
| `arms` / `use_left` / `use_right` | `true` | 腕（`arms.launch.py`） |
| `right_device` / `left_device` / `head_port` | `/dev/ttyUSB2` / `/dev/ttyUSB0` / `/dev/ttyUSB1` | 3枚の基板（最初の接続時、[記録](../../../docs/context/motor.md)）。番号はUSBの認識順で変わるので、差し直した後は `/dev/serial/by-path/`（ハブの口）で確かめる |
| `trajectory_bridge` / `record` | `true` / `false` | 腕の軌道ブリッジ（`aha_manipulation` の `arm_trajectory_bridge.py`）と rosbag（`arms.launch.py`、そちらの既定は `true`） |
| `head` | `servo` | `servo`、`manual`（指令した角度を頭の状態にする）、`imu`（サーボなしでtiltをD435iのIMUから求める）、`none` |
| `camera` / `planes` / `rviz` | `true` | D435i（IMUは `head:=imu` のときだけ）と点群／平面検出／RViz |
| `furniture_config` | なし | 平面検出の家具の設定（[aha_perception](../../../overlay_ws/src/aha_perception/README.md#家具の設定)） |

軌道ブリッジのスクリプトは実行権限なしでインストールされ、`--symlink-install` のビルドでは `ros2 run` から起動できないため、
`arms.launch.py` のブリッジは止め（`trajectory_bridge:=false`）、同じノード名と設定で `python3` から起動する。
腕には `upstream/astra_controller` のホスト側の符号のパッチが要る（[patches](../../../patches/README.md#astra_controller用パッチ)。ないと腕の角度が鏡写しになる）。
頭の基板は電源を入れるたびに透過通信（Serial Forwarding）にする（[aha_perception](../../../overlay_ws/src/aha_perception/README.md#接続)）。

## 骨組みのモデル

[`aha_test_frame/frame.py`](aha_test_frame/frame.py) は `aha_description` の xacro（`sim:=false`）のURDFを、YAMLの `frame_config` で書き換える。
机の高さ、頭のpan軸、カメラの取り付け（任意）、腕ごとの肩の軸の位置と付け根のyaw（昇降は固定関節になる）、
鉛直軸の腕の関節の読みのオフセット（`joint_offsets`、任意）、骨組みの部材（溝付きの断面で描き、衝突判定は外形の箱）を指定する。
頭の土台（AstraのHeadYawの部品）は、ビルド時に上流の胴体のmeshから切り出す（[crop_head_base.py](scripts/crop_head_base.py)、`astra_description` と同じBSD）。
YAMLには、測った値と写真から見積もった値の区別を書く。

## 校正

`real.launch.py` を起動した状態で行う。腕はトルクを切る（`ros2 topic pub --once /<side>/arm/torque_enable std_msgs/msg/UInt8 "{data: 0}"`）。
`--write` で `frame_config` を書き換えた後は、launchを起動し直す。

```bash
# 腕の付け根のyaw：腕ごとに、上腕を前に向け前腕を一直線にした姿勢、続いて外向きの姿勢
ros2 run aha_test_frame frame_calibrate.py --config <source>/aha_test_frame/config/frame_table_202610.yaml [--write]
# カメラの傾きと高さ：天板（0.38 m）か床（0）の空いた所に頭を向けて
ros2 run aha_test_frame camera_plane_calibrate.py --reference 0.38 --config <frame yaml> [--write]
```

`frame_calibrate.py` は肩の符号（左腕は外向きで+90°、右腕は−90°）と肘の0点も確かめる。
`camera_plane_calibrate.py` は `camera`（link_head_tilt → camera_link）を書く。平面からはカメラの水平位置とyawは分からず、結果は測ったときの頭の姿勢で有効。

カメラの水平位置とyawは、次にグリッパ先端のクリックから求める（頭は使う姿勢にする）。
`real.launch.py` のRVizはPublish Pointを頭の注視点に送る（クリックで頭が動く）ため、RVizは別に起動する。

```bash
ros2 launch aha_test_frame real.launch.py rviz:=false
rviz2 -d $(ros2 pkg prefix --share aha_perception)/rviz/sim_view.rviz -f base_footprint
ros2 run aha_test_frame camera_arm_calibrate.py --config <frame yaml> [--write]
```

RobotModelの表示は消す（meshをクリックするとカメラではなくモデルの点が返る）。
両腕の届く範囲に散らした6姿勢以上で、グリッパ（トルクオフ）の先端を下に向けて止め、点群上の先端をPublish Pointでクリックする。
クリックはTFで最も近い `link_ree` / `link_lee` と組にし（`--offset` でリンクの座標系でずらす、`--radius` でカメラ側の表面へずらす）、
グリッパや頭が動いていた組は外し、`q` でカメラの水平のずれとyawを求める（`u` で直前の組を消す）。
組ごとの残差と高さの差（適用しない）を表示し、`--write` で傾きと高さを保ったまま `camera` を書く。
先端を下に向けると、グリッパの軸方向の先端の位置の誤差は高さの差にだけ出る。

腕のモデルと点群の腕がまだ合わないとき（付け根のyawか関節の0点のずれ）は、点群から求める。
カメラの傾きと高さ（両腕が使えれば水平の位置も）を校正した後に行う。

```bash
ros2 run aha_test_frame arm_cloud_calibrate.py --config <frame yaml> --arm right [--free shoulder,elbow[,wrist][,height]] [--save snaps.npz] [--write]
```

腕（トルクオフ）を机の上で、肩と肘を変えたはっきり違う3姿勢以上に止め、前腕とグリッパをカメラに入れて（前に物を置かない）姿勢ごとにEnterを押す
（`u` で直前を消す、`q` で計算）。1姿勢は `base_footprint` の点群と、0.5 sで平均した腕の関節。
関節が動いた、点群が古い、前と同じ姿勢、頭が動いた、のどれかなら撮り直す
（関節の読みが1秒全桁同じなら警告だけ出す。止まった腕も、基板のリセット後の止まった読みもそうなる）。
計算は点群の点（腕の下の机より上、モデルの近く。`--exclude-fingers` で指を除く）をモデルの表面に合わせ、
姿勢ごとのRMS距離と対応した点の数、ずれ（`shoulder` は `yaw` に、`elbow` / `wrist` は joint_<l|r>3 / 4 の `joint_offsets` に、`height` は `shoulder_xyz` のzに足す）、
その相関と1姿勢を抜いたときの変化を表示する。
チェックが警告している間（1姿勢では肩と肘をほとんど分けられない。`--force` で書く）と、`/robot_description` が設定と違うときは `--write` しない。
`--load snaps.npz` で保存した姿勢を計算し直す。
片腕では、カメラのyawのずれは肩のずれと同じように腕を動かす。`--free camera`（水平のずれとyaw、`camera` に書く）を `shoulder` と同時に使うと分けられず、警告が出て `--write` しないので、肩を固定する（例 `--free elbow,camera`）。

## テスト

`colcon test --base-paths src ../tests/ros --packages-select aha_test_frame`（[tests](../../README.md)）。
