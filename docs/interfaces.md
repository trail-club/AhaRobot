# ROSインターフェース

標準シミュレーション、頭部カメラ、実機の頭部の入出力。controller設定は
[controllers.yaml](../overlay_ws/src/aha_description/config/controllers.yaml)を参照。

## Controller

| 名前 | 種類 |
| --- | --- |
| `joint_state_broadcaster` | JointStateBroadcaster |
| `diff_drive_controller` | DiffDriveController |
| `left_arm_controller` / `right_arm_controller` | JointTrajectoryController |
| `lift_controller` / `head_controller` | JointTrajectoryController |
| `left_gripper_controller` / `right_gripper_controller` | ForwardCommandController（position） |

`controller_manager` の更新周期は100 Hz。odom frameは `odom`、base frameは `base_footprint`。
ロボットのlink / joint定義は [URDF](../overlay_ws/src/aha_description/urdf/aha_robot.urdf.xacro)を参照。
`/clock` はGazeboからROSへbridgeする。

## シミュレーションの入出力

| 用途 | 名前 | 型 |
| --- | --- | --- |
| ベース速度指令 | `/diff_drive_controller/cmd_vel` | `geometry_msgs/msg/TwistStamped` |
| 車輪odom | `/diff_drive_controller/odom` | `nav_msgs/msg/Odometry` |
| 関節状態 | `/joint_states` | `sensor_msgs/msg/JointState` |
| シミュレーション時刻 | `/clock` | `rosgraph_msgs/msg/Clock` |
| TF | `/tf` / `/tf_static` | `tf2_msgs/msg/TFMessage` |
| 軌道指令 | `/<controller>/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| 軌道action | `/<controller>/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` |
| グリッパ位置指令 | `/left_gripper_controller/commands` / `/right_gripper_controller/commands` | `std_msgs/msg/Float64MultiArray` |

軌道の `<controller>` は `left_arm_controller` / `right_arm_controller` / `lift_controller` /
`head_controller`。関節配列の名前・順序は `controllers.yaml` に従う。
`lift_controller` は部分関節goalも受け付け、指定しない軸を保持する。
グリッパは同ファイルの2関節の順序で位置を指定する。
`teleop_base.sh` はteleopの `/cmd_vel` を速度指令topicへremapする。

標準launchに `/cmd_vel` / `/odom` へのremapや `map` frameの配信はない。

## タスク評価

[aha_sim_tasks](../overlay_ws/src/aha_sim_tasks/README.md) はepisodeごとに独立したGazebo partitionで
シミュレーションを起動する。通常のpolicyと評価用のROS domainは別々だが、それぞれ全episode共通。
policyは別プロセス・別Gazebo partitionで動き、
通常は観測とActionだけをIPCで交換する。既存controllerへの指令は評価側のROS adapterが配信するため、
標準シミュレーションのtopic / frameは変更しない。評価用topicをpolicy側のdomainへbridgeしない。
domain設定・policy API・分離の保証範囲はパッケージのREADMEを参照。

| 用途 | 名前 | 型 |
| --- | --- | --- |
| 評価用state（pose・指接触数・時刻） | `/evaluation/state` | `std_msgs/msg/String`（`gz.msgs.StringMsg`のJSONからbridge、20 Hz） |
| 評価用シミュレーション時刻 | `/clock` | `rosgraph_msgs/msg/Clock`（Gazebo `/evaluation/clock` の `gz.msgs.Clock`からbridge、20 Hz） |
| step実行時のGazebo制御 | `/world/task_tables/control` | Gazebo Transport service: `gz.msgs.WorldControl` → `gz.msgs.Boolean` |

`--step-duration` 使用時は評価側だけがworld controlを呼び、推論中にpauseし、
Actionを100 msごとのphysics実行に適用する。policyのObservation / Action型やROS topicは共通。
Codexへの観測変換・判断ログと実行方法は [policy評価](../overlay_ws/src/aha_sim_tasks/README.md#codexによるpolicy評価)を参照。
評価側の全カメラのRGB・深度・CameraInfo購読はbridgeと同じreliable（履歴10）とし、
pauseや動画記録の負荷がある場合も同一stampの組を受け取る。

`/evaluation/state` は1つのJSONに `schema_version: 1`、`frame_id: "world"`、
`stamp` / `contact_window_start`（それぞれ整数の `sec` / `nanosec`）、
`robot_pose`（world x / y / yaw）、`object_position`（world x / y / z）、
`finger_contacts`（0 / 1 / 2）を持つ。poseと接触数を別topicから組み合わせない。
`environment.launch.py ground_truth:=true` のときだけ、同じサンプルに
`ground_truth.robot_world_pose`（URDF rootの `base_footprint` のworld姿勢、x / y / z / qx / qy / qz / qw）と
`ground_truth.joint_positions`（Gazebo physicsの関節名→位置、m / rad）を追加する。
既定は `false` で、通常の評価stateにこの追加情報はない。
指接触数は直前の50 ms窓内にappleへ接触した右グリッパの指の数。
窓内の和集合とholdの意味は [Grasp model](../overlay_ws/src/aha_sim_tasks/README.md#grasp-model) を参照。
姿勢・時刻は窓の終端で、成功判定はこのstateの `stamp` を使う。
重複・逆順のstateは速度履歴とfreshnessも更新しない。
pickは2本の接触と持ち上げ、placeのreleaseは0本の接触で判定する。
この観測系は物体の運動を変更しない。通常のpolicyには `/joint_states`、車輪odom、シミュレーション時刻、
タスクID・言語指示を渡す。既定では頭部RGB-D・左右の手首RGBカメラを描画し、画像・校正・画像時刻のbase_linkからのTFを
policyへ渡す。`--no-cameras` で描画とカメラ観測を除く。外部の画像topicは
`--head-image-topic` の指定時のみsensor-data QoSで購読する。このとき内蔵カメラは無効になり、
`--cameras` との併用はエラーになる。
world poseの座標はworld、通常のpolicyのodom座標は `odom`。
真値IK評価では、評価側がphysics関節位置と同じstampの
`/evaluation/joint_states`（`sensor_msgs/msg/JointState`、20 Hz）を配信する。
MoveItの状態入力はこのtopicへremapする。
MoveItへ渡す状態のみ、関節下限・上限の外にあるGazeboの物理限界（0.005 m / rad外側）と
その数値誤差までの値を境界へ補正する。
IPCの観測真値は補正せず、それ以上の逸脱も隠さない。
真値policyの観測・実行・制約は
[Pick with ground-truth IK](../overlay_ws/src/aha_sim_tasks/README.md#pick-with-ground-truth-ik) を参照。
評価はcontrollerの速度・関節位置指令を発行する。
評価launchは `sim.launch.py bridge_clock:=false` とし、stateと同じ周期の評価用clockを配信する。
標準launchの `bridge_clock` は既定で `true`。

## MoveIt

`sim.launch.py use_manip:=true` で [aha_manipulation](../overlay_ws/src/aha_manipulation/README.md) が起動する。
モデルframeは `base_footprint`、右腕groupは `right_arm`、IK/FKのtipは `link_ree`。

| 用途 | 名前 | 型 |
| --- | --- | --- |
| IK | `/compute_ik` | `moveit_msgs/srv/GetPositionIK` |
| FK | `/compute_fk` | `moveit_msgs/srv/GetPositionFK` |
| 計画・軌道実行 | `/move_action` | `moveit_msgs/action/MoveGroup` |
| Planning sceneの取得・更新 | `/get_planning_scene` / `/apply_planning_scene` | `moveit_msgs/srv/GetPlanningScene` / `ApplyPlanningScene` |
| 監視用planning scene | `/monitored_planning_scene` | `moveit_msgs/msg/PlanningScene` |
| MoveIt用の関節状態 | `/aha/manipulation/joint_states` | `sensor_msgs/msg/JointState` |

状態入力は既定で `/joint_states`。`manip_joint_states_topic` で変更できる。
move_groupは、この入力の関節限界の外 `limit_tolerance`（既定1e-6 m / rad）以内の値を限界へ補正した
`/aha/manipulation/joint_states` を読む。
`sim.launch.py` はGazeboの物理限界までを補正する値（`aha_gazebo.model.LIMIT_TOLERANCE`）を渡す。
実行先は `/right_arm_controller/follow_joint_trajectory` と `/lift_controller/follow_joint_trajectory`。
同じcontrollerへのtopic指令とMoveItのaction実行を同時に行わない。

## SLAM（シミュレーションのみ）

`sim.launch.py use_nav:=true` で [aha_navigation](../overlay_ws/src/aha_navigation/README.md) が起動する。
`use_nav:=false`（既定）では `/scan` のbridge、`/map`、`map` frameはない。

| 用途 | 名前 | 型 |
| --- | --- | --- |
| 2D LiDAR（frame `laser_link`、10 Hz） | `/scan` | `sensor_msgs/msg/LaserScan` |
| 占有格子地図 | `/map` | `nav_msgs/msg/OccupancyGrid` |

slam_toolboxは `/scan` と `odom` → `base_footprint` から `map` → `odom` のTFを配信する。
LiDARはシミュレーション専用で、実機のURDF（`sim:=false`）には含まれない。

## 固定地図のnavigation（シミュレーションのみ）

[localized_nav.launch.py](../overlay_ws/src/aha_navigation/README.md#固定地図での移動シミュレーションのみ)はSLAMの代わりに
map_serverが `/map`、AMCLが `map → odom` を配信する。SLAMと同時に起動しない。
Nav2は `/diff_drive_controller/odom` を読み、`/diff_drive_controller/cmd_vel`（TwistStamped）へ送信するため、
teleopなどから同時に速度指令を送らない。ゴールは `/navigate_to_pose`（`nav2_msgs/action/NavigateToPose`、frame `map`）。

## サブゴールタスクの実行

[aha_tasks](../overlay_ws/src/aha_tasks/README.md)の `task_runner` が提供する。runnerはロボット1台につき1つ。

| 用途 | 名前 | 型 |
| --- | --- | --- |
| タスクの状態 | `/aha/tasks/status` | `std_msgs/msg/String`（JSON） |
| 実行の操作 | `/aha/tasks/command` | `aha_msgs/srv/TaskCommand` |
| 地点の表示（frame `map`） | `/aha/tasks/locations` | `visualization_msgs/msg/MarkerArray` |
| Web GUI（`task_monitor`） | `http://localhost:8765/` | HTTP |

topicはいずれもRELIABLE / TRANSIENT_LOCAL / KEEP_LAST(1)。状態は変化のたびに全体を配信し、
形式は `aha_tasks/core/engine.py` の `TaskEngine.status()`（`schema_version: 1`）。
GUIのHTTP APIは `GET /api/status`、`GET /api/events`（SSE）、`POST /api/command`（`TaskCommand` と同じフィールドのJSON）。

## 頭部カメラ

| Topic | 型 |
| --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image`（rgb8） |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/camera/depth_registered/image_rect` | `sensor_msgs/msg/Image`（32FC1、m単位） |
| `/camera/depth/points` | `sensor_msgs/msg/PointCloud2` |

| 経路 | 起動 | 上3topicの配信元 |
| --- | --- | --- |
| シミュレーション | `sim.launch.py use_perception:=true` | Gazeboの `rgbd_camera`（640×480、15 Hz）を `ros_gz_bridge` で変換 |
| 実機カメラ | [カメラツール](../tools/perception/macos/pointcloud/README.md)、`camera_view.launch.py` | `stream_realsense.py` からrosbridge経由 |
| 実機の頭部とカメラ | `real_head_camera.launch.py camera:=true` | 同上 |

`/camera/depth/points` はどの経路も [pointcloud.launch.py](../overlay_ws/src/aha_perception/launch/pointcloud.launch.py)
の `depth_image_proc` が生成する。全topicは `camera_color_optical_frame` を使い、
RGB・CameraInfo・RGBへ位置合わせしたDepthは同じ時刻を持つ。
`stream_realsense.py` の3topicがrosbridgeへ要求するQoSはRELIABLE / VOLATILE / KEEP_LAST(5)。

TFは `link_head_tilt` → `camera_link` → `camera_color_optical_frame`（固定joint）で、
`robot_state_publisher` が配信する。定義は [sensors.xacro](../overlay_ws/src/aha_description/urdf/sensors.xacro)、
取付位置は [頭部ブラケット](../overlay_ws/src/aha_perception/hardware/head_cam_mount_d435i/README.md)の設計値。

## 手首カメラ（シミュレーション）

| Topic | 型 | frame |
| --- | --- | --- |
| `/camera/left_wrist/image_raw` | `sensor_msgs/msg/Image`（rgb8） | `left_wrist_camera_optical_frame` |
| `/camera/left_wrist/camera_info` | `sensor_msgs/msg/CameraInfo` | 同上 |
| `/camera/right_wrist/image_raw` | `sensor_msgs/msg/Image`（rgb8） | `right_wrist_camera_optical_frame` |
| `/camera/right_wrist/camera_info` | `sensor_msgs/msg/CameraInfo` | 同上 |

両カメラは640×480、15 Hz。画像とCameraInfoは同じ時刻・optical frameを持つ。
TFは `link_l6` / `link_r6` → `{left,right}_wrist_camera_link` →
`{left,right}_wrist_camera_optical_frame`（固定joint）で、robot_state_publisherが配信する。
仮想取付位置・光学特性・表示手順は [aha_perception](../overlay_ws/src/aha_perception/README.md#シミュレーション)を参照。
実機のURDFには手首カメラはない。

`sim.launch.py use_perception:=true` またはタスク評価の既定設定で
`camera_bridge.launch.py` が頭部と手首のカメラをbridgeする。
シミュレーションの頭部3topicはRELIABLE / VOLATILE / KEEP_LAST(10)、
手首の画像・CameraInfoはreliable（RELIABLE / VOLATILE / KEEP_LAST(10)）。
既存のsensor-data QoSの購読とも互換性がある。
`sim.launch.py cameras:=false` / 評価の `--no-cameras` は全カメラセンサを除く。

## 頭部の注視

`sim_view.launch.py`（`use_perception:=true` と `real_head_camera.launch.py` の既定 `rviz:=true` で起動）の
`head_look_at.py` が扱う。

| 用途 | Topic | 型 |
| --- | --- | --- |
| 注視点の指令（任意のframe） | `/aha/perception/look_at` | `geometry_msgs/msg/PointStamped` |
| 注視点と視線の表示 | `/aha/perception/look_at/markers` | `visualization_msgs/msg/MarkerArray` |

`head_look_at.py` は `/head_controller/joint_trajectory` へ軌道を送る。
RVizのPublish Point（`/clicked_point`）は `/aha/perception/look_at` へremapしている。

## 実機の頭部

`real_head_camera.launch.py` は [aha_servo](../overlay_ws/src/aha_servo/README.md) の
`servo_trajectory_bridge.py` で、シミュレーションと同じ名前・型の `/head_controller/joint_trajectory` と
`/head_controller/follow_joint_trajectory` を提供する。
頭の関節状態は `/head_controller/joint_states` から `joint_state_publisher` が `/joint_states` へ合流させ、
頭以外の関節は0を配信する。シミュレーションと同じ名前を使うため、同じ `ROS_DOMAIN_ID` で同時に起動しない。

## 独自型

[aha_msgs](../overlay_ws/src/aha_msgs/) に `DetectedObject` / `DetectedObjectArray`、
`SetNavGoal`、`PickObject`、`TaskCommand` の型定義がある。型定義だけでは対応するnodeやserverの起動を意味しない。
シミュレーションやカメラ経路の実際のtopic・型・QoSは起動後に `ros2 topic list -t` / `ros2 topic info -v` で確認する。
