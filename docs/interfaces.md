# ROSインターフェース

標準シミュレーションと、macOSカメラ経路の入出力。controller設定は
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
グリッパは同ファイルの2関節の順序で位置を指定する。
`teleop_base.sh` はteleopの `/cmd_vel` を速度指令topicへremapする。
標準launchに `/cmd_vel` / `/odom` へのremapや `map` frameの配信はない。

## macOSカメラ経路

[カメラツール](../tools/perception/macos/README.md)で別途起動する。
`sim.launch.py use_perception:=true` はログを出すスタブで、この経路は起動しない。

| Topic | 型 | 配信元 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image`（rgb8） | macOSの `stream_realsense.py` |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 同上 |
| `/camera/depth_registered/image_rect` | `sensor_msgs/msg/Image`（32FC1、m単位） | 同上 |
| `/camera/depth/points` | `sensor_msgs/msg/PointCloud2` | コンテナの `depth_image_proc` |

RGB・CameraInfo・RGBへ位置合わせしたDepthは同じ時刻と `camera_color_optical_frame` を使用する。
このカメラframeとロボットのTF接続は未実装。
macOS側の3topicがrosbridgeへ要求するQoSはRELIABLE / VOLATILE / KEEP_LAST(5)。
[配信処理](../tools/perception/macos/stream_realsense.py)と
[変換launch](../overlay_ws/src/aha_perception/launch/camera_view.launch.py)を参照。

## 独自型

[aha_msgs](../overlay_ws/src/aha_msgs/) に `DetectedObject` / `DetectedObjectArray`、
`SetNavGoal`、`PickObject` の型定義がある。型定義だけでは対応するnodeやserverの起動を意味しない。
シミュレーションやカメラ経路の実際のtopic・型・QoSは起動後に `ros2 topic list -t` / `ros2 topic info -v` で確認する。
