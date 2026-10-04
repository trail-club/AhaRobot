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
グリッパは同ファイルの2関節の順序で位置を指定する。
`teleop_base.sh` はteleopの `/cmd_vel` を速度指令topicへremapする。
標準launchに `/cmd_vel` / `/odom` へのremapや `map` frameの配信はない。

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
| 実機カメラ | [カメラツール](../tools/perception/macos/README.md)、`camera_view.launch.py` | `stream_realsense.py` からrosbridge経由 |
| 実機の頭部とカメラ | `real_head_camera.launch.py camera:=true` | 同上 |

`/camera/depth/points` はどの経路も [pointcloud.launch.py](../overlay_ws/src/aha_perception/launch/pointcloud.launch.py)
の `depth_image_proc` が生成する。全topicは `camera_color_optical_frame` を使い、
RGB・CameraInfo・RGBへ位置合わせしたDepthは同じ時刻を持つ。
`stream_realsense.py` の3topicがrosbridgeへ要求するQoSはRELIABLE / VOLATILE / KEEP_LAST(5)。

TFは `link_head_tilt` → `camera_link` → `camera_color_optical_frame`（固定joint）で、
`robot_state_publisher` が配信する。定義は [sensors.xacro](../overlay_ws/src/aha_description/urdf/sensors.xacro)、
取付位置は [頭部ブラケット](../hardware/head_cam_mount_d435i/README.md)の設計値。

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
`SetNavGoal`、`PickObject` の型定義がある。型定義だけでは対応するnodeやserverの起動を意味しない。
シミュレーションやカメラ経路の実際のtopic・型・QoSは起動後に `ros2 topic list -t` / `ros2 topic info -v` で確認する。
