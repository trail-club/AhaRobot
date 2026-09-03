# AhaRobot 班間 Interface ガイド

navigation / manipulation / perception 3班が並行開発するときの **共有メモ**。
「まず参照する場所」を決めておくのが目的で、厳密なルールではない。
食い違いが出たら Issue か Slack で相談し、決まったらここを更新する。

- 対象: ROS 2 Jazzy
- 適用: `overlay_ws/src/aha_*`

---

## 1. TF Tree（想定）

```
map
 └── odom                       ← nav 系 (slam_toolbox / amcl)
      └── base_footprint        ← EKF or ground truth
           └── base_link        ← URDF fixed
                ├── left_wheel_link, right_wheel_link, caster_link
                ├── (arm chain) link_l1..l7 / link_r1..r7
                ├── head_pan_link → head_tilt_link
                │    └── camera_link
                │         └── camera_optical_frame   ← REP-103 optical
                └── (lidar_link) ← 実機搭載時
```

REP-105 / REP-103 に沿った命名を基本にする。新しい frame を足したくなったら
`aha_description/urdf/` に入れて、ここにも追記する（できれば）。

## 2. Topic 命名の目安

| 用途 | Topic | Type |
|---|---|---|
| 速度指令 | `/cmd_vel` | `geometry_msgs/TwistStamped`（Nav2 Jazzy 標準） |
| Wheel odom | `/odom` | `nav_msgs/Odometry` |
| Joint状態 | `/joint_states` | `sensor_msgs/JointState` |
| LiDAR | `/scan` | `sensor_msgs/LaserScan` |
| RGB | `/camera/color/image_raw` | `sensor_msgs/Image` |
| Depth | `/camera/depth/image_rect_raw` | `sensor_msgs/Image` |
| Camera info | `/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| Map | `/map` | `nav_msgs/OccupancyGrid` |
| 検出物体 | `/aha/perception/objects` | `aha_msgs/DetectedObjectArray` |
| 検出人物 | `/aha/perception/people` | `aha_msgs/DetectedObjectArray` |

- 生 sensor は vendor 標準寄せ (`/camera/*`, `/scan`, `/imu/*`)
- 班固有の内部/デバッグ topic は `/aha/<squad>/*` に置くと衝突しにくい

## 3. Action / Service（現時点の想定）

| 名前 | Type | 用途 |
|---|---|---|
| `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Nav2 標準 |
| `set_nav_goal` | `aha_msgs/srv/SetNavGoal` | 上位タスクからの薄いラッパ |
| `pick_object` | `aha_msgs/action/PickObject` | 掴む一連の動作 |
| `move_group` | `moveit_msgs/action/MoveGroup` | MoveIt 標準 |

## 4. Frame ID 命名

- `map` / `odom` / `base_footprint` / `base_link` — REP-105
- カメラ: `<camera>_link`（機械的）/ `<camera>_optical_frame`（光学系, REP-103）
- 検出結果は原則 `camera_optical_frame` で出す。map/base_link への変換は consumer 側で TF

## 5. QoS の目安

- **sensor_data**: `BEST_EFFORT`, `KEEP_LAST(5)` — image / depth / scan / odom / joint_states
- **reliable**: `RELIABLE`, `KEEP_LAST(10)` — cmd_vel, detections, service response
- **transient_local**: latch — map, static_tf, camera_info

subscriber 側で publisher に合わせる。詰まったら `ros2 topic info -v` で確認。

## 6. Node / パラメータ命名の目安

- 班 node は `aha_<squad>_<機能>` にすると grep しやすい
  - 例: `aha_nav_docking`, `aha_perception_yolo`, `aha_manip_pick_server`

## 7. Launch 統合ポイント

`aha_bringup/launch/sim.launch.py` から各班の launch を include する形を想定:

```python
IncludeLaunchDescription(...aha_navigation/launch/nav2.launch.py, condition=IfCondition(use_nav))
IncludeLaunchDescription(...aha_manipulation/launch/move_group.launch.py, condition=IfCondition(use_manip))
IncludeLaunchDescription(...aha_perception/launch/detection.launch.py, condition=IfCondition(use_perception))
```

launch arg: `use_nav`, `use_manip`, `use_perception`（デフォルト false）。

## 8. 変更するとき

- topic 名 / msg フィールド / frame 名を変える時は、他班の影響が出やすいので一言相談する
- 決まったらこのドキュメントを更新
- 既存名を消すときは、しばらく alias を残せると安全
