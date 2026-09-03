# aha_navigation

SLAM (slam_toolbox) と Nav2 の設定・launch を集約する。

## ディレクトリ

- `launch/` — `slam.launch.py`, `nav2.launch.py`, `bringup.launch.py`（未実装）
- `config/` — `nav2_params.yaml`, `slam_toolbox_params.yaml`, EKF params
- `maps/` — 保存済み地図 (`.yaml` + `.pgm`)
- `rviz/` — `nav.rviz`

## 契約（他班との interface）

- **subscribe**: `/scan` (sensor_msgs/LaserScan), `/odom` (nav_msgs/Odometry), `/tf`
- **publish**: `/cmd_vel` (geometry_msgs/Twist), `/map` (nav_msgs/OccupancyGrid)
- **action**: `navigate_to_pose` (nav2_msgs/action/NavigateToPose)
- **service**: `set_nav_goal` (aha_msgs/srv/SetNavGoal) — task planner 用薄いラッパ

frame 規約は [`docs/interfaces.md`](../../../docs/interfaces.md) を参照。

## TODO

- [ ] Nav2 params の初期セット
- [ ] slam_toolbox online_async config
- [ ] `robot_localization` EKF (wheel odom + IMU)
- [ ] docking behavior
