# overlay_ws

チーム独自コード置き場。`upstream/*` を触らずに、上から機能を追加していく。

## パッケージ

| package | 役割 |
| --- | --- |
| `aha_description` | 上流 URDF を xacro で wrap し、差動二輪ベース / `<ros2_control>` / (将来) センサを追加 |
| `aha_gazebo` | Gazebo Harmonic world と Gazebo 起動 launch |
| `aha_bringup` | sim / real の統合起動 launch |

## 前提

- Ubuntu 24.04
- ROS 2 Jazzy (`sudo apt install ros-jazzy-desktop`)
- Gazebo Harmonic (`sudo apt install ros-jazzy-ros-gz`)
- `ros-jazzy-gz-ros2-control` `ros-jazzy-joint-trajectory-controller` `ros-jazzy-diff-drive-controller` `ros-jazzy-joint-state-broadcaster` `ros-jazzy-forward-command-controller` `ros-jazzy-xacro`

`rosdep` でまとめて入れる:

```bash
cd <repo root>
rosdep install --from-paths upstream overlay_ws/src --ignore-src -r -y
```

## ビルド

```bash
cd overlay_ws
colcon build --symlink-install --packages-up-to aha_bringup
source install/setup.bash
```

`upstream/astra_description` などが依存関係で必要な場合は super repo の `upstream/` を
`AMENT_PREFIX_PATH` / `COLCON_PREFIX_PATH` で見えるようにする。最も簡単には super repo 直下で:

```bash
colcon build --symlink-install \
  --paths upstream/astra_description upstream/astra_controller_interfaces \
  --paths overlay_ws/src/aha_description overlay_ws/src/aha_gazebo overlay_ws/src/aha_bringup
```

または単一ワークスペースに揃える運用として、`overlay_ws/src` の中に `upstream/*` への
シンボリックリンクを張っても良い (upstream の git submodule 実体はそのまま)。

## 起動

**URDF を RViz で確認 (Gazebo 不要):**

```bash
ros2 launch aha_description view_robot.launch.py
```

**Gazebo Harmonic で sim 全体を起動:**

```bash
ros2 launch aha_bringup sim.launch.py
```

引数:

- `world:=empty.sdf` (default)
- `use_sim_time:=true` (default)

**動作確認:**

```bash
# 前進コマンド
ros2 topic pub /diff_drive_controller/cmd_vel geometry_msgs/msg/TwistStamped \
  '{twist: {linear: {x: 0.2}}}' -r 10

# joint 状態
ros2 topic echo /joint_states --once

# 頭部を少し動かす
ros2 action send_goal /head_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{trajectory: {joint_names: [joint_head_pan, joint_head_tilt],
                 points: [{positions: [0.3, 0.0], time_from_start: {sec: 1}}]}}'
```

## 既知の制限 (Phase 1 の残タスク)

- 車輪寸法 / キャスターオフセットは仮の値 (実測して置換)
- 上流 URDF の `<limit effort=0 velocity=0>` は未修正 (controller 側 joint_limits で防御予定)
- 左右昇降を 2 DoF として扱っている (実機は共通軸; 後で mimic か融合コントローラ化)
- センサ (RGB-D / LiDAR / IMU) 未実装
- 実機 Hardware Interface (`aha_hardware/AhaSystem`) 未実装 → `sim:=false` は起動できない
