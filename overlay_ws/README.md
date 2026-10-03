# overlay_ws

チーム独自コード置き場。`upstream/*` を触らずに、上から機能を追加していく。

## パッケージ

| package | 役割 |
| --- | --- |
| `aha_description` | 上流 URDF を xacro で wrap し、差動二輪ベース / `<ros2_control>` / (将来) センサを追加 |
| `aha_gazebo` | SOBITS Japan Open world準備、Gazebo起動、回帰テスト |
| `aha_bringup` | sim / real の統合起動 launch |

## 前提

- Ubuntu 24.04
- ROS 2 Jazzy (`sudo apt install ros-jazzy-desktop`)
- Gazebo Harmonic (`sudo apt install ros-jazzy-ros-gz`)
- `ros-jazzy-gz-ros2-control` `ros-jazzy-joint-trajectory-controller` `ros-jazzy-diff-drive-controller` `ros-jazzy-joint-state-broadcaster` `ros-jazzy-forward-command-controller` `ros-jazzy-xacro`

`rosdep` でまとめて入れる:

```bash
cd <repo root>
rosdep install --from-paths overlay_ws/src \
  upstream/astra_description upstream/astra_controller_interfaces \
  upstream/sobits_gazebo_worlds upstream/tmc_wrs_gz/tmc_wrs_gz_worlds \
  --ignore-src -r -y --skip-keys "gz_human_sim sobits_interfaces"
```

## ビルド

```bash
# 開発コンテナ内で
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_bringup
source install/setup.bash
```

依存は `git clone --recursive` で取得する。開発コンテナがworkspaceへリンクし、通常ビルドに含める。
[Japan Openの起動設定](../docs/sobits-rcjo2026.md)を参照。

## 動作確認済み (2026-08-30)

以下はデフォルトをJapan Openへ変更する前の空worldでの確認結果。

Docker (macOS/Apple Silicon) で以下を確認:
- `colcon build` 成功 (astra_description / aha_description / aha_gazebo / aha_bringup)
- `ros2 launch aha_bringup sim.launch.py headless:=true` で:
  - 8 controllers すべて active
  - `/joint_states` @ 100 Hz
  - `/diff_drive_controller/cmd_vel` に 0.3 m/s → `/odom` が前進を報告
  - `ros2 run aha_bringup demo_arms.sh` で頭 / 左右腕 / 昇降 / グリッパが指令値通り動く

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

- `world:=rcjo2026` (default) — SOBITS Japan Open 2026、spawn `(-2.0, 1.5, 0.10)` m
- `world:=empty.sdf` — 空world、spawn `(0.0, 0.0, 0.05)` m
- `world_path:=/absolute/path/world.sdf` — 展開済みSDFで上書き、空worldと同じ既定spawn
- `spawn_x` / `spawn_y` / `spawn_z` / `spawn_yaw` — 初期位置の上書き
- `use_sim_time:=true` (default)
- `headless:=true` — GUI 無し (macOS / CI 推奨)

**ベースをキーボードで走らせる (別シェルで):**

```bash
docker exec -it aharobot-aha_project-1 bash
aha_teleop   # i/j/k/l/, で操作
```

**腕・頭・グリッパのデモ:**

```bash
aha_demo     # 頭 pan +0.4, 左右腕を対称ポーズ, 昇降 +0.2m, 右グリッパ open
```

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
