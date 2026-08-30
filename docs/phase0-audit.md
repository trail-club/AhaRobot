# Phase 0 監査結果

対象 submodule commit は本 doc 追加時点。upstream 追従で SHA が進めば再監査する。

## 1. ROS 2 package 一覧 (upstream/)

| package | build type | 目的 | Jazzy 対応リスク |
| --- | --- | --- | --- |
| `astra_description` | ament_cmake | URDF / meshes / RViz / (旧) Gazebo launch | package format=2, `gazebo`/`gazebo_ros` (Classic) 依存 → `ros_gz` 系へ差し替え必要 |
| `astra_controller` | ament_python | 実機 driver (arm/lift/base/head/cam/teleop/ik/moveit_relay) | pyserial 依存, tty デバイス直叩き, ros2_control 非採用 |
| `astra_controller_interfaces` | ament_cmake | `msg/JointCommand.msg` のみ | 影響小 |
| `astra_moveit_config` | ament_cmake | MoveIt2 設定 | Humble 生成品, `gazebo_ros_control` comment out 済み, SRDF 再生成推奨 |

上流本体 (`hilookas/astra_ws`) の他 submodule で今回取り込んでいないもの:
`usb_cam`, `simple_cap` (video capture), `astra_teleop*`, `lerobot`, `aiortc`, `PyAV`,
`odrive-can`, `pyribbit`, `urchin`, `mr_urdf_loader`, `ModernRobotics`.
Phase 1〜3 の "Digital Skeleton / Nav / MoveIt" では不要。teleop / 学習が必要になった段階で追加 fork。

## 2. `astra_controller` の node 構成

`setup.py` entry_points:

- `arm_node`, `lift_node`, `base_node`, `head_node` — 左右アーム / 昇降 / 差動ベース / 頭部 (それぞれ tty デバイス経由)
- `cam_node` — カメラ
- `ik_node`, `moveit_relay_node` — IK / MoveIt 中継
- `teleop_node`, `teleop_web_node`, `dry_run_node`

各 node が `sensor_msgs/JointState` を独自に組み立てて publish する構造。
**ros2_control (controller_manager + Hardware Interface) は未採用** — Phase 1 で上位 API を統一するため、
overlay 側で `AstraSystem` (Hardware Interface) を新規実装し、これら node を段階的に置き換えるのが本筋。

## 3. URDF (`astra_description.urdf`) の内訳

- 23 links / 22 joints (fixed 4 含む)
- **モバイルベース (差動二輪) は URDF に無い** — `base_link` に直接 lift/head がぶら下がる
- `<sensor>` / `<gazebo>` / `<ros2_control>` / `<transmission>` タグ: **0 個**
- 全ての `<limit effort="0" velocity="0">` — Gazebo/ros2_control で使えない値

### joint 表

| joint | type | range | 備考 |
| --- | --- | --- | --- |
| `joint_r1` / `joint_l1` | prismatic | 0 – 1.2 m | 左右昇降 (実機は共通軸だが URDF では独立 2 軸) |
| `joint_r2..r6` / `joint_l2..l6` | revolute | ±1.57 / ±3.14 (r4/l4 のみ) | SCARA 型 5 自由度アーム |
| `joint_r7r` / `joint_r7l` | prismatic | ±0.06 m | 右グリッパ (2 指) |
| `joint_l7r` / `joint_l7l` | prismatic | ±0.06 m | 左グリッパ |
| `joint_r_ee` / `joint_r_ee_teleop` | fixed | — | EEF フレーム |
| `joint_l_ee` / `joint_l_ee_teleop` | fixed | — | EEF フレーム |
| `joint_head_pan` | revolute | ±1.57 | 頭部 pan |
| `joint_head_tilt` | revolute | ±3.14 | 頭部 tilt |

### Phase 1 で必要な URDF 追加作業

1. Xacro 化 (実機/sim, センサ有無を argument で切替)
2. 差動二輪ベース (`base_footprint` → `base_link`, wheel joints, caster) を追加
3. inertia 実測 or 概算値 / 凸分解 collision mesh
4. joint 全体に `effort` / `velocity` を実測値で設定
5. `<ros2_control>` 定義 + Gazebo 用 `gz_ros2_control` plugin
6. センサ (RGB-D, LiDAR, IMU) の `<gazebo><sensor>` と topic bridge

## 4. LICENSE

| repo | LICENSE ファイル | README 条項 |
| --- | --- | --- |
| `astra_description` | なし | package.xml で `BSD` (テンプレート値の可能性) |
| `astra_controller` | なし | package.xml で `TODO: License declaration` |
| `astra_controller_interfaces` | なし | 同上 |
| `astra_moveit_config` | なし | package.xml で `BSD` (MoveIt Setup Assistant 生成値) |
| `AstraFirmwares` | GPL-3.0 | **GPL-3.0 + 非商用条項** (README で明示) |
| `Astra_Hardwares` | GPL-3.0 | **GPL-3.0 + 非商用条項** (README で明示) |

**注意点:**
- ROS 2 package 4 本には LICENSE ファイルが無く、package.xml の記載も TODO/BSD テンプレート。
  元の `astra_ws` README (GPL-3.0 + 非商用) が super-repo として全体を覆う意図かは不明。
  fork 側 README で「本 fork では non-commercial research use に限る」旨を明記して防御する。
- Firmware / CAD の非商用条項は明確。RoboCup 参加自体は非商用研究用途で問題無し。
  スポンサー展示や将来の製品化前に upstream 著者に確認が必要。
- 派生物 (overlay で生成する URDF / launch / world 等) を公開する際は
  「どの upstream 由来か」を明記し、GPL-3.0 継承側とそれ以外を分離管理する。

## 5. Phase 1 直前の主要リスク

1. **ros2_control 不採用** → 実機とシミュレーションで API が揃わない。overlay 側で必ず統一する。
2. **base 未モデル化** → Nav2 に進めない。URDF に追加が必須。
3. **センサ未定義** → RoboCup 用 sensor suite (RGB-D / LiDAR / IMU / mic) の実機選定と URDF 追加を並行。
4. **MoveIt config が Humble 生成品** → Jazzy で再生成 (Setup Assistant) が現実的。
5. **LICENSE 未整備** → 派生物公開範囲を早めに決めないと Phase 4 で困る。
