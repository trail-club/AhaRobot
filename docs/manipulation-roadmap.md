# Manipulation 班 開発ロードマップ

対象パッケージ: `overlay_ws/src/aha_manipulation/`
関連 msg: `overlay_ws/src/aha_msgs/action/PickObject.action`
班間 interface: [`docs/interfaces.md`](interfaces.md)

---

## 現状サマリ

| 項目 | 状態 |
|---|---|
| URDF (両腕 5 DoF + lift + head + gripper) | ✅ Gazebo で spawn 済 |
| ros2_control (JointTrajectoryController × 4, ForwardCommandController × 2) | ✅ sim で active |
| `/joint_states` 配信 | ✅ 50Hz |
| MoveIt 一式 | ❌ 未 (`ros-jazzy-moveit*` すら未インストール) |
| SRDF / kinematics.yaml | 🟡 upstream に Humble 版あり (`upstream/astra_moveit_config/`) — Jazzy 対応で書き直し要 |
| move_group ノード起動 | ❌ 未 |
| Pick / Place logic | ❌ 未 (`pick_object.action` msg のみ) |
| Perception 連携 (`/aha/perception/objects` → PlanningScene) | ❌ 未 |
| 実機連携 (astra_controller hardware_interface) | ❌ 未 |

---

## Phase M1: MoveIt を最低限動かす

**ゴール**: RViz の MotionPlanning plugin で右腕をマウス操作 → Plan → Execute で Gazebo 内の腕が動く。

- [ ] `docker/Dockerfile` に MoveIt 一式を追加 (`ros-jazzy-moveit`, `ros-jazzy-moveit-py`, `ros-jazzy-moveit-setup-assistant`, `ros-jazzy-moveit-resources`) → イメージ rebuild
- [ ] MoveIt Setup Assistant で SRDF を新規作成
  - `ros2 launch moveit_setup_assistant setup_assistant.launch.py`
  - URDF: `aha_description/urdf/aha_robot.urdf.xacro`
  - Planning groups: `right_arm`, `left_arm`, `right_hand`, `left_hand`, `head` (`both_arms` は要検討)
  - **判断ポイント**: `joint_l1`/`joint_r1` (lift 共有軸) をどのグループに含めるか
- [ ] 生成物を `aha_manipulation/config/` に配置 (SRDF, kinematics.yaml, joint_limits.yaml, moveit_controllers.yaml, ompl_planning.yaml)
- [ ] `aha_manipulation/launch/move_group.launch.py` を実装 (`manip.launch.py` スタブを置き換え)
- [ ] `aha_bringup sim.launch.py use_manip:=true` で move_group が起動することを確認
- [ ] `aha_manipulation/rviz/manip.rviz` を MotionPlanning display 対応に更新
- [ ] RViz からマウス操作で右腕の Plan/Execute 成功を確認

**成果物**: 「pose を指定 → 腕が動く」が sim で成立。

---

## Phase M2: スクリプトで動かす

**ゴール**: Python で任意の pose に腕を動かせる。

- [ ] `aha_manipulation/scripts/demo_move.py` — moveit_py で `right_arm` を home / 特定 pose に動かすサンプル
- [ ] `aha_manipulation/scripts/demo_gripper.py` — gripper open/close を JointTrajectory topic で送るサンプル
- [ ] home pose / ready pose を SRDF の `group_state` として定義

**成果物**: 開発者が「python3 demo_move.py」で腕を動かせる。

---

## Phase M3: pick_object action server

**ゴール**: `pick_object` action を送ると、Gazebo 内の bottle を実際に持ち上げる。

- [ ] `aha_manipulation/pick_server/` (Python) — `aha_msgs/action/PickObject` の server 実装
  - approach (物体の上 10cm へ移動)
  - grasp pose へ Cartesian path で直線移動
  - gripper close
  - attach (Planning Scene の `attached_collision_object` に切替)
  - lift (10cm 持ち上げ)
  - retreat
- [ ] grasp pose 生成: 当初は「物体真上から掴む」ハードコード。GraspNet などは Phase M5 で
- [ ] `aha_manipulation/launch/pick.launch.py` で server 起動
- [ ] Gazebo 内 bottle をダミー DetectedObject として goal に渡すテスト script
- [ ] 実機性能を意識した速度・加速度制約を joint_limits.yaml に設定

**成果物**: `ros2 action send_goal /pick_object aha_msgs/action/PickObject "..."` で bottle が持ち上がる。

---

## Phase M4: Perception 連携 + Planning Scene 動的更新

**ゴール**: perception が publish する物体が MoveIt の障害物として認識され、それを避けて経路生成される。

- [ ] `/aha/perception/objects` (aha_msgs/DetectedObjectArray) を subscribe → PlanningScene に `CollisionObject` として反映するブリッジ node
- [ ] object frame → planning frame (base_link) の TF 解決
- [ ] 把持で `attach` / 離す時に `detach` を自動化
- [ ] perception 側のダミー publisher と組み合わせた e2e テスト

**成果物**: perception → manipulation の完全なパイプラインが sim で流れる。

---

## Phase M5: 高度化 (競技会が近づいたら)

- [ ] 双腕協調動作 (`both_arms` group、両腕でトレイを持つ 等)
- [ ] Cartesian path constraint (orientation を保ったまま移動 = 水をこぼさない)
- [ ] Grasp pose 生成の高度化 (GraspNet / anygrasp / GPD)
- [ ] Impedance / compliance control (押し引き動作)
- [ ] 実機 hardware_interface 対応 (upstream `astra_controller` との統合)
- [ ] 障害物回避性能改善 (プランナ tuning、TRAC-IK / BioIK への切替検討)

---

## 早めに決めておくべき設計判断

Phase M1 の途中で議論し、決まったら [`docs/interfaces.md`](interfaces.md) にも反映:

1. **IK plugin**: KDL / TRAC-IK / BioIK — 5 DoF なので KDL は苦しい可能性
2. **Planning group 構成**: 左右独立 or `both_arms` も定義するか
3. **Lift 軸 (joint_l1/r1) の扱い**: どの group に含めるか / 独立 group にするか
4. **開発言語**: Python (moveit_py) メイン or C++ (MoveGroupInterface) — 速度は C++、開発速度は Python
5. **Grasp pose の生成方針**: 幾何ベース (hard-code) / GraspNet / anygrasp / GPD

---

## 契約 (他班との interface)

[`docs/interfaces.md`](interfaces.md) より抜粋。変更したい場合は先にそちらの PR を立てる。

**Subscribe**:
- `/joint_states` (sensor_msgs/JointState)
- `/aha/perception/objects` (aha_msgs/DetectedObjectArray) — Phase M4 で必要
- `/tf`, `/tf_static`

**Publish / Action server**:
- `pick_object` action (aha_msgs/action/PickObject) — 班のメイン API
- `move_group` action (moveit_msgs/action/MoveGroup) — MoveIt 標準
- `/*_arm_controller/follow_joint_trajectory` action (control_msgs) — 内部利用

**Planning frame**: `base_link`

---

## 環境準備 (最初にやる)

コンテナに MoveIt 一式が入っていないので、まず入れる:

```bash
# コンテナ内で一時的に
sudo apt update
sudo apt install ros-jazzy-moveit \
                 ros-jazzy-moveit-py \
                 ros-jazzy-moveit-setup-assistant \
                 ros-jazzy-moveit-resources
```

恒久対応として `docker/Dockerfile` に追加 → イメージ rebuild → PR。→ **Phase M1 の最初のタスク**。

---

## 参考リソース

- **MoveIt 2 Concepts**: https://moveit.picknik.ai/main/doc/concepts/concepts.html
- **MoveIt 2 Tutorials**: https://moveit.picknik.ai/main/index.html — Getting Started, Setup Assistant, Move Group Python/C++ Interface, Pick and Place
- **MoveIt Setup Assistant**: `ros2 launch moveit_setup_assistant setup_assistant.launch.py`
- **ros2_control docs**: https://control.ros.org/
- **PickNik YouTube**: MoveIt 2 チュートリアル動画

Astra 独自:
- `upstream/astra_moveit_config/` — Humble 時代の SRDF (`right_arm` のみ、参考程度)
- `upstream/astra_controller/` — 実機 hardware_interface (Phase M5 で参照)
