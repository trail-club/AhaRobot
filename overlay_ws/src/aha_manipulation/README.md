# aha_manipulation

MoveIt 2 設定と pick/place ロジックを集約する。

## ディレクトリ

- `srdf/` — MoveIt 用 semantic description（`upstream/astra_moveit_config` から必要分を移植予定）
- `config/` — `kinematics.yaml`, `ompl_planning.yaml`, `moveit_controllers.yaml`
- `launch/` — `move_group.launch.py`, `demo.launch.py`
- `rviz/` — `manip.rviz`

## 契約

- **subscribe**: `/joint_states`, `/aha/perception/objects` (aha_msgs/DetectedObjectArray)
- **publish**: `/*_arm_controller/joint_trajectory`（joint_trajectory_controller 経由）
- **action server**: `pick_object` (aha_msgs/action/PickObject)
- **planning frame**: `base_link`（[`docs/interfaces.md`](../../../docs/interfaces.md)）

## TODO

- [ ] Astra 両腕 SRDF を Jazzy 対応に更新
- [ ] `pick_object` action server 実装
- [ ] Gazebo での grasp シミュレーション動作確認
