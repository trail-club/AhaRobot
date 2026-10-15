# aha_manipulation

AhaRobot右腕のMoveIt 2設定。IKはKDL、経路計画はOMPLのRRTConnectを使い、
時間付けした軌道を既存のros2_controlへ送る。

## 起動

[開発コンテナ](../../../docs/docker.md)内でビルド後、標準シミュレーションに追加する。

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_bringup aha_manipulation
source install/setup.bash
ros2 launch aha_bringup sim.launch.py use_manip:=true
```

起動済みのシミュレーションへ追加する場合は
`ros2 launch aha_manipulation manip.launch.py use_sim_time:=true`。
重複して起動しない。controllerがactiveになってから計画・実行する。
単独起動の `cameras` 引数はシミュレーション側に合わせる（既定 `true`）。MoveItのサービス・actionは[ROSインターフェース](../../../docs/interfaces.md#moveit)を参照。
真値による把持の実行は[aha_sim_tasks](../aha_sim_tasks/README.md#pick-with-ground-truth-ik)を参照。
必要な依存パッケージは `package.xml` に定義する。既存コンテナへ依存を追加する場合は
workspaceで `rosdep install --from-paths src --ignore-src -r -y` を実行する。

## モデルと実行

- URDFは `aha_description/urdf/aha_robot.urdf.xacro`、意味情報は [SRDF](srdf/aha_robot.srdf)。
- `right_arm` は `base_footprint` → `link_ree` の直列チェーン。
  `joint_r1`（右リフト）と `joint_r2`～`joint_r6`を解く。ベース・左腕・頭部は計画対象に含めない。
- [kinematics.yaml](config/kinematics.yaml)がKDL、[ompl.yaml](config/ompl.yaml)が計画・検証・時間付けを設定する。
  速度制限はURDF、加速度制限は [joint_limits.yaml](config/joint_limits.yaml)。
- [controllers.yaml](config/controllers.yaml)がMoveItSimpleControllerManagerと
  `right_arm_controller` / `lift_controller` の `FollowJointTrajectory` actionを接続する。
  右リフトのみの軌道を受け取れるよう、lift controllerは部分関節goalを許可する。
  左リフトは既存位置を保持する。グリッパは既存のposition topicを使う。
- 既定の状態入力は `/joint_states`。`joint_states_topic` で変更でき、標準bringupでは
  `manip_joint_states_topic` が対応する。真値評価は `/evaluation/joint_states` を指定する。
  `joint_state_bounds.py` がこの入力を `/aha/manipulation/joint_states` へ中継し、
  move_groupはこれを読む。URDFの関節限界の外 `limit_tolerance` 以内の値だけを限界へ補正し、
  それより大きい逸脱はそのまま渡す。MoveItは限界の外にある開始状態をわずかでも拒否するが、
  限界で止まった関節は限界のわずかに外側と報告される（Gazeboでは約1e-14）。
  `limit_tolerance` の既定は数値誤差分の1e-6 m / rad。
  MoveItは補正した状態から計画し、実行開始時の確認もこの状態で行う。
  軌道の始点と実際の位置の差（最大 `limit_tolerance`）は、軌道のcontrollerが実行の始めに詰める。
- SRDFは上流Astraの衝突除外とAhaのベース・頭部の隣接リンクを記述する。
  外部物体の衝突形状は利用側がplanning sceneに追加する。標準worldの机等は自動登録しない。

MoveIt 2.12.4ではcontroller pluginの破棄時に終了クラッシュが再現するため、
このバージョンのみmove_groupの `LD_PRELOAD` で同pluginを保持する。
[上流の終了時障害報告](https://github.com/moveit/moveit2/issues/3680)も参照。
他バージョンには適用しない。

実機の共通リフト軸とhardware interfaceへの対応は含まない。
