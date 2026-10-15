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
  move_groupとRVizはこれを読む。URDFの関節限界の外0.005 m / rad以内の値だけを限界へ補正する。
  MoveItは限界の外にある開始状態をわずかでも拒否するが、Gazeboは限界で止まった関節を約1e-14外側と報告し、
  移動中には限界の外へ数mm出ることもある。補正幅は軌道実行の開始位置の許容差（`allowed_start_tolerance` 0.01）より小さい。
- SRDFの登録姿勢は `ready`（腕を伸ばす）、`standby`（腕を畳み、手先を前へ水平、昇降0.2 m）、
  `carry`（同じく畳んだまま昇降0.5 m）。
- SRDFは上流Astraの衝突除外とAhaのベース・頭部の隣接リンクを記述する。
  外部物体の衝突形状は利用側がplanning sceneに追加する。標準worldの机等は自動登録しない。

## RViz操作画面

シミュレーション・MoveIt・RViz・操作パネルを1つのlaunchで起動する。

```bash
ros2 launch aha_bringup sim.launch.py use_manip:=true manip_rviz:=true
```

起動済みのシミュレーションとmove_groupに画面だけ追加・再起動する場合は
`ros2 launch aha_manipulation manip_view.launch.py`。画像は `use_perception:=true` か
`manip_view.launch.py camera_bridge:=true` でbridgeしたときに表示される（`manip_rviz:=true` は自動でbridgeする）。

| 表示・操作 | 内容 |
| --- | --- |
| MotionPlanning | 現在の姿勢、手先目標（橙）、計画した軌道（紫）。左のパネルで計画・実行する |
| `EefGoalError` | 直近の計画の終端の手先姿勢（座標軸）、現在の手先からの線、誤差 `err=<距離>mm/<角度>deg` |
| `HeadCamera` / `RightWristCamera` | `/camera/color/image_raw` / `/camera/right_wrist/image_raw` |
| 操作パネル | 台車・頭に加えて、右グリッパの開閉と右昇降の高さ。詳細は[aha_perception](../aha_perception/README.md#ノード) |

手先目標を指定して動かす手順:

1. MotionPlanningパネルの **Planning** タブで、Planning Groupが `right_arm`、Start Stateが `<current>` であることを確認する。
2. 3D表示の手先のマーカー（橙の腕の先）をドラッグする。球は画面と平行に、矢印は軸方向に、輪は軸周りに動かす。
   IKが解けた姿勢が橙で表示される。登録姿勢へ動かす場合はGoal Stateで `standby` / `carry` / `ready` を選ぶ。
3. **Plan** を押すと、軌道が紫の腕で繰り返し再生される。下部のTrajectory Sliderで途中の姿勢も確認できる。
4. 問題がなければ **Execute** を押す。計画と実行を続けて行う場合は **Plan & Execute**。
   Velocity / Accel. Scaling（既定0.5）で速度を下げられる。
5. 実行後、`EefGoalError` の誤差で目標への到達を確認する。

誤差を測る目標は、`/display_planned_path` に配信された直近の計画の終端である。
計画だけで実行していない場合も、この終端と現在の手先の差を表示する。
RViz以外（`/move_action` を使うスクリプト等）の計画でも更新される。
`eef_goal_monitor.py` が終端の関節値を `/compute_fk` で `base_footprint` 基準の `link_ree` 姿勢に変換し、
TFの現在姿勢と比べて `/aha/manipulation/eef_goal/markers` に描く。
RVizのテキストは日本語を描けないため、誤差は英数字で表示する。空白は環境により広く描かれるため使わない。

グリッパはMoveItの計画対象ではなく、操作パネルから位置指令で動かす。
パネルの昇降とMoveItの軌道実行は同じ `lift_controller` を使うため、同時に操作しない。

MoveIt 2.12.4ではcontroller pluginの破棄時に終了クラッシュが再現するため、
このバージョンのみmove_groupの `LD_PRELOAD` で同pluginを保持する。
[上流の終了時障害報告](https://github.com/moveit/moveit2/issues/3680)も参照。
他バージョンには適用しない。

実機の共通リフト軸とhardware interfaceへの対応は含まない。
