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
  move_groupとRVizはこれを読む。URDFの関節限界の外 `limit_tolerance` 以内の値だけを限界へ補正し、
  それより大きい逸脱はそのまま渡す。MoveItは限界の外にある開始状態をわずかでも拒否するが、
  限界で止まった関節は限界のわずかに外側と報告される（Gazeboでは約1e-14）。
  `limit_tolerance` の既定は数値誤差分の1e-6 m / rad。
  MoveItは補正した状態から計画し、実行開始時の確認もこの状態で行う。
  軌道の始点と実際の位置の差（最大 `limit_tolerance`）は、軌道のcontrollerが実行の始めに詰める。
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
`use_perception:=true` と併用すると、点群用のRViz（aha_perceptionの `sim_view`）も開く。
操作パネルはこの画面のものだけを起動する。点群用のRVizが不要なら `rviz:=false` を加える。

| 表示・操作 | 内容 |
| --- | --- |
| MotionPlanning | 現在の姿勢、手先目標（橙）、計画した軌道（紫）。左のパネルで計画・実行する |
| `EefGoalError` | 直近の計画の終端の手先姿勢（座標軸）、現在の手先からの線、誤差 `err=<距離>mm/<角度>deg` |
| `HeadCamera` / `RightWristCamera` | `/camera/color/image_raw` / `/camera/right_wrist/image_raw` |
| 操作パネル | 台車・頭に加えて、左右のグリッパの開閉と昇降の高さ。詳細は[aha_perception](../aha_perception/README.md#ノード) |

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
終端の手先姿勢を求められなかった計画では、前の計画の目標を表示し続けないよう表示を消す。
`eef_goal_monitor.py` が終端の関節値を `/compute_fk` で `base_footprint` 基準の `link_ree` 姿勢に変換し、
TFの現在姿勢と比べて `/aha/manipulation/eef_goal/markers` に描く。
RVizのテキストは日本語を描けないため、誤差は英数字で表示する。空白は環境により広く描かれるため使わない。

グリッパはMoveItの計画対象ではなく、操作パネルから位置指令で動かす。
左手はMoveItの計画対象ではなく、パネルからグリッパと昇降だけを動かす。
パネルの左右の昇降とMoveItの軌道実行は同じ `lift_controller` を使うため、同時に操作しない。

RVizの開始状態 `<current>` のうち、`right_arm` の関節（昇降 `joint_r1` を含む）は、
RVizからの実行が終わったときとStart Stateを選び直したときだけ現在値に更新される（MoveItのRViz pluginの仕様）。
パネルで昇降を動かした後は **Plan & Execute** を使うか、Start Stateを一度別の項目にして `<current>` に戻してから **Plan** する。
古い開始状態で計画した軌道を **Execute** すると、`start point deviates from current robot state` で中断される。

MoveIt 2.12.4ではcontroller pluginの破棄時に終了クラッシュが再現するため、
このバージョンのみmove_groupの `LD_PRELOAD` で同pluginを保持する。
[上流の終了時障害報告](https://github.com/moveit/moveit2/issues/3680)も参照。
他バージョンには適用しない。

実機の共通リフト軸への対応は含まない。

## 実機の腕の軌道実行

`aha_bringup` の `arms.launch.py` が左右の `arm_node` と一緒に `arm_trajectory_bridge.py` を起動する
（`trajectory_bridge:=false` で止める）。ブリッジはJointTrajectoryControllerと同じ
`/<side>_arm_controller/joint_trajectory` と `follow_joint_trajectory` を受け、
`arm_node` の `/<side>/arm/joint_command` へ5関節の位置指令を `rate_hz`（50 Hz）で送る。
関節状態は `arm_node` の `/<side>/arm/joint_states` から読む（`/joint_states` には他のnodeも腕の関節を載せるため）。
`arm_node` は指令ごとに腕の基板へ目標を書き、基板が台形速度で目標へ動かす。

昇降も同じスクリプトを [config/lift.yaml](config/lift.yaml) で起動する（node名 `aha_lift_bridge`）。
`arms.launch.py` は `use_<side>_lift:=true` で左右の `lift_node` とこのブリッジを起動する。
`lift_node` のデバイス名の既定（`/dev/tty_puppet_lift_<side>`）はupstream AstraFirmwaresのudev規則の名前で、
違うときは `<side>_lift_device` で渡す。
`/lift_controller/joint_trajectory` と `follow_joint_trajectory`（`joint_l1` / `joint_r1`、m）を受け、
左右の `lift_node` の `/<side>/lift/joint_command` へ1関節ずつ送り、`/<side>/lift/joint_states` を読む。
軌道が名前を含む側の昇降だけに指令し、その側の関節状態だけを必要とするため、片側の昇降だけでも動く。
制限は昇降の基板（AstraLiftController）に合わせた0〜1.19 m、0.3 m/s、0.1 m/s²。
基板はAccelStepperの `moveTo` で目標へ動き、目標で止まれる速さまでしか出さないため、補間の現在値をそのまま送ると
v²/(2a) 遅れる。速度付きの軌道（MoveItの軌道）では、昇降（`board_acceleration` を設定した関節）に
補間の現在値より進行方向へ v²/(2 `board_acceleration`) 先の位置を送る。直線補間の軌道は速度が不連続で先回りしない。
フィードバックと許容誤差は補間の現在値で判定する。

- 軌道は [arm_plan.py](aha_manipulation/arm_plan.py) が実測の姿勢から補間する。
  全点に速度があれば3次Hermite、なければ直線で、静止から始まり静止で終わる。
  3次Hermiteでは、点の間の速度・加速度の最大値も制限と比べる。
  時刻0の点は現在の姿勢とみなし、`start_tolerance` 以内であることを確認する。
  実測との差は軌道の形を変えずに上乗せし、最終点の時刻で0になるよう直線的に減らす
  （最初の区間が短くても、差が急な加速度にならない）。
- 関節の位置・速度・加速度の制限を超える軌道は全体を拒否する（actionはgoalを受け付けず、topicは無視する）。
  制限は [config/arm.yaml](config/arm.yaml) と [config/lift.yaml](config/lift.yaml)。腕の位置の制限は `arm_node` の可動域
  （[パッチ](../../../patches/README.md)の `ARM_PROFILES`）に合わせる。`arm_node` は1関節でも範囲外の指令を捨てる。
- actionは、実測が補間の現在値から `path_tolerance` を超えて離れたとき、または関節状態が `state_timeout` 途絶えたときに中断する。
  最終点から `goal_tolerance` 以内に入れば成功、最終点の時刻から `goal_time` 秒以内に入らなければ中断する。
  goalの `path_tolerance` / `goal_tolerance` / `goal_time_tolerance` が指定されていればそれを使う。
- 取り消し・許容誤差による中断では、そのgoalの関節の指令グループに、空の軌道では全グループに、実測の位置を指令して止める
  （昇降は実測から、実行中の軌道の速度で止まれる位置）。
  関節状態の途絶では指令を止める。
  新しい軌道は実行中のgoalを中断する。実行中の軌道の指令グループのうち、新しい軌道が名前を含まないものは実測の位置で保持する
  （JointTrajectoryControllerが新しいgoalに含まれない関節を保持するのと同じ）。
- 起動時と待機中は指令を送らない。関節状態が届くまで軌道を受け付けない。

MoveItの右腕の計画はjoint_r1（昇降）を含み、`lift_controller` へも送られる。
MoveItの `joint_limits.yaml` の昇降の加速度（0.5 m/s²）は昇降の基板（0.1 m/s²）より大きく、
昇降を大きく動かす計画は加速度の倍率を下げないとブリッジに拒否される。
