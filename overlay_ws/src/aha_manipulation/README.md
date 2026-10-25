# aha_manipulation

manipulation用のROSパッケージ。実機アーム（Astra）を ROS のモーションインターフェースに接続する。

## 起動

```bash
ros2 launch aha_manipulation real_arm.launch.py \
    device_left:=/dev/ttyUSB0 device_right:=/dev/ttyUSB1
```

左右それぞれの `arm_node`（upstream astra_controller、`/<side>/arm` namespace）と、
それを `/<side>_arm_controller` インターフェース（sim の JointTrajectoryController と
同じ名前・型）で使う `aha_<side>_arm_bridge` を起動する。
`sides:=left` で片側、`use_bridge:=false` で arm_node だけ起動。
`device_:=/dev/null` ならシリアル接続なしで起動する（フィードバック待ちで停止する）。

`launch/manip.launch.py` はログを出すスタブ（MoveIt等は未起動）。

## ブリッジ（`arm_trajectory_bridge.py`）

設定は `config/arm.yaml`（各sideのノードは `aha_<side>_arm_bridge`）。

- 入力: `/<controller>/joint_trajectory`（fire&forget）、
  `/<controller>/follow_joint_trajectory`（action）。
  出力: `/<controller>/joint_states`。
- upstream `arm_node` は受けた目標をそのままサーボに書き込むだけで、
  firmware のP制御が目標へ追従するのみ。目標1つではアームが突進するため、
  運動整形はすべてここで行う: `arm_plan`（`aha_manipulation/arm_plan.py`）が軌道を
  キーフレーム化し（関節速度 `max_speed` rad/s を超えないよう時間を引き伸ばし、
  関節制限でクランプ）、ブリッジが `rate_hz` 周期で `position_at(t)` をサンプリングして
  `/<side>/arm/joint_command` へ位置指令を配信する。
- `closed_loop: true`（既定）では `arm_node` の関節フィードバックで到達を確認し、
  `goal_tolerance` 内かつ `goal_timeout` 以内なら success。
  状態が `state_timeout` 以上途切れた場合は最後の指令値にフォールバック。
  `closed_loop: false` では計画完了 `settle_time` 後に success。
- キャンセル / `/<controller>/release` service は動作を止め、最後の安全位置を再発行する。

## テスト

```bash
pytest test/test_arm_plan.py test/test_arm_bridge.py
```

`test_arm_plan` は純Pythonの単体テスト。
`test_arm_bridge` は `fake_arm`（目標を `max_rate` rad/s でスローする arm_node 代替）と
別プロセスのブリッジをテスト domain（`ROS_DOMAIN_ID` 97）で起動し、
到達・goal success・キャンセル時の停止・固定レートの指令配信を e2e で検証する。
