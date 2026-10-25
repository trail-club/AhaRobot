# tests

本番のパッケージ（`overlay_ws/src`）の外に置く試験用のコード。ロボットでは使わない。

| 場所 | 内容 |
| --- | --- |
| `perception/` | ホストで実行するperceptionのツールの単体テスト（`make test`）。`make test` の環境にはscipyがなく、`test_plane_bench_noise.py` はスキップされる |
| `ros/` | 試験用のROSパッケージ（下表）。`overlay_ws` の通常のビルドには含まれず、`make test` でビルド・テストする |
| `manipulation/place_sim/` | 置く動作の試行・ストレス試行（Gazebo） |
| `orchestration/place_llm/` | LLMで置く場所を決めて置く試行（Gazebo） |

| パッケージ（`ros/`） | 内容 |
| --- | --- |
| [aha_test_msgs](ros/aha_test_msgs/) | 置く動作のaction `PlaceObject` |
| [aha_test_description](ros/aha_test_description/) | 試験用のロボットのxacro（把持物体の固定、MoveItの昇降の下限より上から始める初期位置0.01 m。`sim.launch.py` の `initial_lift` は使わない）と、本番の `controllers.yaml` に昇降のcontrollerの差分を重ねてビルド時に作る設定 |
| [aha_test_manipulation](ros/aha_test_manipulation/README.md) | 両腕のMoveIt設定、perceptionの置き場所の候補へ持っている物体を置くaction server、シミュレーションの把持 |
| [aha_test_orchestration](ros/aha_test_orchestration/README.md) | ローカルLLMで置く段と場所を決めて置く処理、台車の仮の移動（odomのみ、障害物の確認なし） |
| [aha_test_ui](ros/aha_test_ui/README.md) | 状態・LLMとの会話・イベントを表示するモニタ |

perceptionの置き場所の候補（`/aha/perception/get_place_candidates`）を、物体を持って置くところまで試すための仮の実装。
manipulation・orchestrationの本番の実装は各担当のパッケージ（`overlay_ws/src/aha_manipulation` など）にある。

## ビルド

開発コンテナで、本番のパッケージに加えて `tests/ros` もビルドする。
`tests/ros` の依存（MoveItなど）はコンテナの起動時にrosdepで入る。入っていなければホストで `docker restart aharobot_aha_project_1` を実行する。

```bash
cd /app/overlay_ws
colcon build --symlink-install --base-paths src ../tests/ros
source install/setup.bash
colcon test --base-paths src ../tests/ros \
  --packages-select aha_test_description aha_test_manipulation aha_test_orchestration aha_test_ui
```

## 置く動作の試験環境の起動

試験用のxacroでシミュレーションを起動し、試験用のMoveItと置く動作のserverを起動する。

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true objects:=true \
  depth_noise:=d435i_filtered world_tf:=true \
  furniture_config:=$(ros2 pkg prefix aha_perception)/share/aha_perception/config/furniture/rcjo2026_arena.yaml \
  robot_xacro:=$(ros2 pkg prefix aha_test_description)/share/aha_test_description/urdf/aha_robot_test.urdf.xacro
ros2 launch aha_test_manipulation manip.launch.py
```

各試行ツールは `--launch` でこれらを自動で起動する（[place_sim](manipulation/place_sim/README.md)、[place_llm](orchestration/place_llm/README.md)）。
