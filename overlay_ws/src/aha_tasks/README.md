# aha_tasks

SOBITS RCJO2026でHuman Robot Interaction Challengeの移動フローを実行する。
Nav2 actionを利用し、到着地点での対話・認識・操作はログに `SKIPPED` を出して通過する。
`aha_sim_tasks` のpolicy評価とは独立した、通常のシミュレーション用パッケージ。

## 起動

[開発コンテナ](../../../docs/docker.md)で実行する。
このlaunchがGazebo・ロボット・Nav2・タスクをまとめて起動するため、別のsim / SLAMと重ねて起動しない。

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_tasks
source install/setup.bash
export DISPLAY=display:0  # macOS / LinuxのNoVNC
ros2 launch aha_tasks human_interaction.launch.py
# GUIなし: ros2 launch aha_tasks human_interaction.launch.py headless:=true
```

初期位置は `(-2.0, 1.5, yaw=0.0)` に固定し、AMCLの初期姿勢と一致させる。
カメラは無効。LiDAR、map_server、AMCL、Nav2は有効。
Nav2がACTIVEになってから開始し、各ゴールの成功を待って次へ進む。
拒否・失敗・キャンセル時は後続工程を実行しない。起動待ちの上限は `startup_timeout:=300.0`、
各移動の上限は `goal_timeout:=600.0`（いずれも実時間の秒数）。タイムアウト・Ctrl+Cでは実行中のゴールをキャンセルする。
完了・失敗時はlaunch全体を終了する。結果は `COMPLETED` / `FAILED` またはエラーログとrunnerの終了コードで確認する。

## フローと地点の変更

[設定](config/human_interaction.json)に地点 `[x, y, yaw]`（m、rad、`map` frame）と実行順を定義する。
地図の座標はGazeboのworld座標に合わせてある。家具の中心ではなく、ロボットが立つ位置を指定する。

| ラベル | 用途 |
| --- | --- |
| `waiting` | Study Room内の客の到着待ち位置 |
| `entrance` | 入口の内側、ドア前 |
| `study_chair_left` | Study Roomの `chair_left` の手前。1人目を案内 |
| `study_chair_right` | Study Roomの `chair_right` の手前。2人目を案内 |
| `host` | Study Room内のホストとの合流を代用する固定位置 |
| `bag_drop` | Living Room / Laundry Area内の荷物置き場を代用する固定位置 |

移動順は `waiting → entrance → study_chair_left → waiting → entrance → study_chair_right → host → bag_drop`。
ドアベル・開扉・挨拶・名前/飲み物の聴取・注視・客の特徴説明・着席案内・紹介・荷物の受渡しはスキップする。
人物追従は `bag_drop` への通常のnavigationで代用する。人や荷物のモデルは追加しない。
SOBITSの入口ドアは閉じた配置のまま、内側の地点に到着するだけで通過しない。

別設定は `task_config:=/absolute/path/task.json` で指定する。
各stepは一意な `id` と、`navigate`（既存地点のラベル）または `skip`（理由）のどちらか一方を持つ。
未知の地点、不正な座標、未対応の工程は起動時にエラーにする。
設定内容だけを確認する場合は、ROS・Gazeboなしでリポジトリ直下から実行できる。

```bash
PYTHONPATH=overlay_ws/src/aha_tasks python3 overlay_ws/src/aha_tasks/scripts/human_interaction \
  --config overlay_ws/src/aha_tasks/config/human_interaction.json --dry-run
```

地点名と向きは `/aha/tasks/locations` のMarkerArrayに配信する。
起動中の別コンテナシェルで以下を実行すると、地図とラベルを表示できる。

```bash
rviz2 -d "$(ros2 pkg prefix --share aha_navigation)/rviz/nav.rviz" --ros-args -p use_sim_time:=true
```

地図の作成方法・対象範囲は [aha_navigation](../aha_navigation/README.md)を参照。

## ルールとの対応

参照は [公式ルールブック §5.1](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)
（確認日2026-10-11、2027 Rev-1 / 2026-08-31のドラフト）。
2人の客を順に迎えて席へ案内し、双方を紹介し、2人目の荷物をホスト経由で運ぶ流れを移動のみで再現する。
競技の成功判定や制限時間の採点は行わない。固定の客席・ホスト・荷物置き場はデモ用の代替であり、
実際の人の検出や追従、動的な空席選択には対応しない。
