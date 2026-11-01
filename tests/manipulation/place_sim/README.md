# 置く動作の試行（シミュレーション）

[aha_test_manipulation の置く動作](../../ros/aha_test_manipulation/README.md#置く動作)をGazeboで繰り返し、
置いた物体の姿勢を測るツール。出力（既定は `results/`）はGit対象外。

| スクリプト | 内容 |
| --- | --- |
| `run_trials.py` | 決まった家具（world `rcjo2026`）に置く試行。`results/trials.csv` に追記 |
| `stress.py` | [stress_scenarios.py](stress_scenarios.py) の家具・物体を出して失敗する条件を探す。`results/stress.csv` に追記 |
| `plot_evidence.py` | `stress.py --evidence` の1試行を上から見た図と画像にする |

## 実行

開発コンテナで、overlayをsourceしてリポジトリのルートから実行する。
同じコンテナで別のシミュレーションが動いている場合は、別の `ROS_DOMAIN_ID` と `GZ_PARTITION` を使う。

```bash
export ROS_DOMAIN_ID=99 GZ_PARTITION=place99
python3 tests/manipulation/place_sim/run_trials.py --launch                      # 全シナリオ
python3 tests/manipulation/place_sim/run_trials.py --launch --scenario shelf_060 --trials 0 1
```

| オプション | 内容 |
| --- | --- |
| `--launch` | シナリオごとにheadlessのシミュレーション（試験用のxacro `robot_xacro:=`、`use_perception:=true planes:=true depth_noise:=d435i_filtered`、spawnはシナリオの値）、`aha_test_manipulation` の `manip.launch.py` と `head_look_at.py` を起動し、終わったら止める。ログは `$TMPDIR/place_trials_<scenario>.log` |
| `--scenario` / `--trials` | 実行するシナリオと試行の番号（既定はすべて） |
| `--out` | 出力先（既定 `results/`）。列の違う既存のCSVは今の列で書き直してから追記する |
| `--settle` | 置いてから測るまでの時間（既定3 s、シミュレーション時刻） |
| `--run` | `run` 列の値（既定は開始時刻） |

`--launch` なしでは、同じworld・spawnで起動済みのシミュレーションを使う。
`shelf_060` と `shelf_090`、`dining_table` と `dining_clutter` は同じspawn
（`dining_clutter` の障害物は消さないため後に実行する）。

```bash
ros2 launch aha_bringup sim.launch.py headless:=true rviz:=false use_perception:=true planes:=true \
  depth_noise:=d435i_filtered spawn_x:=2.95 spawn_y:=2.90 spawn_yaw:=1.5708 \
  robot_xacro:=$(ros2 pkg prefix aha_test_description)/share/aha_test_description/urdf/aha_robot_test.urdf.xacro
ros2 launch aha_test_manipulation manip.launch.py rviz:=false
ros2 run aha_perception head_look_at.py --ros-args \
  --params-file $(ros2 pkg prefix aha_perception)/share/aha_perception/config/head.yaml
```

試行ごとに物体を `carry` で持たせ（[把持物体](../../ros/aha_test_manipulation/README.md#シミュレーションの把持物体)）、
新しい平面検出を待って `PlaceObject` を送り、`--settle` 後にGazeboの姿勢とplanning sceneの障害物の箱を読む。
手順の詳細は `run_trials.py` のdocstring。

## シナリオ（`run_trials.py`）

| シナリオ | spawn (x, y, yaw) | 支持面 |
| --- | --- | --- |
| `dining_table` | (−2.0, −1.05, −π/2) | `dining_table` 上面 0.80 m |
| `dining_clutter` | 同上 | 同上。既定の基準点の周りに箱2個と円柱1個 |
| `shelf_060` | (2.95, 2.90, π/2) | `shelf` の0.61 mの棚板 |
| `shelf_060_back` | (2.95, 2.80, π/2) | 同じ棚板（0.1 m遠い） |
| `shelf_090` | (2.95, 2.90, π/2) | `shelf` の0.91 mの棚板 |
| `shelf_furniture` | (2.95, 2.80, π/2) | 家具の設定で `shelf` の段3（0.61 m）・段4（0.91 m）を指定。高すぎる物体（NO_SPACE）、見えない `dining_table`（NO_SURFACE）、serviceを直接呼ぶ試行（`probe`）を含む |
| `dining_furniture` | (−2.0, −1.05, −π/2) | `dining_table` を家具で指定し、`dining_clutter` の障害物の隣に `clearance` 0 / 0.05 mで置く |

各試行の持ち方・接近・腕・物体・`near`・高さ・家具・`clearance`・期待するstatusの指定は `run_trials.py` の `SCENARIOS` にある。
`*_furniture` は `--launch` で `world_tf:=true furniture_config:=<aha_perception>/config/furniture/rcjo2026_arena.yaml` を付けて起動する。

家具の設定なしの棚では期待どおりにならない試行がある。
`shelf_090` は上の段の棚板の前の縁が障害物の箱になって0.91 mの候補がすべて衝突し、0.61 mの棚板に置く（`shelf_furniture` の段4は置ける）。
`shelf_060` の上から置く試行は、planning sceneにない上の段の棚板に腕が当たってロボットが動くことがあり、その後の試行も位置がずれる。
`near` だけを指定する棚の試行は、0.61 m・0.91 mのどちらの棚板に置くかが実行ごとに変わることがある。

## 出力の列（`trials.csv`）

| 列 | 内容 |
| --- | --- |
| `run` / `restrict_height` | 実行の名前、goalの `surface_height`（`restrict_height` のとき） |
| `furniture` / `clearance` / `expect` / `as_expected` | goalの家具と段・`clearance`、期待するstatus、その通りか（`OK` は `success` も） |
| `action_success` / `status` / `message` / `candidates_tried` / `approach_used` / `surface_height` / `surface_id` | actionの結果（`probe` ではserviceの結果、`candidates_tried` は候補の数） |
| `view_point` / `view_err` | `probe` の応答の `view_point`、worldに直した点と設定の段の中心の距離 |
| `min_gap` | 置いた物体と障害物の底面の水平距離の最小（障害物のあるシナリオ） |
| `on_surface` / `upright` / `tilt_deg` | 物体の底面が支持面の上（高さの差0.02 m未満）か、直立（傾き10°未満）か、その傾き |
| `success` | `action_success` かつ `on_surface` かつ `upright` |
| `err_xy` / `err_z` / `surface_err` / `bottom_z` | `place_pose` との水平・高さの差、検出した支持面の高さの誤差、底面の高さ（world） |
| `hold_robot_yaw_deg` / `robot_moved` / `robot_yaw_change_deg` | 持たせる間のロボットの回転、`PlaceObject` の間の移動と回転 |
| `furniture_moved` / `obstacles_moved` / `obstacle_moves` | 家具・障害物の移動量（最大、障害物ごと） |
| `scene_obstacles` / `clutter_in_scene` / `clutter_boxes` | planning sceneの障害物の箱の数、箱の中にある障害物の数、障害物ごとの箱の大きさ（広げる前） |
| `wall_s` / `sim_s` | `PlaceObject` の所要時間（wall clock / シミュレーション時刻） |

## ストレス試験

```bash
export ROS_DOMAIN_ID=72 GZ_PARTITION=stress72
python3 tests/manipulation/place_sim/stress.py --list
python3 tests/manipulation/place_sim/stress.py --launch --scenario edge under_hand --evidence /tmp/stress_ev
python3 tests/manipulation/place_sim/plot_evidence.py /tmp/stress_ev/edge_03   # /tmp/stress_ev/edge_03/plot.png
```

worldは `rcjo2026`、spawnは前方が空いている既定の (−2.0, 1.5, 0)。家具・物体はシナリオの開始時のロボットの姿勢を基準に出し、
同じlaunch（spawn・Depthノイズ）のシナリオは1つのシミュレーションで続けて実行する。

| オプション | 内容 |
| --- | --- |
| `--launch` / `--scenario` / `--trials` / `--out` / `--settle` / `--run` | `run_trials.py` と同じ |
| `--list` | シナリオの一覧 |
| `--csv` | 出力のファイル名（既定 `stress.csv`） |
| `--log` | `--launch` なしで、起動済みのlaunchのログをevidenceに含める |
| `--evidence DIR` | 試行ごとに `DIR/<シナリオ>_<試行>/` へ `result.json`（CSVの行、平面、候補と障害物の箱、planning scene、モデルの前後の姿勢）、`color.png`・`depth.png`、`server.log` を保存する |

ネットワークが切れているとgz-transportのmulticastが失敗してモデルを出せないため、`GZ_IP=127.0.0.1` も設定する。
シナリオの要素（`Prop` / `Scenario` / `Trial` / `Perturb`）と座標の決め方は `stress_scenarios.py` を参照。

出力の列（`run_trials.py` と同じ列は省略）：

| 列 | 内容 |
| --- | --- |
| `noise` / `mass` / `near` / `perturb` / `perturb_result` / `after_perturb_s` | 条件、goal中の変化とその結果、変化から結果までの時間 |
| `phases` / `returned` | 受け取ったfeedbackのphaseの順、結果の `returned` |
| `target` / `ended_on` / `on_target` | 意図した支持面、物体が載った面（`floor` / `hand` / `other` を含む）、両者の一致 |
| `released` | 把持物体の `detach` が出たか |
| `props_moved` / `prop_moves` / `contact_phase` | 出したモデルの移動量（最大、モデルごと）、モデルかロボットが最初に動いたphase |
| `outcome` | `ok`（意図した面に直立）/ `wrong_place`（成功だが別の所・倒れた）/ `fail` / `released`（失敗だが離した）。モデルかロボットが動いた場合（`drive` / `turn` を除く）は `+contact` |
