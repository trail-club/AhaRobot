# aha_test_manipulation

試験用（[tests](../../README.md)）：AhaRobotの両腕のMoveIt 2設定とlaunch、perceptionの置き場所の候補へ持っている物体を置くaction server。
シミュレーションで、perceptionの候補を物体を置くところまで試すための仮の実装で、ロボットでは使わない。

## 起動

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true depth_noise:=d435i_filtered \
  robot_xacro:=$(ros2 pkg prefix aha_test_description)/share/aha_test_description/urdf/aha_robot_test.urdf.xacro
ros2 launch aha_test_manipulation manip.launch.py            # rviz:=false でRVizなし
```

`manip.launch.py` が `moveit.launch.py`（`move_group` とRViz）、`place_server.py`、把持物体のbridgeを起動する。
`robot_state_publisher`・`/joint_states`・controllerは `sim.launch.py` が用意するため、シミュレーションを先に起動する。
ロボットは [aha_test_description](../aha_test_description/) の試験用xacro（把持物体の固定、昇降の初期位置、昇降のcontrollerの設定）で起動する。

| 引数（`manip.launch.py`） | 既定 | 内容 |
| --- | --- | --- |
| `use_sim_time` | `true` | `/clock` を使う |
| `rviz` | `true` | MotionPlanningパネル付きRViz（`rviz/manip.rviz`）。`moveit.launch.py` の引数も同じ |
| `place_server` | `true` | [置く動作](#置く動作)のaction server |
| `sim_held_object` | `true` | [把持物体](#シミュレーションの把持物体)のtopicのbridgeと、`place_server.py` の `release_topic` の設定 |

`rviz` は `perception.launch.py` と共通で、`rviz:=false` は両方のRVizと頭を向ける `head_look_at.py` を止める。
`rviz:=false` で置く動作を使う場合は `head_look_at.py` を別に起動する（[置く動作の試行](../../manipulation/place_sim/README.md#実行)にコマンド例）。

## 設定

| ファイル | 内容 |
| --- | --- |
| `srdf/aha_robot.srdf` | group、named state、end effector、衝突判定の除外 |
| `config/kinematics.yaml` | KDL（timeout 0.05 s） |
| `config/joint_limits.yaml` | 速度・加速度の上限、既定のscaling 0.5、昇降の下限 0.005 m |
| `config/ompl_planning.yaml` / `pilz_cartesian_limits.yaml` | OMPL（RRTConnect、薄い障害物向けの細かい衝突確認）、Pilz LIN / CIRCのTCP速度・加速度 |
| `config/moveit_controllers.yaml` | 実行に使うcontroller（腕と `lift_controller`。グリッパはMoveItから実行しない） |

`robot_description` は [aha_test_description](../aha_test_description/) の試験用xacro（`sim:=true`、`.setup_assistant`）から生成する。planning frameはURDFのroot `base_footprint`（virtual jointなし）。

| group / named state | 内容 |
| --- | --- |
| `right_arm` / `left_arm` | `base_link` → `link_ree` / `link_lee` のchain。昇降 `joint_[r\|l]1` と `joint_[r\|l]2`〜`6` |
| `right_hand` / `left_hand` | 指 `joint_[r\|l]7r` / `joint_[r\|l]7l`（衝突判定とnamed state用）。`open` / `closed` は指 ±0.06 m / 0 |
| `home`（腕） | 初期姿勢。腕の関節0、昇降0.01 m |
| `ready`（腕） | 昇降0.6 m、前腕を前に向け、TCPのxが前方水平 |
| `carry`（腕） | 昇降0.8 m、肘を後ろに畳み、TCPは `base_footprint` の (0.22, ∓0.30, 0.92) m。机・棚の手前で物体を持つ姿勢 |

TCPは `link_ree` / `link_lee`（指先、+xが接近方向、+zは `home` で上）。真上から下向きのTCP（`joint_[r|l]5` = π/2）は
上限1.57 radをわずかに超えるため、`/compute_ik` には1 mrad以上の姿勢の許容を付ける。

衝突判定の除外は次のコマンド（`ros-jazzy-moveit-setup-assistant` を別途入れる）で生成し、頭部と腕のlinkの組は除外せずに残した。再生成時も同じ組を除く。

```bash
/opt/ros/jazzy/lib/moveit_setup_assistant/collisions_updater \
  --config-pkg aha_test_manipulation --output /tmp/aha_robot.srdf --default --trials 300000
```

## 置く動作

`place_server.py` はaction `/aha/manipulation/place`（[aha_test_msgs/action/PlaceObject](../aha_test_msgs/action/PlaceObject.action)）で、
グリッパが持っている物体を平面検出の支持面に置く。物体はgoalの時点で手にある前提で、把持は扱わない。
平面検出（`planes:=true`）と `head_look_at.py` を使う（`head_look_at.py` がないとwarningを出して今の頭の向きで検出する）。

1. 物体の大きさがグリッパに合うか確かめ（[持ち方](#物体の持ち方)）、頭を置く所へ向けて、止まった後の検出から
   置き場所の候補と障害物の箱を得る（腕・手・持っている物体は検出から除く）。
2. 支持面・壁・障害物の箱と持っている物体をplanning sceneに入れ、腕が届く候補を `near`（なければ腕側の既定点）に近い順に試す。
3. 候補ごとに、接近（`side`：横から入って下ろす / `top`：真上から下ろす）と向きを変えて試す。
   腕を動かす前に、その向きで物体と指が下りる柱が分かっている障害物の箱に掛からないか確かめ、掛かる向きは飛ばす
   （perceptionが足跡を確かめたのは最初の向きだけ）。手前の姿勢へ動き、そこで検出をやり直して、
   その間に変わったこと（腕が押した物など）で置く所が塞がっていれば次の候補へ進む。
4. 置く姿勢へ下ろし、グリッパを物体の幅より少し開いて離す。退避し、置いた物体をplanning sceneに残して `return_state`（`carry`）へ戻る。
5. 置いた所へ頭を向けて検出し直し、その支持面の上の置いた所に障害物の箱（物体）があるか確かめる（`verify_place`）。
   なければ `STATUS_NOT_AT_SPOT` で失敗にする（腕が棚などに当たって台車が押されても、車輪の `odom` には出ない）。

支持面の箱は検出した縁から `support_padding`（0.03 m）横に広げてplanning sceneに入れ、縁をかすめる経路を避ける
（Gazeboで、手前の姿勢へ動く途中に持った物体が上の棚板の縁に下から当たり、台車ごと押されて棚の手前で離した例があった）。

検出の後に台車が動いた場合（`odom` → `base_footprint`）は手前の姿勢へ戻り、止まるのを待って1からやり直す（`max_replans` 回まで）。
失敗時は、腕が動いていれば物体を持ったまま `return_state` へ戻る。cancelは実行中の動作を止め、腕はその場に残る。
実行中の別のgoalは拒否する。手順の詳細は `scripts/place_server.py` と `aha_test_manipulation/*.py` のdocstringにある。

```bash
ros2 action send_goal --feedback /aha/manipulation/place aha_test_msgs/action/PlaceObject \
  "{arm: right, object_size: {x: 0.06, y: 0.06, z: 0.12}, grasp: side, approach: auto}"
```

| goal | 空・0のとき | 内容 |
| --- | --- | --- |
| `arm` | `right` | `right` / `left` |
| `object_size` | （必須） | 底面 x（接近方向）, y（指が閉じる方向）と高さ z（m）。グリッパに合わなければ断る |
| `surface_labels` | 机と棚板 | 置く支持面のラベル |
| `near` | 腕側の既定点 | 近くに置きたい点（任意のframe） |
| `restrict_height` / `surface_height` / `height_tolerance` | false / – / 0.03 m | trueなら高さ（`base_footprint` のz）がこの範囲の支持面だけに置く |
| `surface_ids` | どの面でも | この `Surface.id` の面だけに置く |
| `furniture` / `furniture_level` | どの家具でも / すべての段 | perceptionの[家具の設定](../../../overlay_ws/src/aha_perception/README.md#家具の設定)の家具と段（1が一番下）に置く。`near` がなければ、段の中心（perceptionの `view_point`）が `furniture_look_distance` 以内ならそこを、遠ければ `look_point` を段の高さで見る。段が検出になければ段の中心に向け直して1回やり直す |
| `clearance` | 0 | 物体の周りの障害物との間にperceptionの余裕に加えて空ける距離（m）。朝食の配膳の5 cmなど |
| `avoid` | なし | `near` と同じframeの点。別のグループ（同じ段の別カテゴリの物体など）。`near` よりこれらに近い候補を後に回す |
| `group` | なし | `near` と同じframeの点。`near` と同じグループの他の物体。`near` からの距離（下の条件と列）は `near` とこれらのうち最も近い物から測る |
| `near_max_distance` | 0 | >0なら、`near` からこの距離以内で `avoid` より `near` に近い候補だけを試す。届くものがなければ `STATUS_UNREACHABLE` と `suggested_base_pose`。その中では `near` と同じ列（台車の前方向の差が `next_to_row_tolerance` 0.04 m以内、横並び）の候補を手前の候補より先に試す（奥の物を隠さず、手前を後の物に残す） |
| `grasp` | `side` | 物体の持ち方 `side` / `top` |
| `object_in_hand` | 持ち方の既定の姿勢 | 手の中の物体の中心の姿勢（軸は `object_size` と同じ、直立でz上）。手に固定のframe（右TCPの `link_ree` など）で、pickが実際に掴んだ所を渡す。置く姿勢（物体を直立させて底面を `release_height` に）、持っている物体の箱、検出から除く箱に使う。`side` の横からの進入は手の向き（TCPのx）に沿う |
| `approach` | `auto` | `side` / `top` / `auto`（候補ごとに side → top） |
| `release_height` | 0.01 m | 離す高さ（支持面から） |

似た物体の隣に置く場合は、呼び出し側がその物体の位置を `near` に渡す。
結果の `place_pose` は離した時の物体底面の中心（`base_footprint`、xは接近方向）、`surface_id` は置いた支持面の `Surface.id`（未追跡なら0）、
`returned` は最後に腕が `return_state` にあるか、`verified` は5の確認で物体が見えたか（検出がなく確かめられなければfalseで、`message` に書く）。
`suggested_base_pose` は `near_max_distance` で届かなかったときの台車の姿勢（`odom`、今の向き）で、
条件に合う最良の候補が `base_spot`（腕の側の置きやすい位置）に来る。呼び出し側が台車を動かして同じgoalを送り直す。
feedbackは `phase`（`perceiving` / `planning` / `checking` / `approaching` / `releasing` / `retreating` / `returning` / `verifying`）と候補の番号。

`/aha/manipulation/place/markers`（RVizの `sim_view.rviz` の Place order）は、試す順の番号（1が最初、検出をやり直しても続き番号）と状態の色
（青：待ち、黄：試行中、赤：失敗、緑：置いた）、失敗の理由（`IK` / `plan` / `path` / `blocked` / `not seen`）を出す。
試す前に外した候補（届かない・柱が塞がる）は番号のない灰色の小さな点、`near` は白、`group` は緑、`avoid` は赤の円盤。
perceptionの `/aha/perception/place/markers` は問い合わせごとの候補の空き（円盤）と障害物の箱で、順位は持たない。

| `status` | 内容 |
| --- | --- |
| `STATUS_OK` | 置いた（`success` はtrue）。退避・復帰の失敗は `returned` と `message` に出る |
| `STATUS_NO_SURFACE` | ラベル（と高さ・面・家具の段）に合う支持面が検出にない |
| `STATUS_NO_SPACE` | 支持面はあるが物体を置ける所がない。物体が家具の段の上の空きより高い |
| `STATUS_UNREACHABLE` | 候補はあるが、どれも届かない・計画できない |
| `STATUS_EXECUTION_FAILED` | 候補がすべて失敗し、少なくとも1回は軌道の実行で失敗した |
| `STATUS_ERROR` | goalの値が不正、perception・move_groupのserviceがない、検出がない・古い、家具や段が設定にない |
| `STATUS_CANCELED` | 取り消された。離した後なら `message` に書く |
| `STATUS_BASE_MOVED` | 検出の後に台車が動き続けた（`max_replans` 回やり直した、または止まらなかった） |
| `STATUS_INVALID_OBJECT` | `object_size` がグリッパに合わない。腕は動かさない |
| `STATUS_NOT_AT_SPOT` | 離したが、戻った後の検出で置いた所に物体がない（落ちた・押された）。物体はもう手にない |

### 物体の持ち方

`side` はTCPのxが水平・zが上（`carry` と同じ）で、物体の中心が指のパッドの中心にある。
`top` はTCPのxが下向きから接近方向へ `top_tilt`（0.05 rad）傾き、物体の上面がパッドの中心の0.02 m上にある。
どちらも物体を直立させて置く。受け付ける大きさは指の間（y）0.11 m以下、`side` 把持では接近方向（x）0.104 m以下。
届く範囲は肩から水平に `side` 約0.75 m、`top` 約0.45 m（高さ0.8 mの机で `base_footprint` の x 0.45 m 以内）。

### パラメータ

| 主なパラメータ（他は `place_server.py` の宣言とコメント） | 既定 | 内容 |
| --- | --- | --- |
| `default_arm` / `default_grasp` / `default_approach` | `right` / `side` / `auto` | goalが空のときの値 |
| `preferred_point` / `preferred_point_top` | [0.45, 0.25] / [0.38, 0.2] | `near` がないときの基準点 x, \|y\|（腕の側へ符号を付ける） |
| `reach_limits` | [0.80, 0.55] m | 候補を残す肩からの水平距離（`side` / `top` 把持） |
| `aim_head` / `look_point` / `look_height` | true / [0.55, 0.15] / 0.75 m | 検出の前に頭を向ける、`near` がないときの注視点 |
| `furniture_look_distance` | 0.9 m | goalの家具の段の中心を見る水平距離の上限（遠い大きな机は手前を見る） |
| `late_check` / `late_check_margin` | true / 0.005 m | 手前の姿勢で検出をやり直して置く所を確かめる、下りる柱を物体と指から広げる量（指の厚さ0.015 mと合わせて片側0.02 m、perceptionが箱から空ける量と同じ） |
| `base_max_shift` / `base_max_turn` / `max_replans` | 0.01 m / 1.0° / 2 | 台車の移動とみなす量、やり直しの回数 |
| `velocity_scaling` / `cartesian_velocity_scaling` | 0.5 / 0.2 | 関節空間・Cartesian pathの速度・加速度のscaling |
| `release_topic` | 空 | 離す前に送る `std_msgs/Empty` のtopic（`{arm}` を腕の名前に置き換える）。`manip.launch.py` は `sim_held_object:=true` で `/aha/sim/held_object/{arm}/detach` を渡す |
| `return_state` | `carry` | 最後に戻るnamed state |
| `support_padding` | 0.03 m | planning sceneの支持面の箱を検出の縁から横に広げる量 |
| `base_spot` / `base_spot_top` | [0.55, 0.45] / [0.38, 0.2] | `suggested_base_pose` で候補を置く台車からの位置 x, \|y\|。`side` は腕の外側寄り（肩から約30°）で、手先の外側に張り出す手首が物の後ろに回り、隣の物や側板に当たりにくい |
| `group_ratio` | 0.6 | `near` までの距離が `avoid` の各点までの距離のこの倍未満の候補だけを `near` のグループとする（中間の位置はどちらのグループか分からない） |
| `verify_place` / `verify_margin` / `verify_height_tolerance` | true / 0.03 m / 0.04 m | 戻った後の確認、置いた所から箱までの水平距離と、箱の底と支持面の高さの差の上限 |

## ROSインターフェース

planning groupは `right_arm` / `left_arm`（昇降を含む）、TCPは `link_ree` / `link_lee`、planning frameは `base_footprint`。
planning pipelineは `ompl`（既定）と `pilz_industrial_motion_planner`（PTP / LIN / CIRC）。
軌道は `right_arm_controller` / `left_arm_controller` と `lift_controller`（計画した腕の昇降関節だけ）のactionで実行する。
グリッパはMoveItから動かさず、位置指令topicを使う。
`lift_controller` は片方の昇降関節だけの軌道も受け付け（含まれない関節は現在位置を保つ）、
試験用xacroでは軌道の終了から1 s以内に目標の±0.01 mへ届かなければabortする。

| 用途 | 名前 | 型 |
| --- | --- | --- |
| 計画・実行 | `/move_action` | `moveit_msgs/action/MoveGroup` |
| 軌道の実行 | `/execute_trajectory` | `moveit_msgs/action/ExecuteTrajectory` |
| IK / FK | `/compute_ik` / `/compute_fk` | `moveit_msgs/srv/GetPositionIK` / `GetPositionFK` |
| 直線経路 | `/compute_cartesian_path` | `moveit_msgs/srv/GetCartesianPath` |
| 計画のみ | `/plan_kinematic_path` | `moveit_msgs/srv/GetMotionPlan` |
| Pilzの連続動作 | `/sequence_move_group` | `moveit_msgs/action/MoveGroupSequence` |
| planning sceneの取得 / 変更 | `/get_planning_scene` / `/apply_planning_scene` | `moveit_msgs/srv/GetPlanningScene` / `ApplyPlanningScene` |
| 状態の衝突確認 | `/check_state_validity` | `moveit_msgs/srv/GetStateValidity` |
| planning scene | `/monitored_planning_scene` | `moveit_msgs/msg/PlanningScene` |
| 持っている物体を置く（action） | `/aha/manipulation/place` | `aha_test_msgs/action/PlaceObject` |
| 置く候補の試す順と状態（RViz） | `/aha/manipulation/place/markers` | `visualization_msgs/msg/MarkerArray`（transient local） |
| 把持物体の固定（ROS → Gazebo） | `/aha/sim/held_object/<arm>/attach` | `std_msgs/msg/Empty`（`gz.msgs.Empty`） |
| 把持物体の解除（ROS → Gazebo） | `/aha/sim/held_object/<arm>/detach` | `std_msgs/msg/Empty`（`gz.msgs.Empty`） |
| 把持物体の状態（Gazebo → ROS） | `/aha/sim/held_object/<arm>/state` | `std_msgs/msg/String`（`gz.msgs.StringMsg`、`attached` / `detached`） |

`place_server.py` は上のmove_groupのinterface、perceptionの `/aha/perception/get_place_candidates`・`/aha/perception/planes`・
`/aha/perception/look_at`、`/joint_states`、`odom` → `base_footprint` のTF、グリッパの位置指令topicを使う。
planning sceneの `aha_place_surface_*` / `aha_place_wall_*` / `aha_place_obstacle_*` / `aha_held_object_<arm>` / `aha_placed_object`
（`<arm>` は `right` / `left`）はgoalごとにこのserverが書き換える。状態と結果は [aha_test_ui](../aha_test_ui/README.md) のモニタにも出す。
RVizで置く順を見るには、MarkerArrayの表示を `/aha/manipulation/place/markers`（transient local）で足す。

## シミュレーションの把持物体

試験用xacroは、腕ごとにGazeboの `DetachableJoint` でモデル `aha_held_object_<arm>` を
`link_[r|l]6` に固定する（topicは[上の表](#rosインターフェース)、`manip.launch.py` の `sim_held_object:=true` でbridge）。
`scripts/sim_hold_object.py` は前の物体を外して消し、腕を `carry`（`--state`）へ動かし、[持ち方](#物体の持ち方)の位置に
箱か円柱を出して固定し、指を物体の幅まで閉じる。gz-transportを直接使うため、シミュレーションと同じ `GZ_PARTITION` で実行する。

```bash
ros2 run aha_test_manipulation sim_hold_object.py --size 0.06 0.06 0.12            # 右手、side
ros2 run aha_test_manipulation sim_hold_object.py --arm left --shape cylinder --size 0.065 0.065 0.12 --grasp top
ros2 run aha_test_manipulation sim_hold_object.py --release                         # 外して消す
```

持たせて置く試行を繰り返すツールは [置く動作の試行](../../manipulation/place_sim/README.md)。

## 制限

- 障害物は検出（goalの初めと手前の姿勢）で見えた範囲の箱だけ。物体の陰・視野の外・床の上・厚さ約2 cm未満の物体、
  確認の後に置かれた物体、前のgoalで見えない位置に置いた物体は避けない。
- `near` は優先する点で、`restrict_height` なしでは近くに置けなければ別の面（別の棚板など）に置く。
- 届く範囲に空きがなければ `STATUS_UNREACHABLE` で、呼び出し側が台車を近づける。
- 物体は箱として扱い、置く向きは肩から候補への方向で決まる。
- 台車の移動は各動作の前に確かめるだけで、動作中の移動は次の確認まで分からない。

## テスト

`test/test_moveit_config.py` はSRDF・関節上限・controller設定と試験用のロボット（`aha_test_description`）の一致、他の `test/test_*.py` は置く動作のROSに依らない部分を確かめる。
