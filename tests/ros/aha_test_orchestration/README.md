# aha_test_orchestration

タスクの判断をローカルLLM（OpenAI互換のchat completions、既定は `http://10.99.0.1:8080/v1` の `qwen3.8-27b`）で行う。
今は「持っている物体をどの棚のどこに置くか」（RoboCup@Home Pick and Placeの「似た物の隣、なければ空いた段」）だけ。
認識（[aha_perception](../../../overlay_ws/src/aha_perception/README.md)）と置く動作（[aha_test_manipulation](../aha_test_manipulation/README.md#置く動作)）の間に入り、
LLMには文字だけを渡す（サーバーは画像を受け付けない）。Gazeboでの比較は [place_llm](../../orchestration/place_llm/README.md)。

## 使い方

ロボットが物体を持って家具の前にいる状態で実行する（シミュレーションでは `sim_hold_object.py` で持たせる）。
平面検出（家具の設定つき）、`/aha/perception/objects`、`head_look_at.py`、置く動作のserverが要る。

```bash
ros2 launch aha_bringup sim.launch.py use_perception:=true planes:=true objects:=true use_manip:=true \
  world_tf:=true furniture_config:=$(ros2 pkg prefix aha_perception)/share/aha_perception/config/furniture/rcjo2026_arena.yaml
ros2 run aha_test_orchestration place_with_llm.py --label apple --variant C            # 判断して置く
ros2 run aha_test_orchestration place_with_llm.py --label milk --variant A --dry-run   # 判断だけ
```

今の状態（観測・判断・置く・台車の移動）、LLMとの会話（置いた結果を回答の後に注記）、イベントは
[aha_test_ui](../aha_test_ui/README.md) のtopicに発信元 `place_runner` で出す（`ros2 run aha_test_ui robot_monitor.py` で見る）。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `--label` / `--size` | （必須）/ 物体リストの大きさ | 持っている物体のラベルと大きさ x y z（m） |
| `--variant` | `C` | LLMに渡す情報（下の表） |
| `--furniture` / `--furniture-config` / `--objects-config` | `shelf` / arenaの設定 / パッケージの物体リスト | 置く家具、家具の設定、物体リスト（`aha_perception` の `config/`） |
| `--arm` / `--grasp` | `right` / `side` | `PlaceObject` の腕と持ち方 |
| `--llm-url` / `--model` | 上記 | LLMのサーバー |
| `--no-space-hint` / `--dry-run` / `--log` | – | 段ごとの空きの確認をしない / 置かない / 報告（プロンプト・回答・要約）をJSONの1行で追記 |
| `--back-off` | – | 全段の奥まで見える所まで下がって観察し、戻ってから置く（下の1。台車は障害物を確かめずに動く） |

## 処理

`aha_test_orchestration.place_runner.PlaceLlmRunner.run`（後のオーケストレーターから呼ぶ）：

1. 観測：家具の設定で床より高い段（0.1 m以上）を見る。`back_off`（`--back-off`）では、近くからだと上の棚板に隠れる
   低い段の奥まで見える所へ台車をまっすぐ下げる（`aha_test_orchestration.viewpoint`：カメラの高さH、段の上面h、上の棚板の下面b、
   奥行きDから、カメラと前面の距離 (H − b)·D / (b − h) の最大に0.05 mを足し、1.5 mまで。arenaの棚は段2のため約1.25 m）。
   そこから、カラー画像の縦の画角（約43.4°、両端4°ずつ除く）に収まるだけの段をまとめて見る（arenaの棚は2方向）。
   台車の移動は障害物を確かめない（`base_motion`、Nav2が動くまでの代わり）。`back_off` なしでは今の位置から段ごとの中心を見る。
   頭を向けるたびに、頭の関節が止まるまで（0.3 s以上）待ってから面と物体の検出を2回ずつ待ち、その方向の段ごとに
   `get_place_candidates`（家具と段、持っている物体の大きさ）を呼んで空きの有無（`yes` / `no` / `not seen`）を記録する。
   物体はtrack IDごとに位置を平均し、ラベルは多数決で、観測の間ためる。要約が面の検出3回の間変わらなければ終わり、
   下がったなら元の位置へ戻る（置くのはそこから）。
2. 要約（`aha_test_orchestration.scene`）：物体の底面（中心z − 高さ/2）と面の高さの差が0.04 m以内で、中心が面の凸包から0.03 m以内なら、
   その面の上の物体とする。物体の段は、家具の外形の中なら底面に最も近い設定の段の高さ（0.05 m以内）、
   そうでなければperceptionがその面に対応させた段。棚板が検出されない段（物体や上の棚板に隠れる）でも物体は一覧に残り、
   2つが食い違う物体は報告の `level_conflicts` に出る。
3. 判断（`aha_test_orchestration.place_decision`）：関数 `choose_place` の呼び出しを強制し（`tool_choice`）、回答を検証する。
   ない面・段・高さ・物体、別の段の物体などは、誤りを伝えて1回だけやり直し、それでも不正なら失敗にする。
4. 実行：回答から `PlaceObject` のgoalを作って送る。隣の物体があれば、その段の同じカテゴリの他の物体を `group` に、
   別カテゴリの物体を `avoid` に、`next_to_distance`（0.15 m）を `near_max_distance` に入れる。届かなければ、`move_base` なら返ってきた
   `suggested_base_pose` へ台車を動かして（`aha_test_orchestration.base_motion`：向きを変えて直進し向きを戻す、0.6 m まで、
   障害物の確認なし。Nav2が動くまでの代わり）同じgoalを送り直し、それでも届かないか動かないなら
   `near_max_distance` 0（届く候補のうち同じグループの側を先に）で置く。
   物体を持ったままの失敗（`NO_SURFACE` / `NO_SPACE` / `UNREACHABLE` /
   `EXECUTION_FAILED`）は、置こうとした所と理由を関数呼び出しの結果としてLLMに返して選び直させる（2回まで、失敗した所は受け付けない）。
   離した後の失敗（`NOT_AT_SPOT`）などはやり直さない。

棚の中身はロボットが触った分しか変わらないので、要約は保持し（`memory`）、置けたら置いた物体を加える。
`run(..., reuse=True)` は1の観測をせずにこの要約を使う（空きの有無は前の物体のものなので消す）。
棚から取り出した物体は `scene.without_object` で除く。

| variant | LLMに渡す情報 | 回答 | ロボットが送るgoal |
| --- | --- | --- | --- |
| `A`（全部） | 追跡したすべての水平面（ID・ラベル・家具の段・高さ・重心・範囲）とすべての検出物体（ID・ラベル・中心・大きさ）の数値、odomでのロボットの位置 | 面のID、隣に置く物体のIDか目標点 x, y | `surface_ids` = [面]、`near` = 物体か点 |
| `B`（高さだけ） | 家具の設定の段の高さと、各高さで見えた物体のラベル（家具の外形の中の物体を底面の高さで分ける） | 高さ（m） | `restrict_height`、`surface_height` = その段の高さ、`near` なし |
| `B2` | `B` と同じ | `B` と同じ | 高さの段の `furniture` / `furniture_level`、`near` なし（ロボットが高さを家具の段に直す） |
| `C`（IDの要約） | 家具の段ごとの物体のIDとラベル、持っている物体の空きの有無 | 家具・段・隣に置く物体のID（なければ `empty`） | `furniture` / `furniture_level`、`near` = 物体（`empty` はなし） |

systemプロンプトはすべて共通で、規則（同じカテゴリの物の隣、なければ物のない段に新しいグループ）と物体リストのカテゴリを渡す。
物体のカテゴリはロボットが物体リストで知っていて、LLMにはリストとして渡す（検出のラベルからの対応はLLMが行う）。
LLMの呼び出しは温度0、`max_tokens` 1000、`chat_template_kwargs.enable_thinking` false。呼び出しごとの時間とトークン数を報告に入れる。

## ライブラリ

- `aha_test_orchestration.llm`：`LlmClient`（標準ライブラリだけ）、`ask_tool`（関数呼び出しの強制・検証・1回のやり直し）、
  `tool_results`（関数呼び出しへの結果のメッセージ）。
- `aha_test_orchestration.scene`：面と物体の対応、家具の段の要約と、置いた・取り出した物体での更新（ROS非依存）。
- `aha_test_orchestration.place_decision`：variantごとのプロンプト・関数のJSON schema・回答の検証と `PlaceRequest`（ROS非依存）。
- `aha_test_orchestration.place_runner`：上の処理（ROS）。
- `aha_test_orchestration.base_motion`：odometryでの台車の短い移動（ROS）。

## 制限

- 対象は1つの家具。家具の設定（外形と段の高さ）と、perceptionの面と段の対応を前提にする。
- 物体の検出はシミュレーションの `sim_object_detector.py` だけで試した。実機の検出器はない。
- 段の空きは観測の時点で見えている範囲だけ（`get_place_candidates`）で、腕が届くか・置く姿勢の手が届く高さかは確かめない。
- 「隣」は `PlaceObject` の `near` と `avoid`。台車を動かさないと（`move_base` false、CLIの既定）届く候補がそこから遠いと離れた所に置く。
- 台車の移動は障害物を見ない。棚の前の短い移動だけに使う。

## テスト

開発コンテナの `/app/overlay_ws` で `colcon test --base-paths src ../tests/ros --packages-select aha_test_orchestration`（[tests](../../README.md)）。ROSに依存しないテストはホストでも実行できる。

```bash
cd ../../..   # repo root
PYTHONPATH=overlay_ws/src/aha_perception:tests/ros/aha_test_orchestration uv run --no-project --python 3.12 \
  --with numpy==1.26.4 --with scipy==1.11.4 --with pyyaml --with pytest \
  python -m pytest tests/ros/aha_test_orchestration/test/test_scene.py tests/ros/aha_test_orchestration/test/test_place_decision.py tests/ros/aha_test_orchestration/test/test_llm.py
```

`test/test_place_runner.py`（goalとメッセージの変換、失敗後の選び直しと要約の更新）はROSのメッセージが必要で、colconでだけ実行する。
