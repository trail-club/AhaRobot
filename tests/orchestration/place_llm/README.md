# LLMで置く場所を決める試行（シミュレーション）

[aha_test_orchestration](../../ros/aha_test_orchestration/README.md) のvariant（LLMに渡す情報の分け方、`A` / `B` / `B2` / `C`）を、
物体を持って棚の前にいる状態から置き終わるまでGazeboで比べるツール。
平面検出・面の追跡・家具の段の対応・`PlaceObject` は本物を使い、物体の検出だけGazeboの真の姿勢から作る
（[シミュレーションの物体検出](../../../overlay_ws/src/aha_perception/README.md#シミュレーションの物体検出)）。
把持と移動は扱わない（持たせた状態から始める）。出力（既定は `results/`）はGit対象外。

| ファイル | 内容 |
| --- | --- |
| `scenario.py` | 棚に置く物体（`PROPS`）と持たせる物体・正解（`CASES`） |
| `run_trials.py` | 試行。`results/trials.csv` に1試行1行、`results/reports.jsonl` にプロンプト・回答・要約を追記 |
| `summarize.py` | `trials.csv` をnoise・variantごとの表にする |

## 実行

開発コンテナで、overlayをsourceしてリポジトリのルートから実行する。LLMのサーバー（既定 `http://10.99.0.1:8080/v1`）に届くこと。
同じコンテナで別のシミュレーションが動いている場合は、別の `ROS_DOMAIN_ID` と `GZ_PARTITION` を使う。

```bash
export ROS_DOMAIN_ID=98 GZ_PARTITION=placellm98 GZ_IP=127.0.0.1
python3 tests/orchestration/place_llm/run_trials.py --launch                       # 全部（2 × 8 × 4 = 64試行）
python3 tests/orchestration/place_llm/run_trials.py --launch --cases apple cola --variants C --noise clean
python3 tests/orchestration/place_llm/summarize.py --run "2026-10-11" --cases apple peach cola milk
```

| オプション | 内容 |
| --- | --- |
| `--launch` | headlessのシミュレーション（world `rcjo2026`、spawn (2.95, 2.80, π/2)、`world_tf:=true`、arenaの家具の設定、`planes:=true objects:=true depth_noise:=d435i_filtered`、試験用のxacro `robot_xacro:=`）、`aha_test_manipulation` の `manip.launch.py` と `head_look_at.py` を起動し、終わったら止める。ログは `$TMPDIR/place_llm_trials.log`。なしでは同じ条件で起動済みのシミュレーション（[起動の例](../../README.md#置く動作の試験環境の起動)）を使う |
| `--cases` / `--variants` / `--noise` | 実行する持たせる物体・variant（`A` / `B` / `B2` / `C`）・検出の条件（`clean` / `noisy`）。既定はすべて |
| `--dry-run` | 判断だけで置かない |
| `--arm` / `--spawn X Y YAW` | 置く腕（既定 `right`）、ロボットの初期位置（world、既定は `scenario.SPAWN`）。棚の物体はworldの座標で置くため、初期位置を変えると届く範囲・見える段が変わる |
| `--settle` / `--out` / `--run` | 置いてから測るまでの時間（既定3 s）、出力先、`run` 列の値 |
| `--llm-url` / `--model` | LLMのサーバー |
| `--ycb` | 棚の物体と持たせる物体を `tmc_wrs_gz_worlds` のYCBのメッシュにする（`scenario.YCB`。飲み物は形の近い缶・ボトルで代用：緑茶＝マスタード、レモネードとコーラ＝トマト缶、牛乳＝砂糖の箱。お菓子はなく、物体リストの箱・円柱のまま）。見た目・衝突ともYCBのメッシュ、質量もYCBの値（転がる・倒れる物はそのとおり動く）。持たせる物体の大きさはメッシュの外形で、ロボットにもそれを渡す。物体の名前は今までどおりGazeboの真の値 |
| `--physics-step` | 試行の前に、動いているシミュレーションの物理の刻み（s）をGazeboの `set_physics` で変える（既定0で変えない、`rcjo2026` は0.004）。0.008で約2倍速い（Radeon 680M内蔵のホストで、headless・RVizなしの実時間比が約0.35倍から約0.7倍） |
| `--move-base` | 届かなければ `suggested_base_pose` へ台車を動かして置き直す（`RunnerOptions.move_base`）。台車が動いた試行の後は初期位置（odomの原点）へ戻す |
| `--back-off` | 全段の奥まで見える所まで下がって観察し、戻ってから置く（`RunnerOptions.back_off`）。列 `back_off` にカメラから棚の前面までの距離、見た方向の数、戻った誤差 |
| `--next-to-distance` | `RunnerOptions.next_to_distance`（既定0.15 m、0で制限なし） |
| `--physical-hold` | 持たせた物を固定（DetachableJoint）せず、指で挟んだ摩擦だけで持つ（`SimHolder.hold(physical=True)`）。運ぶ途中のずれや離すときの挙動が実機に近くなる |
| `--next-to LABEL` | LLMを使わず、variant `C` に「このラベルの最初の物体の隣」と答える（`--variants C`）。台車の移動などロボット側だけを試す |

最初に両腕を畳んで `carry` へ上げ、棚に物体を出す。試行ごとに、2 cm以上動いた物体を元に戻し、検出器の `miss_rate` / `mislabel_rate` / `seed`
（`noisy` は10 % / 5 %、seedは持たせる物体ごとに同じで、3つのvariantは同じ見落とし・誤りを受ける）を設定し、物体を右手に `side` で持たせて
`PlaceLlmRunner.run` を呼び、`--settle` 後にGazeboの姿勢を読む。置いた物体は次の試行で持たせる前に消す。
要約は初期位置からのodomで、押されたロボット（車輪のodomに出ない）では正解と合わないため、試行の開始時にロボットが初期位置から
2 cm / 2°以上動いていれば止める（シミュレーションを起動し直す）。`summarize.py` は同じ条件・物体・variantの行が複数あれば後の行を使う。

## 場面

棚（`model://shelf`、段の上面 0.31 / 0.61 / 0.91 / 1.20 m）の前、0.5 mに立つ（place_simの `shelf_060_back` と同じ）。
この位置では段3–5に置け、段2は手前の帯しか見えず候補がない。

| 段 | 物体 | カテゴリ |
| --- | --- | --- |
| 2（0.31 m） | cleanser, sponge | cleaning supplies（置けない段の目印） |
| 3（0.61 m） | green tea, lemonade（右寄りの手前） | drink |
| 4（0.91 m） | orange, lemon（右寄り）、tuna can, spam（左端）、どちらも手前から0.18 m奥 | fruit / food |
| 5（1.20 m） | なし | – |

この位置の右手は奥行き約0.15 m・ロボットのy −0.2〜−0.45 m（worldのx 3.15〜3.40）の帯にしか置けない。
drinkとfruitはその右に置き、foodは帯の外で段4の2つ目のグループ（foodを持つケースはない）。
持たせる物体と正解：apple・peach → 段4のfruitの隣、cola・milk → 段3のdrinkの隣、
chocolate jello・strawberry jello（snack、棚にない、高さ0.08 m）→ 空いた段5。
pringles（0.23 m）・cookies（0.15 m）も段5が正解だが、段5で手が届かない高さ（約0.12 m超）。
大きさは物体リスト（`aha_perception/config/objects/robocup_objects.yaml`）。

## 出力の列（`trials.csv`）

| 列 | 内容 |
| --- | --- |
| `noise` / `case` / `held` / `variant` / `expected_level` | 条件と正解の段 |
| `detected` / `missed` / `mislabelled` | 要約の物体のうち棚の物体に対応した数（6 cm以内）、要約にない棚の物体、違うラベルの物体 |
| `decision_level` / `decision_target` | 回答の段と、隣に置く物体（対応した棚の物体）・点・`empty` / `level only`（B） |
| `level_ok` / `next_ok` / `decision_ok` | 段が正解か、隣の指定が正解か（同じカテゴリの物体の隣、点は同じカテゴリの物体から0.15 m以内で他より近い、グループがない場合は `empty`。Bは空）、両方 |
| `llm_calls` / `llm_error` | LLMの呼び出し回数（やり直しで2）、最後の検証の誤り |
| `place_status` / `place_success` / `place_s` | `PlaceObject` の結果と時間（wall clock） |
| `goals` / `base_move` / `compromise` | 送った `PlaceObject` の結果の順、台車の移動の誤差（または失敗の理由）、近さの制限を外して置いたか |
| `placed_level` / `d_same` / `d_other` | 置いた物体の底面がある段（棚の外は0）、その段の同じカテゴリ・別のカテゴリの最も近い物体との中心間の水平距離 |
| `placed_ok` | 置けて、正解の段で、グループがあれば `d_same` 0.15 m以内かつ `d_same` < `d_other` |
| `knocked` / `robot_moved` | 2 cm以上動いた棚の物体、試行の間にodometryが見ていないロボットの移動（押された、車輪のずれ） |
| `observe_s` / `llm_s` / `trial_s` | 観測（頭を段ごとに向けて要約が安定するまで）、LLM（呼び出しの合計）、試行全体の時間（wall clock） |
| `stable` / `stale_looks` | 観測の終わりに要約が新しい面の検出で変わらなくなっていたか、新しい検出が届かなかった頭の向きの数 |
| `prompt_tokens` / `completion_tokens` | LLMのトークン数（呼び出しの合計） |
