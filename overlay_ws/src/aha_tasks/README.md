# aha_tasks

タスクをサブゴールに分けて実行・可視化する共通基盤と、RoboCup@Homeのタスク定義。

| 構成 | 内容 |
| --- | --- |
| `aha_tasks/core/` | ROS非依存の実行基盤。タスク定義の検証、サブゴールの選択、チェックポイント・ロールバック |
| `aha_tasks/skills.py` | 組み込みスキル `navigate` と `robot.location` の復元 |
| `aha_tasks/people.py` | 人の認識（[person_perception](../aha_perception/README.md#人の認識)）を使うスキル |
| `aha_tasks/objects.py` | 物体の検出（`/aha/perception/objects`）から台の上の物体と棚の分類を求めるスキル（[物体の認識](#物体の認識)） |
| `aha_tasks/gaze.py` | スキルの頭の向き先。人・Nav2の経路の先を追い続ける（[注視](#注視)） |
| `aha_tasks/gestures.py` | 右腕で席を指し示す `offer_seat`（[席の案内](#席の案内)） |
| `aha_tasks/guests.py`、`sim_guests` | シミュレーションの客の動作（ROS非依存）と、Gazeboで客を動かすnode（[シミュレーションの客](#シミュレーションの客)） |
| `aha_tasks/sim_objects.py`、`sim_objects` | シミュレーションの物体のモデルと配置の検証（ROS非依存）と、Gazeboの家具に物体を置くプログラム（[シミュレーションの物体](#シミュレーションの物体)） |
| `aha_tasks/teleport.py`、`sim_teleport` | デバッグ用。Nav2の代わりにGazeboでロボットを瞬間移動させる `NavigateToPose` のserver（[e2eの確認](#e2eの確認)） |
| `aha_tasks/handling.py` | 物体を取る・置く・人へ渡すスキル（[物体を取る・置く・渡すスキル](#物体を取る置く渡すスキル)） |
| `aha_tasks/sim_manipulation.py`、`sim_manipulation` | デバッグ用。取る・置く・渡すserverの代わりにGazeboで物体を動かすnode（[シミュレーションの把持](#シミュレーションの把持)） |
| `run_task` | runnerの準備を待って実行を始め、終了まで待って各サブゴールの所要時間を出すCLI（[e2eの確認](#e2eの確認)） |
| `aha_tasks/plan.py` | 計画の検証と、計画の手順を各スキルへ振り分ける `run_plan_step` |
| `aha_tasks/display.py`、`web/display/` | ロボットの画面の状態（ROS非依存）とROSの入出力、画面のページ（[ロボットの画面](#ロボットの画面)） |
| `aha_tasks/interaction.py` | スキルからの発話とタッチの質問、スキル `say` / `wait_start_signal` |
| `task_runner` | タスク定義を実行し、状態を配信・操作を受け付けるnode |
| `task_monitor` | 進捗表示・部分実行・失敗時の操作を行うWeb GUIと、ロボットの画面 |
| `config/<task>.json` | タスク定義 |
| `config/sim/<task>.json` | シミュレーションの客とドアの配置 |
| `config/sim/objects/<scene>.json`、`config/sim/object_models.json` | シミュレーションの物体の配置と、ラベルごとのモデル |
| `launch/task.launch.py` | RCJO2026のGazebo、固定地図のNav2、runner、GUI、`head_look_at.py` を起動 |

ROSの入出力は [ROSインターフェース](../../../docs/interfaces.md#サブゴールタスクの実行)を参照。

## 起動

[開発コンテナ](../../../docs/docker.md)で実行する。別のsim / SLAMと同時に起動しない。

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_tasks
source install/setup.bash
ros2 launch aha_tasks task.launch.py task:=human_interaction
```

GUIは `http://localhost:8765/`（macOSは [`make forward`](../../../docs/docker.md#2-guiに接続する)）。
ロボットの画面は同じportの `/display/` で、GUIの右の欄にも同じ画面を表示する。
Nav2がACTIVEになると待機状態になり、GUIから実行する。runnerが終了するとlaunch全体が終了する（`e2e:=debug` を除く）。
ロボットの初期位置（Gazeboのspawn、AMCLの初期姿勢）はタスク定義の `initial_state` の `robot.location`。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `task` | 必須 | `config/` のタスク名、またはJSONのパス |
| `e2e` | `final` | `final` はNav2とDGXのモデル、`debug` はシミュレーションの代替（[e2eの確認](#e2eの確認)） |
| `navigation` / `perception` | 空（`e2e` による） | `e2e:=debug` で代替を個別に戻す。`nav2` / `teleport`、`model` / `truth`。`final` は `nav2` / `model` だけ |
| `manipulation` | 空（`e2e` による） | `sim` は `sim_manipulation` を起動（`debug` の既定）、`none` は起動しない（`final` の既定。manipulation担当のserverを別に起動する） |
| `headless` | `false` | Gazebo GUIを起動しない |
| `gui_port` | `8765` | Web GUIのport |
| `people` | `false` | 頭部カメラのbridgeと[person_perception](../aha_perception/README.md#人の認識)（DGX Spark）を起動 |
| `objects` | `false` | 頭部カメラのbridgeと物体の検出（`/aha/perception/objects`、frame `map`）を起動し、`world` → `map` の恒等の静的TFを配信する（[物体の認識](#物体の認識)） |
| `ssh_host` / `ssh_port` / `ssh_user` | `10.99.0.1` / `2222` / 空 | `people:=true`・`objects:=true` のDGXの接続先 |
| `worker_command` | 空 | `people:=true` でDGXの代わりに起動する人のworkerのコマンド（[人の認識](../aha_perception/README.md#人の認識)） |
| `object_worker_command` | 空 | `objects:=true`（`perception:=model`）でDGXの代わりに起動する物体のworkerのコマンド（[物体の認識](../aha_perception/README.md#物体の認識)） |
| `guests` | `false` | [シミュレーションの客](#シミュレーションの客)を動かす `sim_guests` を起動。`config/sim/<task>.json` が必要 |
| `people_scene` | 空 | [姿勢を付けた人](#姿勢を付けた人)を置く `sim_people` を起動。`config/sim/people/` の名前、またはJSONのパス |
| `object_scene` | 空 | [シミュレーションの物体](#シミュレーションの物体)を家具に置く `sim_objects` を起動。`config/sim/objects/` の名前、またはJSONのパス |
| `subgoals` | 全て | 既定の選択 |
| `on_failure` | `pause` | 失敗時のポリシー |
| `max_retries` | `1` | ポリシー `retry` の再実行回数 |
| `startup_timeout` / `goal_timeout` | `300.0` / `600.0` | Nav2の起動待ちと各移動の上限（実時間の秒） |

## 実行の仕組み

- **部分実行**: 選択外のサブゴールは実行せず、スキルの期待状態と `mock` をタスクの記憶へ適用する。未実装のスキルも同様に通過する。
- **状態の同期**: スキルの実行前に、記憶上の `robot.location` と実際の位置が異なればNav2で移動する。
- **ロールバック**: 失敗するとタスクの記憶とロボットの位置をそのサブゴールの開始前へ戻す。人やドアなどは手で戻す（`guests:=true` の客とドアは戻したサブゴールに合わせて移動する）。
- **失敗時のポリシー**: `pause` は停止してGUIで再試行・巻き戻し・スキップ・中断を選ぶ。`retry` は `max_retries` 回まで再実行、`skip` は `mock` を仮定して続行、`abort` は終了する。

経過時間は制限時間と並べて表示するが、採点・時間切れの判定はしない。

![1人目の客を部分実行したWeb GUI](../../../docs/images/aha-tasks-web-gui.png)

## e2eの確認

シミュレーションのタスクの通しは、`e2e` で2つを使い分ける。

| | `e2e:=final`（既定） | `e2e:=debug` |
| --- | --- | --- |
| 移動 | Nav2（0.2 m/s） | `sim_teleport`：先にAMCLへ `/initialpose` を送り、AMCLのTFが目標の0.05 m・0.05 rad以内になった時点でGazeboの `set_pose` でロボットを移す。1.5 s後に成功。`guests:=true` では、移した後に客の歩く速さ（1.0 m/s）で直線距離を進む時間（1.5 s以上）待って成功する（ついて歩く客は、ロボットが移ってから歩き出す）。待ちは模擬時間で数え、`/clock` が10 s進まないときに失敗する |
| 人の認識（`people:=true`） | `person_perception.py` とDGXのSAM 3.1・VLM | [sim_person_detector.py](../aha_perception/README.md#シミュレーションの人の検出)：Gazeboの真の位置から、頭部カメラに見える客を出す。DGXを使わない |
| 物体の検出（`objects:=true`） | `aha_perception` の `objects.launch.py`（画像からの検出器） | [sim_object_detector.py](../aha_perception/README.md#シミュレーションの物体検出)：Gazeboの真の姿勢から、頭部カメラに見える物体リストの物体を出す |
| 取る・置く・渡す | manipulation担当のserver（このlaunchは起動しない） | [sim_manipulation](#シミュレーションの把持)：把持を計画せず、Gazeboで物体のモデルを手先へ移す・置く・消す |
| runnerの終了 | launch全体が終わる | runnerと人の認識のnodeは止めると2 s後に再起動し、Gazebo・AMCL・客はそのまま |

`perception:=model` を付けると、`debug` でも人の認識と物体の検出は `final` と同じになる（移動だけを省く）。
`people_scene` の人を探す・数えるには `perception:=model` が要る（`sim_person_detector.py` は `classify_person` を持たず、客だけを出す）。`people:=true` と `perception:=truth` に `people_scene` を付けると起動時にエラーにする。
`debug` で確かめられないため、`final` で確かめること：

- **移動**: 経路の有無（開かないドア、壁の向こうの地点）、Nav2の失敗・時間切れ、移動中のAMCLのずれ、移動中に見える人。`sim_teleport` は目標の、地図上のロボットの半径（0.40 m）に占有・未知のセルがないことだけを見る。AMCLのTFは2 s先の時刻で配信されるため、1回の移動に約4 sかかる。移した直後の1–2枚の画像は、前後の位置が混ざったTFで測られる。
- **人の認識**（`perception:=truth` のとき）: SAM 3.1の見落とし・誤検出・並んだ人の結合、マスクとDepthから求める位置、VLMの特徴・照合の誤り、DGXの往復の遅れ。人は高さ1.75 mの立った体の軸として扱い、名前は覚えたモデルに直ちに付く。
- **客の動き**: ロボットは待った後に一度に移るため、案内される客はロボットの経路をたどらず、壁を抜けて歩く。
- **物体の検出**（`perception:=truth` のとき）: 見落とし・誤ったラベル・遮蔽された物体の位置と大きさの誤差。真値は物体リストの大きさとGazeboのモデルの姿勢を出す。
- **取る・置く・渡す**（`manipulation:=sim` のとき）: 把持・置く動作の計画と失敗、腕と家具・物体の衝突、落下、人が受け取ったことの検知。代替は届く距離だけを見る。

```bash
# 1回目：起動して全体を実行（コンテナ内）
ros2 launch aha_tasks task.launch.py task:=human_interaction people:=true guests:=true e2e:=debug headless:=true
ros2 run aha_tasks run_task --on-failure skip --output /tmp/run.json  # 別のシェル

# 2回目以降：変えたnodeだけを止め（launchが再起動する）、対象のサブゴールだけを実行する
pkill -INT -f 'lib/aha_tasks/[t]ask_runner'  # スキル・タスク定義を変えた
pkill -INT -f '[p]erson_perception.py'       # perception:=model で認識を変えた（DGXのworkerも作り直す）
ros2 run aha_tasks run_task --subgoals 'introduce.*' --on-failure skip
```

HICの最初の `start.signal` は画面の `Start` のタッチを待つため、GUIの右の欄の画面で押すか、`--subgoals 'guest1.wait..'` で除く。
`run_task` はrunnerがReadyになるのを待って（再起動中なら次のrunnerを待って）始め、止まると各サブゴールの実時間（`robot.location` の復元を含む）、スキルごとの合計、全体の時間を出す。終了・失敗のほか、ポリシー `pause` で一時停止したとき（GUIで選ぶ）と、実行中にrunnerが再起動したときも止まる。完了のときだけ終了コード0。
`--output` は要約・最後の状態・ログをJSONに書く。`final` でも同じように使える（runnerの終了でlaunchが終わるため、毎回起動する）。
Pythonだけの変更は `--symlink-install` のビルドならそのまま反映される。ファイル・msgの追加ではビルドしてから止める。
`run_task` が `Runner not ready in time` で終わるときは、launchのログでrunnerの起動エラーを見る（タスク定義・スキルの誤りで起動前に終わると、状態を配信しないまま再起動を繰り返す）。
再起動したrunnerはロボットが `initial_state` の位置にいるとみなすため、最初に実行するサブゴールの記憶上の位置が初期位置なら、ロボットは移動しない。
人の認識のnodeは覚えた名前を実行をまたいで保つ。客を覚えるより前のサブゴールから走らせ直すときは、人の認識のnodeも止める（`pkill -INT -f '[s]im_person_detector.py'` か上の `person_perception.py`）。名前の付いた客は `find_person` の対象にならないため。

## タスク定義

[human_interaction.json](config/human_interaction.json) を参照。未知のキー・地点、不正な値は起動時にエラーになる。

| キー | 内容 |
| --- | --- |
| `task` / `title` / `rulebook` / `time_limit` | 識別子、表示名、ルールの出典、制限時間（秒） |
| `locations` | `map` frameの地点 `[x, y, yaw]`（m、rad）。障害物から0.6 m以上離す（壁や家具に寄せるとNav2が止まる） |
| `initial_state` | 開始時のタスクの記憶 |
| `skills` | スキル名と実装 `"module:Class"`。`null` は未実装 |
| `subgoals` | 実行順のサブゴール。`id`、`title`、`group`、`skill`、`params`、`mock`、`points`、`note` |

選択は `,` 区切りで、id（`guest1.ask`）、glob（`guest1.*`）、範囲（`guest1.find..guest1.seat`、片側省略可）を組み合わせる。
ROSなしで定義と選択を確認できる。

```bash
# リポジトリ直下
PYTHONPATH=overlay_ws/src/aha_tasks python3 overlay_ws/src/aha_tasks/scripts/task_runner \
  --config overlay_ws/src/aha_tasks/config/human_interaction.json --dry-run --subgoals 'guest2.*'
```

## スキルの追加

`aha_tasks.core.Skill` を継承し、タスク定義の `skills` に登録する。
インスタンスはスキル名ごとに1つで、`self.runtime` から `node`・`spec`・`navigation`・`gaze`（[注視](#注視)）・`skills`（名前からインスタンス。未実装は `None`）を使える。

```python
from aha_tasks.core import Skill, SkillResult


class AskNameAndDrink(Skill):
    @classmethod
    def validate(cls, params, spec):  # 起動時。不正ならValueError
        ...

    def ready(self):  # 依存するserverを待つ
        ...

    def execute(self, ctx):  # ctx.params / memory / cancelled / log
        guest = ctx.params["guest"]
        return SkillResult(True, "", {f"{guest}.name": name, f"{guest}.drink": drink})
```

成功時の `outputs` はタスクの記憶へ追加され、後続が `ctx.memory` から読む。
`expected(params)` は成功後とmock時に記憶へ追加する状態。ロボットを動かすスキルは `robot.location` を出力する。
実行しない場合の状態が記憶で変わるスキルは `assumed(params, memory)` を上書きする（既定は `expected(params)`）。
`execute` はタスク用threadで呼ばれる。`ctx.cancelled` を監視し、中断時は速やかに戻る。

## ロボットの画面

`task_monitor` が `/display/` でロボットのタッチディスプレイの画面を出す。
シミュレーションでは、GUIの右の欄の画面をクリックしてタッチの代わりにする。
文言は英語。ROSの入出力は [ROSインターフェース](../../../docs/interfaces.md#ロボットの画面)を参照。

| 表示 | 内容 |
| --- | --- |
| 字幕 | ロボットの発話の最新3件。最新を大きく表示する |
| 画像 | `/aha/display/image` の1枚（認識の結果など）。質問と同時なら横に並べる |
| 質問 | 文と1〜6個のボタン。タッチすると押したボタンを上部に表示する |
| 待機 | 何も出していなければ `AhaRobot` の文字 |

サーバーとの接続が切れると上部に `Not connected` を出し、ボタンを押せなくする。

スキルは `aha_tasks.interaction.Interaction.of(self.runtime)` から使う。

```python
interaction = Interaction.of(self.runtime)
ok, message = interaction.say("What is your name?", ctx.cancelled)
ok, message = interaction.show(
    jpeg_bytes, "jpeg", "I found the cup on the table.", ctx.cancelled
)
index, label = interaction.ask("Ready to start", ["Start"], ctx.cancelled)
```

- `say` は字幕を出し、TTS（`/aha/perception/audio/speak`）が起動していれば読み上げ終わるまで待つ（上限60 s）。TTSがなければ字幕だけで戻る（シミュレーション）。起動したばかりのrunnerはTTSをDDSで見つけるまで時間がかかるため、最初の1回だけ最大2 s待つ（`say` と `wait_start_signal` は `ready()` で待つ）。
- `show` は画像（JPEGかPNGのbytes）を出し、`say` で説明する。画像は次の画像か `hide()` まで残る。認識の結果は `show` で審判に示すだけで、タッチで答えを求めない。
- `ask` は画面にボタンを出し、タッチされると番号と文言を返す。タッチがなければ `index` が `None` で、`label` が理由（中断、`timeout` 秒の経過、置き換え、画面の終了）。`timeout` は実時間で、0以下は中断まで待つ。
- 画面に出る質問は1つで、新しい質問は表示中の質問を置き換え、前の質問はタッチなしで終わる。
- serverを見つけた直後に送ったgoalやその応答は失われることがある。`ask` は受け付けの応答が5 s以内になければ、serverが見えている間は質問を送り直す。送り直す前の質問がタッチされていれば、画面に出し直さずにそのタッチを返す。`say` は二重に読み上げないよう送り直さない。
- 画面の起動を5 s待っても `/aha/display/ask` がなければ、`ask` はタッチなしで終わる。待つ間も中断を受け付ける。
- `say` と `ask` は、待っている間にserver（TTS・`task_monitor`）が5 s続けて見えなくなると諦める。異常終了したプロセスがROSのグラフから消えるまで、DDSのリースの約20 sがかかる。

| スキル | 動作 | 使うサブゴール |
| --- | --- | --- |
| `say` | `text` を字幕に出し、TTSがあれば読み上げる | GPSRの `command.mode` |
| `wait_start_signal` | 準備ができたことと押すボタンを `text` で伝え、`button`（既定 `Start`）のタッチを待つ。どちらも省略できる | HICの `start.signal` |

TTSで読み上げられなかった場合も、`wait_start_signal` はボタンを出して待つ。
タッチを待つスキルは、HICの開始合図の `wait_start_signal` だけ。`show` を使うスキルはまだない。

`test/test_interaction.py` のROSのテストは `ROS_DOMAIN_ID=95`（`AHA_TEST_DOMAIN_ID` で変更可）で動かす。

## シミュレーションの物体

`object_scene:=<名前>` は [config/sim/objects/](config/sim/objects/) の配置で、物体をGazeboの家具の上に置く（`sim_objects`）。
物体は動く剛体で、ロボットが押したり取ったりできる。スキルには配置を渡さず、物体は頭部カメラと物体検出からだけ得る。

```bash
ros2 launch aha_tasks task.launch.py task:=pick_and_place object_scene:=pick_and_place
ros2 run aha_tasks sim_objects --config pick_and_place          # 起動中のGazeboで、動いた物体を置き直す
ros2 run aha_tasks sim_objects --config pick_and_place --check  # 置き直さずに、無い・動いた・倒れた・配置にない物体を出す
```

| 配置 | 内容 |
| --- | --- |
| `pick_and_place` | [ルールブック §5.2](https://robocupathome.github.io/RuleBook/rulebook/master.pdf) の配置（24個）。食卓（`dining_table`）にcutlery・mug・plate・ゴミ（`ppc.trash_category` のカテゴリ）・その他2個、ゴミ箱の近くの床にゴミ、`tall_table` にボウル・スプーン・洗剤タブ、補助台（`long_table`）にCommon Objects 2個。棚（`shelf`）は段ごとに1カテゴリで、2段目がcleaning supplies、3段目がfood（シリアルを含む）、4段目がdrink（牛乳を含む）、5段目（天板）がfruit。1段目（床の高さ）は空き。食卓のその他2個（apple、cola）は棚の同じカテゴリの段の隣に置ける |
| `gpsr` | GPSRのmockの計画用（18個）。`kitchen_table` にコーラ・果物・ボウル、`desk` に飲み物3個とルービックキューブ、`side_tables` にお菓子2個、`shelf` にfood・drink・fruit |

配置のJSONは、物体ごとに次のキーを持つ。

| キー | 内容 |
| --- | --- |
| `name` | Gazeboのモデル名。[物体リスト](../aha_perception/config/objects/robocup_objects.yaml)のパターンに合わせ、`sim_object_detector.py` がそのラベルで出すようにする（例 `red_bull`、`cola_2`） |
| `label` | 物体リストのラベル |
| `on` / `level` | 置く家具（[家具の設定](../aha_perception/README.md#家具の設定) `rcjo2026_arena.yaml` の名前）か `floor`。段が複数ある家具は `level`（1が一番下）も書く |
| `position` / `yaw` | 物体の中心の `[x, y]`（Gazeboのworld。`map` と同じ、m）と向き（rad）。高さは段の高さと物体の大きさから決まる |
| `role` | 省略可。正解の役割。PPCでは記憶のキー（`table.cutlery`、`trash.floor`、`breakfast.milk` など） |

読み込み時に、家具の縁から1 cm以内に掛かる物体、上の棚板との隙間が1 cm未満の物体、同じ面で1 cm以内に近づく物体（向きを付けた外接箱で判定）、家具の外形に掛かる床の物体を拒否する。
正解（物体の中心・大きさ・カテゴリ・段・役割）は `aha_tasks.sim_objects.load_scene` で読める。

モデルは [config/sim/object_models.json](config/sim/object_models.json) でラベルに割り当てる。

- `ycb`: `tmc_wrs_gz_worlds` のYCBのモデル（メッシュ・質量・慣性）。`yaw`（度）でメッシュをモデルの中で回して辺をx・yに揃え、原点を衝突メッシュの外接箱の中心へ移す。外接箱と物体リストの大きさの差は各軸1.5 cm以内（テストで確かめる）。
- `printed`: YCBにない物体。物体リストの形・大きさの箱・円柱をcollisionとし、`text` と色を印刷したテクスチャのメッシュを付ける。メッシュとテクスチャは `~/.cache/aharobot/sim_objects` に作る。

原点は外接箱の中心で、モデルの姿勢を物体の中心とする `sim_object_detector.py` の前提に合う。摩擦係数は1.0。
`sim_objects` は配置の物体を作り（同じ名前のモデルが残っていれば位置を戻す）、3 s後に置いた位置から2 cm以上ずれた・10°以上傾いた物体を警告して終わる。全部の物体があれば終了コード0。
物体リストのパターンに合う名前で配置にないモデル（同じGazeboで先に置いた別の配置の物体など）は、置く前に消す。家具・ロボット・客・人のモデルは消さない。`--check` は消さずに警告する。

制限：

- 失敗で巻き戻しても物体の位置は戻らない。`sim_objects` を実行し直して戻す。
- 頭部カメラのセンサは `people:=true` か `objects:=true` のときだけ有効になる。
- 印刷した物体は箱・円柱で、ロゴや形の細部はない。YCBの物体と物体リストの大きさは最大1.2 cm違う（tuna canの高さ）。
- 食洗機と、その中の物体はない。

## Human Robot Interaction Challenge

[ルールブック §5.1](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を28サブゴールに分ける。
`points` は得点表の最大点で、実行中に継続して求められる項目（注視など）はそのサブゴールに含める。
実装済みのスキルは `navigate`、[ロボットの画面](#ロボットの画面)を使う `wait_start_signal`、人の認識を使う `find_person` / `memorize_person` / `locate_guests`（`people:=true` が必要）と、`offer_seat`。

ロボットは場内から始めるため、最初の `start.signal` で審判の手動の開始合図（§3.6.8）として画面の `Start` のタッチを待つ。
部分実行で `start.signal` を選ばなければ合図を待たない。

客席・ホスト・荷物置き場は固定の代替地点。ロボットは内側の `entrance` まで移動する。
`guests:=true` でなければ入口のドアは閉じたままで、人のモデルは配置しない。ホスト・荷物のモデルは配置しない。
地点は `/aha/tasks/locations` に配信し、RVizで表示できる。

```bash
rviz2 -d "$(ros2 pkg prefix --share aha_navigation)/rviz/nav.rviz" --ros-args -p use_sim_time:=true
```

### 人の認識を使うスキル

`/aha/perception/people` の人をスキル間でIDと名前で受け渡す。出力の `<役割>.position` は人の体の中心（`<役割>.frame` の座標）。
person_perceptionが起動していない場合は、実行時に失敗する（起動時の待機はしない）。発話は含まない。

| スキル | 動作 | 出力 |
| --- | --- | --- |
| `find_person` | 名前のない人のうち最も近い人を見て、`distance`（既定1.0 m）手前まで近づき向き合う。人が `map` 以外の座標なら見るだけ | `<role>.track` / `.position` / `.frame`、`robot.location` |
| `memorize_person` | `<guest>.track` の人（いなければ名前のない最も近い人）の外見を `guest` の名前で覚え、VLMで特徴を求める | `<guest>.appearance`（英語句のリスト）、`.track` / `.position` / `.frame` |
| `locate_guests` | 頭を正面・左・右へ `dwell`（既定10 s）ずつ向け、`guests` の全員を名前付きで見るまで待ち、最後に頭を正面へ戻す。名前はVLMが外見で照合して付けるため、席を入れ替えても区別する | 各客の `.track` / `.position` / `.frame` |

`find_person` と `memorize_person` は、終了後も頭をその人へ向け続ける（[注視](#注視)）。

待つ時間は `timeout`（`find_person` は30 s、`locate_guests` は60 s、`memorize_person` は10 s）、VLMは `describe_timeout`（180 s）で変更できる。`timeout` と `dwell` は模擬時間が有効ならノードの時計、それ以外はシステム時刻の補正に影響されない単調時計で数える。
模擬時間が実時間で10 s進まなければ、`/clock` の停止を理由に失敗する。
`locate_guests` は `map` 座標の観測だけを時点をまたいで蓄積する。他の座標では1つのメッセージ内で全員が見える必要がある。
同じtrack、または位置が0.4 m未満（`map` では水平距離）の客は重複とみなし、古い観測を捨てて探し続ける。
出力の `.track` は破棄済みのことがあるため、後続スキルは名前か位置を使って客を再取得する。
`memorize_person` をmockにした場合は外見を覚えないため、後の `locate_guests` はその客を見分けられない。

### シミュレーションの客

```bash
ros2 launch aha_tasks task.launch.py task:=human_interaction people:=true guests:=true ssh_user:=<Unixユーザー名>
```

`sim_guests` が `/aha/tasks/status` のサブゴールの状態に合わせ、[config/sim/human_interaction.json](config/sim/human_interaction.json) の配置で2人の客とドアをGazeboの中で動かす。
スキルには真値を渡さず、客の位置は頭部カメラと `person_perception` からだけ得る。
未実装のスキルがmockで一瞬で通過しても、状態から段階を決めるため客の動作は飛ばない。

| サブゴールの開始 | 客の動作 |
| --- | --- |
| 開始前 | 床下に隠れる。worldの壁は高さ0.8 mで、頭部カメラから壁の外の人が見えるため |
| `<guest>.doorbell` | 家の外からドアの外側へ歩き、中を向いて待つ |
| `<guest>.open_door` | ドアが外へ開く。以後は開いたまま |
| `<guest>.ask` | ドアの外側で左右に0.2 m（周期6 s）動き、ロボットを向く。ルールのとおり、会話が始まるまで家には入らない |
| `<guest>.guide` | ロボットとの距離を1.2 m保ち、ロボットが通った経路を歩いて家に入る |
| `<guest>.seat` | 席の前へ歩き、部屋を向いて立つ |

巻き戻しと、選択外で歩く間のない段階への移行は、歩かずに移動する。歩く速さ1.0 m/sはシミュレーション時刻。

制限：

- 客は静止した立ち姿のモデル（`person_standing`）を動かすもので、脚は動かず、座らない。シャツなどの色で2人を区別し、その他の外見は同じ。
- 客はvisualだけでcollisionがなく、ロボットを押さない。LiDARとカメラには写る。障害物は避けず、固定の経路と追従で歩く。
- 客の測定位置はロボットの自己位置の誤差の影響を受ける。
- 待っている他の客・ホスト・荷物は含まない。

### 注視

runnerの `gaze`（`aha_tasks.gaze.Gaze`）が、スキルの頭の向き先を `/aha/perception/look_at` へ送る。
`task.launch.py` の `head_look_at.py` が頭を向け、`map` の点は台車が動いても向け直し続ける。
追う動作は次の指示まで続き、サブゴールをまたぐ。
タスクが一時停止・中断・終了すると追う動作をやめ、操作パネルやteleopで頭を動かせるようにする。再開後は、スキルが次に指示するまで最後の向き先を保つ。

| 指示 | 向き先 | 使うスキル |
| --- | --- | --- |
| `follow_person(people, track, name)` | 人の顔（`head` の0.1 m下。`head` はマスクの上端）。trackが消えたら名前で、それもなければ直前の位置から0.5 m以内の名前のない人を引き継ぐ。見失っている間は最後の点を保つ | `find_person`、`memorize_person`、`offer_seat` の後 |
| `follow_path(goal)` | Nav2の経路 `/plan` に沿って1.5 m先、高さ1.2 m。経路の終わりの先は経路の始点から終点への向き（進行方向）へ延ばし、始点と終点が0.3 m未満（その場で目標の向きへ回る）なら目標の向きへ延ばすため、到着時は正面を見る。新しい経路が届くまでは、目標地点から目標の向きへ1.5 m先（すでに目標にいて経路を待たずに終わっても、頭が真下や真横を向かない） | `navigate` と `robot.location` の復元（到着後は `hold`） |
| `look_at(frame, point)` / `look_towards(pan)` | 固定の点。追う動作をやめる | `locate_guests`、`offer_seat` |
| `hold()` | 最後の向き先を保つ | `navigate` の到着後、タスクの停止時（runner） |

`find_person` で向けた客を、未実装の `ask_name_and_drink` の間も見続ける（ルールの「会話中に動く客を追う」）。
`navigate` の間は進行方向を見る（「進行方向または目的地を見る」）。
`navigate` と `robot.location` の復元はどのタスクでも頭を動かし、到着後は経路の先の点を `map` で保つ。到着後に机や床などを見るスキルは、`look_at` か `look_towards` で向き先を指示する。

### 席の案内

`offer_seat` は `position`（`map` の席の点 `[x, y, z]`）を頭で見て、右腕で指し示す。

```json
{"guest": "guest1", "seat": "chair_left", "position": [1.75, 3.5, 0.5], "height": 0.8, "hold": 3.0}
```

1. 席が右腕で指せる方向（肩から見て右100°〜左60°）になければ、Nav2でその場で席へ向く。
2. `lift_controller` があれば手首を床から `height` m（既定0.8 m）へ上げ、腕を席へ伸ばす。手先のx軸が席の点を通る関節値を [aha_manipulation.pointing](../aha_manipulation/aha_manipulation/pointing.py) が求める。
3. 腕が着いたら現在のTFで関節値を求め直し、移動直後に台車がずれていれば向け直す。`hold` 秒（既定3 s、模擬時間）保ち、腕と昇降を開始時の姿勢へ戻す。頭は客（`<guest>.track`、名前）を追う。指し示す途中で失敗・キャンセル・例外（controllerやTFの時間切れ）があっても、腕と昇降を開始時の姿勢へ戻してから失敗を返す。

出力は `<guest>.seat`（席の名前）。`robot.holding` があるとき（`guest2` は荷物を持つ）は腕を動かさず、頭だけで席を示す。
腕は `right_arm_controller` / `lift_controller` へ関節空間の直線の軌道を送り、衝突は確認しない。開始時の姿勢から指す姿勢までの間に障害物がないことが前提で、シミュレーションの初期姿勢（全関節0）からの動作だけを確認している。
空席の判定は含まず、席は固定の代替地点。

## Pick and Place Challenge

[ルールブック §5.2](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を71サブゴールに分ける。
実装済みのスキルは `navigate` と[物体の認識](#物体の認識)の `perceive_objects` / `perceive_shelf`（`objects:=true` が必要）で、他はmockで通過する。

```bash
ros2 launch aha_tasks task.launch.py task:=pick_and_place subgoals:='dishwasher.*'
```

既定の順序は、ゴミ、食洗機（扉の開閉は1回）、棚、朝食、補助台。物体は1個ずつ運ぶ。
`points` は得点表の最大点で、合計は総計の3515。負の項目は `note` に書く。
得点表の認識・取得・配置は12個分で物体は14個あるため、補助台の2個を枠の外として0 ptにする（`note` に数える場合の点を書く）。
First Pick Bonusは既定の順序で最初に取る `trash.table.pick` に載せる。

| 地点 | 使い方 | sim上の場所 |
| --- | --- | --- |
| `dining_table` | 食卓 | `dining_table` の+y側 |
| `dishwasher` | 食洗機 | worldにないため、食卓の近くの壁（`inside_wall_center_vertical_front`）の前に置いた代替地点 |
| `trash_box` | ゴミ箱と床のゴミ | `trash_box` の−x側 |
| `shelf` | 棚 | `shelf` の−y側 |
| `tall_table` | ボウル・スプーン・洗剤タブの指定の台 | `tall_table` の−y側 |
| `long_table` | 補助台 | `long_table` の−y側 |

入口のドアは閉じたままで、ロボットは内側の `spawn` から始める。
物体は `object_scene:=pick_and_place` で置く（[シミュレーションの物体](#シミュレーションの物体)）。

### 記憶の受け渡し

`perceive_objects` は `<output>.<役割>` に物体を出力し、`pick_object` / `place_object` の `object` はそのキーを指す。
把持中の物体は `robot.holding` に持ち、置くと `null` になる。

| サブゴール | 出力 |
| --- | --- |
| `table.perceive` | `table.cutlery` / `.cup` / `.plate` / `.trash` / `.other1` / `.other2` |
| `trash.floor.perceive` | `trash.floor` |
| `tab.perceive` | `dishwasher.tab` |
| `cabinet.perceive` | `cabinet.categories` / `.groups` / `.placement`（棚の段・左右のカテゴリと、`table.other1` / `.other2` の置き場所） |
| `breakfast.perceive_tableware` / `.perceive_food` | `breakfast.bowl` / `.spoon` / `.cereal` / `.milk` |
| `extra.perceive` | `extra.obj1` / `.obj2` |

物体の値は次の辞書（[物体の認識](#物体の認識)）。見つからなかった役割のキーは出さない。

```json
{"label": "mug", "category": "tableware", "position": [-1.76, -1.55, 0.841], "frame": "map",
 "size": [0.106, 0.091, 0.081], "track": 12, "confidence": 1.0, "yaw": 0.5}
```

`position` は物体の中心、`size` は物体自身の軸に沿った外接箱（m、物体リストの底面 x, y と高さ z）、`track` は検出器のtrack ID、`confidence` は同じラベルの検出の平均。
`yaw` は `frame` のz軸まわりの物体の向き（rad、同じラベルの検出の円周平均）で、`size` の軸の向きを表す。
真値の検出器はモデルの向きを出す。画像からの検出器（`object_perception.py`）は `frame` の軸に沿った箱（単位四元数）を出すため、`yaw` は0になる。
ゴミのカテゴリはセットアップ日に発表されるため、`initial_state` の `ppc.trash_category` に設定する。
タスク固有のスキルのparamsは `open_container`（`object`）、`pour`（`object`、`target`）。

### 物体の認識

`aha_tasks/objects.py` の2つのスキルは `/aha/perception/objects`（[ROSインターフェース](../../../docs/interfaces.md#物体検出)）を購読し、
台を頭で見回して検出を集め、台の上の物体を求める。検出器の起動はlaunchの `objects:=true`。真値の検出器も画像からの検出器も同じtopicに出す。

```bash
ros2 launch aha_tasks task.launch.py task:=pick_and_place object_scene:=pick_and_place objects:=true e2e:=debug headless:=true
ros2 run aha_tasks run_task --subgoals '*.perceive*' --on-failure skip  # 別のシェル
```

選択しない移動はmockで通過し、各認識の前にrunnerがロボットをその地点へ移す（[実行の仕組み](#実行の仕組み)の状態の同期）。

| スキル | params | 動作 |
| --- | --- | --- |
| `perceive_objects` | `surface`、`output`、`roles`、`margin`、`exclude` | `surface` の上の物体を、`roles` の順に役割へ割り当てる。1つ以上の役割が見つかれば成功し、見つからない役割は結果のメッセージに出す。終了後は `roles` の順で最初に見つかった役割の物体の方を見る（1段の台では下記の見る点のうちロボットに最も近い点、棚ではその物体の段の0.1 m上、床ではその物体） |
| `perceive_shelf` | `surface`、`objects`、`output` | 棚の各段の物体を左右の群に分け、群のカテゴリと、`objects`（記憶のキー）の物体の置き場所を出す。頭は最上段を向いたまま終える |

- `surface` は家具の設定（[aha_perceptionの家具の設定](../aha_perception/README.md#家具の設定)）の家具の名前か `floor`。
  家具の外形を `margin` m広げた範囲（食卓は審判が50 cmまで動かすため0.5 m）、`floor` はロボットから `margin` m以内を探す。
- `exclude` の家具（家具の設定の名前）の外形を0.05 m広げた範囲の物体は、高さによらず候補にしない。
  `trash.floor.perceive` は `["trash_box"]` で、直前にゴミ箱へ入れた食卓のゴミ（床の近くにあり、ゴミのカテゴリ）を床のゴミとして選ばない。
- 物体はその中心が段の高さ − 0.02 m以上で、底（中心 − 箱の高さの半分）が段の高さ + 0.15 m以下なら、その段にある（皿の上の物体、横に倒れた物体も含む）。
  条件を満たす段が複数あれば、最も高い段とする。
- 頭は、1段の台では上面の中心（長さ1.3 mを超える辺は両端から0.3 m内側の点も）をロボットに近い順に、棚では各段の0.1 m上を下の段から、床ではロボットの前方0.8 mを正面・左0.5 rad・右0.5 radの順に見る。
  各向きで1.5 s待ってから2 s、頭が止まった後に撮られた画像の検出が届くまで（最大20 s）検出を集める（模擬時間）。
  画像からの検出器は1枚に数秒かかり、頭を向ける前の画像の検出が後から届くため、検出の時刻（画像の時刻）で待つ。
- 物体は、頭が向きごとに止まってから次の向きへ動き出すまで（最後の向きは集め終わるまで）に撮られた画像の検出だけから求める。
  最初の向きへ動く前（移動の終わり、前のサブゴールの視野）と頭が動く間の画像の検出は使わない。
- 検出はtrack IDごと（trackのないものは距離）にまとめ、同じラベルで0.08 m未満の塊は1つにする（検出器がtrackを付け直した物体）。同じメッセージで一緒に検出された塊は、並んだ別の物体としてまとめない。ラベルは最も多いもの。
  2回以上検出された物体だけを使う（使う検出が向きごとに平均2回未満の遅い検出器では1回）。
- 家具の設定のframe（シミュレーションの設定は `world`）から検出のframeへはTFで変換する。`task.launch.py` の `objects:=true` は、固定地図がGazeboのworld座標で描かれているため、`world` → `map` の恒等の静的TFを出す。
- 家具の設定とカテゴリの物体リストは、runnerのパラメータ `furniture_config`・`objects_config`（空はaha_perceptionの[シミュレーションの家具](../aha_perception/config/furniture/rcjo2026_arena.yaml)と物体リスト）。
  起動時（runnerの準備）に、サブゴールの `surface` が家具の設定にあるかを確かめる。

`roles` は役割の名前から割り当ての条件への辞書。

| 条件 | 選ぶ物体 |
| --- | --- |
| `{"labels": ["mug", "cup"]}` | ラベルが一致する物体 |
| `{"category": "cutlery"}` | 物体リストのカテゴリが一致する物体 |
| `{"category_key": "ppc.trash_category"}` | 記憶のキーの値のカテゴリの物体（ゴミ） |
| `{}` | 他の条件のどれにも当たらない、物体リストにある物体（食卓の「その他」2個） |

条件のある役割は書いた順に、当たる物体のうち最も多く検出されたもの（同じなら確信度、ロボットとの近さの順）を取る。
`{}` の役割は、残りの物体から同じ順に選び、ロボットに近い順に割り当てる。
カテゴリは大文字小文字と最後の語の末尾の `s` を区別しない（`snacks` と `snack` は同じ）。

`perceive_shelf` は、段ごとに、棚の正面から見て左右（家具の局所yの正負）で物体を群に分け、群のカテゴリを多数決で決める。

| 出力 | 内容 |
| --- | --- |
| `<output>.categories` | 段ごと（下から）の群のカテゴリ。左から。空の段は `[]` |
| `<output>.groups` | 群の `level`・`side`（`left` / `right`）・`category`・`labels`・`position`（物体の中心の平均、zは段の高さ）・`frame` |
| `<output>.placement` | `objects` のキーごとの置き場所。`level`・`side`・`height`・`position`・`frame`・`category`・`reason` |

置き場所の `position` は、置く物体の中心を置く点（zは段の高さ）で、段の物体と側板から棚の左右方向に0.03 m離す。
置く物体の幅は記憶の `size` の水平の長い辺（`size` がなければ0.1 m）、段の物体の幅は向きと `size` から求める。

- 同じカテゴリの群があれば、大きい群から順に、群の物体の外側（側板の側、入らなければ内側）の隣で幅が入る所（`reason` が `category`、`side` は群の側）。
- 入る群がなければ、物体のない段の片側（高さ0.2–1.0 mの段で0.6 mに近い順、左から。`empty`）。
  空いた所もなければ、高さ0.2–1.0 mで物体の最も少ない所（`least occupied`）。`position` はその片側で最も広い空きの中央。
- 後の物体は先に決めた物体の群に加わり、その隣に置く。

カテゴリの意味の近さ（似た物体）は判断しない。

## General Purpose Service Robot Challenge

[ルールブック §5.3](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を29サブゴールに分ける。
3命令をまとめて受け取り、`plan.generate` が記憶へ出力した計画を `exec.step01`〜`exec.step20` で1手順ずつ実行する。
実行の枠は命令に結びつかないため、命令の解決（3×250）とInterleaved Task Bonus（200）の950点は `finish.report` に計上する。
実装済みのスキルは `navigate`、`run_plan_step`、`say`（[ロボットの画面](#ロボットの画面)）、人の認識を使う `find_described_person` / `count_people`（`people:=true` が必要）、[物体を取る・置く・渡す](#物体を取る置く渡すスキル) `pick_object` / `place_object` / `deliver_object`（`objects:=true` と取る・置く・渡すserverが必要）。`plan_commands` と `listen_command` はmockの命令3件とその計画で通過する。

### 計画

```json
{"command": 1, "action": "pick_object", "params": {"object": "cola", "surface": "kitchen_table"}, "mock": {"robot.holding": "cola"}}
```

`command` は1〜3、`action` は `skills` のスキル名。`mock` は省略できる。
`run_plan_step`（params `{"index": i}`）は、計画の `i` 番目の手順を同名のスキルへ振り分ける。

- **検証**: `validate_plan(plan, spec)` はROSなしで呼べる。不正な形、枠の数を超える長さ、未宣言の `action`、タスクの流れ用の `action`（`run_plan_step` など）、`locations` にない `location` と `mock` の `robot.location`、実装済みスキルの `validate` に通らないparamsを拒否する。`mock` の計画は起動時に、記憶の計画は枠の実行ごとに全手順を検証する。不正なら最初の枠で移動前に失敗する。
- **手順がない枠**: 即座に成功する。記憶に計画がなければ失敗する。
- **未実装のスキル**: 成功として通過し、手順の `mock` を記憶へ追加する。メッセージに未実装と出る。`mock` の `robot.location` にはNav2で移動する。
- **実装済みのスキル**: `execute` の `outputs` と `expected(params)` が `run_plan_step` の出力として記憶へ入る。失敗はそのサブゴールの失敗として扱われ、ロールバック・再試行が手順単位で働く。
- **選択外・スキップ**: 枠を実行しない場合は、手順の `expected(params)` と `mock` を記憶へ追加する。`exec.step05` だけを選ぶと、手前の手順の `robot.location` へNav2で移動してから実行する。到達できない地点へ移動する手順をスキップすると、次に実行する枠はその地点への移動で失敗する。

| 手順 | params | 生成器の命令 |
| --- | --- | --- |
| `navigate` | `location` | goToLoc |
| `find_object` | `object`、`location` | findObjInRoom、findObj |
| `pick_object` / `place_object` | `object`、`surface` | takeObjFromPlcmt、takeObj、bringMeObjFromPlcmt / placeObjOnPlcmt、putObjInTrash（`surface: trashbin`） |
| `deliver_object` | `object`、`person`（`operator` は命令した人） | bringMeObjFromPlcmt、deliverObjTo* |
| `find_described_person` | `gesture` / `pose` / `clothing` / `description` / `name`、`location` | findPrsInRoom、meetPrsAtBeac、greet*、talkInfoToGestPrsInRoom、tellPrsInfo* |
| `say` / `answer_question` | `text` / `question` | 挨拶 / talkInfo、talkInfoToGestPrsInRoom |
| `count` | `target`、`location`、`output` | countObjOnPlcmt |
| `count_people` | `output`、`gesture` / `pose` / `clothing` / `description`（省略で全員）、`location` | countPrsInRoom、countClothPrsInRoom |
| `describe` | `target`、`location`、`output` | tellObjPropOnPlcmt、tellCatPropOnPlcmt、tellPrsInfo* |
| `tell_info` | `key`（記憶のキー）、`person` | `count` / `describe` の結果を伝える |
| `follow_person` / `guide_person` | `person`、`location`（`follow_person` は省略可） | follow* / guide* |

`find_described_person` は名前・ジェスチャ・姿勢・服で探す。HICの `find_person`（名前のない最も近い人）とは別のスキル。

### 人を探す・数えるスキル

人の条件は、[コマンド生成器](https://github.com/RoboCupAtHome/CommandGenerator)の句をそのまま渡す（`"waving person"`、`"person raising their left arm"`、`"persons pointing to the right"`、`"sitting person"`、`"person wearing a red t shirt"`、`"people wearing gray sweaters"`）。値だけ（`"waving"`、`"sitting"`、`"red t shirt"`）も受け付ける。
`gesture` / `pose` / `clothing` / `description` は組み合わせられ、矛盾する指定は検証で拒否する。左右は本人の左右。
句の解釈は [person_description.py](aha_tasks/person_description.py)、人の判定は [person_perception](../aha_perception/README.md#姿勢ジェスチャ服) の `ClassifyPerson`。

- **範囲**: `location` の地点（省略時はロボットの位置）から `radius`（既定3.0 m）以内の人だけを対象にする。人が `map` 以外の座標なら範囲で絞らない。部屋の形は使わない。
- **見回し**: 頭を `pans`（既定 `[0.0, 0.4, -0.4]` rad）へ `dwell`（既定6 s）ずつ、2 m先の高さ `look_height`（既定1.0 m）へ向ける。
- **判定**: 1人につき、頭をその人の頭へ向けて0.75 s待ち、その時に見えた頭の位置へ向け直してさらに0.75 s待つ（頭が画像の端で切れると頭の位置が低く出るため。挙げた手も写るように）。次に `skip_vlm` で深度とキーポイントの確かな答えだけを聞く。条件の成否が決まらなければ（服、不確かな姿勢・ジェスチャ）VLMに聞く。VLMの答えは `describe_timeout`（既定180 s、person_perceptionの `vlm_timeout` に合わせる）より30 s長く待つ（サーバーは先に動きを待つため）。判定に失敗した人（見失った、VLMの失敗、時間切れ）はもう1回判定し、それでも失敗すれば「判定できず」として続ける。中断は `Cancelled` で返す。
- **名前**: `name` は person_perceptionが覚えた名前とだけ照合する。名前がまだ付いていない人（新しいtrackには数秒後に付く）は、付くのを待つ。名前を尋ねる会話は含まない。
- **制限時間**: `timeout` に達すると、1人の判定（注視と `ClassifyPerson` の待ち）の途中でも打ち切る。

| スキル | 動作 | 出力 |
| --- | --- | --- |
| `find_described_person` | 見回しながら、範囲内の人を近い順に1回ずつ判定し、最初に条件に合った人を見て、`approach`（既定 `true`）なら `distance`（既定1.0 m）手前まで近づく。`timeout`（既定60 s）で失敗 | `<role>.track` / `.position` / `.frame`、見えた `<role>.pose` / `.gesture` / `.garments`、`robot.location`（`role` の既定は `person`） |
| `count_people` | `pans` を1巡し、見えている人を1人ずつ判定する。その向きに判定していない人が残る間は次の向きへ進まず、人を判定したらその向きを `dwell` だけ見直す。1人の判定は1回（失敗した場合は2回まで）で、判定できなかった人と、答えで条件の成否が決まらなかった人は数えない。`map` で0.5 m未満の観測は同じ人とする。`timeout`（既定300 s）で失敗 | `<output>`（人数）、`<output>.positions`（条件に合った人の最後の位置） |

trackは視野を外れると3 sで破棄され、見直すと別のIDになる。そのため判定は見えている間に行い、判定した人の `map` の位置から0.5 m未満に現れた新しいtrackは、同じ人として判定し直さない（数え直さない）。ただし、判定した人のtrackが同じ画像にまだ見えていれば、近くの新しいtrackは別の人として判定する。

### 物体を取る・置く・渡すスキル

[handling.py](aha_tasks/handling.py) の3つのスキルは、物体の選択・置き場所・台車の位置・人への声かけを受け持ち、腕の動作は取る・置く・渡すaction server（[ROSインターフェース](../../../docs/interfaces.md#サブゴールタスクの実行)）に任せる。
台の上は[物体の認識](#物体の認識)と同じ走査で見るため、`objects:=true` が必要。

```bash
ros2 launch aha_tasks task.launch.py task:=gpsr objects:=true object_scene:=gpsr e2e:=debug headless:=true
ros2 run aha_tasks run_task --subgoals 'exec.step01,exec.step02' --on-failure skip  # 別のシェル
```

| スキル | params | 動作 | 出力 |
| --- | --- | --- | --- |
| `pick_object` | `object`、`surface`（省略可）、`arm`（`right`）、`margin`（0.1）、`retries`（1）、`approach`（`true`） | `surface` の上を走査して `object` に合う物体を選び、腕から届かなければ台車を寄せて `PickObject` を送る。失敗すると走査からやり直す | `robot.holding`（`object` の値）、`robot.held_object`（物体の辞書に `surface` を加えたもの）、寄せたときは `robot.location` |
| `place_object` | `object`、`surface`、`spot`（省略可）、`arm`、`clearance`（0.05）、`retries`、`approach` | 置き場所を決め、届かなければ台車を寄せて `PlaceObject` を送る。別の置き場所・台車の位置で変わりうる失敗（`STATUS_NO_SURFACE` / `NO_SPACE` / `UNREACHABLE` / `EXECUTION_FAILED`）だけ、置き場所からやり直す。`STATUS_NOT_AT_SPOT`（離したが置いた所に見えない）は物体がもう手にないため成功とし、メッセージに書く（失敗では記憶が戻り、`robot.holding` が残る） | `robot.holding` と `robot.held_object` を `null` |
| `deliver_object` | `object`、`person`、`role`（`person`）、`distance`（1.0）、`timeout`（20）、`retries`（1）、`arm` | 人の前へ行き、`Here is the cola. Please take it from my hand.` と言って `HandOver` を送る | 同上 |

- **物体**: `object` は命令の語（`cola`、`a drink`、`drinks`。ラベル、その複数形、カテゴリ。`object` などの総称は何にでも合う）か、物体の辞書を持つ記憶のキー（`perceive_objects`・`find_object` の出力）。合う物体のうち、キーなら記憶の位置から0.3 m以内の同じラベルの物体、なければ検出回数・確信度・ロボットとの近さの順で選ぶ。
- **surface**: 家具の設定の家具の名前、`floor`、またはタスクの地点。地点は、その向きの前方90°以内で1.2 m以内にある最も近い家具とする（GPSRの `kitchen_table` は `dining_table`、`desk` は `long_table`、`side_tables` は `tall_table`、`trashbin` は `trash_box`、`shelf` は `shelf`）。`trash` などの語はカテゴリ `trash bin` の家具。
- **台車を寄せる**: 物体（置き場所）が台車の中心から0.75 mより遠いか後ろにあれば寄せる。家具ではロボットのいる側の縁に向き、台車の中心を縁から0.65 m（タスクの地点と同じく障害物から0.6 m以上。それより近いとNav2はその姿勢から経路を作れない）、物体を右腕の側（`base_footprint` の y −0.2 m、`arm: left` は +0.2 m）に置く。台の縁から約0.15 mより奥の物体は、実機の腕（肩から約0.75 m）では届かないことがある。床では物体へ向き、物体が (0.45, −0.2) m に来る位置。移動に失敗しても、その場からserverへ送る。
- **置き場所**: ゴミ箱（カテゴリ `trash bin` の家具）は外形の中心で高さの半分の点に `drop` で入れる。棚（段が複数の家具）は `perceive_shelf` と同じ規則（同じカテゴリの群の隣など）。台は、見えた物体と互いの外接円が `clearance` 以上離れ、縁から0.03 m内側で、右腕の前の点（台車の (0.45, −0.2) m）に最も近い点（2 cm格子）。床は台車の0.5 m前。`spot`（記憶のキー）があれば、その `perceive_shelf` の `placement`（物体のキーで引く）、`position` を持つ辞書、`map` の `[x, y, z]` に置く。
- **人**: `person` が `operator` か `me` なら、`instruction_point` へ移動し、地点から2.5 m以内で人の認識が3 s以内に見た最も近い人へ向く。見えなければ地点の向きの1.0 m先へ差し出す。それ以外は `<role>.position` / `.frame`（`find_described_person`）の人へ `distance` まで近づく（1.3 m以内なら向きだけ合わせる）。記憶になければ失敗する。
- **受け渡し**: 手を出す点は人の中心で、高さを0.7–1.1 mに収める。serverが受け取りを感じ取る（`release: take`）と離す。`touch: true` なら画面の `Release` のタッチでも離す（HIC以外ではタッチを使わない方針のため既定はfalse）。タッチでは `take` のgoalを取り消して（腕は出したまま）`release: now` で送り直す。取り消す間に `take` のgoalが終わっていれば（受け取られた後のタッチなど）、送り直さずにその結果を使う。`timeout` 秒で受け取られなければ、言い直して `retries` 回まで送る。
- 失敗しても人に助けを求めない（§3.7の補助は最後の手段で、まず手順を飛ばす）。失敗はサブゴールの失敗としてポリシーに従う。
- 持てる物体は1つで、持っている間の `pick_object` は失敗する。

### シミュレーションの把持

`e2e:=debug`（`manipulation:=sim`）では `sim_manipulation` が取る・置く・渡すserverの代わりに、把持を計画せずGazeboで物体のモデルを動かす。

- **pick**: 目標の点から0.2 m以内の同じラベル（なければ他のラベル）の物体モデルを選ぶ。台車から水平に1.0 m（`reach`）より遠いと失敗する。右腕を物体の上へ伸ばし、モデルを同じ名前の静的なモデル（collisionなし）に作り直して手先（`link_ree`）に置き、腕を `carry` の姿勢へ動かす。手先に置いた後は、`carry` へ動けなくても（取り消し・controllerの失敗）成功を返し、メッセージに書く。持った物体は20 Hzで手先を追い、直立のまま手先の向きで回る。
- **place**: `near` が `reach` 以内なら腕をその上（`drop` は家具の設定の容器の天板の上）へ伸ばし、静的なモデルを `near` に立てて（`drop` は中心を `near` にして）腕をpickの前の姿勢へ戻した後、動く物体のモデルとして作り直す（`drop` では落ちる）。
- **handover**: 腕を目標の点へ伸ばし、`release: take` では2 s（`take_delay`、模擬時間）後に人が受け取ったとみなしてモデルを消し、腕を戻す。パラメータ `take` が `false` なら受け取られず、`timeout` で `STATUS_TIMEOUT` になり、持ったまま腕を `carry` へ戻す。`reach` / `take` / `take_delay` は実行中に `ros2 param set /sim_manipulation` で変えられる。
- 腕は関節空間の直線で動き、衝突を確かめない。上げるときは昇降、下げるときは腕を先に動かし、手は物体の上0.08 mを通る。右腕だけを扱う。終了時、持っている物体はその場で動く物体に戻す。
- 指は閉じず、持った物体は指を通り抜ける。巻き戻しても物体は戻らない（`sim_objects` を実行し直す）。`sim_manipulation` が持ったまま異常終了すると、物体は静的なまま宙に残る。

### 姿勢を付けた人

`people_scene:=<名前>` は [config/sim/people/](config/sim/people/) の配置で、姿勢やジェスチャを付けた人をGazeboに置く（`sim_people`）。人はタスクの間、同じ場所で同じ姿勢を保つ。スキルには配置を渡さない。

```bash
ros2 launch aha_tasks task.launch.py task:=gpsr people:=true people_scene:=gpsr ssh_user:=<Unixユーザー名>
```

| 配置 | 内容 |
| --- | --- |
| `gpsr` | GPSRのmockの計画用。寝室に左腕を挙げた人と立つ人、オフィスに手を振る人・椅子に座る人・左を指す人 |
| `attributes` | 人の認識の確認用。寝室・オフィス・リビングに、9種類の姿勢・ジェスチャとシャツの色が異なる10人（[person_attributes_report.py](../aha_perception/README.md#姿勢ジェスチャ服) の `--truth` に使う） |

人ごとに `label`（`standing` / `sitting` / `lying` / `waving` / `raising_left_arm` / `raising_right_arm` / `raising_both_arms` / `pointing_left` / `pointing_right`）、`shirt`（`purple` / `red` / `blue` / `yellow` / `orange` / `green` / `white` / `gray` / `black`）、`position`（m。立つ・座る人は足元、寝る人は体の中央。`z` は足の裏または寝た背中の高さ）、`yaw`（体の正面の向き）を書く。

- モデルはGazebo Fuelの [Male visitor](https://app.gazebosim.org/OpenRobotics/fuel/models/Male%20visitor)（version 2、CC0）。初回に取得してハッシュを確かめ、骨格の重みで姿勢を付けたメッシュとシャツの色を変えたテクスチャを `~/.cache/aharobot/sim_people` に作る。
- 手を振る人は、前腕を左右に25°傾けた2つのメッシュを `wave_period`（模擬時間の秒）で交互に見せる。
- 座る人は脚を椅子の座面の高さに合わせた形で、`z` 0の椅子の位置に置く。寝る人は仰向けで、頭が体の向きの後ろ（yaw 0なら−x）になる。`attributes` の寝る人は、頭が枕に埋もれないよう足をベッドの枕側へ向けている。
- 体型は1種類（男性）で、服は半袖のTシャツと半ズボンだけ。シャツ以外の服の種類はsimでは確かめられない。
- visualだけでcollisionがなく、ロボットを押さない。LiDARとカメラには写る。
- Gazeboを動かしたまま `sim_people` を起動し直すと、同じ名前で残っている人は作り直さず位置だけ合わせる。作れない人がいても他の人は続ける。

### 地点

部屋・家具のラベルは[CompetitionTemplate](https://github.com/RoboCupAtHome/CompetitionTemplate)の語彙を `_` 区切りにしたもので、RCJO2026 simの部屋・家具を割り当てる。simにない語彙は登録しない。

| 部屋 | simの部屋 | 家具（simのモデル） |
| --- | --- | --- |
| `bedroom` | Bed Room | `bed` |
| `kitchen` | Dining Room | `kitchen_table`（`dining_table`）、`sofa`（`couch_right`） |
| `office` | Study Room | `chairs`（`chair_left` / `chair_right`）、`shelf` |
| `living_room` | Living Room / Laundry Area | `desk`（`long_table`）、`side_tables`（`tall_table`）、`trashbin`（`trash_box`） |

`instruction_point` は固定の代替地点で、発表された地点に合わせて座標を変える。入口のドアは閉じたままで、ロボットは内側の `spawn` から始める。
物体は `object_scene:=gpsr` で家具に置く（[シミュレーションの物体](#シミュレーションの物体)）。
審判がタイマーを止める間も、GUIの経過時間は止まらない。

## Doing Laundry Challenge

[ルールブック §5.4](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を41サブゴールに分ける。
`points` は得点表の最大点で、合計は総計の4415。減点は `note` に書く。
実装済みのスキルは `navigate` のみで、洗濯機と衣類の操作はmockで通過する。

| 地点 | 位置 |
| --- | --- |
| `washing_machine` | worldの洗濯機の前 |
| `laundry_basket` | worldのバスケットの前 |
| `folding_table` | 畳む台の固定の代替地点。worldの `long_table` の前 |

入口のドアは閉じたままで、ロボットは内側の `spawn` から始める。
worldの `pile_of_clothes` と `laundry_in_drum` は剛体で、畳める衣類のモデルはない。

順序は、洗濯機の扉を開ける、1枚をバスケットへ移す、バスケットを台の横へ運ぶ（`basket.carry` は持つ・移動・置くを1スキルにまとめる）、台の1枚を畳む（`shirt0.*`）、バスケットの衣類を取って畳み重ねる（`shirt1.*`〜`shirt6.*`）。
台への `navigate` は `shirt0.*` と `shirt1.*` の先頭のみ。`shirt2.*` 以降は記憶上の位置が台なので、実装済みのスキルを実行する前に移動する（状態の同期）。
`shirt0.*` は0点で、`shirt0.state` のみを変える。台が空の配置では選択から外しても後続の記憶は変わらない。

洗濯機とバスケットを飛ばす場合は `start.*,shirt1.*` を選ぶ。バスケットは記憶上台の横にあるため、実物は手で置く。

| 記憶 | 内容 |
| --- | --- |
| `robot.holding` | 持っている衣類（`washer_shirt`、`shirtN`）。置くと `null` |
| `basket.location` | バスケットのある地点。`laundry_basket`、運ぶと `folding_table` |
| `table.stack` | 重ねた枚数。`shirtN.stack` で `N` |
| `shirtN.state` | `placed`、`spread`、`folded`、`stacked` |
