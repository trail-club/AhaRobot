# aha_tasks

タスクをサブゴールに分けて実行・可視化する共通基盤と、RoboCup@Homeのタスク定義。

| 構成 | 内容 |
| --- | --- |
| `aha_tasks/core/` | ROS非依存の実行基盤。タスク定義の検証、サブゴールの選択、チェックポイント・ロールバック |
| `aha_tasks/skills.py` | 組み込みスキル `navigate` と `robot.location` の復元 |
| `aha_tasks/people.py` | 人の認識（[person_perception](../aha_perception/README.md#人の認識)）を使うスキル |
| `aha_tasks/objects.py` | 物体の検出（`/aha/perception/objects`）から台の上の物体と棚の分類を求めるスキル（[物体の認識](#物体の認識)） |
| `aha_tasks/describe.py`、`appearance.py` | 客の外見を別の客に伝える `describe_person` と、VLMの外見の句の扱い（[外見の説明](#外見の説明)） |
| `aha_tasks/gaze.py` | スキルの頭の向き先。人・Nav2の経路の先を追い続ける（[注視](#注視)） |
| `aha_tasks/doorbell.py` | ドアベル検出（[aha_audio_doorbell](../aha_perception/docs/audio.md#ドアベル検出)）を待つスキル |
| `aha_tasks/gestures.py` | 右腕で席を指し示す `offer_seat`（[席の案内](#席の案内)） |
| `aha_tasks/seats.py` | 空いている席を探す `find_free_seat`（[空席の判定](#空席の判定)） |
| `aha_tasks/guests.py`、`sim_guests` | シミュレーションの客の動作（ROS非依存）と、Gazeboで客を動かすnode（[シミュレーションの客](#シミュレーションの客)） |
| `aha_tasks/sim_objects.py`、`sim_objects` | シミュレーションの物体のモデルと配置の検証（ROS非依存）と、Gazeboの家具に物体を置くプログラム（[シミュレーションの物体](#シミュレーションの物体)） |
| `aha_tasks/object_models.py` | データセットのスキャンと実測の形状から物体のモデルを作る（ROS非依存） |
| `aha_tasks/sim_sounds.py`、`sim_sounds` | シミュレーションのマイク音（背景雑音と客のドアベル・ノック） |
| `aha_tasks/teleport.py`、`sim_teleport` | デバッグ用。Nav2の代わりにGazeboでロボットを瞬間移動させる `NavigateToPose` のserver（[e2eの確認](#e2eの確認)） |
| `run_task` | runnerの準備を待って実行を始め、終了まで待って各サブゴールの所要時間を出すCLI（[e2eの確認](#e2eの確認)） |
| `aha_tasks/plan.py` | 計画の検証と、計画の手順を各スキルへ振り分ける `run_plan_step` |
| `aha_tasks/display.py`、`web/display/` | ロボットの画面の状態（ROS非依存）とROSの入出力、画面のページ（[ロボットの画面](#ロボットの画面)） |
| `aha_tasks/interaction.py` | スキルからの発話・聞き取りとタッチの質問、スキル `say` / `wait_start_signal` |
| `aha_tasks/talk.py`、`aha_tasks/conversation.py` | GPSRの `answer_question` / `tell_info` と、名前・答え・質問の文（ROS非依存）（[名前を尋ねる・情報を伝える](#名前を尋ねる情報を伝える)） |
| `aha_tasks/show_objects.py`、`aha_tasks/object_view.py` | 認識した物体を画面に1個ずつ示すスキル `show_objects` と、頭部カメラの画像に物体の枠を描く処理（ROS非依存）（[認識した物体を示す](#認識した物体を示す)） |
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
| ドアベルの検出（`guests:=true`） | 音声コンテナの `aha_audio_doorbell`（[ドアベルの検出](#ドアベルの検出)） | `sim_sounds` が、客の鳴らした音をそのまま検出（`SoundEvent`、`model` は `sim_truth`）として出す。音声コンテナを使わない |
| runnerの終了 | launch全体が終わる | runnerと人の認識のnodeは止めると2 s後に再起動し、Gazebo・AMCL・客はそのまま |

`perception:=model` を付けると、`debug` でも人の認識・物体の検出・ドアベルの検出は `final` と同じになる（移動だけを省く）。
`people_scene` の人を探す・数えるには `perception:=model` が要る（`sim_person_detector.py` は `classify_person` を持たず、客だけを出す）。`people:=true` と `perception:=truth` に `people_scene` を付けると起動時にエラーにする。
`debug` で確かめられないため、`final` で確かめること：

- **移動**: 経路の有無（開かないドア、壁の向こうの地点）、Nav2の失敗・時間切れ、移動中のAMCLのずれ、移動中に見える人。`sim_teleport` は目標の、地図上のロボットの半径（0.40 m）に占有・未知のセルがないことだけを見る。AMCLのTFは2 s先の時刻で配信されるため、1回の移動に約4 sかかる。移した直後の1–2枚の画像は、前後の位置が混ざったTFで測られる。
- **人の認識**（`perception:=truth` のとき）: SAM 3.1の見落とし・誤検出・並んだ人の結合、マスクとDepthから求める位置、VLMの特徴・照合の誤り、DGXの往復の遅れ。人は高さ1.75 mの立った体の軸として扱い、名前は覚えたモデルに直ちに付く。
- **ドアベルの検出**（`perception:=truth` のとき）: マイク音からの検出の見落とし・遅れ・誤報。
- **客の動き**: ロボットは待った後に一度に移るため、案内される客はロボットの経路をたどらず、壁を抜けて歩く。
- **物体の検出**（`perception:=truth` のとき）: 見落とし・誤ったラベル・遮蔽された物体の位置と大きさの誤差。真値は物体リストの大きさとGazeboのモデルの姿勢を出す。

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

    def execute(self, ctx):  # ctx.params / memory / cancelled / log / statuses
        guest = ctx.params["guest"]
        return SkillResult(True, "", {f"{guest}.name": name, f"{guest}.drink": drink})
```

成功時の `outputs` はタスクの記憶へ追加され、後続が `ctx.memory` から読む。
`expected(params)` は成功後とmock時に記憶へ追加する状態。ロボットを動かすスキルは `robot.location` を出力する。
`ctx.statuses` は開始時の各サブゴールの状態（`/aha/tasks/status` の `status`）。`skipped` のサブゴールの出力のキーには、結果でなく `mock` が入っている。
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
text, why = interaction.listen(["Charlie", "Jane"], ctx.cancelled)
```

- `say` は字幕を出し、TTS（`/aha/perception/audio/speak`）が起動していれば読み上げ終わるまで待つ（上限60 s）。TTSがなければ字幕だけで戻る（シミュレーション）。起動したばかりのrunnerはTTSをDDSで見つけるまで時間がかかるため、最初の1回だけ最大2 s待つ（`say` と `wait_start_signal` は `ready()` で待つ）。
- `show` は画像（JPEGかPNGのbytes）を出し、`say` で説明する。画像は次の画像か `hide()` まで残る。認識の結果は `show` で審判に示すだけで、タッチで答えを求めない。
- `ask` は画面にボタンを出し、タッチされると番号と文言を返す。タッチがなければ `index` が `None` で、`label` が理由（中断、`timeout` 秒の経過、置き換え、画面の終了）。`timeout` は実時間で、0以下は中断まで待つ。
- `listen` はASR（`/aha/perception/audio/listen`）が聞いた次の発話1つを返す。`hotwords` は名前などの語で、`timeout`（既定10 s）は話し始めを待つ時間。ASRが5 s見つからなければ文字列なしで `No ASR` で始まる理由を返す。
- 画面に出る質問は1つで、新しい質問は表示中の質問を置き換え、前の質問はタッチなしで終わる。
- serverを見つけた直後に送ったgoalやその応答は失われることがある。`ask` は受け付けの応答が5 s以内になければ、serverが見えている間は質問を送り直す。送り直す前の質問がタッチされていれば、画面に出し直さずにそのタッチを返す。`say` は二重に読み上げないよう送り直さない。
- 画面の起動を5 s待っても `/aha/display/ask` がなければ、`ask` はタッチなしで終わる。待つ間も中断を受け付ける。
- `say` と `ask` は、待っている間にserver（TTS・`task_monitor`）が5 s続けて見えなくなると諦める。異常終了したプロセスがROSのグラフから消えるまで、DDSのリースの約20 sがかかる。

| スキル | 動作 | 使うサブゴール |
| --- | --- | --- |
| `say` | `text` を字幕に出し、TTSがあれば読み上げる | GPSRの `command.mode` |
| `wait_start_signal` | 準備ができたことと押すボタンを `text` で伝え、`button`（既定 `Start`）のタッチを待つ。どちらも省略できる | HICの `start.signal` |
| `show_objects` | 認識した物体を1個ずつ、頭部カメラの画像に枠を描いて `show` で示す（[認識した物体を示す](#認識した物体を示す)） | PPCの `*.show` |

TTSで読み上げられなかった場合も、`wait_start_signal` はボタンを出して待つ。
タッチを待つスキルは、HICの開始合図の `wait_start_signal` だけ。

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
| `pick_and_place` | [ルールブック §5.2](https://robocupathome.github.io/RuleBook/rulebook/master.pdf) の配置（24個）。食卓（`dining_table`）にcutlery・mug・plate・ゴミ（`ppc.trash_category` のカテゴリ）・その他2個、ゴミ箱の近くの床にゴミ、`tall_table` にボウル・スプーン・洗剤タブ、補助台（`long_table`）にCommon Objects 2個。棚（`shelf`）は段ごとに1カテゴリで、2段目がcleaning supplies、3段目がfood（シリアルを含む。箱の高さ28 cmが段の隙間に入らないため横倒し）、4段目がdrink（牛乳を含む）、5段目（天板）がfruit。1段目（床の高さ）は空き。食卓のその他2個（apple、cola）は棚の同じカテゴリの段の隣に置ける |
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
物体は [RoboCup Japan Open 2026 DSPLの物体リスト](https://github.com/RoboCupAtHomeJP/AtHome2026/blob/main/rules/DSPL/dspl_2026_object_list.pdf) の物体（YCBと指定の製品）に合わせる。YCBにない物体は公開データセットの実物のスキャンを使い、スキャンがない物体は実測の形状に画像を貼って作る（`aha_tasks/object_models.py`）。

- `ycb`: `tmc_wrs_gz_worlds` のYCBのモデル（メッシュ・質量・慣性）。`yaw`（度）でメッシュをモデルの中で回して辺をx・yに揃え、原点を衝突メッシュの外接箱の中心へ移す。
- `scanned`: データセットのテクスチャ付きメッシュ（アーカイブ内のOBJ・PLY・DAE、またはGLB）。`url` から取得して `sha256` で確かめ、`crop`（スキャンの座標）で台を切り落とし、`rpy`（度）・`scale` でz上・mにする。
- `measured`: スキャンのない物体。形状（回転体の断面 `revolve`、`box`、実寸のスキャン `scan`、スキャンの点群から測った断面 `profile`）に、側面のラベル・上下の円・箱の面の画像（`paint`）を貼る。

`scanned` と `measured` は初回に取得・変換して `$AHA_CACHE_DIR/sim_objects` に置く（取得するファイルは合計約200 MB、ネットワークが必要。`--check` は取得しない）。開発コンテナの `AHA_CACHE_DIR` はリポジトリの `.cache/aharobot`（gitignore）で、コンテナを作り直しても残る。未設定なら `~/.cache/aharobot`。取得・変換できない物体は置かずに続け、最後に無い物体として報告する（終了コード1）。
衝突形状は凸包で、慣性は凸包を一様な密度として `mass` から求める。変換後の外接箱が `size` と2 mm以上違えば失敗する。ラベルの文字はDejaVu Sans BoldとDroid Sans Fallback（Dockerfileで入れる）で描き、なければ失敗する。
各モデルの外接箱と物体リストの大きさの差は各軸1.5 cm以内（テストで確かめる。最大はtuna canの高さの1.2 cm）。取得元とライセンスは [THIRD_PARTY.md](../../../dependencies/THIRD_PARTY.md#sobits--tmcのシミュレーション資源) に記載する。

原点は外接箱の中心で、モデルの姿勢を物体の中心とする `sim_object_detector.py` の前提に合う。摩擦係数は1.0。
`sim_objects` は配置の物体を作り（同じ名前のモデルが残っていれば、1個ずつ動かすと別の物体に重なることがあるため、消してから作り直す）、3 s後に置いた位置から2 cm以上ずれた・10°以上傾いた物体を警告して終わる。全部の物体があれば終了コード0。
物体リストのパターンに合う名前で配置にないモデル（同じGazeboで先に置いた別の配置の物体など）は、置く前に消す。家具・ロボット・客・人のモデルは消さない。

制限：

- 失敗で巻き戻しても物体の位置は戻らない。`sim_objects` を実行し直して戻す。
- 頭部カメラのセンサは `people:=true` か `objects:=true` のときだけ有効になる。
- 緑茶・レモネード・カップ麺（どん兵衛）・歯磨き粉の箱・食洗機タブは、再利用できる写真がないため、ラベルを製品の配色と文字から描いている。食洗機タブの大きさは推定（29×20×8.5 mm）、ボトルの断面は製品画像から測った値（約2 mm）。
- milk・cookiesはHOPEのおもちゃの食品のスキャンで、RCJOの物体（明治おいしい牛乳など）と見た目が違う。colaとcookiesはRCJOの物体リストにない。
- red bullはRCJOの物体リストの指定どおり中身のない空き缶（15 g）で、他の飲料（中身込み）より軽く、触れると倒れやすい。
- 食洗機と、その中の物体はない。

## Human Robot Interaction Challenge

[ルールブック §5.1](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を30サブゴールに分ける。
`points` は得点表の最大点で、実行中に継続して求められる項目（注視など）はそのサブゴールに含める。
実装済みのスキルは `navigate`、[ロボットの画面](#ロボットの画面)を使う `wait_start_signal`、人の認識を使う `find_person` / `memorize_person` / `locate_guests` / `find_free_seat` / `describe_person`（`people:=true` が必要）、`offer_seat`、
ドアベルを待つ `detect_doorbell`（[ドアベルの検出](#ドアベルの検出)）。

ロボットは場内から始めるため、最初の `start.signal` で審判の手動の開始合図（§3.6.8）として画面の `Start` のタッチを待つ。
部分実行で `start.signal` を選ばなければ合図を待たない。

席の候補・ホスト・荷物置き場は固定の代替地点。ロボットは内側の `entrance` まで移動する。
`guests:=true` でなければ入口のドアは閉じたままで、人のモデルと追加の椅子は配置しない。荷物のモデルは配置しない。
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
| `memorize_person` | `<guest>.track` の人（いなければ名前のない最も近い人）の外見を `guest` の名前で覚え、VLMで特徴を求める。`views`（既定3）枚の画像を `view_interval`（既定0.5 s、画像の時刻）以上空けて聞き、過半数が一致した特徴を残す（[外見の説明](#外見の説明)） | `<guest>.appearance`（英語句のリスト）、`.track` / `.position` / `.frame` |
| `locate_guests` | 頭を正面・左・右へ `dwell`（既定10 s）ずつ向け、`guests` の全員を名前付きで見るまで待ち、最後に頭を正面へ戻す。名前はVLMが外見で照合して付けるため、席を入れ替えても区別する | 各客の `.track` / `.position` / `.frame` |

`find_person` と `memorize_person` は、終了後も頭をその人へ向け続ける（[注視](#注視)）。

待つ時間は `timeout`（`find_person` は30 s、`locate_guests` は60 s、`memorize_person` は10 s）、VLMは `describe_timeout`（180 s。`memorize_person` は全画像の合計）で変更できる。`timeout` と `dwell` は模擬時間が有効ならノードの時計、それ以外はシステム時刻の補正に影響されない単調時計で数える。
模擬時間が実時間で10 s進まなければ、`/clock` の停止を理由に失敗する。
`locate_guests` は `map` 座標の観測だけを時点をまたいで蓄積する。他の座標では1つのメッセージ内で全員が見える必要がある。
同じtrack、または位置が0.4 m未満（`map` では水平距離）の客は重複とみなし、古い観測を捨てて探し続ける。
出力の `.track` は破棄済みのことがあるため、後続スキルは名前か位置を使って客を再取得する。
`memorize_person` をmockにした場合は外見を覚えないため、後の `locate_guests` はその客を見分けられない。

### 外見の説明

`describe_person`（`aha_tasks/describe.py`）は、`memorize_person` がVLMから得た `<subject>.appearance` の句から特徴を選び、`listener` の客を見ながら英語の1文で伝える。句の扱いは `aha_tasks/appearance.py`。

```json
{"subject": "guest1", "listener": "guest2", "max_features": 4}
```

ルールブックは伝えた特徴を4つまで数え、正しければ+20、誤りなら−20とする（特徴の種類は指定していない）。

- `memorize_person` は、聞いた画像の過半数（3枚なら2枚）が一致した特徴だけを残す。失敗した画像は反対に数えるため、1枚しか答えなければ何も残さない（ログに出す）。
- 体の部位が同じ句は部位ごとに、服の種類と色で数える（`red t-shirt`・`red shirt`・`red top` は同じ）。最も多い種類と色が過半数になければ部位ごと捨てる（`black pants` と `dark grey pants` に分かれた下衣など）。言う句は過半数が答えた言い方、なければ色と種類だけ（`red shirt`、`black hair`）にする。
- 部位ごとに1つの句を、上衣、眼鏡、帽子、下衣、髪、ひげ、かばん、首回り、靴の順に `max_features`（既定4）まで選ぶ。部位は句の最後の名詞で決める。
- 推量・否定を含む句（`possibly glasses`、`may have a beard`、`looks like a hat`、`no hat`）、部位の分からない句（`smiling`）、5語以上の句は言わない。文が自分で付ける先頭の動詞・冠詞（`wearing glasses`、`has a beard` → `beard`）と、`with` などより後の細部（`red t-shirt with a logo` → `red t-shirt`）は省いてから語数を数える。
- 持っている物（`carrying` / `holding` の付く句、手・腕・肩などにある物：`carrying a black jacket`、`cap in hand`）は、かばんでなければ言わない（`bag on shoulder` は言う）。
- 名前 `<role>.name` を使い、なければ `the other guest` と言う。例：`Emma, Charlie is wearing a red t-shirt and blue jeans and has short black hair.`
- 頭は `<listener>.track`（なければ名前）の人を追い、[ロボットの画面](#ロボットの画面)の `say` で話す（シミュレーションは字幕だけ）。頭がすでにその人を追っていれば（`memorize_person` の直後）、位置で引き継いだ新しいtrackのまま追い続ける。まだ追っていなければ、頭が向くまで1.5 s待ってから話す。言える特徴がなければ何も言わずに成功する。
- 出力は `<subject>.told`（伝えた句のリスト）。`<subject>.appearance` がなければ失敗する。

`e2e:=debug` の真値の人の特徴は `ground truth <モデル名>` だけなので、何も言わない。言う内容は `perception:=model` か `final` で確かめる。

### ドアベルの検出

`detect_doorbell` は、開始してから届いた最初の `/aha/perception/audio/doorbell`（`aha_msgs/SoundEvent`）で成功し、
`doorbell.heard` に `doorbell` か `knock` を出力する。開始前の検出は使わない。
`timeout`（既定120 s、実時間）で失敗する。
検出器が検出できる状態でなければ（publisherがない。モデルの読み込みと音声入力の開始を終えるまで作らず、`doorbell_input:=topic` では音声が2 s 届かないと消す）すぐに失敗する。待っている間に検出器が止まったときも、そこで失敗する。

検出器は音声コンテナで起動する（[ドアベル検出](../aha_perception/docs/audio.md#ドアベル検出)）。
シミュレーションでは `guests:=true` の `sim_sounds` が `/aha/sim/audio` にマイク音を出すので、検出器を `doorbell_input:=topic` で起動する。`perception:=truth`（`e2e:=debug` の既定）では `sim_sounds` が検出の代わりも出すので、検出器は起動しない。
ASR・TTSも使うために `make audio-run` を使うときは、`DOORBELL_INPUT=topic`（`perception:=truth` では `USE_DOORBELL=false`）を付ける。マイクを読む検出器が残ると、シミュレーションの音を聞かないまま検出器があるとみなされ、マイクの物音でも成功する。

```bash
# 音声コンテナ（同じ ROS_DOMAIN_ID）
ros2 launch aha_perception audio.launch.py use_asr:=false use_tts:=false use_debug_ui:=false \
  doorbell_input:=topic
```

### シミュレーションの客

```bash
ros2 launch aha_tasks task.launch.py task:=human_interaction people:=true guests:=true ssh_user:=<Unixユーザー名>
```

`sim_guests` が `/aha/tasks/status` のサブゴールの状態と記憶に合わせ、[config/sim/human_interaction.json](config/sim/human_interaction.json) の配置で2人の客とドアをGazeboの中で動かし、座っているホストと追加の椅子を置く。
スキルには真値を渡さず、客の位置は頭部カメラと `person_perception` からだけ得る。
未実装のスキルがmockで一瞬で通過しても、状態から段階を決めるため客の動作は飛ばない。

| サブゴールの開始 | 客の動作 |
| --- | --- |
| 開始前 | 床下に隠れる。worldの壁は高さ0.8 mで、頭部カメラから壁の外の人が見えるため |
| `<guest>.doorbell` | 家の外からドアの外側へ歩き、中を向いて待つ。着いていて、ロボットが `detect_doorbell` で待っている間に、設定の `bell` を1回鳴らす（`chime` は毎回ランダムなチャイム、`knock` はノック）。再試行や新しい実行では鳴らし直す |
| `<guest>.open_door` | ドアが外へ開く。以後は開いたまま |
| `<guest>.ask` | ドアの外側で左右に0.2 m（周期6 s）動き、ロボットを向く。ルールのとおり、会話が始まるまで家には入らない |
| `<guest>.guide` | ロボットとの距離を1.2 m保ち、ロボットが通った経路を歩いて家に入る |
| `<guest>.seat` | 記憶の `<guest>.seat`（`find_free_seat` が選んだ席）の前へ歩き、部屋を向いて立つ。記憶にない席なら客の設定の `seat` |

巻き戻しと、選択外で歩く間のない段階への移行は、歩かずに移動する。歩く速さ1.0 m/sはシミュレーション時刻。

`seats` の席ごとに、客が立つ位置 `pose`（席の0.45 m前）、座った人の位置 `sit`（座面の中央）と途中の点 `via` を書く。`chair` のある席には、worldにない椅子（`model://chair`）を置く（審判が椅子を追加する場面）。
`seated` の人（ホスト）は起動時から席に座る。モデルは `people_scene` と同じ座った人（`aha_tasks.sim_people`）で、`shirt` はその色の名前。メッシュを作れない（取得・キャッシュの失敗）ときはホストを置かずに客とドアを動かす。ホストは `chair_left` にいるため、`find_free_seat` は1人目にホストの席を除いて `chair_right` を、2人目にさらに1人目の席を除いて `chair_extra` を選ぶ。各サブゴールの `mock` はこの席。

制限：

- 客は静止した立ち姿のモデル（`person_standing`）を動かすもので、脚は動かず、座らない（座った姿は別のモデルで、覚えた外見と変わるため）。シャツなどの色で2人を区別し、その他の外見は同じ。
- 客はvisualだけでcollisionがなく、ロボットを押さない。LiDARとカメラには写る。障害物は避けず、固定の経路と追従で歩く。
- 客の測定位置はロボットの自己位置の誤差の影響を受ける。
- ホストは座ったまま動かない（`bag.*` の追従はmock）。待っている他の客・荷物は含まない。
- マイク音（`sim_sounds`、実時間）は −55 dBFS のピンク雑音と、合成のチャイム・ノック（ピーク −20 dBFS）だけで、距離・残響・人の声はない。チャイムは ding-dong・pinpon（日本のインターホン型）・westminster・melody・single_ding から毎回ランダムに選び、音の高さ・音色・間隔もランダム。検出器が苦手なブザー型は入れていない。検出のつながりを確かめるためのもので、検出性能は [ドアベル検出の比較](../../../tools/perception/doorbell_bench/README.md) で評価する。

### 注視

runnerの `gaze`（`aha_tasks.gaze.Gaze`）が、スキルの頭の向き先を `/aha/perception/look_at` へ送る。
`task.launch.py` の `head_look_at.py` が頭を向け、`map` の点は台車が動いても向け直し続ける。
追う動作は次の指示まで続き、サブゴールをまたぐ。
タスクが一時停止・中断・終了すると追う動作をやめ、操作パネルやteleopで頭を動かせるようにする。再開後は、スキルが次に指示するまで最後の向き先を保つ。

| 指示 | 向き先 | 使うスキル |
| --- | --- | --- |
| `follow_person(people, track, name)` | 人の顔（`head` の0.1 m下。`head` はマスクの上端）。trackが消えたら名前で、それもなければ直前の位置から0.5 m以内の名前のない人を引き継ぐ。見失っている間は最後の点を保つ | `find_person`、`memorize_person`、`offer_seat` の後、`describe_person` |
| `follow_path(goal)` | Nav2の経路 `/plan` に沿って1.5 m先、高さ1.2 m。経路の終わりの先は経路の始点から終点への向き（進行方向）へ延ばし、始点と終点が0.3 m未満（その場で目標の向きへ回る）なら目標の向きへ延ばすため、到着時は正面を見る。新しい経路が届くまでは、目標地点から目標の向きへ1.5 m先（すでに目標にいて経路を待たずに終わっても、頭が真下や真横を向かない） | `navigate` と `robot.location` の復元（到着後は `hold`） |
| `look_at(frame, point)` / `look_towards(pan)` | 固定の点。追う動作をやめる | `locate_guests`、`find_free_seat`、`offer_seat` |
| `hold()` | 最後の向き先を保つ | `navigate` の到着後、タスクの停止時（runner） |

`find_person` で向けた客を、未実装の `ask_name_and_drink` の間も見続ける（ルールの「会話中に動く客を追う」）。
`navigate` の間は進行方向を見る（「進行方向または目的地を見る」）。
`navigate` と `robot.location` の復元はどのタスクでも頭を動かし、到着後は経路の先の点を `map` で保つ。到着後に机や床などを見るスキルは、`look_at` か `look_towards` で向き先を指示する。

### 空席の判定

`find_free_seat` は席の候補を順に頭で見て、人のいない最初の席を `<guest>.seat`（名前）と `<guest>.seat_position` に出力する。

```json
{"guest": "guest2", "seats": "living_room.seats", "radius": 0.8, "dwell": 2.5}
```

- **候補**: `seats` は座る場所ごとの `map` の点（座面の中央 `[x, y, z]`、m）のobjectか、それを入れた記憶のキー。並びが優先順。HICでは2人の客で同じ候補を使うため、`initial_state` の `living_room.seats` に書く。ソファは座る場所ごとに分ける。人の認識は家具を出さないため、地点と同じくセットアップ日に測って書く。審判が追加した椅子や、動かされた家具は反映されない。
- **占有**: `/aha/perception/people` の見えている人は、水平距離が `radius`（既定0.8 m）以内で最も近い席を占める。人の位置は体の見える面の中央で、席より手前に出る。席の前に立つ人も席をふさぐとみなす。探す間に見えた人をすべて数え、すでに埋まっている席は見ずに飛ばす。案内している客（`guest` の名前、`<guest>.track`）は数えない。ホストや先に座った客は、記憶ではなく見えた人として除く（着席後に席を入れ替えることがあるため）。
- **見た判定**: 頭を席の0.4 m上（座った人の胴）へ向け、向けてから2 s（頭の旋回）以降に撮った画像のメッセージが届いたら、その席を見たとする。人の認識の配信は撮影より遅れるため、撮った時刻（`header.stamp`）をnodeの時計と比べる。2 s + `dwell`（既定2.5 s、計4.5 s。撮った画像が届くまでの遅れを見込む）の間に誰もいなければ空席として終わり、頭は席を向いたままにする。待ち時間は他の人の認識のスキルと同じ時計で数える。
- **見ない席**: `robot.location` の向きから左右90°（頭のpanの範囲）を超える席は見ない（台車は回さない）。案内先は席が前に来る位置にする。
- **空席が見えないとき**: 埋まっていると見えなかった最初の席（見えなかった席、見ない席を含む）を出力し、席ごとの理由（`chair_left: person 3 sits there`、`not seen`、`not looked at` など）をメッセージに付ける。1つの席を見る間に人の認識のメッセージが1通も届かないか、どれも `map` 以外の座標なら、残りの席は見ない。失敗するのは全席が埋まっていると見えたときだけ（以後は失敗時のポリシーに従い、`skip` なら `mock` の席で続ける）。

制限：

- 案内中に客のtrackが切れ、名前のない新しいtrackになると、客は占有者として数えられる。客が席の近くに立っていればその席を飛ばし、別の席を示す。

### 席の案内

`offer_seat` は席の点（`map` の `[x, y, z]`）を頭で見て、右腕で指し示す。席は記憶の `<guest>.seat` と `<guest>.seat_position`（`find_free_seat` の出力）で、paramsに `seat` と `position` を与えるとそれを使う。記憶にもparamsにもなければ、動かずに失敗する。

```json
{"guest": "guest1", "height": 0.8, "hold": 3.0}
```

1. 席が右腕で指せる方向（肩から見て右100°〜左60°）になければ、Nav2でその場で席へ向く。
2. `lift_controller` があれば手首を床から `height` m（既定0.8 m）へ上げ、腕を席へ伸ばす。手先のx軸が席の点を通る関節値を [aha_manipulation.pointing](../aha_manipulation/aha_manipulation/pointing.py) が求める。
3. 腕が着いたら現在のTFで関節値を求め直し、移動直後に台車がずれていれば向け直す。`hold` 秒（既定3 s、模擬時間）保ち、腕と昇降を開始時の姿勢へ戻す。頭は客（`<guest>.track`、名前）を追う。指し示す途中で失敗・キャンセル・例外（controllerやTFの時間切れ）があっても、腕と昇降を開始時の姿勢へ戻してから失敗を返す。

出力は `<guest>.seat`（席の名前）。`robot.holding` があるとき（`guest2` は荷物を持つ）は腕を動かさず、頭だけで席を示す。
腕は `right_arm_controller` / `lift_controller` へ関節空間の直線の軌道を送り、衝突は確認しない。開始時の姿勢から指す姿勢までの間に障害物がないことが前提で、シミュレーションの初期姿勢（全関節0）からの動作だけを確認している。

## Pick and Place Challenge

[ルールブック §5.2](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を77サブゴールに分ける。
実装済みのスキルは `navigate`、[物体の認識](#物体の認識)の `perceive_objects` / `perceive_shelf`（`objects:=true` が必要）と、認識した物体を画面に示す `show_objects`（[認識した物体を示す](#認識した物体を示す)）で、他はmockで通過する。

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
  各向きで1.5 s待ってから、2 s経ち、かつ頭が止まった後に撮られた画像の検出が届くまで（最大20 s）検出を集める（模擬時間）。
- 物体は、頭が向きごとに止まってから次の向きへ動き出すまで（最後の向きは集め終わるまで）に撮られた画像の検出だけから求める。
  画像からの検出器は1枚に数秒かかり、移動の終わり・前のサブゴールの視野・頭が動く間の画像の検出が後から届くため、待つのも選ぶのも検出の時刻（画像の時刻）による。
- 検出はtrack IDごと（trackのないものは距離）にまとめ、同じラベルで0.08 m未満の塊は1つにする（検出器がtrackを付け直した物体）。同じメッセージで一緒に検出された塊は、並んだ別の物体としてまとめない。ラベルは最も多いもの。
  2回以上検出された物体だけを使う（使う検出が向きごとに平均2回未満の遅い検出器では1回）。
- 家具の設定のframe（シミュレーションの設定は `world`）から検出のframeへはTFで変換する。`task.launch.py` の `objects:=true` は、固定地図がGazeboのworld座標で描かれているため、`world` → `map` の恒等の静的TFを出す。
- 家具の設定とカテゴリの物体リストは、runnerのパラメータ `furniture_config`・`objects_config`（空はaha_perceptionの[シミュレーションの家具](../aha_perception/config/furniture/rcjo2026_arena.yaml)と物体リスト）。
  起動時（runnerの準備）に、サブゴールの `surface`・`exclude` が家具の設定にあるかを確かめる。

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

### 認識した物体を示す

ルール16（Communicating Perception）は、認識した物体を審判へはっきり示すことを求める。
指差し、取ろうとする動作、1個ずつの表示のどれでもよく、表示を使う場合は周りの様子が見える画像にし、表示したことを伝えて審判が見たことを確かめる。
`perceive_objects` の各サブゴールの直後の `*.show`（0 pt。認識の点は直前のサブゴールに含める）で、`show_objects` が物体を1個ずつ[ロボットの画面](#ロボットの画面)に出す。

```json
{"objects": ["table.cutlery", "table.cup"], "surface": "dining_table", "dwell": 2.0}
```

`objects` は物体の記憶のキー（値は[記憶の受け渡し](#記憶の受け渡し)の辞書）。
`surface` は発話の場所（`dining_table` は「on the dining table」）、`dwell`（省略時2 s、実時間）は画像を残す時間。
物体ごとに次を行い、最後に画像を消す。

1. 頭を物体の中心へ向け（[注視](#注視)）、1.5 s（模擬時間）待つ。その後に撮られた `/camera/color/image_raw` を5 sまで待つ。
2. 立てた外接箱の8つの角を、画像の時刻のTF（`frame` → カメラの光学座標系）と `camera_info` の内部パラメータで画像へ投影し、外接する長方形を4 px広げる（`size` に0の成分がある薄い物体も枠が残る）。歪みは無視する。画像の外にはみ出た部分は切る。
   箱は `yaw` で回し、`yaw` がなければどの向きでも物体が入るよう、底面を外接円に外接する正方形にする。
3. 画像全体（切り抜かない）に、長方形の外側へ緑の枠（黒の縁取り）と `label` の札を描き、JPEGで `show` に渡す。
4. 「I found the mug on the dining table.」と字幕・TTSで伝える。最初の物体では前に「Please look at my screen.」を付ける。ゴミの役割（キーに `trash` を含む）では「That goes in the trash.」を加える。文はラベルの単数・複数（`cookies`、`milk`）によらない。発話の後、`dwell` 秒画像を残す。

審判が見たことの確かめ方はルールに定めがない。画面の表示は審判に答えを求めない方針（[ロボットの画面](#ロボットの画面)）のため、タッチや発話の返事は求めず、画面を見るよう伝えて `dwell` 秒表示を保つ。
これで「審判が見たことの確認」を満たすかは未確認。

| 状態 | 動作 | 結果 |
| --- | --- | --- |
| キーがない・`null` | 飛ばす（何も示していない物体） | 影響しない |
| 失敗してスキップした認識の `mock` の値 | 示さず、発話もしない（認識していない物体） | 結果の文に `not perceived (<認識> skipped)` と残すが、失敗にはしない（認識の失敗は操作者が対応済み） |
| 位置や大きさのない値（文字列のmockなど） | 「I recognized the fork on the dining table.」と発話だけ | 成功 |
| 画像が届かない、TFがない、頭を向けても視野外（頭の可動範囲外など）、頭を向ける間に `/clock` が止まる（模擬時間が実時間で10 s進まない） | 発話だけで伝える | 失敗（失敗時のポリシーに従う） |
| 一部が視野外 | 見えている部分に枠を描く | 成功 |

発話だけで伝える物体の前には、前の物体の画像を消す（前の枠がその物体を指して見えないように）。中断したときも画像を消す。
選択しなかった認識の `mock` は記録と同じく示すため、`--subgoals '*.show*'` で表示だけを試せる。

頭部カメラの画像とTFを使う。シミュレーションでは `objects:=true` がカメラとbridgeを起動する。認識と続けて試すには、[物体の認識](#物体の認識)のコマンドの選択を `--subgoals '*.perceive*,*.show*'` にする。
`yaw` のない記憶では、細長い物体（フォークなど）の枠が実物より大きくなる。

## General Purpose Service Robot Challenge

[ルールブック §5.3](https://robocupathome.github.io/RuleBook/rulebook/master.pdf)（2027 Rev-1、2026-10-14ドラフト）を29サブゴールに分ける。
3命令をまとめて受け取り、`plan.generate` が記憶へ出力した計画を `exec.step01`〜`exec.step20` で1手順ずつ実行する。
実行の枠は命令に結びつかないため、命令の解決（3×250）とInterleaved Task Bonus（200）の950点は `finish.report` に計上する。
実装済みのスキルは `navigate`、`run_plan_step`、`say`（[ロボットの画面](#ロボットの画面)）、人の認識を使う `find_described_person` / `count_people`（`people:=true` が必要）、[名前を尋ねる・情報を伝える](#名前を尋ねる情報を伝える) `answer_question` / `tell_info`。`plan_commands` と `listen_command` はmockの命令3件とその計画で通過する。

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
| `find_described_person` | `gesture` / `pose` / `clothing` / `description` / `name`、`info`（伝える情報）、`location` | findPrsInRoom、meetPrsAtBeac、meetName、greet*、meetNameAtLocThenFindInRm、talkInfoToGestPrsInRoom、tellPrsInfo*（`info`）、名前の人への deliver* / follow* / guide* |
| `say` / `answer_question` | `text` / `question` | 挨拶 / talkInfo、talkInfoToGestPrsInRoom |
| `count` | `target`、`location`、`output` | countObjOnPlcmt |
| `count_people` | `output`、`gesture` / `pose` / `clothing` / `description`（省略で全員）、`location` | countPrsInRoom、countClothPrsInRoom |
| `describe` | `target`、`location`、`output` | tellObjPropOnPlcmt、tellCatPropOnPlcmt |
| `tell_info` | `key`（記憶のキー）、`person`、`role` / `location`（省略可） | count*、tell*（`count` / `count_people` / `describe` / `find_described_person` の結果を伝える） |
| `follow_person` / `guide_person` | `person`、`location`（`follow_person` は省略可） | follow* / guide* |

`find_described_person` は名前・ジェスチャ・姿勢・服で探す。HICの `find_person`（名前のない最も近い人）とは別のスキル。

### 人を探す・数えるスキル

人の条件は、[コマンド生成器](https://github.com/RoboCupAtHome/CommandGenerator)の句をそのまま渡す（`"waving person"`、`"person raising their left arm"`、`"persons pointing to the right"`、`"sitting person"`、`"person wearing a red t shirt"`、`"people wearing gray sweaters"`）。値だけ（`"waving"`、`"sitting"`、`"red t shirt"`）も受け付ける。
`gesture` / `pose` / `clothing` / `description` は組み合わせられ、矛盾する指定は検証で拒否する。左右は本人の左右。
句の解釈は [person_description.py](aha_tasks/person_description.py)、人の判定は [person_perception](../aha_perception/README.md#姿勢ジェスチャ服) の `ClassifyPerson`。

- **範囲**: `location` の地点（省略時はロボットの位置）から `radius`（既定3.0 m）以内の人だけを対象にする。人が `map` 以外の座標なら範囲で絞らない。部屋の形は使わない。
- **見回し**: 頭を `pans`（既定 `[0.0, 0.4, -0.4]` rad）へ `dwell`（既定6 s）ずつ、2 m先の高さ `look_height`（既定1.0 m）へ向ける。
- **判定**: 1人につき、頭をその人の頭へ向けて0.75 s待ち、その時に見えた頭の位置へ向け直してさらに0.75 s待つ（頭が画像の端で切れると頭の位置が低く出るため。挙げた手も写るように）。次に `skip_vlm` で深度とキーポイントの確かな答えだけを聞く。条件の成否が決まらなければ（服、不確かな姿勢・ジェスチャ）VLMに聞く。VLMの答えは `describe_timeout`（既定180 s、person_perceptionの `vlm_timeout` に合わせる）より30 s長く待つ（サーバーは先に動きを待つため）。判定に失敗した人（見失った、VLMの失敗、時間切れ）はもう1回判定し、それでも失敗すれば「判定できず」として続ける。中断は `Cancelled` で返す。
- **名前**: `name` はperson_perceptionが覚えた名前と照合し、その名前の人がいれば先に選ぶ。`ask`（既定 `true`）なら、名前のない人にも近づいて名前を尋ね（[名前を尋ねる](#名前を尋ねる情報を伝える)）、本人なら覚えさせる。この実行で覚えた名前を探すときは、名前のない人を尋ねる前に `pans` を1巡してその名前の人を探し、新しいtrackには3 s待つ（VLMが名前を付けるため）。`ask: false` は覚えた名前だけを待つ。
- **伝える情報**: `info`（`name` / `pose` / `gesture`）は後で伝える人の情報で、他の条件がなければ範囲内の誰でもよい。`pose` / `gesture` は分かるまでVLMに聞き、`name` は覚えた名前がなければ尋ねる。
- **制限時間**: `timeout` に達すると、1人の判定（注視と `ClassifyPerson` の待ち）や、尋ねるための移動・会話の途中でも打ち切る。

| スキル | 動作 | 出力 |
| --- | --- | --- |
| `find_described_person` | 見回しながら、範囲内の人を近い順に1回ずつ判定し、最初に条件に合った人を見て、`approach`（既定 `true`）なら `distance`（既定1.0 m）手前まで近づく。`timeout`（既定60 s、名前を尋ねるときは120 s）で失敗 | `<role>.track` / `.position` / `.frame`、見えた `<role>.pose` / `.gesture` / `.garments`、分かった `<role>.name`、`<role>.<info>.answer`（文）、`robot.location`（`role` の既定は `person`） |
| `count_people` | `pans` を1巡し、見えている人を1人ずつ判定する。その向きに判定していない人が残る間は次の向きへ進まず、人を判定したらその向きを `dwell` だけ見直す。1人の判定は1回（失敗した場合は2回まで）で、判定できなかった人と、答えで条件の成否が決まらなかった人は数えない。`map` で0.5 m未満の観測は同じ人とする。`timeout`（既定300 s）で失敗 | `<output>`（人数）、`<output>.positions`（条件に合った人の最後の位置）、`<output>.answer`（`"There are 2 waving people in the office."`） |

trackは視野を外れると3 sで破棄され、見直すと別のIDになる。そのため判定は見えている間に行い、判定した人の `map` の位置から0.5 m未満に現れた新しいtrackは、同じ人として判定し直さない（数え直さない）。ただし、判定した人のtrackが同じ画像にまだ見えていれば、近くの新しいtrackは別の人として判定する。

### 名前を尋ねる・情報を伝える

[conversation.py](aha_tasks/conversation.py) が文を作り、答えを解釈する（ROS非依存）。発話は [ロボットの画面](#ロボットの画面) の `say`、聞き取りは `listen`（ASR、`/aha/perception/audio/listen`）。
名前の一覧・ロボットとチームの情報はタスク定義の `initial_state` に書く。

| 記憶 | 内容 |
| --- | --- |
| `gpsr.names` | 人の名前。既定は[CompetitionTemplate](https://github.com/RoboCupAtHome/CompetitionTemplate)の10人。ASRの `hotwords` に渡し、聞いた語と照合する |
| `gpsr.robot_name` / `gpsr.team_name` / `gpsr.team_country` / `gpsr.team_affiliation` | `AhaRobot` / `TRAIL` / `Japan` / `the University of Tokyo` |
| `gpsr.utc_offset` | 時刻・日付の時差（時間）。省略時はシステムの時間帯 |

- **名前の照合**: 返答の語を `gpsr.names` と照合し、綴りの類似度0.8以上か、Soundexが同じで0.54以上の語も名前とする（`"Jewels"` は Jules）。一覧にない語は名前としない（`"I'm fine"`、`"What?"`）。名前は「I'm Charlie」のようなcueの後か、名前だけの返答のときだけ答えとみなす（「Charlie is over there」は答えでない）。
- **名前を尋ねる**（`find_described_person` の `name`）: 人に近づいて頭を向け、`"Excuse me, are you Charlie?"` と聞く。否定（no、`"not me"`、`"I'm not"`。`"I didn't hear you"` は否定としない）、名前（`"I'm Jane"` は別人）、肯定（yes、`"That's me"`）の順に判断する。本人なら `remember_person` でその名前を覚えさせる。近づけない人にはその場から尋ね、本人と分かった後に近づき直す。それでも近づけなければ、その人を見つけた結果としてその場から返す（失敗にしない）。聞き取れない・判断できない答えは1回聞き直し、それでも分からない人は尋ね直さずに次の人へ進む。`touch: true` なら、その前に画面の `Yes` / `No` のタッチを10 s待つ（音声認識の迂回として −50）。
- **名前を聞く**（`info: "name"`）: `"Excuse me, what is your name?"` と聞く。2回聞いて分からなければその人を諦める。
- **`answer_question`**: `question` は生成器の `{talk}`（`"the time"`、`"your teams name"` など9種）か言い換え。キーワードで話題を決め（明日、月、週、時刻、所属、国、チーム名、今日、自己紹介の順）、決まらない質問は未対応としてその手順だけをmockで通過する。答えは `"It is 3:05 PM."`、`"Today is Monday, November 16th, 2026."`、`"My team's affiliation is the University of Tokyo."` など。時刻は実時間。
- **`tell_info`**: `<key>.answer` があればそれを、なければ `<key>` の値を文にして言う（`<role>.name` / `.pose` / `.gesture` は人の情報、他は `"The answer is 3."`）。値がなければ失敗する。言う前に相手を向く。`person` が `operator`（`me`）なら `instruction_point` へ戻り、`location` があればそこへ行き、その地点から2.0 m以内に見えた最も近い人を向く（頭を正面・左右0.6 radへ2 sずつ、6 sまで探す）。それ以外は `<role>.track` / `.name` / `.position`（`role` の既定は `person`）の人を向く。1.5 mより遠い人には `distance`（既定1.0 m）手前まで近づく。人が見えないときと、向く・近づく移動に失敗したときは、その場で言う。

伝える手順の例（tellPrsInfoAtLocToPrsAtLoc）：

```json
[{"command": 1, "action": "navigate", "params": {"location": "bed"}},
 {"command": 1, "action": "find_described_person", "params": {"info": "gesture", "location": "bed"}},
 {"command": 1, "action": "navigate", "params": {"location": "desk"}},
 {"command": 1, "action": "tell_info", "params": {"key": "person.gesture", "person": "person at the desk", "location": "desk"}}]
```

制限：

- 人の答えは1発話だけで判断し、言い直しや「はい、でも…」のような文の意図は解釈しない。名前は一覧の語に寄せるため、一覧にない似た名前（Jean）は一覧の名前（Jane）として扱う。
- 尋ねる人には `distance` 手前まで近づくため、人が多い部屋では移動と会話に時間がかかる（1人あたり移動・発話・最大2回の聞き取り）。
- 覚えさせた名前が後の手順で付くかは、person_perceptionのVLMの照合による。照合されなければ後の手順で再び尋ねる。
- 名前を尋ねる手順（`name`、`info: "name"`）はASRが要り、なければ（`touch: true` の `name` では画面もなければ）その場で失敗する（mockにしない）。シミュレーションにはASRもTTSもなく、人の声は代わりのListenのserverが要る。`gpsr.json` のmockの計画には名前の手順はない。

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

- モデルはGazebo Fuelの [Male visitor](https://app.gazebosim.org/OpenRobotics/fuel/models/Male%20visitor)（version 2、CC0）。初回に取得してハッシュを確かめ、骨格の重みで姿勢を付けたメッシュとシャツの色を変えたテクスチャを `~/.cache/aharobot/sim_people` に作る。`sim_guests` の座ったホストと同じキャッシュで、ファイルは一時ファイルに書いてから置き換えるため、同時に起動しても書きかけを読まない。
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
