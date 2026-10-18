# aha_sim_tasks

Gazebo Harmonicで接近・把持・配置を評価するパッケージ。
Pythonのpolicyを実行し、物体位置と指の接触から独立して成功を判定する。
[開発コンテナ](../../../docs/docker.md)内で使用する。

## 実行

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_sim_tasks
source install/setup.bash
ros2 run aha_sim_tasks evaluate --task all --policy scripted --episodes 2 \
  --output /tmp/aha-evaluation/results.json
```

各試行でロボット・controller・物体を含むシミュレータを新しく起動し、終了時に関連プロセスを停止する。
全タスクは2台の机とYCB物体を共有する。配置とpolicy設定は
[scene.json](config/scene.json)、world構成は [worlds/](worlds/) にある。

| オプション | 用途 |
| --- | --- |
| `--task pick_object` | 接近・把持・配置を選択。`--list-tasks` で判定条件を表示 |
| `--objects apple orange soup_can gelatin_box` | 対象物体。`all` は全4種。既定はapple |
| `--randomize --seed 100` | 物体の位置・yaw、ロボット初期姿勢、照明を試行ごとに変更 |
| `--distractors 2` | 対象とは別種のYCB物体を追加（0～3個） |
| `--scene-config /absolute/path/scene.json` | 配置・物理・センサ設定を指定。保存済みsceneの再利用にも使用 |
| `--no-cameras` | カメラセンサ・描画・画像観測を無効化し、CPUのみで評価 |
| `--no-headless` | Gazeboの3D画面を表示 |
| `--camera-view` | 頭部・左右手首のRGB画像を別窓で表示。カメラとGUI環境が必要 |
| `--record-video` | 3視点のRGB動画とAction・判断メモをH.264 MP4へ保存。内蔵カメラが必要 |
| `--policy noop` | 無操作でタイムアウトを確認 |

既定では頭部RGB-D・左右手首RGBカメラを描画する。headlessでもEGLの描画環境が必要で、
ソフトウェア描画では評価が遅くなる。GUIの設定は開発コンテナの手順を参照。
配置動作を画面で確認する例：

```bash
ros2 run aha_sim_tasks evaluate --task place_apple --policy scripted \
  --episodes 1 --no-headless --camera-view --output /tmp/aha-evaluation/watch.json
```

JSONレポートに成功率・判定条件・各試行の状態と測定値を記録し、ログを
`<レポート名>_episodes/` に保存する。全試行成功時の終了コードは0、それ以外は1。
状態は `success`、`timeout`、`wall_timeout`、`error` のいずれか。

`--record-video` は各episodeの `episode.mp4` に頭部・左右手首のRGBを横並びで保存する。
10 fpsのシミュレーション時間で再生し、step間の動作を記録する。推論中のpause時間は動画に含めず、
最後に結果を1秒表示する。経過時間・各画像のstamp・現在のActionを表示し、
Codexでは `situation` と `decision_note` も重ねる。
`video_frames.jsonl` に動画frameとセンサstamp・decision番号の対応、`video.json` に形式・長さを保存し、
評価JSONにも動画の情報を記録する。例外時も、それまでの動画を終了処理で保存する。
OpenCVのH.264 encoderが必要で、利用できなければpolicy実行前にエラーにする。

## 成功条件と制限

[タスク定義](config/tasks.json)に従い、次の条件を保持時間中すべて満たすと成功する。
距離はm、角度はrad、保持時間とタスク期限はシミュレーション時間。

| タスク | 成功条件 | 保持 | 期限 |
| --- | --- | --- | --- |
| `approach_object` | 接近目標との平面距離0.04以内、目標yawとの差0.08以内、速度0.02 m/s以下 | 1秒 | 60秒 |
| `pick_object` | 同じphysics stepで両指が対象に接触し、回転後の衝突メッシュの最下点が元の天板から0.10以上離れる | 1秒 | 90秒 |
| `place_object` | 上記の把持・持ち上げを1秒保持した後、指接触なし・物体全体が配置先の天板内・配置先天板との接触あり・底面の高さ差0.015以内・速度0.03 m/s以下・角速度0.15 rad/s以下 | 2秒 | 120秒 |

`approach_apple` / `pick_apple` / `place_apple` はapple専用の互換名。
`--task all` は3つの汎用タスクを実行する。判定の境界は、衝突メッシュの全頂点を現在の姿勢でworld座標へ変換して求める。
物体が回転しても、実形状の底面・投影範囲を使って判定する。
`xy_tolerance: "table_half_size"` は机の半幅へ解決し、数値または `[x, y]` も指定できる。

<a id="grasp-model"></a>

判定はGazeboの姿勢・接触・時刻を同じサンプルで受け取り、policyの完了申告には依存しない。
指接触数は直前50 msの窓内の和集合で、把持にはさらに同じphysics stepでの両指接触を要求する。
物体の全collisionを集計する。解放には窓全体で指接触がないことが必要。
重複・逆順のサンプルは保持時間を進めず、0.5秒を超える受信間隔で保持をやり直す。

物体は自由剛体で、衝突・摩擦・重力によって把持や落下が決まる。評価処理は物体を固定・移動しない。
[objects.json](config/objects.json) は既存の `tmc_wrs_gz_worlds` のYCBモデルを参照する。
元のmesh・質量・慣性を使い、物体のscaleは変更しない。モデルの原点は衝突形状の中心へ移し、
visual・collision・重心の相対関係を保つ。資源とライセンスは
[THIRD_PARTY](../../../dependencies/THIRD_PARTY.md)を参照。

机は幅0.50 m・奥行0.60 m・天板高0.74 mで、4本の脚にもcollisionを持つ。
タスク起動時の左右liftは `initial_lift`（既定0.02 m）から開始する。
初期落下によって下限の外へ出て動かなくなる問題を避け、物理の関節限界は変更しない。
ロボットの `base_footprint` は床面（spawn z=0）から開始し、起動時の落下を小さくする。
`scene.json` の `source_position` / `destination_position` のx/yが物体の目標位置で、
zは天板高と物体寸法から計算する。机は `table_offset` 分だけ物体目標から離す。

`scripted` は関節位置・車輪odomと、保存したepisodeの初期配置を使う診断用baseline。
平面の位置・yawを合わせて接近・把持・配置する。ライブの物体真値は使わないが、
サンプリングした配置を知っているため、レポートでは `sensors_and_known_episode_layout` と明記する。
通常の独自policyとCodexへsceneファイルを環境変数で渡すことはない。

## 再現可能な評価とセンサ設定

```bash
ros2 run aha_sim_tasks evaluate --task place_object --objects all --policy scripted \
  --randomize --seed 100 --distractors 2 --episodes 20 --no-cameras \
  --output /tmp/aha-evaluation/ycb.json
```

seedは試行ごとに1増え、同じ物体・試行番号はタスクやpolicyを変えても同じ条件になる。
物体のxyは±0.025 m、yawは全周、ロボットのxyは±0.025 m・yawは±0.08 rad、照明も変える。
机は固定し、物体が天板内に収まり、物体同士が重ならない配置を選ぶ。
JSONに全体と物体別の成功率を保存する。各試行の `scene.json` は実際の配置・物理・センサ設定を含み、
`state_samples.jsonl` は判定サンプルと測定値を記録する。保存sceneを `--scene-config` に指定すると、物体・clutter・seedを引き継いで配置を再利用する。
`--randomize` を加えると再サンプリングする。Gazeboの乱数seedも設定するが、
physics・描画・ROSの実行順による差までbit単位で保証するものではない。

`sensors` はpolicy用頭部深度に加えるGaussian noiseの標準偏差（既定0.005 m）、
欠測率（既定0.01）、Gazeboの頭部・手首RGB noise標準偏差（既定0.003、正規化輝度）を指定する。
深度の乱数はseedとセンサstampで決まり、元のROS messageと校正・TFは変更しない。
深度の欠測はNaN。Gazebo/ROSの生depth表示には、このpolicy用の追加noiseを反映しない。
noiseを0にしたsceneで理想センサ条件も評価できる。
標準bringupのRGB noiseは `camera_rgb_noise_stddev` で変更できる。

机寸法・摩擦・noise・把持高さのオフセット・駆動応答はシミュレーション用の設定で、
実機の到達範囲・摩擦・深度誤差への校正結果ではない。カメラ取付の設計値は既存のURDFを使う。

観測停止、policyの例外・不正な指令は `error` となる。推論中も観測と判定を継続する。
実時間の期限は次のとおり。遅い描画やモデルの読み込み・推論では対応する値を延ばす。

| 対象 | 既定の期限 | オプション |
| --- | --- | --- |
| シミュレータの準備 | 90秒 | 固定 |
| 準備後の試行 | 240秒 | `--wall-timeout` |
| policyの読み込み・生成・`reset` | 120秒 | `--policy-startup-timeout` |
| `act` 1回 | 30秒（`codex` は150秒） | `--policy-timeout` |
| カメラの受信・利用可能な画像組の待機 | 10秒 | `--camera-timeout` |

## Codexによるpolicy評価

`codex` は各stepでCodex CLIを呼び、`gpt-6.1-sol`・reasoning effort `medium` で
次のActionと短い状況説明・判断理由を生成する。
開発コンテナ内で `codex` がPATH上にあり、`codex login status` で認証済みであることが必要。
CLIの画像添付・構造化出力は [OpenAI公式ドキュメント](https://learn.chatgpt.com/docs/developer-commands?surface=cli)を参照。

```bash
ros2 run aha_sim_tasks evaluate --task place_apple --policy codex \
  --episodes 1 --wall-timeout 1800 \
  --camera-timeout 30 --record-video --output /tmp/aha-evaluation/codex-place.json
```

step開始時の関節位置・速度、odom、時刻・指示、頭部・左右手首のRGB画像、
各画像のCameraInfoとbase_linkからのTFを渡す。
頭部深度は16列×12行以下の画素位置・距離のグリッドにしてプロンプトへ含め、
無効値は `null` とする。元の深度配列もログへ保存する。
Gazeboの物体真値・接触数・評価値は渡さず、scripted policyへのフォールバックも行わない。
各呼び出しは独立したCLIセッションで、直近8回のAction・状況説明・判断理由を渡す。

`--step-duration` はActionを適用するシミュレーション秒数で、0.1秒の正の倍数を指定する。
Codexの既定は2秒。推論中はGazeboをpauseし、Actionの適用中だけ1 msのphysicsを100回ずつ進める。
各100 msで同じActionを再配信するため、ベース指令がcontrollerの0.5秒の期限で失効することを防ぐ。
成功判定は従来と同じphysicsの観測に基づき、保持時間・タスク期限は推論待ち時間を含まない。
実時間の `--wall-timeout` と `--policy-timeout` は有効で、CLI自体の呼び出し期限は120秒。
Codexの `--policy-timeout` の既定は150秒で、120秒以下の指定はエラーとする。
他のpolicyも `--step-duration` を使えるが、`ground_truth_ik` は連続実行のみ対応する。
他のpolicyの既定は0（連続実行）。Codexでは3台の内蔵カメラが必須。

各episodeの `codex_steps/decisions.jsonl` と `policy.log` に、step番号・時刻・モデル・
推論時間・Action・`situation`・`decision_note` を記録する。
`codex_steps/0001/` などには入力画像・深度配列・観測JSON・プロンプト・出力JSON・CLIログを保存する。
不正なAction、CLI失敗、期限超過はepisodeを `error` とする。
このpolicyはカメラと関節指令による実験用で、タスクの成功を保証しない。

<a id="pick-with-ground-truth-ik"></a>

## 真値IKによる把持

ビルド・setupの読み込み後に実行する。

```bash
ros2 run aha_sim_tasks evaluate --task pick_apple --policy ground_truth_ik \
  --episodes 1 --output /tmp/aha-evaluation/ik-pick.json
```

`pick_object`（互換名 `pick_apple`）専用で、カメラ観測・センサ描画は無効。`--no-headless` で3D画面を表示できる。
Gazebo真値のロボット姿勢・関節位置・物体位置・指接触を使い、
[MoveIt](../aha_manipulation/README.md)で右腕のIK・衝突検証・軌道実行を行う。
ベースを停止してから物体の上方・中心へ腕を動かし、両指の接触後に持ち上げる。
腕の段階遷移は実行成功と真値の到達判定を待ち、失敗や期限超過はエラーにする。

planning sceneには両机の天板と対象物体のbounding boxを登録する。指との接触を許可し、把持後は対象物体の衝突形状を
MoveIt側だけで腕に追従させる。固定シーンの静止物体と初期方向に沿う接近を前提とし、
未知の障害物検出・配置は対応しない。高さ・許容誤差・段階期限は `scene.json` の `ground_truth_ik` にある。

このpolicyだけ評価と同じROS domainを使い、`--policy-ros-domain-id` は `--ros-domain-id` と同じ値にする。
真値のデータ形式とMoveItの状態入力は [ROSインターフェース](../../../docs/interfaces.md#タスク評価)を参照。
レポートの `observation_source` は `gazebo_ground_truth` となる。

## 独自policyの接続

`--policy my_package.my_policy:create_policy` で、Pythonパス上のfactoryを指定する。
試行ごとに別プロセスで生成し、`reset` 後に最大10 Hz（実時間）で `act` を呼ぶ。
型と関節名・制限は [api.py](aha_sim_tasks/api.py)を参照。

```python
from aha_sim_tasks.api import Action, Observation


class MyPolicy:
    def reset(self, task_id: str, instruction: str) -> None:
        self.instruction = instruction

    def act(self, observation: Observation) -> Action:
        return Action(joint_positions={"joint_r1": 0.18})


def create_policy():
    return MyPolicy()
```

通常の `Observation` は時刻、タスクID・指示、関節位置・速度、odomのx/y/yaw、カメラを含む。
関節・カメラの辞書は読み取り専用で、物体の真値・接触数・スコアは渡さず、`ground_truth` は `None`。
真値IKでは `ground_truth` に姿勢・物体位置・接触数を渡し、関節位置と時刻も同じphysicsサンプルを使う。
odomはworld x/y/yaw、関節速度は空となり、センサtopic・TFは購読しない。

`cameras` のキーは `head`、`left_wrist`、`right_wrist`。各要素はRGB画像、CameraInfo、
画像時刻の `base_link` → optical frameのTF、頭部のみの位置合わせ済み深度（32FC1、m）を持つ。
3視点の時刻差は0.1秒以内、画像の遅れは0.5秒以内（clock受信遅れによる0.1秒の先行は許容）。
関節・odomは最新値で、画像時刻とは同期しない。`head_image` は頭部RGB画像を指す。
`--no-cameras` では `cameras` は空、`head_image` は `None`。
外部画像は `--head-image-topic /your/image/topic` で指定し、内蔵カメラとは併用できない。
この場合は最新画像だけを渡し、校正・TFの組み合わせは行わない。
カメラ構成は [aha_perception](../aha_perception/README.md#シミュレーション)を参照。

`Action` はベースの並進・旋回速度と関節位置を指定する。省略した速度は0、関節は直前の指令値
（初回は測定値）を保持する。位置軌道は2秒で、未到達なら3秒後に再送する。
速度制限は0.3 m/s・0.5 rad/s。関節制限外、未知の関節名、非有限値はエラーになる。

通常policyのROS domainは評価と分け、Gazebo partitionも試行ごとに分離する。
既定のdomainは評価87、policy88。`--ros-domain-id` と `--policy-ros-domain-id` で変更でき、
policy側を省略すると評価側+1（232の次は0）になる。他のROSアプリと重複させない。
観測と指令はIPCで交換し、policyの標準出力・例外は `policy.log` に保存する。
この分離は真値の誤用を防ぐためのもので、セキュリティ上の隔離ではない。

## タスク追加と検証

既存の判定方式（`approach`・`pick`・`place`）は `config/tasks.json` に追加して使う。
新しい判定方式は [evaluation.py](aha_sim_tasks/evaluation.py)に実装し、対応するpolicyも用意する。
単体テストはリポジトリの [make test](../../../AGENTS.md#動作確認)、
`colcon test --packages-select aha_sim_tasks` は全4種の静置・押し動作・自由落下も検証する。
ロボットの把持・配置は上記の評価コマンドを使う。`noop` はタスク期限で失敗し、終了コード1となる。
