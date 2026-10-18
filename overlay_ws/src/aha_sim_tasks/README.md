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
全タスクは2台の机とappleを共有する。配置とpolicy設定は
[scene.json](config/scene.json)、world構成は [worlds/](worlds/) にある。

| オプション | 用途 |
| --- | --- |
| `--task pick_apple` | タスクを個別に選択。`--list-tasks` で判定条件を表示 |
| `--no-cameras` | カメラセンサ・描画・画像観測を無効化し、CPUのみで評価 |
| `--no-headless` | Gazeboの3D画面を表示 |
| `--camera-view` | 頭部・左右手首のRGB画像を別窓で表示。カメラとGUI環境が必要 |
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

## 成功条件と制限

[タスク定義](config/tasks.json)に従い、次の条件を保持時間中すべて満たすと成功する。
距離はm、角度はrad、保持時間とタスク期限はシミュレーション時間。

| タスク | 成功条件 | 保持 | 期限 |
| --- | --- | --- | --- |
| `approach_apple` | 接近目標との平面距離が0.04以内、初期方向とのyaw差が0.08以内、速度0.02 m/s以下 | 1秒 | 45秒 |
| `pick_apple` | 右の両指がappleに接触し、apple中心の床からの高さが0.24以上 | 1秒 | 60秒 |
| `place_apple` | 両指で接触して高さ0.24以上に持ち上げた後、接触なし・配置目標との差がx/y各0.08以内、zが0.02以内・速度0.03 m/s以下 | 2秒 | 90秒 |

<a id="grasp-model"></a>

判定はGazeboの姿勢・接触・時刻を同じサンプルで受け取り、policyの完了申告には依存しない。
接触数は直前50 msにappleへ触れた右指の種類数（0～2）で、両指が同時に触れたことは要求しない。
解放には窓全体で接触がないことが必要。重複・逆順のサンプルは保持時間を進めず、
0.5秒を超える受信間隔で保持をやり直す。

appleは自由剛体で、衝突・摩擦・重力によって把持や落下が決まる。評価処理は物体を固定・移動しない。
形状・摩擦・駆動応答は実機に合わせて校正していない。

`scripted` は関節位置と車輪odom、固定の配置設定を使って接近・把持・配置する。
ライブの物体位置は参照せず、初期方向に沿う前進のみ対応する。横移動・後退が必要な配置はエラーになる。

観測停止、policyの例外・不正な指令は `error` となる。推論中も観測と判定を継続する。
実時間の期限は次のとおり。遅い描画やモデルの読み込み・推論では対応する値を延ばす。

| 対象 | 既定の期限 | オプション |
| --- | --- | --- |
| シミュレータの準備 | 90秒 | 固定 |
| 準備後の試行 | 240秒 | `--wall-timeout` |
| policyの読み込み・生成・`reset` | 120秒 | `--policy-startup-timeout` |
| `act` 1回 | 30秒 | `--policy-timeout` |
| カメラの受信・利用可能な画像組の待機 | 10秒 | `--camera-timeout` |

<a id="pick-with-ground-truth-ik"></a>

## 真値IKによる把持

ビルド・setupの読み込み後に実行する。

```bash
ros2 run aha_sim_tasks evaluate --task pick_apple --policy ground_truth_ik \
  --episodes 1 --output /tmp/aha-evaluation/ik-pick.json
```

`pick_apple` 専用で、カメラ観測・センサ描画は無効。`--no-headless` で3D画面を表示できる。
Gazebo真値のロボット姿勢・関節位置・物体位置・指接触を使い、
[MoveIt](../aha_manipulation/README.md)で右腕のIK・衝突検証・軌道実行を行う。
ベースを停止してから物体の上方・中心へ腕を動かし、両指の接触後に持ち上げる。
腕の段階遷移は実行成功と真値の到達判定を待ち、失敗や期限超過はエラーにする。

planning sceneには両机とappleを登録する。指との接触を許可し、把持後はappleの衝突形状を
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
物理接触を含む検証は上記の評価コマンドを使う。`noop` はタスク期限で失敗し、終了コード1となる。
