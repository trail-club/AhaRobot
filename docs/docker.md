# 開発コンテナ

## 起動フロー

### 1. 取得・起動（ホスト）

```bash
git clone --recursive git@github.com:trail-club/AhaRobot.git
cd AhaRobot
./run_docker_container.py --rebuild  # 初回・Dockerfile変更時
```

既にclone済みなら、リポジトリ直下で `git submodule update --init --recursive` を実行する。
通常起動は `./run_docker_container.py`。起動後にコンテナ内シェルへ入る。
起動だけなら `--no-enter`、後から入る場合はホストで `make shell` を使う。
コンテナ名は `aharobot_aha_project_1`、リポジトリは `/app` にマウントされる。

### 2. GUIに接続する

LinuxはX11、macOS / WSL2はNoVNCを自動使用する。NVIDIA GPUはLinux / WSL2で自動検出する。
LinuxでNoVNCを使うには `--novnc`、GPUを無効にするには `--no-gpu` を起動コマンドに付ける。

1. NoVNCでは、ホストのブラウザで [http://localhost:8080/vnc.html](http://localhost:8080/vnc.html) を開く。
2. **Connect** を押す（パスワード不要）。コンテナ内で起動したGazebo / RVizが表示される。

macOS / LinuxのNoVNCでは、GUIを起動するコンテナ内シェルで `export DISPLAY=display:0` を実行する。
WSL2の表示先は自動設定される。

### 3. ビルド・シミュレーション起動（コンテナ内）

```bash
cd /app/overlay_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch aha_bringup sim.launch.py
```

world・初期位置・GUIなしの設定は [Japan Open起動設定](sobits-rcjo2026.md)を参照。

### 4. 別ターミナルでの操作・終了

ホストの別ターミナルで、リポジトリ直下から `make shell` で入り、コンテナ内で実行する。

```bash
source /app/overlay_ws/install/setup.bash
ros2 run aha_bringup teleop_base.sh
```

RVizも別ターミナルで `make shell` と同じsetupの読み込み後、`ros2 launch aha_description view_robot.launch.py` で起動する。
各プログラムを `Ctrl+C`、シェルを `exit` で終了する。コンテナの停止・削除はホストで `make down`。

## ショートカット（ホストのリポジトリ直下）

| コマンド | 内容 |
| --- | --- |
| `make build` | イメージをビルド |
| `make up` | base Composeのみで起動。NoVNC・GPU・WSLは起動スクリプトを使う |
| `make ws-build` | 起動済みコンテナで `aha_bringup` までビルド |
| `make sim` | 起動済みコンテナでシミュレーション起動 |
| `make clean` | workspaceのbuild / install / logを削除 |
| `make test` | [PR前の動作確認](../AGENTS.md#動作確認) |

## 問題の切り分け

- LinuxのX11でGUIが出ない: ホストで `xhost +local:docker`、または `--novnc` を使う。
- NoVNCが黒い・接続できない: 10–20秒待って再接続し、ホストで `docker logs aharobot_novnc_1` を確認する。
- GPUが使えない: ホストで `docker exec aharobot_aha_project_1 nvidia-smi` を確認する。
- rosdepの未解決: コンテナ初期化ログを確認する。
