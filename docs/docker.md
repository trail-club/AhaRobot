# Docker で開発する

## 前提

- Docker Engine 24+ / Docker Desktop 4.30+
- docker compose v2
- 対応 OS: **Linux (native X11)** / **macOS (Docker Desktop)** / **Windows (WSL2 + Docker Desktop)**

## 初回

```bash
git clone --recursive git@github.com:trail-club/AhaRobot.git
cd AhaRobot

# イメージ build (10〜20 分)
./run_docker_container.py --rebuild
```

## 起動 & シェルへ

```bash
./run_docker_container.py            # ホスト自動判定 (NoVNC / X11, NVIDIA GPU も自動検出)
./run_docker_container.py --no-enter # 起動だけ (別ターミナルで make shell)
./run_docker_container.py --no-gpu   # NVIDIA 検出されても CPU で動かす
./run_docker_container.py --novnc    # Linux でも NoVNC を強制
```

コンテナ名は常に `aharobot_aha_project_1`。

## プラットフォーム別の挙動

| ホスト | GUI 経路 | 追加 compose |
|---|---|---|
| Linux (X11)          | ホストの X server            | なし |
| Linux + NVIDIA       | ホスト X + GPU passthrough  | `docker-compose.gpu.yml` |
| macOS                | NoVNC (auto)                 | (base の `--profile darwin`) |
| WSL (Windows)        | NoVNC (auto)                 | `docker-compose.wsl-novnc.yml` |
| WSL + NVIDIA         | NoVNC + GPU                  | `.gpu.yml` + `.wsl-novnc.yml` |

**NoVNC の使い方 (macOS / WSL)**:
1. `./run_docker_container.py` 起動後にブラウザで [http://localhost:8080/vnc.html](http://localhost:8080/vnc.html)
2. **Connect** クリック（パスワード不要）
3. コンテナ内で起動した GUI（Gazebo / RViz など）がそこに映る

## コンテナ内での典型作業

```bash
# build
cd /app/overlay_ws
colcon build --symlink-install
source install/setup.bash

# sim
ros2 launch aha_bringup sim.launch.py world:=home.sdf

# 各班 launch 併用
ros2 launch aha_bringup sim.launch.py use_nav:=true use_perception:=true

# rviz (別 shell から)
rviz2 -d /app/overlay_ws/install/aha_navigation/share/aha_navigation/rviz/nav.rviz
```

## Makefile ショートカット

| コマンド | 内容 |
|---|---|
| `make build` | イメージ build |
| `make up` | base compose のみで起動（NoVNC/GPU/WSL は `run_docker_container.py`）|
| `make shell` | 既存コンテナに shell |
| `make down` | コンテナ停止・削除（全 overlay + profile）|
| `make clean` | overlay_ws の build/install/log 削除 |

## トラブルシューティング

- **Linux で GUI が出ない**: `xhost +local:docker` または `LIBGL_ALWAYS_SOFTWARE=1 ./run_docker_container.py --novnc`
- **macOS で Gazebo が重い**: GPU 無し + software rendering なので想定内。開発は headless smoke test + 実描画は Linux 機で。
- **WSL で NoVNC 画面が真っ黒**: `./run_docker_container.py --no-enter` で up 後、`docker logs aharobot_novnc_1` を確認。NoVNC が起動しきる前にブラウザを開くと空になることがある（10〜20 秒待って再接続）
- **NVIDIA GPU が使われない**: `docker exec aharobot_aha_project_1 nvidia-smi` で見えるか確認。見えなければ nvidia-container-toolkit の設定が不足
- **`rosdep` の deps 未解決**: コンテナ内で `sudo apt update && rosdep update` → `rosdep install --from-paths /app/overlay_ws/src --ignore-src -r -y`
- **`upstream/*` が空**: ホスト側で `git submodule update --init --recursive`
