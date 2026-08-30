# Docker で開発する

参考: `~/trail/kachaka_challenge_trail2026` の構成を踏襲。差分は Kachaka bridge を外し、
Gazebo Harmonic + ros2_control 一式を同梱、upstream/ と overlay_ws/ の 2 段構成に対応。

## 前提

- Docker + docker compose v2
- Linux: X サーバ (X11)
- macOS: Docker Desktop + ブラウザ (GUI は NoVNC 経由)

## 初回

```bash
# イメージ build (10〜20 分)
make build
# もしくは:
./run_docker_container.py --rebuild
```

## 起動 & シェルへ

```bash
./run_docker_container.py
```

これで:
1. `aharobot-aha_project-1` コンテナが起動 (無ければ compose up)
2. 初回のみ `upstream/*` を `overlay_ws/src/` へシンボリックリンク + `rosdep install`
3. `docker exec -it ... bash` で入る

macOS の場合はブラウザで `http://localhost:8080/vnc.html` を開き Connect。RViz / Gazebo はそこに映る。

## コンテナ内での典型作業

```bash
# 初回 or 変更後の build
aha_build           # alias: colcon build --symlink-install --packages-up-to aha_bringup

# sim 起動 (Gazebo + spawn + controllers)
aha_sim             # alias: ros2 launch aha_bringup sim.launch.py

# URDF だけ RViz で確認
aha_view

# 走行コマンド (別ターミナルで再度 shell に入る: docker exec -it aharobot-aha_project-1 bash)
ros2 topic pub /diff_drive_controller/cmd_vel geometry_msgs/msg/TwistStamped \
  '{twist: {linear: {x: 0.2}}}' -r 10
```

## ホスト側 Makefile ショートカット

| コマンド | 内容 |
| --- | --- |
| `make build` | イメージ build |
| `make up` | compose up (バックグラウンド起動) |
| `make shell` | 既存コンテナに shell で入る |
| `make ws-build` | コンテナ内で colcon build |
| `make sim` | コンテナ内で sim.launch.py |
| `make down` | コンテナ停止・削除 |
| `make clean` | overlay_ws の build/install/log 削除 |

## トラブルシューティング

- **Gazebo GUI が起動しない (Linux):**
  `xhost +local:docker` をホストで実行、または `LIBGL_ALWAYS_SOFTWARE=1 make up` で
  ソフトウェアレンダリング。
- **macOS で Gazebo が重い / 落ちる:**
  Gazebo Harmonic は GPU を強く要求する。ヘッドレスで smoke test → 実描画は Linux 機で。
- **`rosdep` が deps を解決できない:**
  コンテナ内で `sudo apt update && rosdep update` を再実行。
- **`upstream/*` が空:**
  ホスト側で `git submodule update --init --recursive` 実行後にコンテナ起動。
