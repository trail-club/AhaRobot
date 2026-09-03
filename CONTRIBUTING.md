# Contributing to AhaRobot

## 前提

- 開発は必ず `docker/` の dev コンテナ内で行う（[`docs/docker.md`](docs/docker.md)）
- ROS 2 Jazzy / Gazebo Harmonic / Ubuntu 24.04 を想定
- 変更は必ず PR。`main` への直 push は禁止

## Branch 戦略

```
feature/<squad>-<short-desc>      例: feature/nav-slam-toolbox-init
fix/<squad>-<short-desc>          例: fix/manip-srdf-joint-name
docs/<short-desc>                 例: docs/interfaces-update
```

`<squad>` = `nav` | `manip` | `perception` | `bringup` | `desc` | `infra`

## パッケージ担当

| ディレクトリ | 担当班 |
|---|---|
| `overlay_ws/src/aha_navigation/` | navigation |
| `overlay_ws/src/aha_manipulation/` | manipulation |
| `overlay_ws/src/aha_perception/` | perception |
| `overlay_ws/src/aha_msgs/` | 全班（追加は cross-review 必須） |
| `overlay_ws/src/aha_description/` | infra リード |
| `overlay_ws/src/aha_bringup/` `aha_gazebo/` | infra リード |
| `upstream/*` | 触らない（fork 側で PR） |

## Commit メッセージ

[Conventional Commits](https://www.conventionalcommits.org/) を推奨:

```
<type>(<scope>): <subject>

<body 任意>
```

- `type`: `feat` / `fix` / `docs` / `refactor` / `test` / `chore`
- `scope`: パッケージ名 or 班名（`aha_navigation`, `nav`, `docker` 等）
- 例: `feat(aha_navigation): add slam_toolbox online_async config`

## PR ルール

- 1 PR = 1 目的。跨り物は 2 PR に分ける
- 破壊的変更（topic 名 / msg フィールド / frame 名）は必ず [`docs/interfaces.md`](docs/interfaces.md) 更新 PR を先にマージ
- レビュワーは該当班の誰か 1 名以上
- Squash merge を基本とする

## Interface 変更

`aha_msgs/` の msg/srv/action を追加・変更する場合:
1. `docs/interfaces.md` を先に更新
2. 影響班（producer + consumer 両方）の approve
3. `aha_msgs/` の実装 PR
4. 各班の追随 PR

## ローカル動作確認

PR 前に最低限:

```bash
# コンテナ内で
cd /workspace/overlay_ws
colcon build --symlink-install
colcon test --event-handlers console_direct+
source install/setup.bash

# 変更したパッケージの launch が最低限 die しないこと
ros2 launch aha_bringup sim.launch.py    # など
```

## Issue / 議論

- バグ・機能提案は GitHub Issue（テンプレあり）
- 設計相談は Issue に `type:design` ラベル
- 質問だけなら Discussions（作成予定）
