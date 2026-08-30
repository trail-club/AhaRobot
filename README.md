# AhaRobot Simulation & Competition Stack

RoboCup@Home (OPL) 向け AhaRobot ソフトウェアスタック。
基盤: **ROS 2 Jazzy + Ubuntu 24.04 + Gazebo Harmonic**。

## リポジトリ構成

```
.
├── upstream/                         # 公式由来コード (trail-club fork を submodule)
│   ├── astra_description                 # URDF / meshes / RViz / Gazebo launch
│   ├── astra_controller                  # 実機制御 ROS 2 node
│   ├── astra_controller_interfaces       # custom msg / srv
│   ├── astra_moveit_config               # MoveIt 2 設定 (Humble, deprecated 予定)
│   ├── AstraFirmwares                    # ESP32 / ODrive ファームウェア
│   └── Astra_Hardwares                   # CAD (STEP / STL)
└── (overlay_ws/)                     # チーム overlay workspace (Phase 1 以降で追加)
```

`upstream/*` は trail-club org の fork を submodule として固定。
改変は fork 側 branch で行い、super repo は SHA を進めるだけとする。
Upstream 追従は各 fork で `git remote add upstream https://github.com/hilookas/<repo>.git` → `git fetch upstream`。

## セットアップ

```bash
git clone --recursive git@github.com:trail-club/AhaRobot.git
# 既に clone 済みなら
git submodule update --init --recursive
```

## ライセンス

元の `hilookas/astra_ws` README は GPL-3.0 に加え非商用条項を記載している。
派生物の公開・配布時は各 upstream の LICENSE を確認すること。

なお `hilookas/astra_ws` 本体 (super-repo) は本リポジトリでは submodule として保持せず、
Phase 1〜3 で必要な package のみを個別 fork として `upstream/` 直下に並べる方針。
Phase 4 以降で teleop / lerobot / WebRTC 系 (`astra_teleop*`, `lerobot`, `aiortc` 等) が
必要になった時点で fork を追加する。

## ドキュメント

- [Phase 0 監査結果](docs/phase0-audit.md) — package / URDF / LICENSE の現状と Phase 1 リスク
- [Upstream 追従ワークフロー](docs/upstream-workflow.md) — fork の同期手順

## ロードマップ (概要)

- Phase 0: 監査 (依存 / topic / joint / センサ / ライセンス整理)
- Phase 1: Digital Skeleton — Jazzy + Harmonic で spawn し全自由度を制御
- Phase 2: Navigation (SLAM Toolbox / Nav2)
- Phase 3: Manipulation (MoveIt 2)
- Phase 4: Competition Harness (GPSR / HRI / 評価)
