# AhaRobot Simulation & Competition Stack

RoboCup@Home (OPL) 向け AhaRobot ソフトウェアスタック。
基盤: **ROS 2 Jazzy + Ubuntu 24.04 + Gazebo Harmonic**。

## リポジトリ構成

```
.
├── upstream/           # 公式 AhaRobot 由来コード (trail-club fork を submodule)
│   ├── astra_ws           # ROS 2 workspace (Humble)
│   ├── astra_description  # URDF / meshes / RViz / Gazebo launch
│   ├── AstraFirmwares     # ESP32 / ODrive ファームウェア
│   └── Astra_Hardwares    # CAD (STEP / STL)
└── (overlay_ws/)       # チーム overlay workspace (Phase 1 以降で追加)
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

`upstream/astra_ws` は GPL-3.0 に加え非商用条項が README に記載されている。
派生物の公開・配布時は各 upstream の LICENSE を確認すること。

## ロードマップ (概要)

- Phase 0: 監査 (依存 / topic / joint / センサ / ライセンス整理)
- Phase 1: Digital Skeleton — Jazzy + Harmonic で spawn し全自由度を制御
- Phase 2: Navigation (SLAM Toolbox / Nav2)
- Phase 3: Manipulation (MoveIt 2)
- Phase 4: Competition Harness (GPSR / HRI / 評価)
