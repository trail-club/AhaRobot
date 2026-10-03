# SOBITS Japan Open 2026

標準シミュレーションはSOBITSのJapan Open 2026 worldを使う。
対象環境はROS 2 Jazzy / Gazebo Harmonic。

## ビルド・起動

`git clone --recursive` でworldとモデルのsubmoduleも取得される。
既存cloneではリポジトリ直下で次を実行する。

```bash
git submodule update --init --recursive
```

開発コンテナ内で通常のビルドと起動を行う。

```bash
cd /app/overlay_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch aha_bringup sim.launch.py
# 別ターミナル（同じsetup）で操作
ros2 run aha_bringup teleop_base.sh
```

## 起動設定

初期位置は `(-2.0, 1.5, 0.10)` m、yawは `0.0` rad。
`spawn_x` / `spawn_y` / `spawn_z` / `spawn_yaw` で変更できる。

- `headless:=true`：GUIなし
- `world:=empty.sdf`：空world、初期位置 `(0.0, 0.0, 0.05)` m
- `world_path:=/absolute/path/world.sdf`：展開済みSDFで上書き、空worldと同じ既定初期位置

worldは `sobits_gazebo_worlds`、床・机のモデルは `tmc_wrs_gz_worlds` を使う。
専用ビルドは不要。Xacroは起動時にSDFへ展開し、[preset](../overlay_ws/src/aha_gazebo/config/rcjo2026.json)のチェックサムと照合する。

2Dマップ・SLAM・Nav2は未追加。ビルド・回帰テスト・SDF検証済みで、実走行は未検証。
[ライセンス・第三者資源の出典](../dependencies/THIRD_PARTY.md)
