# AhaRobot

RoboCup@Home（OPL）向けのROS 2ソフトウェアスタック。
開発環境はUbuntu 24.04 / ROS 2 Jazzy / Gazebo Harmonic。

## 構成

| パス | 内容 |
| --- | --- |
| `overlay_ws/` | AhaRobot独自のROSパッケージ |
| `upstream/` | Astraコード・ファームウェア・CAD、SOBITS / TMC資源、SAM 3.1のsubmodule |
| `docker/` | 開発コンテナ |
| `tools/` | 知覚・サーボ・ファームウェア・ローカルLLMのツール |
| `docs/` | 開発・起動手順とインターフェース |
| `docs/context/` | 実機実験・CAD・検証結果の補助資料 |

## 起動

起動・GUI接続・操作は [開発コンテナの起動フロー](docs/docker.md#起動フロー)、
worldと初期位置の変更は
[Japan Open起動設定](docs/sobits-rcjo2026.md)を参照。

## ドキュメント

- [開発ルール](AGENTS.md)
- [ROSパッケージ・制限](overlay_ws/README.md)
- [ROSインターフェース](docs/interfaces.md)
- [昇降機構の配線資料](docs/electrical/lift/README.md) — WireViz YAML・配線図・機器配置図・元PDF
- [両腕キーボード教示](docs/arm-keyboard-teleop.md) — 実機の両腕を `arm_node` 経由でキー操作する
- [submoduleの更新](docs/upstream-workflow.md)
- [ライセンス・出典](dependencies/THIRD_PARTY.md)
- [補助資料](docs/context/README.md) — 実機実験・CAD・検証結果
