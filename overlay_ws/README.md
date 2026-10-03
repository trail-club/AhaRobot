# overlay_ws

AhaRobot独自のROSパッケージ。[開発コンテナ](../docs/docker.md)が必要なsubmoduleを
`src/` へリンクし、同じworkspaceでビルドする。

## パッケージ

| package | 内容 |
| --- | --- |
| `aha_description` | URDF xacro、差動二輪ベース、ros2_control設定 |
| `aha_gazebo` | Gazebo起動、SOBITS worldの展開・検証 |
| `aha_bringup` | ロボット生成とcontrollerの起動 |
| `aha_sobits_bringup` | Japan Openシミュレーションの互換起動入口 |
| `aha_msgs` | msg / srv / action定義 |
| `aha_navigation` | navigation用パッケージ。launchはスタブ |
| `aha_manipulation` | manipulation用パッケージ。launchはスタブ |
| `aha_perception` | 知覚用パッケージ。実行手順は[README](src/aha_perception/README.md)を参照 |

## ビルド・起動

コマンドは [開発コンテナの起動フロー](../docs/docker.md#起動フロー)を参照。

world・spawn設定は [Japan Open起動設定](../docs/sobits-rcjo2026.md)、
controllerとROSの入出力は [インターフェース](../docs/interfaces.md)を参照。

## 制限

- 車輪・キャスターの一部寸法は推定値。[寸法調査の根拠と限界](../docs/context/cad.md)
- 昇降は左右2軸として記述しているが、実機は共通軸。
- センサ、SLAM / Nav2、MoveItの統合は未実装。
- 実機用Hardware Interfaceは未実装で、`sim:=false` による実機制御はできない。
