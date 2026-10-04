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
| `aha_perception` | 頭部カメラの点群・RViz表示・頭部の操作UI。実行手順は[README](src/aha_perception/README.md)を参照 |
| `aha_servo` | STSサーボのJointTrajectoryブリッジとキャリブレーション。[README](src/aha_servo/README.md)を参照 |

## ビルド・起動

コマンドは [開発コンテナの起動フロー](../docs/docker.md#起動フロー)を参照。

world・spawn設定は [Japan Open起動設定](../docs/sobits-rcjo2026.md)、
controllerとROSの入出力は [インターフェース](../docs/interfaces.md)を参照。

## テスト

PR前の標準チェックはホストで [make test](../AGENTS.md#動作確認) を実行する。
上流パッケージも含めて再検証する場合は、開発コンテナ内で以下を実行する。

```bash
cd /app/overlay_ws
colcon build --symlink-install --cmake-force-configure
source install/setup.bash
colcon test-result --delete-yes  # 前回の生成済みテスト結果をクリア
colcon test
colcon test-result --verbose
```

`colcon.meta` はこのworkspaceからの実行時に自動で読み込まれ、上流の既知の失敗を次の範囲で除外する。

- `sobits_gazebo_worlds`: `ament_cmake_flake8`
- `tmc_wrs_gz_worlds`: `test_flake8` / `test_copyright`

SOBITSはCMakeの再構成時に除外を反映するため、設定追加後は上記のビルドから実行する。
除外前の結果ファイルが集計に残らないよう、前回のテスト結果をクリアしてから再実行する。
TMCの著作権・ライセンス本文は変更せず、表記を認識できないチェックを除外する。
上流のその他のチェックと、`aha_gazebo` のworld・launch回帰テストは引き続き実行する。
`astra_controller_interfaces` のlintは除外対象に含めない。

## 制限

- 車輪・キャスターの一部寸法は推定値。[寸法調査の根拠と限界](../docs/context/cad.md)
- 昇降は左右2軸として記述しているが、実機は共通軸。
- Gazeboのセンサは頭部カメラのみ。SLAM / Nav2、MoveItの統合は未実装。
- 実機用Hardware Interfaceは未実装で、`sim:=false` による実機制御はできない。
  実機の頭部は `aha_servo` のブリッジで動かす。
