# ライセンス・第三者資源の出典

## AhaRobot独自コード

`overlay_ws/src/aha_*` の9パッケージは、それぞれの `package.xml` に `Apache-2.0` を宣言している。
例: [aha_description/package.xml](../overlay_ws/src/aha_description/package.xml)、[aha_gazebo/package.xml](../overlay_ws/src/aha_gazebo/package.xml)。
リポジトリ直下にはLICENSEファイルがなく、`tools/`・`docker/`などを含む全体のライセンスは未整備。
上流から取り込んだコード・URDF・mesh・素材への適用範囲は、独自コードの宣言とは別に確認する。
パッケージ内に置いたAstra由来の派生資源は[下記](#astra由来の派生資源)の条件で扱う。

## Astra由来のsubmodule

使用先は `trail-club` のfork、元の開発元は同名の `hilookas` リポジトリ。

| submodule / 使用commit | 確認した記載 | 状態・参照元 |
| --- | --- | --- |
| [astra_description](https://github.com/trail-club/astra_description/tree/a2791e61daaaff227e575b06120424439fe89e2e) | `package.xml`: `BSD` | 直下にLICENSEなし。BSDの種類・適用範囲は未確認。[package.xml](../upstream/astra_description/package.xml) |
| [astra_controller](https://github.com/trail-club/astra_controller/tree/9c1b96e7684f32e90d72805d9e3c982ac38a5de6) | `package.xml`: `TODO: License declaration` | 直下にLICENSEなし。ライセンス未宣言。[package.xml](../upstream/astra_controller/package.xml) |
| [astra_controller_interfaces](https://github.com/trail-club/astra_controller_interfaces/tree/8f6f896ea8b90ff358be2937ff65306f578b8074) | `package.xml`: `TODO: License declaration` | 直下にLICENSEなし。ライセンス未宣言。[package.xml](../upstream/astra_controller_interfaces/package.xml) |
| [astra_moveit_config](https://github.com/trail-club/astra_moveit_config/tree/f053a291345495368cbb125ab400af858ce24533) | `package.xml`: `BSD` | 直下にLICENSEなし。BSDの種類・適用範囲は未確認。[package.xml](../upstream/astra_moveit_config/package.xml) |
| [AstraFirmwares](https://github.com/trail-club/AstraFirmwares/tree/1b4e7db36dea97623fb2735e272aada9c8f23b0f) | LICENSE: GPL v3本文。README: 非商用などの追加制限 | [LICENSE](../upstream/AstraFirmwares/LICENSE)、[README](../upstream/AstraFirmwares/README.md#license) |
| [Astra_Hardwares](https://github.com/trail-club/Astra_Hardwares/tree/27528e9831311ebd2a5b1e9a651940c1c18a08d3) | LICENSE: GPL v3本文。README: 非商用などの追加制限 | [LICENSE](../upstream/Astra_Hardwares/LICENSE)、[README](../upstream/Astra_Hardwares/README.md#license) |

元のsuper-repo [hilookas/astra_wsのREADME](https://github.com/hilookas/astra_ws#license)にもGPL-3.0と非商用の追加制限がある。
このsuper-repo自体はAhaRobotのsubmoduleには含めていない。
個別ROSパッケージへの適用範囲、およびGPL本文とREADMEの追加制限の扱いは未確認。
競技参加・展示・商用利用の可否をこの一覧だけで確定しない。

`AstraFirmwares` 内の各コントローラも[入れ子のsubmodule](../upstream/AstraFirmwares/.gitmodules)。
親リポジトリの宣言だけで各コントローラのライセンスを確定せず、個別の記載を確認する。

### Astra由来の派生資源

| 資源 | 由来 | 扱い |
| --- | --- | --- |
| [aha_perception/hardware/head_cam_mount_d435i](../overlay_ws/src/aha_perception/hardware/head_cam_mount_d435i/README.md) | `Astra_Hardwares` の `HeadCamMount`（[HeadCamMount.STL](../upstream/Astra_Hardwares/Astra/STL/HeadCamMount.STL)）のホーン接続部を作り直したD435i用ブラケット（STL・STEP・f3d・画像） | 上流と同じGPL-3.0と非商用の追加制限。`aha_perception` の `Apache-2.0` 宣言とは別に扱う |
| [aha_description/meshes/head_cam_mount_d435i.stl](../overlay_ws/src/aha_description/meshes/head_cam_mount_d435i.stl) | 上記ブラケットのURDF用mesh | 同上。`aha_description` の `Apache-2.0` 宣言とは別に扱う |

## SOBITS / TMCのシミュレーション資源

TeamSOBITSのリポジトリを直接参照する。Japan Open worldの形状は変更せず、AhaRobotの初期位置と資源参照を設定している。

| 資源 / 使用commit | 確認した記載 | 参照元 |
| --- | --- | --- |
| [sobits_gazebo_worlds](https://github.com/TeamSOBITS/sobits_gazebo_worlds/tree/a62f981651c357367b132fa412b09258554016c5) | BSD-3-Clause。個別assetの条件も確認する | [package.xml](../upstream/sobits_gazebo_worlds/package.xml)、[LICENSE](../upstream/sobits_gazebo_worlds/LICENSE) |
| [tmc_wrs_gz](https://github.com/TeamSOBITS/tmc_wrs_gz/tree/eeff2783fc8c6149129d6e5407341dee57846804)のソフトウェア | Clear BSD。Copyright © 2020 TOYOTA MOTOR CORPORATION | [LICENSE.txt](../upstream/tmc_wrs_gz/LICENSE.txt)、[README](../upstream/tmc_wrs_gz/README.md#ライセンス) |
| `wrc_ground_plane` / `wrc_long_table` / `wrc_tall_table` | CC BY 4.0。作者 Nobuyuki Matsuno | 各model.config: [床](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/models/wrc_ground_plane/model.config)、[長机](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/models/wrc_long_table/model.config)、[高机](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/models/wrc_tall_table/model.config) |
| TMCの `ycb_*` モデル | CC BY 4.0。出典 Yale-CMU-Berkeley Object and Model Set | [モデル別README](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/README.md) |
| TMCの `person_standing` モデル | CC BY 3.0。出典 OSRF gazebo_models | [モデル別README](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/README.md) |
| TMCの `trofast` / その他の `wrc_*` モデル | CC BY 4.0 | [モデル別README](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/README.md) |
| 床の木目テクスチャ | 出典 freestocktextures.com。添付文書にCreative Commons Zeroと記載 | about-photo.txt: [TMC](../upstream/tmc_wrs_gz/tmc_wrs_gz_worlds/models/wrc_ground_plane/materials/about-photo.txt)、[SOBITS](../upstream/sobits_gazebo_worlds/models/floor_plane/materials/about-photo.txt) |

TMCのソフトウェア向けライセンスと、モデル・テクスチャ向けライセンスは分けて扱う。
公開画像・動画などでCC BY素材を使う場合は、素材の作者・出典・ライセンスへのリンク・変更の有無を記載する。
条件の原文: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)、[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。
資源の再配布時は、各LICENSEの条件に従って著作権表示・ライセンス本文・免責事項などを保持する。

## その他の第三者資源

[tools/firmware/backup/](../tools/firmware/backup/) の `stock-waveshare-esp32-*.bin`、
`backup_0927/Aharobot_esp32_backup.bin` と
[esp32_backup/](../tools/firmware/esp32_backup/) の同名ファイルはWaveshare純正デモのフラッシュdump。
取得時の条件は [ファームウェアの検証記録](../docs/context/firmware.md)を参照。
収録バイナリのライセンス記載はなく、再配布条件は未確認。
