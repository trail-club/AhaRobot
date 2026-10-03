# SOBITS / TMC資源の出典

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
