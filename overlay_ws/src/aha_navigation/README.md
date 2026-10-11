# aha_navigation

navigation用のROSパッケージ。

`launch/nav.launch.py`（`sim.launch.py use_nav:=true`）はシミュレーション専用の2D LiDAR
（`aha_description/urdf/lidar.xacro`、frame `laser_link`）の `/scan` をbridgeし、
`launch/slam.launch.py` でslam_toolbox（online async、`config/slam_toolbox.yaml`）を起動する。
`/map` と `map → odom` を配信する。Nav2は起動しない。

## 固定地図での移動（シミュレーションのみ）

`launch/localized_nav.launch.py` は `/scan` のbridge、map_server、AMCL、Nav2を起動する。
SLAMとは同時に起動しない。[Human Interactionタスク](../aha_tasks/README.md)から利用する。
速度指令はJazzyの `enable_stamped_cmd_vel` を有効にし、既存の
`/diff_drive_controller/cmd_vel`（TwistStamped）へ直接接続する。
odomは `/diff_drive_controller/odom`、robot baseは `base_footprint`。
AMCLの初期姿勢は標準RCJO2026のspawnと同じ `(-2.0, 1.5, yaw=0.0)`。
設定は [nav2.yaml](config/nav2.yaml)。速度は0.20 m/s、costmapのrobot radiusは0.40 m。
実機用の設定ではない。
Gazebo GUIのソフトウェア描画時にもactionの応答を待てるよう、BT内のserver応答待ちは1000 msに設定する。
シミュレーションの更新遅延に備え、AMCLのTF有効期間とcontrollerの一時的な失敗の許容時間は2秒にする。

### デモ地図

[rcjo2026.yaml](maps/rcjo2026.yaml) はworld座標と整合する0.05 m/pixelの占有地図。
RCJO2026の4部屋を移動対象とし、場外はunknownにする。plannerはunknownを通行しない。
`map:=/absolute/path/map.yaml` で別の保存済み地図を指定できる。初期姿勢・地点座標も同じ座標系に合わせる。

[rcjo2026_geometry.json](maps/rcjo2026_geometry.json)に、上流の `rcjo2025_frame/model.sdf` の壁、
`rcjo2026_arena.world.xacro` のドア・家具配置、机・椅子・棚・かご・ゴミ箱のcollision形状と、ソファ・ベッド・洗濯機のcollision meshを覆う矩形を記録している。
椅子は傾いた脚と背もたれ、洗濯機はドアも含めた領域を占有させる。
床以外の家具全体を保守的な占有領域に投影したデモ用地図で、SLAMで取得した測量地図ではない。
AMCLはLiDARで位置を補正し、costmapのobstacle layerも `/scan` を使用する。
上流world・家具配置を変えた場合は外接矩形と地点を更新して再生成する。
上流資源の出典・ライセンスは [THIRD_PARTY](../../../dependencies/THIRD_PARTY.md)を参照。

```bash
# リポジトリ直下。標準Pythonのみで再生成できる。
python3 overlay_ws/src/aha_navigation/scripts/generate_demo_map.py
```
