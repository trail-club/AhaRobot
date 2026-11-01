# aha_navigation

navigation用のROSパッケージ。

`launch/nav.launch.py`（`sim.launch.py use_nav:=true`）はシミュレーション専用の2D LiDAR
（`aha_description/urdf/lidar.xacro`、frame `laser_link`）の `/scan` をbridgeし、
`launch/slam.launch.py` でslam_toolbox（online async、`config/slam_toolbox.yaml`）を起動する。
`/map` と `map → odom` を配信する。Nav2は起動しない。

## 固定地図での移動（シミュレーションのみ）

`launch/localized_nav.launch.py` は `/scan` のbridge、map_server、AMCL、Nav2を起動する。
SLAMと同時に起動しない。設定は [nav2.yaml](config/nav2.yaml)（0.20 m/s、robot radius 0.40 m）で、実機用ではない。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `map` | [rcjo2026.yaml](maps/rcjo2026.yaml) | 地図のパス。地点・初期姿勢も同じ座標系に合わせる |
| `initial_pose` | nav2.yamlの値 | AMCLの初期姿勢 `x,y,yaw` |

ソフトウェア描画のGazeboでも動くよう、BTのserver応答待ちは1000 ms、AMCLのTF有効期間とcontrollerの失敗許容時間は2秒。

### デモ地図

`rcjo2026.yaml` はworld座標に合わせた0.05 m/pixelの地図で、4部屋の外はunknown（plannerは通行しない）。
SLAMの測量地図ではなく、[rcjo2026_geometry.json](maps/rcjo2026_geometry.json) に記録した上流worldの壁・ドア・家具のcollisionを覆う矩形を占有領域にしたもの。
上流のworld・家具配置を変えたら矩形を更新して再生成する。出典は [THIRD_PARTY](../../../dependencies/THIRD_PARTY.md)。

```bash
# リポジトリ直下
python3 overlay_ws/src/aha_navigation/scripts/generate_demo_map.py
```
