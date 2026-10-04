# aha_navigation

navigation用のROSパッケージ。

`launch/nav.launch.py`（`sim.launch.py use_nav:=true`）はシミュレーション専用の2D LiDAR
（`aha_description/urdf/lidar.xacro`、frame `laser_link`）の `/scan` をbridgeし、
`launch/slam.launch.py` でslam_toolbox（online async、`config/slam_toolbox.yaml`）を起動する。
`/map` と `map → odom` を配信する。Nav2は起動しない。
