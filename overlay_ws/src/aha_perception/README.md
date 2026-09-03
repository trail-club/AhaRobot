# aha_perception

RGB-D カメラからの物体検出・トラッキング・人物認識を担当する。

## ディレクトリ

- `launch/` — `camera.launch.py`, `detection.launch.py`
- `config/` — モデル/しきい値の YAML
- `rviz/` — `perception.rviz`

## 契約

- **subscribe**: `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`
- **publish**: `/aha/perception/objects` (aha_msgs/DetectedObjectArray) — `header.frame_id = camera_optical_frame`
- **QoS**: sensor_data (BEST_EFFORT / KEEP_LAST 5)
- 詳細は [`docs/interfaces.md`](../../../docs/interfaces.md)

## TODO

- [ ] Gazebo RGB-D センサの launch（sim との突合）
- [ ] YOLO 系 detector node 実装
- [ ] 人物追跡 (`/aha/perception/people`)
