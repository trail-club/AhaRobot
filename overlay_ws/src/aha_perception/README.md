# aha_perception

RGB-D カメラからの物体検出・トラッキング・人物認識と、音声（ASR / TTS）を担当する。

## ディレクトリ

- `launch/` — `camera.launch.py`, `detection.launch.py`, `audio.launch.py`
- `config/` — モデル/しきい値の YAML（音声は `config/audio/`）
- `rviz/` — `perception.rviz`
- `aha_perception/audio/`, `scripts/audio_*` — 音声スタック（別コンテナ `aha_audio` で動かす。詳細は [`docs/audio.md`](docs/audio.md)）

## 契約

- **subscribe**: `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`
- **publish**: `/aha/perception/objects` (aha_msgs/DetectedObjectArray) — `header.frame_id = camera_optical_frame`
- **QoS**: sensor_data (BEST_EFFORT / KEEP_LAST 5)
- 詳細は [`docs/interfaces.md`](../../../docs/interfaces.md)

## TODO

- [ ] Gazebo RGB-D センサの launch（sim との突合）
- [ ] YOLO 系 detector node 実装
- [ ] 人物追跡 (`/aha/perception/people`)
