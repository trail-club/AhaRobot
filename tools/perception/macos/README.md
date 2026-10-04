# macOS 点群変換テスト

リポジトリのルートで次の1コマンドを実行する。

```bash
bash tools/perception/macos/start.sh
```

## rosbridgeへの直接送信

rosbridgeを別に起動している場合（例: `real_head_camera.launch.py camera:=true`）は、
pyrealsense2と [requirements.txt](requirements.txt) の依存が入ったPythonで `stream_realsense.py` だけを実行する。

```bash
python3 tools/perception/macos/stream_realsense.py --list-devices
python3 tools/perception/macos/stream_realsense.py
```

既定は送信先 `ws://127.0.0.1:9090`、640×480、撮影15 fps、送信上限5回/s。
`--ws-url`・`--width` / `--height`・`--fps`・`--rate` で変更する。
