# macOSでのRGB-D点群変換検証

MacにRealSenseを接続し、リポジトリ直下で実行する。

```bash
bash tools/perception/macos/pointcloud/start.sh
```

RGB・位置合わせ済みDepth・CameraInfoをrosbridge経由でROSコンテナへ送り、
`depth_image_proc` で点群に変換してRVizで表示する。
起動後に [NoVNC](http://localhost:8080/vnc.html) を開いて **Connect** を押す。
Ctrl+Cで配信と検証用コンテナを終了する。

MacにはDocker Desktopと[共通の撮影環境](../README.md#依存環境とテスト)が必要。
カメラ配信・RealSense Viewer・SAM検証を終了し、カメラを空けておく。
USBアクセスに管理者認証が必要な場合は、撮影プロセスだけsudoで起動する。

カメラなしで変換・表示を確認する場合は合成RGB-Dを使う。

```bash
bash tools/perception/macos/pointcloud/start.sh --demo
```

`--rebuild` で共通Jazzyイメージを再構築する。カメラ設定は `--` の後へ渡す。
起動オプションは `start.sh --help`、配信オプションは
`stream_realsense.py --help` を参照する。

点群変換の設定は[変換launch](../../../../overlay_ws/src/aha_perception/launch/camera_view.launch.py)、
実機の観測結果は[カメラ検証記録](../../../../docs/context/camera.md)を参照する。

## rosbridgeへの直接送信

rosbridgeを別に起動している場合（例: `real_head_camera.launch.py camera:=true`）は、`stream_realsense.py` だけを実行する。
共通の撮影環境では、`start.sh` の初回起動でビルドしたRealSense SDKのライブラリを指定する。
pyrealsense2と [pyproject.toml](../pyproject.toml) の依存が入った他のPythonでも実行できる。

```bash
venv=.venv-perception-macos
export DYLD_LIBRARY_PATH="$venv/realsense-sdk/lib"
"$venv/bin/python" tools/perception/macos/pointcloud/stream_realsense.py --list-devices
"$venv/bin/python" tools/perception/macos/pointcloud/stream_realsense.py
```

既定は送信先 `ws://127.0.0.1:9090`、640×480、撮影15 fps、送信上限5回/s。
`--ws-url`・`--width` / `--height`・`--fps`・`--rate` で変更する。

Depthには、RGBへの位置合わせの前にlibrealsenseの後処理
（disparity変換 → spatial → temporal → depthへ戻す、オプションはSDKの既定値）を全フレームにかける。
`--no-filter` で後処理なしのDepthを送る。有効な後処理は起動時のログに出る。
