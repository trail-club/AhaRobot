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
実機の観測結果は[カメラ検証記録](../../../../docs/context/perception.md)を参照する。
