# aha_perception

RealSenseのRGB-D表示とSAM 3.1のセグメンテーション検証。

- [Macカメラツール](../../../tools/perception/macos/pointcloud/README.md): RGB-DをROSへ配信し、RVizで点群を表示する。
- [SAM対話検証ツール](../../../tools/perception/macos/sam31/README.md): Macで撮影し、DGX Sparkで推論して結果を保存する。

`aha_perception.sam31` が単一フレームのモデル推論・マスク処理・深度集計を行い、
`aha_perception.sam31_worker` が対話検証用の常駐ワーカーを提供する。
SAM検証はROS topicを使わず、カメラ表示launchとは独立して動作する。
