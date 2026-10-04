# macOS知覚検証ツール

| ディレクトリ | 検証内容 |
| --- | --- |
| [pointcloud/](pointcloud/README.md) | RGB-DをROSへ配信し、RVizで点群変換を確認 |
| [sam31/](sam31/README.md) | RealSenseで撮影し、DGX SparkのSAM 3.1で対話推論 |

両ツールはPython環境とRealSense SDKを共用する。カメラを同時には使用しない。
SDKの準備は `bootstrap_realsense.py` が行い、初回起動時に専用venvへビルドする。
既存バインディングがSDKのdylibを読み込めない場合も再ビルドする。

## 依存環境とテスト

ホストにuv（0.7.11以上）が必要。`pyproject.toml` と `uv.lock` で依存を管理する。

```bash
bash tools/perception/macos/setup.sh
bash tools/perception/macos/test.sh
```

setup・起動・テストは同じ `.venv-perception-macos` を使い、`uv sync --locked` で
ロックした依存を導入する。`--inexact` で別途ビルドしたRealSenseバインディングを保持する。
テストは `uv run` で実行し、カメラ・CUDA・ROSを必要としない。
環境の作成にはPython 3.11を使う。`PERCEPTION_BASE_PYTHON` / `PERCEPTION_PYTHON` で
そのインタープリタを指定でき、`UV_PROJECT_ENVIRONMENT` でvenvの保存先を変更できる。

依存を追加するときは `uv add --project tools/perception/macos --no-sync <package>` を使い、
更新した `pyproject.toml` と `uv.lock` を一緒に管理する。
追加後はsetupまたはテストを実行して専用venvへ同期する。
ROS側のビルド・テストを含めた確認はホストから `make test` を実行する。

実機の観測結果は[カメラ検証記録](../../../docs/context/perception.md)を参照する。
