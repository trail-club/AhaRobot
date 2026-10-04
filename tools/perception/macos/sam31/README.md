# DGX SparkでのSAM 3.1対話検証

MacにRealSenseを接続し、リポジトリ直下で実行する。

```bash
bash tools/perception/macos/sam31/interactive.sh
```

## 準備

- Mac: uv（0.7.11以上）とRealSense SDKのビルド環境。
  Python 3.11の依存・SDKは初回起動時に準備する。
- 接続: Cloudflare One（WARP）をConnectedにし、
  [directory-access](https://github.com/trail-club/directory-access) の `roster.yml` に
  本人のUnixユーザー名・公開鍵を登録する。
- DGX: GPU対応Dockerと共有重み
  `/srv/shared/models/sam3.1/sam3.1_multiplex.pt` が読み取り可能であること。
- カメラ配信やRealSense Viewerを終了し、カメラを空けておく。

起動時にUnixユーザー名を入力すると `10.99.0.1:2222` へ接続する。ユーザー名は保存しない。
標準のSSH鍵またはssh-agentを使い、パスフレーズ付きの鍵は事前に `ssh-add` で読み込む。
初回のSSHホスト鍵は登録し、登録済みの鍵が変わっている場合は接続を拒否する。

## 操作

1. Unixユーザー名を入力する。
   `このホストのパスワード（RealSense撮影用）` と表示されたらMacの管理者認証を行う。
2. `Ready` の後、`bottle` / `cup` / `person` など短い英語の名詞句を入力してEnterを押す。
3. RGB・位置合わせ済みDepthを撮影し、DGXで推論する。
   回収した `overlay.png` が開き、物体数・推論時間・保存先を表示する。
4. 空のEnterで同じプロンプトを再撮影し、別の入力で対象を変更する。
5. Ctrl+Cで終了する。テスト専用コンテナと転送した一時コードを削除し、結果・イメージ・キャッシュを保持する。

モデルは起動時に一度だけ読み込む。共有重みは読み取り専用で使い、HF認証は不要。
撮影・推論エラーは表示して再入力できる。SSH切断・応答タイムアウトでは終了処理へ進む。
停止を確認できない場合は専用コンテナを削除するコマンドを表示する。

主なオプションは以下。全項目は `bash tools/perception/macos/sam31/interactive.sh --help` を参照。

| オプション | 用途 |
| --- | --- |
| `--user <Unixユーザー名>` | 起動時のユーザー名入力を省略 |
| `--host <SSH接続先>` | SSH configの別名などを使用。ユーザー・ポート・鍵もconfigに従う |
| `--identity-file <秘密鍵パス>` | 標準以外の鍵を指定 |
| `--checkpoint <DGX上の絶対パス>` | 別の取得済み重みを使用 |
| `--threshold 0.5` | 検出スコアの閾値（既定0.5） |
| `--serial <camera serial>` | 撮影するRealSenseを指定 |
| `--output <保存先>` | 結果の保存先を変更 |
| `--no-open` | 画像を自動で開かない |

`SAM31_SSH_HOST` でも接続先を指定できる。既定の撮影は640×480・15 FPS、30フレーム待機。

## 保存結果

`.cache-sam31/results/interactive/<実行日時-ID>/<連番>/` に保存する（Git対象外）。
モデルの診断ログは実行ディレクトリの `dgx.log` に残る。

| ファイル | 内容 |
| --- | --- |
| `capture/rgbd.npz` | RGB（uint8）、位置合わせ済みZ深度（float32、m、無効値NaN）、メタデータ |
| `capture/rgb.png` / `capture/capture.json` | 撮影画像と時刻・内部パラメータ・深度スケール・フレーム番号 |
| `result/overlay.png` / `result/rgb.png` | マスクの重ね合わせと入力画像 |
| `result/labels.png` | 16 bitラベル。0=背景、1..N=その画像内の物体 |
| `result/masks.npz` | 元マスク・スコア、重なりを解決したlabels、入力depth_m |
| `result/result.json` | prompt、閾値、推論時間、入力情報、物体ごとのbbox・画素数・有効深度画素数・Z深度中央値 |

重なりはスコアが高い物体を優先する。距離は可視マスク内の有効Z深度の中央値で、
3D位置ではない。有効深度がなければ `null`。未検出時は背景ラベルと空のobjectsを保存する。
画像ごとに独立して処理し、物体の追跡・永続IDは提供しない。
ローカル重みのrevisionは自動検証しないため、`model_revision` は `null` となる。
実機の観測・重みの確認記録は[カメラ検証記録](../../../../docs/context/perception.md)を参照する。

## 実装とテスト

モデル処理・結果保存は `overlay_ws/src/aha_perception/aha_perception/`、
撮影は `tools/perception/macos/sam31/capture_realsense.py` に置く。
公式モデルコードは `upstream/sam3` のsubmoduleを使う。
固定したコード・重みと利用条件は[出典一覧](../../../../dependencies/THIRD_PARTY.md#sam-31)を参照。

DGXへ現在のコードとロックファイルを転送し、NVIDIA PyTorch `26.07-py3` をベースに
Dockerイメージを構築する。Python 3.12の依存は `pyproject.toml` / `uv.lock` で管理し、
uv 0.12.23で導入する。PyTorch / torchvision / Tritonはベースイメージのものを使う。
CUDAを必須とし、FlashAttention 3とcompileは無効。

単体テストはカメラ・SSH・CUDA・重みを使わずにホストで実行する。

```bash
bash tools/perception/macos/test.sh
```

ROSビルドを含む標準チェックは `make test`。
SAM側の依存を変更する場合はuv 0.12.23以上で
`uv add --project tools/perception/macos/sam31 --no-sync <package>` を実行し、
`pyproject.toml` と `uv.lock` を一緒に更新する。次回起動時にイメージを再構築する。
