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
2. カメラが起動すると、ライブプレビュー（`http://127.0.0.1:8765/`）がブラウザで開く。
   左にカラー映像、右に位置合わせ済みDepthを表示し、上部に撮影・推論の状態を表示する。
   推論が終わるたびに、直前の検出結果（マスクを重ねた `overlay.png`）を映像の下に表示する。
   モデルの読み込み中も画角を確認できる。
3. `Ready` の後、`bottle` / `cup` / `person` など短い英語の名詞句を入力してEnterを押す。
4. Enterの直後のフレーム（RGB・位置合わせ済みDepth）を保存し、DGXで推論する。
   回収した `overlay.png` が開き、物体数・推論時間・保存先を表示する。
5. 空のEnterで同じプロンプトを再撮影し、別の入力で対象を変更する。
6. Ctrl+Cで終了する。カメラを解放し、テスト専用コンテナと転送した一時コードを削除し、結果・イメージ・キャッシュを保持する。

カメラはセッション中ずっと開いたままにする（撮影ごとに開き直さない）。
root権限で動くのはカメラのプロセスだけで、プレビューのHTTPサーバーはループバックのみで待ち受ける。
カメラのプロセスが停止した場合は撮影エラーとなるため、Ctrl+Cで終了して起動し直す。

モデルは起動時に一度だけ読み込む。共有重みは読み取り専用で使い、HF認証は不要。
撮影・推論・結果受信のエラーは表示して再入力できる。SSH切断・応答タイムアウトでは終了処理へ進む。
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
| `--no-preview` | ライブプレビューを使わず、撮影ごとにカメラを開く |
| `--preview-port 8765` | ライブプレビューのポート |
| `--vlm` | 商品名・ブランド名向け。DGXのVLMが候補マスクを選ぶ（[VLMモード](#vlmモード)） |
| `--vlm-model qwen3-vl:30b` / `--vlm-url http://127.0.0.1:11434` | VLMモードで使うDGX上のOllamaモデルと接続先 |
| `--candidate-threshold 0.3` | VLMモードでSAMが候補を出す閾値（既定0.3） |
| `--no-vlm-match` | VLMモードの照合（手順4）を省き、速度を優先する |

`SAM31_SSH_HOST` でも接続先を指定できる。既定の撮影は640×480・15 FPS、30フレーム待機。

## VLMモード

`--vlm` を付けると、SAM 3.1が見た目で拾った候補から、DGXのVLM（既定はOllamaの `qwen3-vl:30b`）が
対象を選ぶ。`pringles` や `じゃがりこ` のように、見た目の近い別商品と区別が必要なプロンプトに使う。

1. VLMが画像とプロンプトを見て、SAM向けの一般的な英語句（例: `red snack can`）を最大3つ返す。
   英語のプロンプトはそのままの句も試す。
2. 各句でSAM 3.1を `--candidate-threshold` で実行し、重なる候補（IoU 0.8超）を除いて最大8個にまとめる。
3. 元の画像と番号付きの候補画像（`candidates.png`）をVLMへ渡し、対象の番号をすべて選ばせる。
   斜線でロゴや文字が隠れるため、パッケージは元の画像で確認させる。同じ商品が複数あれば複数選ぶ。
4. 照合: 後ろ向きでロゴが見えない同じ商品を拾うため、選ばれなかった候補を、選ばれた候補（最もスコアの高いもの）と
   切り出して横に並べ（`match_<番号>.png`）、同じ商品かをVLMに1つずつ聞く。
   切り出しは深度で実寸の比率にそろえ、深度から求めた高さ・幅（cm）も伝える。1候補あたり数秒かかる。
   何も選ばれなかった場合は照合しない。
5. 選択または照合で採用した候補だけを結果に残す。VLMの判断は `result.json` の `vlm` に記録する。

各フレームの後に、言い換え・句ごとのSAM検出・候補・VLMの選択と照合（回答と推論過程）・結果を
1ページにまとめた `flow.html` を作成し、`overlay.png` の代わりにブラウザで開く。

OllamaはDGXのループバックで待ち受けるため、VLMモードの専用コンテナは `--network host` で起動する。
VLMの応答が得られない・形式が不正な場合はそのフレームをエラーとし、再入力できる。
VLMは思考過程を出力するため、1フレームに数十秒かかる場合がある。

## 参照画像の撮影

商品ごとに、向きを変えた参照画像を撮影して保存する。

```bash
bash tools/perception/macos/sam31/reference.sh --user <Unixユーザー名> --name pringles
```

1. 対象の商品を1つだけ、無地の机の上で画面の中央に置く。ライブ映像のページで画角を確認する。
2. 表示された向き（`front` → `back` → `left` → `right` の順）に商品を回してEnterを押す。
   `top` など別の向きは名前を入力して撮影する。同じ向きを入力すると撮り直して上書きする。
3. DGXのSAM 3.1が種類の言葉で物を検出し、画面中央の物を参照として保存する。
   言葉は `--phrase` で指定する。省略すると、最初の1枚でDGXのQwenが中央の物を見て決める（約10秒）。
   決めた言葉は `phrase.json` に保存し、次回以降も使う。
   端で切れている・小さい・ほかの物が写っている・深度が取れない場合は注意を表示する。
4. Ctrl+Cで終了する。Macのスリープを防ぐため `caffeinate -i` で実行する。

`.cache-sam31/references/<name>/` に保存する（Git対象外）。`index.html` で全ての向きを確認できる。

| ファイル | 内容 |
| --- | --- |
| `<向き>/rgb.png`・`rgbd.npz` | 撮影したRGBと位置合わせ済みDepth |
| `<向き>/mask.png`・`crop.png` | 参照に選んだ物のマスクと切り出し |
| `<向き>/preview.png` | 参照の位置を太い枠で示した画像 |
| `<向き>/view.json` | bbox、スコア、深度から求めた高さ・幅（cm）、注意 |
| `reference.json` | 全ての向きの `view.json` をまとめたもの |
| `phrase.json` | SAM 3.1に渡す種類の言葉と、決めた方法（`vlm`：Qwen、`user`：`--phrase`） |

参照画像を使った検出は [sam31_dinov2](../sam31_dinov2/README.md) を参照する。

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
| `result/candidates.png` | VLMモードのみ。VLMへ渡した番号付きの候補 |
| `result/match_<番号>.png` | VLMモードのみ。照合でVLMへ渡した切り出し（左が採用済みの候補、右が照合した候補） |
| `result/match_<番号>_boxes.png` | VLMモードのみ。照合した2つの候補（A・B）の位置を写真上に太い枠で示した図（Mac側で作成） |
| `result/flow.html` | VLMモードのみ。検出までの各段階をまとめたページ（Mac側で作成） |
| `result/result.json` | prompt、閾値、推論時間、入力情報、物体ごとのbbox・画素数・有効深度画素数・Z深度中央値。物体ごとのマスク色 `color`。VLMモードでは `vlm`（使った句、句ごとのSAM検出と扱い、候補ごとの句・スコア・採否・採用の経路・色、照合の結果、VLMの回答・推論過程・所要時間） |

重なりはスコアが高い物体を優先する。距離は可視マスク内の有効Z深度の中央値で、
3D位置ではない。有効深度がなければ `null`。未検出時は背景ラベルと空のobjectsを保存する。
画像ごとに独立して処理し、物体の追跡・永続IDは提供しない。
ローカル重みのrevisionは自動検証しないため、`model_revision` は `null` となる。

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
