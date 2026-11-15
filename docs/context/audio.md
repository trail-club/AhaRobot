# 音声（ASR / TTS）の測定・検証記録

音声認識と発話のモデルを別のマシンで動かす方式の、実機での確認を日付順に記録する。
使い方は [音声スタック](../../overlay_ws/src/aha_perception/docs/audio.md)・[Macの対話ツール](../../tools/perception/macos/audio/README.md) を参照する。

## 2026-09-27 — `target:=local` での遅延（trailpc）

### 条件

- `target:=local`（ノードと同じプロセスの GPU）、RTX 5090 Laptop。
- 音声はループバック入力。日本語は JSUT の 3 文。
- このとき `ja` のモデルは Qwen3-ASR だった（今の `ja-qwen`）。
- 測定の回数とばらつきの求め方は記録が残っていない。

### 結果

| 項目 | 値 |
|---|---|
| 英語：話し終わり → 認識結果 | 約 325 ms（確定待ち 224 + Parakeet 33） |
| 日本語：話し終わり → 認識結果 | 460〜670 ms（確定待ち 224 + Qwen3-ASR 210〜410）。JSUT 3 文をすべて正しく認識 |
| TTS：goal → 最初の音 | 約 50 ms（Kokoro。4 秒の文を約 100 ms で合成） |
| Smart Turn / Silero VAD | 27 ms / 回、1 ms / フレーム（CPU） |
| VRAM | ASR ノード 8.5 GB（Parakeet＋Qwen3-ASR）、TTS ノード 1.0 GB |

### 確認していないこと

- 人の声とロボットのマイクからの入力。
- Haas での同じ測定。

## 2026-11-01 — DGX Spark と trailpc でのモデルの動作と速度

### 条件

- DGX Spark（aarch64、NVIDIA GB10）と trailpc（x86-64、RTX 5090 Laptop）。
- 同じ Dockerfile（`tools/perception/audio_worker/`）をそれぞれでビルドした。ベースは `nvcr.io/nvidia/pytorch:26.07-py3`、torch 2.13.0a0、PyPI の torchaudio 2.11.1。
- 音声は Kokoro の合成音声（3.05 秒）と macOS の `say`（英語 Samantha、日本語 Kyoko）。実際のマイクと人の声は使っていない。
- 往復時間は Mac から WARP と SSH を経由した値で、base64への変換と転送を含む。

### 結果

| 項目 | DGX Spark | trailpc |
| --- | --- | --- |
| Parakeet（GPU 上のみ） | 0.03 秒 | 0.02 秒 |
| Parakeet（Mac からの往復） | 0.12〜0.14 秒 | 0.06〜0.11 秒 |
| Qwen3-ASR（GPU 上のみ） | 0.28 秒 | 0.14〜0.21 秒 |
| Qwen3-ASR（Mac からの往復） | 0.35〜0.38 秒 | 0.18〜0.54 秒 |
| Kokoro（3.05 秒の音声、Mac からの往復） | 0.10〜0.13 秒 | 0.07〜0.11 秒 |
| モデル読み込み（重みの取得後） | Kokoro 2 秒、Parakeet 7 秒、Qwen3-ASR 36 秒 | Parakeet 9 秒、Qwen3-ASR 3 秒 |
| GPU メモリのピーク（3 モデル） | 7.9 GB | 7.9 GB |
| Mac の対話ツールの最初の音 | 128 ms | 115 ms |

trailpc の `aha_audio` イメージでノードを動かし、SSH を `worker_command` で同じコンテナ内のワーカーに置き換えて測った、話し終わりから認識結果までの遅延は、英語 0.37〜0.40 秒、日本語 0.70〜0.78 秒だった。英語と日本語の文はどちらも正しく認識した。

### 考察

- 3 モデルは ARM64 の DGX Spark でも動く。torch は `local` のイメージ（2.11.0）より新しい開発版だが、確認した範囲では問題が出なかった。
- 両方で NeMo が `cudaErrorInvalidValue` を報告し、Triton の代わりに PyTorch のデコードへ切り替わる。Parakeet は 0.03 秒で終わるので速度の問題にはなっていない。原因は調べていない。（この時点の NeMo 3.0.0 の動作。原因は下の Haas の節を参照）
- Qwen3-ASR の読み込みが DGX Spark で遅いのは、ディスクからの読み込みの差と考えられる。`preload_languages` に `ja` を入れておけば起動時に一度だけ待つ。

### 接続の制約

trailpc から DGX Spark の `10.99.0.1`（ポート 22・2222・8080）へは接続できなかった。逆方向も同じだった。`10.99.0.x` へのルートは各マシンの LAN のゲートウェイに向いていて、どちらにも `warp-cli` はない。
そのため **trailpc で ROS ノードを動かして `target:=dgx` を選ぶ構成は、現時点では使えない**。Mac からは両方へ接続できる。

### Mac のマイクでの確認

macOS の `say` を MacBook のスピーカーで流し、内蔵マイクで拾った（trailpc、入力のピーク約 -47 dB）。

| 発声 | 認識結果 | 話し終わりから |
| --- | --- | --- |
| 英語 | `Please bring me the orange juice from the kitchen.`（正しい） | 366 ms |
| 日本語 1 回目 | 結果なし | — |
| 日本語 2 回目 | `レンジジュースを持ってきてください。`（文頭が欠けた） | 437 ms |

発声していない区間に `So` が 1 回認識された。いずれも入力が小さいことが原因と考えられるが、人の声では確認していない。

### 確認していないこと

- 人の声とロボットのマイクからの入力。
- SSH を経由した ROS ノードの動作。ノードの確認は上のとおり `worker_command` で行い、SSH の経路は Mac から別に確認した。
- 雑音のある環境での認識精度。

## 2026-11-01 — Haas-ja-v1.0 と Qwen3-ASR の比較

### 条件

- trailpc（RTX 5090 Laptop、ドライバー 595.99.02）と DGX Spark（ARM64、ドライバー 580.178.04）。
- `haas-asr` が `nemo-toolkit[asr]>=2.7.3,<2.8` を求めるので、ワーカーの NeMo を 3.0.0 から 2.7.3 に下げた。
- 音声は macOS の `say -v Kyoko` の 6 文（3.3〜5.0 秒、16 kHz）。人の声ではない。文が少なく合成音声なので、精度の優劣は判断できない。
- 時間は Mac から WARP と SSH を経由した往復を 3 回測った最小値。

### 結果

| | Qwen3-ASR（trailpc） | Haas（trailpc） | Qwen3-ASR（DGX） | Haas（DGX） |
|---|---:|---:|---:|---:|
| 往復時間 | 0.17〜0.30 秒 | 0.08〜0.11 秒 | 0.41〜0.71 秒 | 0.10〜0.15 秒 |
| 6 文の表記が元の文と一致した数 | 6 | 4 | 6 | 4 |
| 読み込み | 3.3 秒 | 6.5 秒 | 30.5 秒 | 99.8 秒（重みの取得を含む） |

Haas の不一致は両方とも同じ表記の違いだけで、「三階 → 3階」「午後二時 → 午後2時」（同じ文）と「ソファ → ソファー」だった。

同じイメージで Parakeet は 0.06〜0.08 秒、Kokoro は 0.06〜0.10 秒で、NeMo を下げても変わらなかった。

語彙の指定（Haas）の結果（下の 1 行目は DGX でも同じ。残りは trailpc のみ）：

| 発話 | 指定なし | 指定あり |
|---|---|---|
| アハロボットはハースを使います。 | アハロボットはバースを使います。 | `AhaRobot=アハロボット`、`Haas=ハース,バース` → AhaRobotはHaasを使います。 |
| すのはらさんを呼んでください。 | 角原さんを呼んでください | `砂原=すのはら` → 砂原さんを呼んでください |
| すのはらさんを呼んでください。 | 角原さんを呼んでください | `春日原`（読みなし） → 変わらない |
| 今日はいい天気です。 | 今日はいい天気です。 | `Haas=…`、`砂原=…` → 変わらない |

### 考察

- NeMo 2.7.3 は、デコーダーの CUDA グラフ用カーネルを実行時に NVRTC でコンパイルする。イメージの CUDA は 13.3 で、trailpc のドライバーは 13.2 までなので、trailpc では `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` で読み込めず、`cudaErrorInvalidValue` で推論が止まる。NeMo 3.0.0 で見えていたエラーも同じ原因の可能性があるが、確認していない。
- `decoding.greedy.use_cuda_graph_decoder` を切って回避した（`without_cuda_graphs`）。`disable_cuda_graphs()` だと、語彙の指定で `change_decoding_strategy` が呼ばれたときに元に戻る。
- 語彙は Haas が書ける綴りでしか強調されないので、読みが必要である。

### 判断

`ja` を Haas に切り替え、Qwen3-ASR は人の声で確認するまで `ja-qwen` として残す。

`target:=local` のイメージ（`docker/audio/`）も NeMo 2.7.3 と `haas-asr` に更新した。trailpc で確認用のタグでビルドし、プロセス内で `Registry` と `load_engine` から動かした結果：Parakeet 0.02〜0.03 秒、Haas 0.05 秒、Qwen3-ASR 0.17〜0.20 秒、Kokoro 0.04 秒。

### 確認していないこと

- 人の声と雑音のある環境での Haas の精度。
- ページで Haas を選んで話す一連の操作。比較はページと同じ `load_remote_engine` と `RemoteEngine` で行った。
