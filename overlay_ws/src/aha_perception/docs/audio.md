# 音声スタック（aha_perception/audio）

AhaRobot の音声スタック。外部の API は使わず、推論は DGX Spark・trailpc・手元の GPU のいずれかで動かす。

```
              手元                                       推論先（target）
マイク ─▶ Silero VAD ─▶ Smart Turn ──発話区間──▶ ASR (en: Parakeet / ja: Haas)
          (発話区間)    (話し終わり)                │
/aha/perception/audio/transcript ◀──────認識結果────┘
/aha/perception/audio/speak ──────────文──────────▶ Kokoro (英語のみ)
スピーカー ◀───────────────────────音声──────────────┘
                 (再生中は /aha/perception/audio/tts_active=True でマイクを無視)
```

## 推論先

launch 引数 `target` で ASR と TTS のモデルを動かす場所を選ぶ。

| `target` | 場所 | 接続 |
|---|---|---|
| `dgx`（既定） | DGX Spark（`10.99.0.1`） | SSH＋Docker |
| `trailpc` | trailpc（`10.99.0.2`） | SSH＋Docker |
| `local` | ノードと同じプロセスの GPU | なし |

マイク・スピーカー・VAD・話し終わり判定は `target` によらず手元で動かす。推論先へ送るのは確定した発話区間（16 kHz）と読み上げる文だけで、推論先では保存しない。

`dgx` と `trailpc` では、SAM 3.1 と同じくノードの起動時に SSH でコードを転送し、イメージをビルドしてコンテナを起動する。ASR と TTS はそれぞれ別のコンテナになり、終了時に削除する。

- 接続の準備は [SAM 3.1 検証ツール](../../../../tools/perception/macos/sam31/README.md#準備)と同じ。WARP と `trail-club/directory-access` への登録が必要。推論先ごとに Docker と GPU が使えること。
- ノードを動かすマシンから推論先へ SSH で届くこと。trailpc から DGX Spark へは届かない（[検証記録](../../../../docs/context/audio.md)）。
- 初回はベースイメージの取得とビルドで数十分かかる。モデルは推論先の `~/.cache/aharobot-audio` に保存する（合計約 7 GB）。
- `aha_perception` は `--symlink-install` でビルドする。
- ssh-agent に鍵を読み込んでから `make audio-up` する。コンテナにはエージェントのソケットだけを渡す。
- 認識 1 回の失敗はログに残して聞き続け、読み上げの失敗は goal を abort する。接続が切れるとノードはエラーで終了するので、起動し直す。
- モデルの依存は [`tools/perception/audio_worker/`](../../../../tools/perception/audio_worker/) の `pyproject.toml` / `uv.lock` で固定する。`local` の依存は `docker/audio/` にある。

### Mac から使う

Mac のマイクとスピーカーで試す、ブラウザで操作するツールがある。ROS のトピックとアクションは出さない。使い方は [`tools/perception/macos/audio/`](../../../../tools/perception/macos/audio/README.md) を参照する。

## 使用モデル

**台帳は [`config/audio/models.yaml`](../config/audio/models.yaml) で、モデルはここにしか書かない。**
ノードは台帳に書いた `repo` と `revision`（コミット SHA）の版だけを取得して読み込む。
モデルを変えるときは `models.yaml` とこの表を同じ PR で更新する。

Haas-ja-v1.0 は Linux + CUDA 専用である。Qwen3-ASR は比較用に残していて、`language:=ja-qwen` で選ぶ。

| 用途 | モデル | リビジョン | ライセンス | 実行先 |
|---|---|---|---|---|
| ASR（英語） | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | `ae9ad07` | CC-BY-4.0 | GPU（NeMo） |
| ASR（日本語） | [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | `660728d` | Apache-2.0 | GPU（haas-asr, bf16） |
| ASR（日本語、比較用） | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `7278e1e` | Apache-2.0 | GPU（qwen-asr, bf16） |
| VAD | [silero-vad](https://github.com/snakers4/silero-vad) 6.2.3（pip 同梱の重み） | — | MIT | CPU |
| 話し終わり判定 | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) `smart-turn-v3.2-cpu.onnx` | `f766f81` | BSD-2-Clause | CPU（onnxruntime） |
| TTS（英語のみ） | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)（音声 `af_heart`） | `f3ff357` | Apache-2.0 | GPU |

実行中に読み込まれている構成は、次の 3 か所で確認できる。

- `/aha/perception/audio/models`（`std_msgs/String` の JSON、latched）：ノードごとのモデル、リビジョン、デバイス、ロード時間、主要 Python パッケージの版
- デバッグ UI の「使用中のモデル」欄
- 各認識結果の `aha_msgs/SpeechTranscript.model`（例：`nvidia/parakeet-tdt-0.6b-v2@ae9ad07`）

Python パッケージの全バージョンは `make audio-lock` で [`docker/audio/requirements.lock`](../../../../docker/audio/requirements.lock) に書き出す。

## 起動

音声スタックは開発コンテナとは別のコンテナ（`aharobot_aha_audio_1`）で動かす。

```bash
make audio-build       # 初回のみ（PyTorch / NeMo などで数 GB）
make audio-up
make audio-ws-build    # aha_msgs + aha_perception を colcon build
make audio-run SSH_USER=<Unixユーザー名>                          # 推論先は DGX Spark
make audio-run SSH_USER=<Unixユーザー名> AUDIO_TARGET=trailpc      # 推論先は trailpc
make audio-run AUDIO_TARGET=local                                  # このマシンの GPU
```

ブラウザで <http://localhost:8765> を開く。`local` の初回起動時は、モデルを named volume `aharobot_aha_audio_hf` に取得する（合計約 7 GB）。

### マイク / スピーカーの選択

コンテナはホストの PulseAudio / PipeWire を使う。入力元は `PULSE_SOURCE` で選ぶ（未指定ならホストの既定入力）。

```bash
docker exec aharobot_aha_audio_1 pactl list sources short     # 入力元の一覧
make audio-run PULSE_SOURCE=alsa_input.usb-XXXX                # USB マイクを使う
# マイクなしでパイプラインを試す: スピーカー出力のモニターを入力にし、別端末から paplay で音声を流す
make audio-run PULSE_SOURCE=alsa_output.pci-0000_80_1f.3.analog-stereo.monitor
```

デバッグ UI の「入力レベル」が無音でも -5 dB 前後から下がらない場合、その入力元にはマイクがつながっていない（ノイズしか来ていない）。

- 推論先を指定する：`ros2 launch aha_perception audio.launch.py target:=trailpc ssh_user:=<Unixユーザー名>`
- 日本語で起動する：`ros2 launch aha_perception audio.launch.py language:=ja`
- デバッグ UI を使わない：`use_debug_ui:=false`

## インターフェース

| 名前 | 型 | 内容 |
|---|---|---|
| `/aha/perception/audio/transcript` | `aha_msgs/SpeechTranscript` | 認識結果。常時出る |
| `/aha/perception/audio/listen` | `aha_msgs/action/Listen` | 次の発話を 1 つ待って返す。`language` と `hotwords` を goal ごとに指定できる |
| `/aha/perception/audio/speak` | `aha_msgs/action/Speak` | 英語テキストを読み上げる。キャンセルできる。複数の goal は順に再生する |
| `/aha/perception/audio/tts_active` | `std_msgs/Bool` | TTS の再生中は `True` |
| `/aha/perception/audio/debug` | `std_msgs/String`（JSON） | デバッグ UI 向けのイベント（フレームごとのレベル・VAD、認識、TTS） |
| `/aha/perception/audio/models` | `std_msgs/String`（JSON） | 読み込んだモデルの記録 |

```bash
# 言語切り替え（ASR のみ。TTS は英語固定）
ros2 param set /aha_audio_asr language ja
# ホットワード
ros2 param set /aha_audio_asr hotwords "['Charlie', 'orange juice']"
# 1 回聞く / 話す
ros2 action send_goal /aha/perception/audio/listen aha_msgs/action/Listen "{timeout: 10.0}"
ros2 action send_goal /aha/perception/audio/speak aha_msgs/action/Speak "{text: 'Hello, I am AhaRobot.'}"
```

## ホットワード（現状は仕組みのみ）

`hotwords` パラメータと `Listen.goal.hotwords` は、どちらも ASR エンジンまで届く。

- **Qwen3-ASR（ja-qwen）**：`context`（system prompt）に語彙として入れる。
- **Haas（ja）**：GPU phrase boosting で語を強調し、認識結果を指定の表記にする。`Haas=ハース` のように読みを付ける。読みを付けないと、モデルが書けない綴り（`Haas`）は強調されない。読みが複数あれば `Haas=ハース` と `Haas=バース` のように分けて並べる。
- **Parakeet（en）**：未実装で、受け取っても無視する。NeMo の GPU phrase boosting（`boosting_tree`）で実装する予定。デバッグ UI では「ホットワード未対応モデル」と表示される。

## デバッグ UI

- **ライブ入力**：直近 10 秒の入力レベル、VAD 確率、発話区間、Smart Turn の話し終わり確率、ミュート区間（TTS 再生中）
- **認識結果**：テキスト、遅延の内訳（確定待ち＋認識）、区切った理由、録音した区間の再生（直近 50 件を `/tmp/aha_audio/segments` に保存）
- **操作**：認識言語の切り替え、ホットワード、Listen テスト、TTS テスト（最初の音までの時間を表示）
- **使用中のモデル**：リビジョンと Hugging Face のリンク、ライセンス、デバイス、ロード時間

UI は既定で `127.0.0.1` にだけ bind する。認証がないので、別 PC から見るとき（`host: 0.0.0.0`）は信頼できるネットワークでだけ使う。

## 実測値（2026-09-27, `target:=local`, RTX 5090 Laptop, ループバック入力）

`ja` が Qwen3-ASR だったときの値である。Haas の値は [測定・検証記録](../../../../docs/context/audio.md) を参照する。

`dgx` / `trailpc` の測定は [音声の検証記録](../../../../docs/context/audio.md) を参照する。

| 項目 | 値 |
|---|---|
| 英語：話し終わり → 認識結果 | 約 325 ms（確定待ち 224 + Parakeet 33） |
| 日本語：話し終わり → 認識結果 | 460〜670 ms（確定待ち 224 + Qwen3-ASR 210〜410）。JSUT 3 文をすべて正しく認識 |
| TTS：goal → 最初の音 | 約 50 ms（Kokoro。4 秒の文を約 100 ms で合成） |
| Smart Turn / Silero VAD | 27 ms / 回、1 ms / フレーム（CPU） |
| VRAM | ASR ノード 8.5 GB（Parakeet＋Qwen3-ASR）、TTS ノード 1.0 GB |

日本語を使わないときは `preload_languages: [en]` にすると、Haas を読み込まずに VRAM を節約できる。その場合は `ja` に切り替えた最初の発話で読み込むので、その 1 回だけ待つ。

## チューニング

発話の区切り方は [`config/audio/params.yaml`](../config/audio/params.yaml) の `segmenter.*` で調整する。

- **話し終わる前に切られる**：`turn_threshold` を上げるか、`max_silence_ms` を延ばす
- **反応が遅い**：`turn_check_ms` を短くする
- **雑音で反応する**：`start_threshold` と `min_start_ms` を上げる

## 既知の制約

- エコーキャンセルはまだない。TTS の再生中と、終了後 `mute_tail_ms` の間はマイク入力を捨てる（バージイン不可）。
- `onnxruntime-gpu` の公式 wheel は RTX 50 系（sm_120）に未対応という報告があるので、Smart Turn は CPU で動かす。
