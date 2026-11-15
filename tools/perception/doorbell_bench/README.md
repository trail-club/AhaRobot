# ドアベル検出の比較

HICのドアベル（RoboCup@Home Rulebook 2027 Rev-1 §5.1）の検出手法を、学習済みモデルのまま比べる。
ルールでは、ロボットが開始位置で待っている間だけ、事前に知らされない一般的なドアベルが鳴り、用意できなければノックになる。
採用した構成は [ドアベル検出](../../../overlay_ws/src/aha_perception/docs/audio.md#ドアベル検出) で動く。
出力（既定は `/data/results`）はGit対象外で、スクリプトで作り直す。

## スクリプト

| スクリプト | 内容 |
| --- | --- |
| `fsd50k.py` | FSD50Kから正例・負例のクリップを選び、32 kHzで `clips/` に書く |
| `synth.py` | 合成のドアベル（7種類）。FSD50Kにない日本のインターホン型などを補う |
| `build_eval.py` | 評価用の音（ストリーム）を作る（[評価データ](#評価データ)） |
| `zoo.py` | 比べるモデルと読み込み（[モデル](#モデル)） |
| `score.py` | 1モデルを全ストリームに0.2 sずつずらして適用し、スコアを `scores/<model>/` に書く |
| `run_all.sh` | 比較に使う全モデル・窓の `score.py`。GPUの作業者2つが1つの列から順に取る（再実行すると終わった分を飛ばす） |
| `systems.py` | スコアのファイルから、システムの指定（モデル・窓・クラスの集合、アンサンブル、CLAPによる再確認）ごとのスコアを作る |
| `evaluate.py` | 後処理・しきい値を dev で決め、eval の表と誤り例を `report/` に書く（[評価](#評価)）。`--auto` でアンサンブルとCLAPによる再確認も作り、`--describe <指定>` は1つのシステムの動作点と種類ごとの検出率を出す |
| `timing.py` | 1窓のGPUでの推論時間 |
| `pipeline.sh` | `run_all.sh` → `evaluate.py --auto` → `timing.py` を続けて実行する。SSHが切れても止まらないよう `docker exec -d` で起動する |

## 準備

trailpc（RTX 5090 Laptop）で動かした。音声は再生しない。

```bash
# イメージ（ベースは audio worker と同じ nvcr.io/nvidia/pytorch:26.07-py3）
docker build -t trail/aharobot-doorbell-bench:<tag> -f tools/perception/doorbell_bench/Dockerfile .
docker run -d --name <container> --gpus all --shm-size 8g -u $(id -u):$(id -g) \
  -v $PWD:/app -v <data>:/data -v <cache>:/cache trail/aharobot-doorbell-bench:<tag> sleep infinity
```

FSD50Kは、正解ラベルとメタデータを [Zenodo 4060432](https://zenodo.org/records/4060432) から、
音声をOpusのミラー [philgzl/fsd50k](https://huggingface.co/datasets/philgzl/fsd50k)（全51,197クリップ、約3 GB）から取得し、
`<data>/fsd50k/` に `FSD50K.ground_truth/`・`FSD50K.metadata/`・`parquet/` を置く。
ZenodoのWAV（約30 GB）は1本あたり0.1 MB/s程度しか出なかったため、ミラーを使った（Opusの符号化による差は未確認）。

```bash
docker exec <container> python fsd50k.py
docker exec <container> python build_eval.py
docker exec -d <container> bash /app/tools/perception/doorbell_bench/pipeline.sh   # 進み具合は <data>/results/pipeline.log
```

## 評価データ

AudioSetは各モデルの学習に使われているため使わない。FSD50K（Freesound由来）のdev / evalの分割をそのまま使い、
後処理・しきい値はdevだけで決め、表はevalで出す。

| 区分 | 内容 |
| --- | --- |
| 正例（チャイム） | FSD50K `Doorbell` 139本（dev 106 / eval 33）と、合成7種類 × 24本 × 2分割 |
| 正例（ノック） | FSD50K `Knock` 373本（dev 270 / eval 103） |
| 負例 ambient | 話し声・人混み・拍手、音楽、生活音（食器・足音・タイピングなど）。dev 2.9 h / eval 4.1 h |
| 負例 hard_chime | ベル・チャイム・風鈴・鉄琴・ピアノ・電話・着信音・警報・電子レンジなど。dev 3.6 h / eval 3.4 h |
| 負例 hard_knock | ドア・閉まる音・タップ・木・ハンマー・ドラム・拍手など。dev 1.4 h / eval 2.1 h |

FSD50Kの `Doorbell` には地下鉄・空港の案内チャイム（announcement）、店の手振りベル（handbell）、メロディ、ブザー型（ring）が含まれる。
タイトルとタグから `fsd50k.py` の `DOORBELL_RULES` で分類し、筆記音・自転車ベル・冷蔵庫の警告音の5本は除いた。分類は聞いて確かめていない。
FSD50Kに日本のインターホンの録音はない（タイトル・タグを検索して0件）。合成は `synth.py` の7種類（ding_dong、pinpon、westminster、melody、buzzer、bell_ring、single_ding）。

各正例は次の4条件で、6 sの背景の後に鳴らす。

| 条件 | 内容 |
| --- | --- |
| clean | −70 dBFSの白色雑音だけ |
| snr10 / snr0 | 同じ分割のambientを背景に、SN比 +10 / 0 dB（正例の有効区間の電力と背景の平均電力の比） |
| room | pyroomacousticsの直方体の部屋（RT60 0.3〜0.8 s）。正例は3〜8 m先、背景は1〜3 m先から、マイクで +5 dB |

ロボットのマイクでの録音、ロボット自身の音（モーター・ファン）は含まない。

## モデル

| 区分 | `zoo.py` の名前 | モデル（重みの版は `zoo.py` で固定） |
| --- | --- | --- |
| 汎用タガー | `sslam` | [SSLAM AS-2M](https://huggingface.co/ta012/SSLAM_AS2M_Finetuned)（入力は10 sに0詰め） |
| | `eat_large` | [EAT-large AS-2M](https://huggingface.co/worstchan/EAT-large_epoch20_finetune_AS2M)（同上） |
| | `beats` | BEATs iter3+ AS2M cpt2（Microsoftの配布はSASトークンが必要なため、SHA-256が4つのミラーで一致したファイル） |
| | `ced_base` / `ced_mini` | [CED-Base](https://huggingface.co/mispeech/ced-base) / [CED-Mini](https://huggingface.co/mispeech/ced-mini) |
| | `effat_mn40` | [EfficientAT](https://github.com/fschmid56/EfficientAT) mn40_as_ext |
| | `yamnet` | YAMNet（TFLite、0.975 sごと） |
| フレーム単位 | `sed_beats` / `sed_atstf` / `sed_mn10` | [PretrainedSED](https://github.com/fschmid56/PretrainedSED) のAudioSet strong（10 s入力、40 msごと） |
| ゼロショット | `clap_fused` / `clap_larger` | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) / [laion/larger_clap_general](https://huggingface.co/laion/larger_clap_general) |

EfficientAT mn10_as は2 s以下の窓で全クラスのlogitが20〜90になり、使えなかった。軽量の比較対象はCED-Miniとsed_mn10にした。
EAT-largeは共有GPUで全データに十数時間かかる見込みだったため、hard_chime / hard_knock の負例を除いて採点した（`score.py --skip-hard`、表のhardの誤報率は「–」）。
同じEATのコードで動くSSLAMは全データで採点した。
LAION-CLAPの学習データ（LAION-Audio-630K）にはFreesoundが含まれ、評価データの音を学習している可能性がある。

## 評価

`evaluate.py` の冒頭に定義を書いた。要点：

- スコアは窓のクラス確率の最大値。クラスの集合は `common.CLASS_SETS` の3つ（`strict`：Doorbell・Ding-dong / Knock、`door`：ほかのドアの合図を加える、`tonal`：ベル・鍵盤打楽器・電子音を加える）。
- 後処理：移動平均（`avg` 回）、`n` 回連続でしきい値以上。その立ち上がりを検出とし、前の立ち上がりから5 s以内は同じ検出とみなす。
- 正解：鳴り始めから4 s以内に検出が始まること。遅延は窓の終わりの時刻 − 鳴り始め（推論時間は含まない）。
- 誤報率：負例1時間あたりの検出数。
- 動作点：60 s待つ間に誤報せず鳴った音を検出する確率（検出率 × exp(−ambientの誤報率 × 60 s)、表の gain）をdevで最大にする。誤報率 ≤ 1 / 5 回/h の検出率も並べる。

## 結果

2026-11-15、trailpc（RTX 5090 Laptop）。数値はeval（チャイム201本・ノック103本 × 4条件、負例 ambient 4.1 h・hard_chime 3.4 h・hard_knock 2.1 h）。
各モデルは窓・クラスの集合・後処理のうちdevのgainが最良のもの。gainは60 s待つ間に誤報せず鳴った音を検出する確率で、その動作点の検出率・誤報率を並べる。
「誤報≤1 / 5」はdevで環境音の誤報が1時間あたり1 / 5回以下になる最も低いしきい値でのevalの検出率。推論時間はバッチ1の1窓（GPU、YAMNetはCPU）。
全行は `report/table_{chime,knock}.md`、誤りの一覧は `report/examples_*.md`。

### チャイム

| システム | gain | 検出率 | clean / snr10 / snr0 / room | 実録 / 合成 | 遅延 中央値 / 90 % (s) | 環境音の誤報/h（話し声 / 音楽 / 生活音） | hard_chime 誤報/h | 誤報≤1 / 5 | 推論 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| アンサンブル（BEATs-SED, ATST-F, mn10, BEATs） | 69 | 75 | 77 / 75 / 72 / 77 | 70 / 76 | 0.2 / 0.8 | 5.2（0.0 / 19.1 / 1.3） | 90 | 50 / 65 | – |
| **BEATs-SED**（`sed_beats:delay0.4:strict`） | 66 | 76 | 80 / 74 / 73 / 77 | 72 / 76 | 0.8 / 1.0 | 7.9（0.6 / 29.2 / 1.3） | 50 | 46 / 69 | 5.4 |
| CLAPで再確認（BEATs-SEDにclap_larger ≥ 0.2） | 66 | 75 | 78 / 74 / 72 / 77 | 73 / 75 | 0.6 / 1.2 | 7.6（0.0 / 28.2 / 2.0） | 58 | 44 / 67 | – |
| ATST-F（`sed_atstf:last0.2:strict`） | 64 | 70 | 71 / 68 / 67 / 75 | 70 / 70 | 0.6 / 1.0 | 5.9（0.0 / 20.1 / 2.6） | 72 | 40 / 55 | 2.3 |
| frame_mn10（`sed_mn10:last0.2:strict`） | 62 | 69 | 80 / 67 / 59 / 68 | 63 / 70 | 0.4 / 0.8 | 6.4（0.0 / 25.1 / 0.7） | 58 | 32 / 56 | 2.4 |
| CED-Base（`w2:strict`） | 57 | 72 | 71 / 72 / 73 / 71 | 70 / 72 | 0.4 / 1.2 | 13.8（2.6 / 37.2 / 9.9） | 114 | 38 / 50 | 2.3 |
| BEATs iter3+（`w2:strict`） | 55 | 59 | 49 / 67 / 61 / 61 | 56 / 60 | 0.6 / 2.6 | 5.2（1.3 / 13.1 / 4.0） | 76 | 45 / 54 | 5.1 |
| CED-Mini（`w2:strict`） | 55 | 66 | 67 / 68 / 61 / 68 | 61 / 67 | 0.8 / 1.4 | 10.6（0.6 / 27.2 / 9.9） | 116 | 32 / 51 | 2.3 |
| SSLAM（`w2:door`） | 54 | 62 | 78 / 64 / 49 / 58 | 61 / 62 | 0.6 / 2.6 | 8.9（0.0 / 28.2 / 5.3） | 103 | 24 / 43 | 3.9 |
| EfficientAT mn40（`w2:strict`） | 54 | 60 | 53 / 66 / 59 / 62 | 62 / 60 | 1.0 / 2.0 | 6.2（0.6 / 21.1 / 2.0） | 93 | 40 / 52 | 2.4 |
| YAMNet（`w1:strict`、基準） | 54 | 65 | 74 / 68 / 58 / 61 | 63 / 66 | 0.6 / 1.6 | 11.4（1.3 / 23.1 / 13.9） | 120 | 11 / 46 | 0.7（CPU） |
| EAT-large（`w2:door`、hardの負例なし） | 53 | 64 | 73 / 65 / 54 / 65 | 69 / 63 | 1.0 / 2.4 | 11.8（2.6 / 28.2 / 10.6） | – | 8 / 31 | 9.8 |
| CLAP（clap_fused / clap_larger、`tonal`） | 32 | 42 / 43 | – | – | – | 16.5 / 18.0 | 89 / 79 | 0 / 26・0 / 0 | 3.6 / 5.0 |

### ノック

| システム | gain | 検出率 | clean / snr10 / snr0 / room | 遅延 中央値 / 90 % (s) | 環境音の誤報/h（話し声 / 音楽 / 生活音） | hard_knock 誤報/h | 誤報≤1 / 5 |
|---|---|---|---|---|---|---|---|
| **BEATs-SED**（`sed_beats:delay0.4:strict`） | 72 | 77 | 88 / 85 / 71 / 62 | 0.6 / 1.2 | 4.2（0.6 / 1.0 / 9.9） | 65 | 53 / 73 |
| アンサンブル（BEATs-SED, ATST-F, BEATs） | 70 | 75 | 86 / 84 / 69 / 60 | 0.6 / 1.2 | 4.2（0.6 / 2.0 / 9.3） | 53 | 57 / 75 |
| ATST-F（`delay0.4:strict`） | 66 | 77 | 90 / 86 / 71 / 60 | 0.6 / 1.0 | 9.1（1.9 / 5.0 / 19.2） | 80 | 55 / 67 |
| BEATs iter3+（`w2:strict`） | 65 | 73 | 81 / 80 / 68 / 65 | 0.4 / 1.4 | 7.4（3.9 / 1.0 / 15.2） | 99 | 38 / 64 |
| frame_mn10（`delay0.4:strict`） | 61 | 72 | 88 / 76 / 59 / 65 | 0.8 / 1.6 | 10.4（7.8 / 3.0 / 17.8） | 100 | 35 / 54 |
| SSLAM（`w2:strict`） | 60 | 66 | 86 / 69 / 51 / 57 | 0.6 / 1.8 | 5.4（3.2 / 2.0 / 9.9） | 73 | 35 / 59 |
| CED-Base（`w2:strict`） | 59 | 67 | 86 / 79 / 60 / 44 | 0.6 / 1.4 | 7.6（2.6 / 2.0 / 16.5） | 106 | 34 / 52 |
| EAT-large（`w2:strict`、hardの負例なし） | 58 | 65 | 85 / 70 / 52 / 53 | 0.6 / 1.8 | 6.9（3.9 / 1.0 / 13.9） | – | 32 / 60 |
| CED-Mini / EfficientAT mn40 | 54 / 53 | 65 / 67 | – | – | 10.6 / 14.8 | 114 / 135 | 35 / 45・33 / 43 |
| CLAP（clap_larger `w1` / clap_fused `w2`） | 44 / 35 | 52 / 51 | – | – | 11.1 / 22.2 | 97 / 66 | 22 / 35・10 / 33 |
| YAMNet（`w1:strict`） | 31 | 35 | 83 / 24 / 18 / 14 | 0.4 / 2.2 | 6.9（3.9 / 0.0 / 14.5） | 95 | 22 / 30 |

### 分かったこと

- AudioSetの時刻付きラベル（strong）で学習したフレーム単位のモデル（PretrainedSED）が、どのクリップ単位のタガーより良い。mAPが最も高いSSLAMより、チャイムでgainが12、ノックで12高い。
- クラスの集合は `strict`（Doorbell・Ding-dong / Knock）が最良。ベルや電子音のクラスを足すと、音楽での誤報が増えてしきい値が上がり、検出率が下がる。
- アンサンブルはチャイムで +3、ノックで −2。CLAPによる再確認は効果がない。CLAP単体は、ドアベルの文章と鐘・電子音の文章を区別できず最下位。
- 環境音の誤報の大部分は、FSD50Kの音楽にある単音の楽器サンプル（ハープ・ギター・ゴング・シンギングボウル・チューブラーベル）。話し声では1時間あたり0〜1回。
- クリップ単位のタガーは2 sの窓が1 sより良い（CED・EfficientAT・BEATsでgain +2〜8）。YAMNetはほぼ同じ。

### 推奨構成

[台帳](../../../overlay_ws/src/aha_perception/config/audio/models.yaml)の `beats_sed`（既定）：BEATs-SED、10 sの窓を0.2 sごと、窓の終わりの0.6〜0.4 s前のフレームの最大、クラスは `strict`。

| 対象 | 動作点 | 後処理（avg, n, τ） | 検出率 | 環境音の誤報/h | hard の誤報/h | 遅延 中央値 / 90 % (s) |
|---|---|---|---|---|---|---|
| チャイム | **dev誤報≤5/h（台帳）** | 1, 3, 0.0122 | 69 % | 2.7（音楽 11.1、話し声 0） | 28 | 1.0 / 1.2 |
| チャイム | gain最大 | 1, 2, 0.0071 | 76 % | 7.9 | 50 | 0.8 / 1.0 |
| チャイム | dev誤報≤1/h | 1, 2, 0.168 | 46 % | 0.25 | 2.4 | 1.0 / 1.4 |
| ノック | **dev誤報≤5/h（台帳）** | 2, 2, 0.0488 | 73 % | 4.0（生活音 9.3） | 47 | 0.8 / 1.4 |
| ノック | gain最大 | 2, 1, 0.0353 | 77 % | 4.2 | 65 | 0.6 / 1.2 |
| ノック | dev誤報≤1/h | 1, 2, 0.243 | 53 % | 0.49 | 8.4 | 0.8 / 1.5 |

台帳は「dev誤報≤5/h」にした。60 s待つ場合のgainはgain最大の点とほぼ同じ（0.66）で、誤報は3分の1になる。
この表と種類ごとの検出率は `python evaluate.py --describe sed_beats:delay0.4:strict` で出る。

### 種類ごとの検出率（BEATs-SED、gain最大の点）

| ding_dong | pinpon | melody | chime（実録） | handbell | single_ding | westminster | ring（実録） | bell_ring | other（実録） | announcement | buzzer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 99 | 100 | 89 | 92 | 86 | 83 | 64 | 61 | 62 | 44 | 5 |

ノックは77 %（clean 88、snr10 85、snr0 71、room 62）。

### 見逃し・誤報の例と原因

- ブザー型（合成のbuzzer、実録の「buzzer.wav」「Doorbell harsh」）はほとんど検出しない。AudioSetのDoorbellはチャイム型が中心で、ブザーは `Buzzer` クラスに入るためと考えられる。ただし `Buzzer` を加えた `door` の集合でも検出率は5 %から7 %にしか上がらず、devのgainは下がった。
- 地下鉄・空港・エレベーターの案内チャイム（announcement）も多くを見逃す。ルールの「一般的なドアベル」からは外れる。
- 部屋の残響で3〜8 m先から叩いたノック（room）は62 %に下がる。
- 環境音の誤報は、音楽の単音（ハープ・ギターの弦、ゴング、シンギングボウル、チューブラーベル）と、生活音の鍋のふた・食器の衝突音（ノック）。
- hard_chimeの誤報は、グロッケンシュピール・カウベル・地下鉄の発車チャイムなどで、ドアベルと区別しにくいものが多い。

### 確認していないこと

- ロボットのマイク・会場の雑音・ロボット自身の音（モーター・ファン）での性能。スピーカーで鳴らした実録の評価はしていない。
- 日本のインターホンの実録（FSD50Kにない。合成の「pinpon」だけ）。
- 曲として流れる音楽での誤報。FSD50Kの音楽は単音や短いフレーズが多い。
- Opusミラーの符号化による影響、`DOORBELL_RULES` による分類の正しさ（聞いて確かめていない）。
- clap_largerの2 sの窓（GPUのメモリ不足で採点できず、1 sの窓だけ）。
