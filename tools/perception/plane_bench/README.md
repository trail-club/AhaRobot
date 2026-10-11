# 平面検出ベンチマーク

[`aha_perception.planes`](../../../overlay_ws/src/aha_perception/aha_perception/planes/__init__.py) の検出器と
[`aha_perception.place`](../../../overlay_ws/src/aha_perception/aha_perception/place.py) の置き場所の候補、
[`aha_perception.surfaces`](../../../overlay_ws/src/aha_perception/aha_perception/surfaces.py) の面の追跡を、
合成シーンと実機で取得したDepthで評価する。ROSは不要で、ホストの uv で実行する（[取得](#取得)だけ開発コンテナ）。
結果のまとめは `results/` の手で整理した資料を参照する。

| 資料 | 内容 |
| --- | --- |
| [comparison.md](results/comparison.md) | 検出器の候補の比較 |
| [tuning.md](results/tuning.md) | 検出器のノイズモデルと断片統合の比較（ノードの既定の根拠） |
| [noise_calibration.md](results/noise_calibration.md) | Depthノイズのプリセットの較正 |
| [real.md](results/real.md) | 実機フレームでの検出 |
| [shelf.md](results/shelf.md) / [place.md](results/place.md) | 棚・置き場所の候補の評価 |
| [tracking.md](results/tracking.md) | 動くカメラでの面の追跡（IDの安定性）の評価 |

## スクリプト

リポジトリのルートで、次の形で実行する（画像を書くスクリプトは `--with matplotlib` を加える）。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 --with matplotlib \
  python tools/perception/plane_bench/<script>.py [options]
```

| スクリプト | 内容 | 既定の出力（`--out`） |
| --- | --- | --- |
| `run_bench.py` | 全検出器を全シーン × ノイズ3段階（×0.5 / ×1 / ×2）で評価。既定値（壁検出あり・なし）、パラメータスイープ、スイープ最良値（`tuned`） | `results/` |
| `run_sequence.py` | 静止カメラの連続フレームでGT精度と安定性を評価 | `results/sequence.md` |
| `run_real.py` | `real_frames/` の実機フレームで検出（GTなし） | `results/real.md` |
| `capture_real.py` | 実機のDepthを `real_frames/` に保存（[取得](#取得)） | `real_frames/` |
| `calibrate_noise.py` | 実機の静止フレームに合わせてノイズのプリセットを求める | `results/noise_calibration.md` |
| `run_shelf.py` | 棚板ごとの置き場所としてノードの出力を評価 | `results/shelf_report.md` |
| `run_place.py` | 置き場所の候補を物体の大きさごとに評価 | `results/place_report.md` |
| `run_tracking.py` | 動くカメラの系列と実機フレームで面の追跡を評価（[追跡の評価](#追跡の評価)） | `results/tracking_report.md` |
| `make_tuning_figures.py` | 設定ごとのラベル画像 | `results/tuning_*.png` |

BLASは1スレッドに固定する。`--jobs N` のあるスクリプトはN個のプロセスで並列に実行する（時間の指標は競合で増える）。
各スクリプトの全オプションは `--help` で確認する。

## 合成シーン

[`scenes.py`](scenes.py) が解析形状を頭部カメラからレイキャストし、
シミュレーションと共通の [`aha_perception.depth_noise`](../../../overlay_ws/src/aha_perception/aha_perception/depth_noise.py) でノイズを加える。
`scene_set()` がベンチと系列評価、`shelf_scene_set()` が棚の評価、`place_scene_set()` が置き場所の評価の追加シーン、
`tracking_scene_set()` が追跡の評価のシーン（`generate_motion_sequence` で台車と頭を動かす）。
GTラベルの作り方とシーンの追加方法は `scenes.py` のdocstringを参照。乱数のシードはシーン名とノイズから決まる。

## 実行

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/run_bench.py
```

- `--no-sweep`：スイープと `tuned` を省く。`--noise 1.0`・`--scenes table_1.5m ...` で対象を絞る。
- `--figures <scene> ...`：ラベル画像を出力するシーン。`--figures` だけを渡すと出力しない。
- `--with open3d` を加えると、Open3D の `segment_plane` と `detect_planar_patches` を参照値として併記する（import できなければ省く。NixOSでは libEGL・libX11・libusb などを `LD_LIBRARY_PATH` に加える）。

## 出力

| ファイル | 内容 |
| --- | --- |
| `report.md` / `summary.csv` / `per_frame.csv` / `sweep.csv` | `run_bench.py`：比較表、手法 × variant × グループの集計、フレームごとの値、スイープ（ノイズ×1） |
| `labels_<scene>_a.png` / `_b.png` | GTと各検出器のラベル画像。対応したGTと同色、誤検出はマゼンタ |
| `sequence.md` / `.csv`、`real.md`、`real_<scene>_<method>[_<設定>].png` | `run_sequence.py`・`run_real.py` |
| `shelf_report.md` / `.csv` / `_planes.csv`、`shelf_*.png` | `run_shelf.py`：表、棚板 × フレーム、出力の水平面 |
| `place_report.md` / `.csv` / `_invalid.csv`、`place_*.png` | `run_place.py`：表、シーン × プリセット × 大きさ × variant、無効な候補 |
| `tracking_report.md` / `.csv` | `run_tracking.py`：表と実機フレーム、GTの面ごとの値 |
| `comparison.md`・`tuning.md`・`shelf.md`・`place.md` など | 手で整理したまとめ（実行では更新しない） |

時間は640×480の1フレームあたり。`cpu` はプロセスCPU時間、`wall` は経過時間。

## 指標

GT平面と検出平面をハンガリアン法で1対1に対応させる（コスト `角度/10° + GT重心から検出平面までの距離/5 cm + (1 − 画素IoU)`、
角度10°・距離5 cm・IoU 0.2 のいずれかを超える組は対応させない、`metrics.MatchParams`）。集計はフレームをまたいだ合計（micro平均）。

| 指標 | 定義 |
| --- | --- |
| recall / precision / F1 | 対応したGT・検出の割合。画素の半分以上が don't care の検出は除く |
| PQ | 対応組のIoU合計 / (TP + FP/2 + FN/2)。スイープの順位付けに使う |
| mIoU / compl. | 対応組の画素IoUの平均 / GT平面の画素のうち検出に含まれる割合の平均 |
| holes/plane / hole px | 対応した検出マスクに囲まれ、GTの有効画素を含む穴の数（検出1つあたり）と穴の中のGT画素の割合 |
| angle / offset / support h | 法線の角度誤差、GT重心から検出平面までの距離、水平GTの重心のxyでの高さ誤差 |
| class acc | 対応組のうちラベルがGTのクラスと一致した割合 |
| over / under | 2つ以上の検出に分割されたGTの数 / 2つ以上のGTをそれぞれ20%以上覆う検出の数 |
| h_* / v_recall | 水平GT・水平検出だけで数えた値 / 垂直GTのrecall |

## 実機フレーム

実機の頭部カメラのDepthを `real_frames/` に保存し、ロボットなしで検出を確認する。GTはない。

### 取得

実機のカメラ（`real_head_camera.launch.py camera:=true`）が動いている状態で、開発コンテナから実行する。
`ROS_DOMAIN_ID` はロボットと同じ値にし、`aha_perception` を別のbuild baseでビルドした場合はその `setup.bash` も読み込む。

```bash
docker exec -it aharobot_aha_project_1 bash -lc '
  source /app/overlay_ws/install/setup.bash &&
  export ROS_DOMAIN_ID=<ロボットと同じ値> TZ=Asia/Tokyo &&
  python3 /app/tools/perception/plane_bench/capture_real.py \
    --scene desk --filter-label filtered --note "机の上から床と壁"'
```

- `--scene`（必須）：ディレクトリ名に入れるシーン名。
- `--frames`（既定5）・`--interval`（既定0.5 s）・`--timeout`（既定30 s、取れた分だけ保存）。
- `--filter-label`：Depthの後処理の有無（`filtered` / `raw`）。記録するだけで、較正でプリセットの選択に使う。
- `--base-height`：`target_frame` の原点の床からの高さ（m、不明なら省く）。
- `--target-frame`（既定 `base_footprint`）・`--note`・`--out-dir`（既定 `real_frames/`）。

ディレクトリ名の日時はプロセスのローカル時刻（コンテナはUTCのため `TZ` を指定する）。

`real_frames/<YYYYMMDD-HHMM>_<scene>/` は `frames.npz`（Depth mm・`camera_to_base`・時刻・内部パラメータ・頭の関節角）、
`meta.json`、`color.jpg`（1フレーム目）を持つ。読み書きは [`real_frames.py`](real_frames.py) の `save_scene` / `load_scene` / `list_scenes`。

### 評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/run_real.py --max-depth 2.0 3.0
```

- 引数なしで全シーン、シーンのディレクトリを並べるとそれだけを実行する。
- `--methods`（既定 `ransac_gravity_refined ransac_gravity`）・`--max-depth`（既定3.0、複数可）・`--min-depth`（既定0.3）。
- `--output`（既定 `both`）：`all` は `detect_planes` の出力、`selected` はノードと同じ `select_output_planes` 後。
- 検出器の設定は [設定の比較](#設定の比較) の引数。`--no-figures` で画像を省く。
- `base_height` があれば床が z=0 になるよう `camera_to_base` を上げる。

| 指標 | 定義 |
| --- | --- |
| planes/frame / small/frame | フレームあたりの平面数（平均、最小–最大）/ 0.1 m²未満の平面の数 |
| assigned px | 深度範囲内の有効画素のうち、いずれかの平面に入った割合 |
| h matched / h std / h IoU | 連続フレームで対応した水平面（高さ差3 cm・xy 20 cm以内）の割合 / 高さの標準偏差（mm）/ インライアのIoU |
| detect ms | `detect_planes` の経過時間（1スレッド） |
| temporal noise | 深度0.5 mごとの時間方向の標準偏差（中央値）とモデルのσ(Z) |

## ノイズの較正

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/calibrate_noise.py
```

実機の静止フレームと合成の静止系列に同じ統計（[`noise_stats.py`](noise_stats.py)、定義はdocstring）をかけ、
`d435i_raw`（`raw` のシーン）と `d435i_filtered`（`filtered` のシーン）の `NoiseParams` を求める。
値は標準出力に出し、`NOISE_PRESETS` と一致するかを表示する。変える場合は `aha_perception.depth_noise` の `NOISE_PRESETS` を手で更新する。

## 系列評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/run_sequence.py
```

- `--scenes`（既定は全シーン）・`--presets`（既定 `default d435i_raw d435i_filtered`）・
  `--methods`（既定 `ransac_gravity_refined ransac_gravity`）・`--frames`（既定10）・`--max-depth`（既定3.0、複数可）・`--jobs`。
- 検出器の設定は [設定の比較](#設定の比較) の引数。

| 指標 | 定義 |
| --- | --- |
| PQ / F1 / recall / precision / h recall / h height | フレームごとの [指標](#指標) の系列内の平均 ± 標準偏差 |
| persistence / persistent | 水平GT平面ごとの対応したフレームの割合の平均 / 全フレームで対応した水平GT平面の割合 |
| planes all / selected | `detect_planes` / `select_output_planes` のフレームあたりの平面数 |
| h matched / h std / h IoU | `run_real.py` と同じフレーム間の安定性（GTを使わない） |

## 設定の比較

`run_sequence.py` と `run_real.py` は検出器の設定を次の引数の直積で並べる（[`configs.py`](configs.py)、設定名は `sub 0.16, merge` の形）。

- `--detector-subpixel S ...`（既定0.08）：検出器の `DepthNoiseModel(focal=fx, subpixel=S)`。
- `--merge off on`（既定 `off`）：params の `merge`。
- `--param KEY=VALUE ...`：全設定に共通の params の値（例 `merge_k=3 refine_k=4`）。
- `--node-stride N ...`（既定1）：ノードのパラメータ `stride` と同じ間引き。GTも同じ画素に間引く。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/run_sequence.py \
  --detector-subpixel 0.08 0.16 --merge off on --jobs 12 --out /tmp/grid.md
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/make_tuning_figures.py \
  --configs sub=0.08 sub=0.16,merge=on --labels current candidate \
  --real tools/perception/plane_bench/real_frames/<scene>
```

`make_tuning_figures.py` の `--configs` は1行ずつの設定（`sub=` が subpixel、ほかは params の `KEY=VALUE` をカンマで並べる）、`--labels` は行の名前。
合成シーンは `--scenes`（既定 `table_1.5m clutter_room`）× `--presets`、実機は `--real <dir> ...`。
`--method`（既定 `ransac_gravity_refined`）・`--max-depth`・`--min-depth`・`--prefix`（既定 `tuning`）。

## 棚の評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 --with matplotlib \
  python tools/perception/plane_bench/run_shelf.py --keep-downward --jobs 12
```

- `--configs`（既定 `sub=0.08 sub=0.12,merge=on`、名前 `current` / `node`）・`--labels` は `make_tuning_figures.py` と同じ形式。
- `--presets`（既定 `d435i_filtered d435i_raw default`）・`--methods`・`--frames`・`--max-depth`・`--scenes`・`--jobs`・`--figures`（既定3シーン）。
- `--keep-downward`：下向きの水平面を残す選択（`ClassifyOptions(drop_downward=False)`）も ` +keep` として並べる。

棚板は面の名前で識別し、出力平面の画素はノイズなしのレイキャストで当たった面（棚板の上面・縁・裏面、床、机、物体の上面）で分類する。

| 指標 | 定義 |
| --- | --- |
| status | target：板のGT画素がある。sliver：上面が見えるがGTの大きさ未満。hidden：深度範囲内に上面の画素がない |
| detected | target のうち、インライアの50%以上がその板で板のGTの30%以上を覆う出力平面がある割合 |
| abs dh / max dh / dh std | 検出した平面の板のGT重心のxyでの高さ誤差（平均・最大）、板ごとのフレーム間の標準偏差 |
| coverage / split | 板のGTを覆う割合 / 板の5%以上を覆う平面が2つ以上ある target の割合 |
| shelf / table / other | 検出した平面のラベルの割合 |
| merge b-b / b-s / board+obj | 2つの棚板 / 棚板と床・机 / 棚板と物体の上面にまたがる平面の数（フレームあたり） |
| false / underside | 棚板・床・机の画素が半分未満の水平面と、そのうち裏面が主のものの数（フレームあたり） |
| hidden det | hidden の板の面を含む平面がある（シーン, フレーム, 板）の数 |

## 置き場所の評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 --with matplotlib \
  python tools/perception/plane_bench/run_place.py --jobs 12
```

ノードの設定（`ransac_gravity_refined`、`sub=0.12,merge=on`、`select_output_planes`）の出力に `find_place_candidates` をかける。
GTの作り方は [`run_place.py`](run_place.py) のdocstringを参照。

- `--sizes`（既定 `0.06x0.06x0.12 0.10x0.10x0.20 0.20x0.15x0.10`、x × y × 高さ m）・`--presets`・`--frames`・`--max-depth`・
  `--groups`（既定 `bench shelf place`）・`--scenes`・`--jobs`・`--figures`（既定3シーン、`<scene>[:<サイズの番号>]`）。
- `--variants`：`PlaceOptions` の上書き（例 `default obstacle_k=3 edge_margin=0.03,obstacle_margin=0.03`）、`--labels` で名前を付ける。
- `--node-stride`（既定1）：[設定の比較](#設定の比較) と同じ間引き。variantは `PlaceOptions.for_input_stride` を通す。
- 1回の呼び出しで最大50候補を返させ、全候補を評価する。

| 指標 | 定義 |
| --- | --- |
| validity / top1 valid | 半径 r の円がGTの支持面内にあり障害物と重ならない候補の割合 / 1位の候補の割合 |
| off surf / on obj | 高さ4 cm以内の面がない候補 / 物体の上面に割り当てた候補の割合 |
| viol cm / max viol cm | 無効な候補のはみ出し・食い込みの平均・最大 |
| avail n / availability / no place | 円全体が見え、縁・障害物から r + 4 cm 以上離れた中心がある支持面 × フレームの数 / そのうち有効な候補がある割合 / 候補がない割合 |
| IoU / pred/GT / false area | 予測の置ける中心（`PlaceGrid.placeable`）と見えるGTの置ける中心のIoU / 面積比 / GTで置けない面積の割合 |
| clear err cm | 有効な候補の clearance − GTの縁・障害物までの距離（負は保守的） |
| top jitter cm / top same | 1位の候補の連続フレーム間の移動距離の平均 / 2 cm未満の割合 |
| ms / max ms | `find_place_candidates` の経過時間（平均・最大） |

## 追跡の評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/run_tracking.py --jobs 8
```

`tracking_scene_set()` の各シーンで、台車と頭を `run_tracking.trajectory` のとおりに動かした47フレームの系列
（静止、頭のpan ±40°、台車が180°回って `--away` 秒（既定60）後に戻る、前進とtilt）を作り、
ノードと同じ設定（`ransac_gravity_refined`、`sub=0.12,merge=on,node_stride=2`、`select_output_planes`）で検出して
`SurfaceTracker` に正しいodomとカメラの視野を渡す。出力平面をGT平面に [指標](#指標) と同じ方法で対応させ、
GTの面（名前、クラス floor / table / shelf / wall）ごとに付いた面のIDを数える。実機フレーム（`real_frames/`）も同じ設定で追跡する。

- `--scenes`・`--presets`（既定 `d435i_filtered d435i_raw`）・`--away`・`--real <dir> ...`（`--real` だけで実機を省く）・`--jobs`。

| 指標 | 定義 |
| --- | --- |
| with ID / coverage / latency | IDが付いたGTの面の数 / GTの面がある（GT平面になる）フレームのうちIDが付いた割合 / 最初に見えてからIDが付くまでのフレーム数 |
| switches / IDs/surface | GTの面のIDが変わった回数 / GTの面に付いたIDの数（理想は1） |
| shared IDs | 2つ以上のGTの面にそれぞれ2フレーム以上付いたIDの数（別の面の取り違え） |
| tracks / false / don't care | IDの数 / GTに対応しないフレームが半分を超えるID / 平面の大半がdon't care（物体の上面など）のID |
| returns kept / max away s | 2フレーム以上見えなくなって戻ったときに同じIDだった回数 / そのうち見えなかった最長の時間 |
| level acc | 棚板で、IDの `level` がそれまでに見えた棚板の中の下からの順番と一致した割合 |
| track ms | `SurfaceTracker.update` の経過時間 |
| 実機：planes with ID / late IDs / visible | 2フレーム目以降の出力平面のうちIDが付いた割合 / 3フレーム目以降に初めて付いたID / 最初に付いてからそのIDの平面があったフレームの割合 |
