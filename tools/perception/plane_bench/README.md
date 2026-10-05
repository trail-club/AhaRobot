# 平面検出ベンチマーク

[`aha_perception.planes`](../../../overlay_ws/src/aha_perception/README.md#平面検出-aha_perceptionplanes)
の全候補を合成シーンで評価し、`results/` にレポートとCSVを書き出す。
ROSとコンテナは不要で、ホストの uv で実行する。
これまでの比較結果と条件は [results/comparison.md](results/comparison.md)、
検出器のノイズモデルと断片統合の比較は [results/tuning.md](results/tuning.md) にまとめている。
実機で取得したDepthでの確認は [実機フレーム](#実機フレーム) を参照。

## 実行

リポジトリのルートで実行する。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/run_bench.py
```

- 全シーン × ノイズ3段階、各候補の既定値（壁検出あり・なし）、パラメータスイープ、
  スイープ最良値（`tuned`）を実行する。BLASは1スレッドに固定する。
  `ransac_gravity` の画像法線版（`ransac_gravity[organized_normals]`）も別の行として既定値で実行する。
- `--no-sweep`：スイープと `tuned` を省く。`--noise 1.0`・`--scenes table_1.5m ...` で対象を絞る。
- `--figures <scene> ...`：ラベル画像のPNGを出力するシーン（matplotlibが必要）。`--figures` だけを渡すと出力しない。
- `--out <dir>`：出力先（既定は `results/`）。
- `--with open3d` を加えると Open3D の `segment_plane` 反復と `detect_planar_patches` を参照値として併記する。
  `open3d` が import できない場合は自動で省く。Open3D の wheel は libEGL・libX11・libusb などを
  ホストから読み込むため、NixOS ではそれらを `LD_LIBRARY_PATH` に加える。
  Open3D の出力も候補と同じ共通後処理（再フィット・画素選別・分類）を通す。

## 出力

| ファイル | 内容 |
| --- | --- |
| `report.md` | 実行ごとに再生成。全体・シーン種別ごとの比較、`ransac_gravity` 系の比較、壁検出あり・なしの水平面精度、スイープ結果 |
| `summary.csv` | 手法 × variant × グループごとの集計値 |
| `per_frame.csv` | 手法 × variant × フレームごとの集計前の値 |
| `sweep.csv` | スイープ設定ごとの集計値（ノイズ×1のフレーム） |
| `labels_<scene>_a.png` / `_b.png` | GTと各候補のラベル画像（a: 初期4候補、b: 追加4候補）。対応したGTと同色、誤検出はマゼンタ |
| `comparison.md` | 手で整理した比較のまとめ（実行では更新しない） |
| `tuning.md` / `tuning_*.png` | 手で整理した検出器の設定の比較と、`make_tuning_figures.py` の画像 |
| `shelf.md` | 手で整理した棚の置き場所の評価（[棚の評価](#棚の評価)） |
| `shelf_report.md` / `shelf_report.csv` / `shelf_report_planes.csv` / `shelf_s*.png` | `run_shelf.py` の出力 |

実行時間は 640x480 入力1フレームあたり、前処理を含む。`cpu` はプロセスCPU時間、
`wall` は経過時間で、内部でマルチスレッド化する Open3D では `wall` が小さくなる。

## 指標

GT平面と検出平面をハンガリアン法で1対1に対応させる。コストは
`角度/10° + GT重心から検出平面までの距離/5 cm + (1 − 画素IoU)` で、
角度10°・距離5 cm・IoU 0.2 のいずれかを超える組は対応させない（`metrics.MatchParams`）。

| 指標 | 定義 |
| --- | --- |
| recall / precision / F1 | 対応したGT・検出の割合。画素の半分以上が don't care の検出は除外する |
| PQ | Panoptic Quality：対応組のIoU合計 / (TP + FP/2 + FN/2)。スイープの順位付けに使う |
| mIoU | 対応組の画素IoUの平均 |
| compl. | 完全性：対応組で、GT平面の画素のうち検出に含まれる割合の平均 |
| holes/plane / hole px | 対応した検出マスクに囲まれた穴のうち、GT平面の有効画素を含むものの数（検出1つあたり）と、その穴の中のGT画素の割合。欠損画素だけの穴は数えない |
| angle / offset | 対応組の法線角度誤差、GT重心から検出平面までの距離 |
| support h | 床・テーブル・棚板（水平GT）での、GT重心のxyにおける検出平面の高さ誤差の絶対値 |
| class acc | 対応組のうちラベルがGTのクラスと一致した割合 |
| over | 2つ以上の検出に分割されたGT平面の数（検出画素の50%以上がそのGT内、GTの5%以上を被覆） |
| under | 2つ以上のGT平面をそれぞれ20%以上覆う検出の数 |
| h_* / v_recall | 水平GT・水平検出だけで数えた値 / 垂直GTのrecall |

集計はフレームをまたいだ合計から計算する（micro平均）。

## シーン

`scenes.py` が解析形状（矩形・箱・円柱）を頭部カメラ（高さ1.43 m、pan/tilt）からレイキャストし、
D435i相当のノイズを加える。座標は base_link（x前方・z上・床 z=0）。

- 深度ノイズ σ(Z) = Z²·0.08 / (f·0.05 m)、半分は空間相関（ガウシアン 1.5 px）。ノイズ倍率は ×0.5 / ×1 / ×2
- ステレオの遮蔽帯（前景の左側の欠損）、深度エッジの flying pixel、ランダムな欠損と穴、外れ値
- 内部パラメータは 640x480、fx = fy = 465.7、cx = 319.5、cy = 239.5（`Intrinsics`）
- 乱数のシードはシーン名とノイズ倍率から決まり、実行ごとに同じフレームになる

`NOISE_PRESETS` に実機に合わせたノイズ（`d435i_raw` / `d435i_filtered`、[ノイズの較正](#ノイズの較正)）があり、
`noise_preset(name, fx)` で f を画像の fx にして使う。`generate_sequence` は静止カメラの連続フレームを返し、
遮蔽帯・穴・静的ノイズ場は系列で共通、それ以外のノイズはフレームごとに引く（[系列評価](#系列評価)）。

GTラベルは平面ごとの画像上の連結成分（400画素以上かつ0.02 m²以上）。
物体の面・脚・棚の側板・小さすぎる面は don't care、円柱の側面・flying pixel・外れ値は非平面、
評価範囲（0.3–4.0 m）外と欠損は無効として扱う。

シーンを追加するには `scene_set()` に `Scene(name, tags, CameraPose(...), surfaces, primitives)` を追加する。
`surfaces` は面の名前からクラス（floor / table / shelf / wall / other）への対応、
`primitives` の `Quad.tag`・`Box.faces`・`Cylinder.top` にその名前を指定した面がGT平面になる。
`tags` はレポートのグループ（table / shelf / coplanar / walls / far / floor / clutter）に使う。
棚の評価用のシーンは `scene_set()` とは別の `shelf_scene_set()` にあり、ベンチと系列評価の既定には含まれない。

## 実機フレーム

実機の頭部カメラのDepthを `real_frames/` に保存し、ロボットなしで検出を確認する。GTはない。

### 取得

実機のカメラ（`real_head_camera.launch.py camera:=true` と [stream_realsense.py](../macos/pointcloud/README.md#rosbridgeへの直接送信)）
が動いている状態で、開発コンテナから実行する。`ROS_DOMAIN_ID` はロボットと同じ値にする。
`aha_perception` を別のbuild baseでビルドした場合（例 `/tmp/plane_ws/install`）はその `setup.bash` も読み込む。

```bash
docker exec -it aharobot_aha_project_1 bash -lc '
  source /app/overlay_ws/install/setup.bash &&
  source /tmp/plane_ws/install/setup.bash &&
  export ROS_DOMAIN_ID=<ロボットと同じ値> TZ=Asia/Tokyo &&
  python3 /app/tools/perception/plane_bench/capture_real.py \
    --scene desk --filter-label filtered --note "机の上から床と壁"'
```

- `--scene`（必須）：ディレクトリ名に入れるシーン名。
- `--frames`（既定5）・`--interval`（既定0.5 s）：保存するフレーム数と間隔。`--timeout`（既定30 s）で打ち切り、取れた分だけ保存する。
- `--filter-label`：Depthの後処理の有無など（`filtered` / `raw`）。`meta.json` に記録するだけ。
- `--base-height`：`target_frame` の原点の床からの高さ（m）。不明なら省く（`null`）。
- `--target-frame`（既定 `base_footprint`）・`--note`・`--out-dir`（既定はこのディレクトリの `real_frames/`）。

Depth（`/camera/depth_registered/image_rect`）・RGB・CameraInfo・`/joint_states` を購読し、
Depthの時刻のTF `target_frame` ← Depthのframe（得られなければ最新）と頭の関節角を記録する。
ディレクトリ名の日時はプロセスのローカル時刻（コンテナはUTCのため `TZ` を指定する）。

### 形式

`real_frames/<YYYYMMDD-HHMM>_<scene>/` に次を置く。読み書きは `real_frames.py`（numpyのみ）の
`save_scene` / `load_scene` / `list_scenes`。`load_scene` はDepthをm（無効はNaN）で返す。

| ファイル | 内容 |
| --- | --- |
| `frames.npz` | `depth_mm`（N×H×W uint16、0は無効）、`camera_to_base`（N×4×4、Depthの光学frame → `target_frame`）、`stamps`（N, s）、`k`（9）、`head_joints`（N×2、pan・tilt rad、不明はNaN） |
| `color.jpg` | 1フレーム目のRGB（cv2かPILがない場合は省く） |
| `meta.json` | scene・note・filter_label・target_frame・Depthのframe・幅と高さ・base_height・ROS時刻・フレーム数・関節名・TFを最新で代用したフレーム |

### 評価

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/run_real.py --max-depth 2.0 3.0
```

- 引数なしで `real_frames/` の全シーン、シーンのディレクトリを並べるとそれだけを実行する。
- `--methods`（既定 `ransac_gravity_refined ransac_gravity`）・`--max-depth`（既定3.0、複数で比較）・`--min-depth`（既定0.3）。
- `DepthNoiseModel` の f はシーンの fx。`base_height` があれば床が z=0 になるよう `camera_to_base` を上げる。
- `--output`（既定 `both`）：`all` は `detect_planes` の出力、`selected` はノードと同じ `select_output_planes`（水平面と壁）後の平面、`both` は両方を `output` 列で並べる。
- 検出器の設定は [設定の比較](#設定の比較) の引数で指定し、表の `config` 列に出す。
- `--out`（既定 `results/real.md`）に表を、同じディレクトリに1フレーム目のラベル画像 `real_<scene>_<method>.png`
  （設定が複数なら `real_<scene>_<method>_<設定>.png`）を書く（`--no-figures` で省く）。
  画像はカラー・Depthの後に max depth ごとに `--output` の各出力を並べ、`selected` でも平面の色は `all` と同じ。

指標は `output` ごとに、その平面の集合で計算する。

| 指標 | 定義 |
| --- | --- |
| planes/frame | フレームあたりの平面数（平均、最小–最大） |
| small/frame | 面積0.1 m²未満の平面の数 |
| assigned px | 深度範囲内の有効画素のうち、いずれかの平面に入った割合 |
| h matched | 連続フレームで対応した水平面の割合。高さ差3 cm以内・重心のxy距離20 cm以内をハンガリアン法で対応させる |
| h std | 対応をつないだ水平面ごとの高さの標準偏差（mm）の平均 |
| h IoU | 対応した水平面のインライア画素のIoUの平均 |
| detect ms | `detect_planes` の経過時間（1スレッド） |
| temporal noise | 全フレームで有効な画素の深度の標準偏差の中央値を平均深度0.5 mごとに集計し、モデルのσ(Z)と並べる |

## ノイズの較正

実機フレーム（静止カメラ）と合成の静止系列に同じ統計（`noise_stats.compute`）をかけ、
`NoiseParams` を実機に合わせる。`d435i_raw` は `filter_label` が `raw`、`d435i_filtered` は `filtered` のシーンから求める。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/calibrate_noise.py
```

- 実機の統計から初期値を求め、合成シーン `calib_desk`（手前の机・2.6 mの壁）と `calib_wall`（2.7 mの壁・机）の
  10フレーム系列の統計との比で3回補正する。
- `--out`（既定 `results/noise_calibration.md`）に、プリセット値と実機・合成の統計の比較表を書く。
  求めた値は標準出力にも出し、`scenes.NOISE_PRESETS` と一致するかを表示する。値を変えたら `NOISE_PRESETS` を手で更新する。

統計の定義（深度はZ²で正規化して比べる）：

| 統計 | 定義 |
| --- | --- |
| temporal subpixel | 深度エッジ・無効画素から2 px以上離れた全フレーム有効画素の時間標準偏差を、平均深度0.25 mごとに σ/Z²·f·b の中央値（10サンプルの偏りを補正）にし、0.75–3.5 mのビンの中央値をとる |
| correlated fraction / correlation px / aspect | 1.0–3.5 mの画素で、フレームと時間平均の差の行・列方向の自己相関（ラグ1–24 px）に c·exp(−d²/4s²) を当てはめる。px は縦の s、aspect は横/縦 |
| static | 時間平均Depthで `ransac_gravity` が検出した15000画素以上の平面（0.75 m以上）に平面を当てはめ、残差の分散から時間ノイズの寄与（σ²/N）を引く。static level は subpixel 単位の標準偏差、static / temporal はその比、相関長は時間ノイズの寄与を除いた自己相関から求める |
| invalid / never-valid / flicker | 無効画素の割合（全フレーム）、どのフレームでも無効な画素、一部のフレームだけ無効な画素（一度でも有効な画素に対する割合） |
| flicker rate / decay px | never-valid 画素からの距離ごとのフリッカー率をフレームあたりの欠損確率に直し、q(d) = rate·exp(−(d−1)/decay) を当てはめる。6 pxより遠い画素は isolated dropout |
| gross outlier | 画素ごとの中央値から 8 MAD かつ深度の10%を超えて外れたサンプルの割合 |

## プリセット

`scenes.NOISE_PRESETS` の `default` は従来の単フレームのノイズ、`d435i_raw` / `d435i_filtered` は較正値。
較正で使う `NoiseParams` の項目：

| 項目 | 内容 |
| --- | --- |
| `subpixel` | フレームあたりの深度ノイズ σ(Z) = Z²·subpixel / (f·b)（時間成分と静的成分の合計） |
| `static_fraction` / `static_correlation_px` | σ² のうち系列で固定する割合と、その静的ノイズ場のガウシアン相関（px） |
| `correlated_fraction` / `correlation_px` / `correlation_aspect` | 時間成分のうち空間相関をもつ割合、縦のガウシアン相関（px）、横/縦の比 |
| `flicker_rate` / `flicker_px` | 固定の無効画素（遮蔽帯・穴・距離範囲外）から距離 d の画素が各フレームで欠ける確率 rate·exp(−(d−1)/px) |
| `dropout` / `holes` / `outlier_rate` | フレームごとの孤立した欠損、系列で固定の楕円の穴の数、フレームごとの外れ値 |
| `quantization` | 深度の丸め幅（m、0で無効） |

`subpixel` と相関長は適用する画像の画素単位（較正は実機の fx = 603）。
fx の異なる画像で `focal = fx` とすると、m単位の σ(Z) は fx の比で変わり、検出器の `DepthNoiseModel(focal=fx)` との比は保たれる。

## 系列評価

合成シーンを静止カメラの連続フレームとして生成し、フレームごとの検出精度（GTあり）と安定性を評価する。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/run_sequence.py
```

- `--scenes`（既定は全シーン）・`--presets`（既定 `default d435i_raw d435i_filtered`）・
  `--methods`（既定 `ransac_gravity_refined ransac_gravity`）・`--frames`（既定10）・`--max-depth`（既定3.0、複数可）。
- 内部パラメータはベンチと同じ `Intrinsics()`、プリセットと検出器のノイズモデルの f は fx。深度範囲とGTの評価範囲は0.3 m–max depth。
  シードはシーン名とプリセットから決まる。1スレッドで実行する。
- `--out`（既定 `results/sequence.md`）にプリセット × 手法 × 設定ごとのシーン平均と、シーンごとの表を書き、同名の `.csv` に行ごとの値を書く。
- 検出器の設定（`config` 列）は [設定の比較](#設定の比較) の引数で指定する。
- `--jobs N`：系列（シーン × プリセット × max depth）ごとにN個のプロセスで並列に実行する。`detect ms` は1より大きいと競合で増える。

| 指標 | 定義 |
| --- | --- |
| PQ / F1 / recall / precision / h recall / h height | `detect_planes` の出力をフレームごとに [指標](#指標) で評価した値の、系列内の平均 ± 標準偏差 |
| persistence | 水平GT平面（面の名前と重心距離15 cm以内でフレーム間をつなぐ）ごとの、存在するフレームのうち対応した割合の平均 |
| persistent | 存在する全フレームで対応した水平GT平面の割合 |
| planes all / selected | `detect_planes` と `select_output_planes` のフレームあたりの平面数（平均、最小–最大） |
| h matched / h std / h IoU | `run_real.py` と同じフレーム間の安定性（GTを使わない） |

## 設定の比較

`run_sequence.py` と `run_real.py` は検出器の設定を次の引数の直積で並べる（`configs.py`）。
設定名は `sub 0.16, merge` のように表す。

- `--detector-subpixel S ...`（既定0.08）：検出器の `DepthNoiseModel(focal=fx, subpixel=S)`。
- `--merge off on`（既定 `off`）：params の `merge`（[断片の統合](../../../overlay_ws/src/aha_perception/README.md#平面検出-aha_perceptionplanes)）。
- `--param KEY=VALUE ...`：全設定に共通の params の値（例 `merge_k=3 refine_k=4`）。全手法の params にない項目はエラー。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  python tools/perception/plane_bench/run_sequence.py \
  --detector-subpixel 0.08 0.16 --merge off on --jobs 12 --out /tmp/grid.md
```

`make_tuning_figures.py` は設定ごとのラベル画像を `--out`（既定 `results/`）に `<prefix>_<scene>.png`・`<prefix>_real_<scene>.png` として書く（matplotlibが必要）。
合成シーン（`--scenes`、既定 `table_1.5m clutter_room`）は `--presets`（既定 `d435i_filtered d435i_raw`）× 設定ごとに1行で、
1フレーム目の深度誤差・GT・検出（1フレーム目の `all` / `selected`、6フレーム目の `all`）を並べ、検出は対応したGTと同色、誤検出はマゼンタ。
実機シーン（`--real <dir> ...`）は設定ごとに1行で、カラー画像と同じ検出の列を並べる。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 \
  --with matplotlib python tools/perception/plane_bench/make_tuning_figures.py \
  --configs sub=0.08 sub=0.16,merge=on --labels current candidate \
  --real tools/perception/plane_bench/real_frames/<scene>
```

- `--configs`：1行ずつの設定。`sub=` が subpixel、ほかは params の `KEY=VALUE` をカンマで並べる。`--labels` で行の名前を付ける。
- `--method`（既定 `ransac_gravity_refined`）・`--max-depth`（既定3.0）・`--min-depth`（既定0.3）・`--prefix`（既定 `tuning`）。

## 棚の評価

`shelf_scene_set()` の棚のシーン（棚の手前までの距離・棚板の間隔・tilt / pan・物体の有無・板厚・横の机）を系列で検出し、
ノードの出力（`select_output_planes` 後の水平面）を棚板ごとの置き場所として評価する。結果のまとめは [results/shelf.md](results/shelf.md)。

```bash
uv run --no-project --python 3.12 --with numpy==1.26.4 --with scipy==1.11.4 --with matplotlib \
  python tools/perception/plane_bench/run_shelf.py --keep-downward --jobs 12
```

- `--configs`（既定 `sub=0.08 sub=0.12,merge=on`、名前 `current` / `node`）・`--labels` は `make_tuning_figures.py` と同じ形式。
- `--presets`（既定 `d435i_filtered d435i_raw default`）・`--methods`（既定 `ransac_gravity_refined`）・`--frames`（既定10）・`--max-depth`（既定3.0）・`--scenes`・`--jobs`。
  シーケンスの生成は `run_sequence.py` と同じ（シードはシーン名とプリセット）。
- `--keep-downward`：各設定の検出に `ClassifyOptions(drop_downward=False)`（下向きの水平面を残す）の選択も適用し、` +keep` を付けた設定として並べる。
- `--out`（既定 `results/shelf_report.md`）に表を、同名の `.csv` に棚板 × フレーム、`_planes.csv` に出力の水平面を書く。
- `--figures`（既定3シーン）：最初のプリセットで `results/<scene>.png` を書く（深度誤差、GT、最初の設定の1フレーム目、最後の設定の1・6フレーム目）。

棚板は面の名前（`shelf_board<i>`）で識別し、物体や遮蔽で画像上に分かれたGTの断片も1つの棚板として扱う。
出力平面の画素は、ノイズなしのレイキャストで当たった面（棚板の上面・縁・裏面、床、机、物体の上面、その他）で分類し、
同じ板の上面と縁はその板として数える。

| 指標 | 定義 |
| --- | --- |
| status | target：その板のGT画素がある。sliver：上面が見えるがGTの大きさ未満。hidden：深度範囲内に上面の画素がない（カメラより高い、視野外） |
| detected | target のうち、インライアの50%以上がその板で、板のGTの30%以上を覆う出力平面がある割合 |
| abs dh / max dh / dh std | 検出した平面の、板のGT重心のxyでの高さと板の高さの差（絶対値の平均・最大）、板ごとのフレーム間の標準偏差の平均 |
| coverage / split | 検出した平面が覆う板のGTの割合、板の5%以上を覆う平面が2つ以上ある target の割合 |
| shelf / table / other | 検出した平面のラベルの割合 |
| merge b-b / b-s | インライアの10%以上かつ100画素以上が2つの棚板（b-b）、棚板と床・机（b-s）にある平面の数（フレームあたり） |
| board+obj | 棚板と物体の上面をそれぞれ10%以上含む平面の数（フレームあたり） |
| false / underside | 棚板・床・机の画素が半分未満の水平面（偽の置き場所）と、そのうち裏面が主のものの数（フレームあたり） |
| hidden det | hidden の板の面（裏面・縁）をインライアの30%以上含む平面がある（シーン, フレーム, 板）の数 |
