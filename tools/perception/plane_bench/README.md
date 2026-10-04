# 平面検出ベンチマーク

[`aha_perception.planes`](../../../overlay_ws/src/aha_perception/README.md#平面検出-aha_perceptionplanes)
の全候補を合成シーンで評価し、`results/` にレポートとCSVを書き出す。
ROSとコンテナは不要で、ホストの uv で実行する。
これまでの比較結果と条件は [results/comparison.md](results/comparison.md) にまとめている。
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

GTラベルは平面ごとの画像上の連結成分（400画素以上かつ0.02 m²以上）。
物体の面・脚・棚の側板・小さすぎる面は don't care、円柱の側面・flying pixel・外れ値は非平面、
評価範囲（0.3–4.0 m）外と欠損は無効として扱う。

シーンを追加するには `scene_set()` に `Scene(name, tags, CameraPose(...), surfaces, primitives)` を追加する。
`surfaces` は面の名前からクラス（floor / table / shelf / wall / other）への対応、
`primitives` の `Quad.tag`・`Box.faces`・`Cylinder.top` にその名前を指定した面がGT平面になる。
`tags` はレポートのグループ（table / shelf / coplanar / walls / far / floor / clutter）に使う。

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
- `--out`（既定 `results/real.md`）に表を、同じディレクトリに1フレーム目のラベル画像 `real_<scene>_<method>.png` を書く（`--no-figures` で省く）。

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
