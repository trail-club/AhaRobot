# 平面検出ベンチマーク

[`aha_perception.planes`](../../../overlay_ws/src/aha_perception/README.md#平面検出-aha_perceptionplanes)
の全候補を合成シーンで評価し、`results/` にレポートとCSVを書き出す。
ROSとコンテナは不要で、ホストの uv で実行する。
これまでの比較結果と条件は [results/comparison.md](results/comparison.md) にまとめている。

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
