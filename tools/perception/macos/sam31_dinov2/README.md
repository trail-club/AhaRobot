# SAM 3.1＋DINOv2 で指定の物を見つける

SAM 3.1 が「缶」のような種類の言葉で出した候補を、探す物の参照画像（4方向）と DINOv2 で見比べ、その物だけを残す。
探す物は自由に決められ、用意するのは名前と参照画像だけ。検出は Qwen を使わず、1枚 約 0.4 秒（DGX）。
比較した方式と経緯は [物体検出の検証記録](../../../../docs/context/perception.md) と [index.html](index.html)（ブラウザで開く）にある。

## 撮影から検出まで

前提：[macOS の準備](../README.md)を済ませ、DGX に SSH で入れること。`<user>` は DGX のユーザー名、`<名前>` は探す物の名前（例：`my_mug`）。

1. **参照画像を撮る**（物ごとに1回）。案内に従って正面・背面・左右を撮る。物は1つだけ無地の机に置き、全体を画面に入れる。

   ```bash
   bash tools/perception/macos/sam31/reference.sh --user <user> --name <名前>
   ```

   最初の1枚で Qwen が中央の物を見て種類の言葉（例：`cylindrical can`）を決め、`phrase.json` に保存する。
   保存先は `.cache-sam31/references/<名前>/` で、`index.html` で確認できる。
2. **場面を撮る**。言葉は何でもよい（確認表示に使うだけ）。写真は `.cache-sam31/results/interactive/<セッション>/` に入る。

   ```bash
   bash tools/perception/macos/sam31/interactive.sh --user <user>
   ```

3. **検出する**。写真はセッションのフォルダか `rgbd.npz` を1つ以上指定する。

   ```bash
   cd tools/perception/macos/sam31_dinov2
   SAM31_SSH_USER=<user> ../../../../.venv-perception-macos/bin/python dino.py detect \
     --reference <名前> ../../../../.cache-sam31/results/interactive/<セッション>
   ```

   `--threshold`（既定 0.52）で採用の下限を、`--phrase` で保存した言葉の代わりを指定できる。1回 約40秒（9枚で38秒）。
4. **結果を見る**。`.cache-sam31/sam31_dinov2/detect/<日時>/report/index.html` に各写真の流れ（候補 → 参照との比較 → 結果）、
   `detections.json` に枠と点数が出る。見逃しや誤りがあれば、点数を見て `--threshold` を変える。

**Qwen を使わない場合**は、登録で種類の言葉を自分で渡す（`--phrase can`）。検出は同じコマンドで動く。
言葉は、探す物も似た物もまとめて候補に出る種類の言葉（`can`・`bottle`・`cup`・`box` など）にする。商品名や `object` では候補に出ないことがある。

## 仕組み

1. SAM 3.1 に保存した種類の言葉を渡し、しきい値 0.3 で候補を全部出す。
2. 候補をマスクで切り出し（背景は灰色、224×224）、参照4方向も同じように切り出す。
3. DINOv2 large（`vit_large_patch14_dinov2.lvd142m`）の全体の特徴（CLS）で参照とのコサイン類似度を測り、最大値を点数にする。
4. 点数がしきい値以上の候補を採用する。裏向きの物も、参照の背面・側面と比べて見つかる。

## 選ばれた経緯

SAM 3.1 は商品名を区別できないため、商品名の物だけを見つける方式を比べ、キャンベル缶の27枚で正解が最も多く（23/27）速い（1枚 0.4 秒）この方式を選んだ。
SAM 3.1 に渡す言葉も比べ、登録時に Qwen が決める方式にした。
条件・結果・不明点は [物体検出の検証記録](../../../../docs/context/perception.md)、各方式の流れは [index.html](index.html) にある。

## 比較の再実行

撮影データ（`.cache-sam31/results/interactive/`）と参照画像（`.cache-sam31/references/campbell/`）が要る。
DGX ではコンテナを裏で動かすので、VPN が切れても処理は続き、つながり直すと結果を取ってくる。

```bash
cd tools/perception/macos/sam31_dinov2
export SAM31_SSH_USER=<user>
PY=../../../../.venv-perception-macos/bin/python
$PY plan_a.py run --set runs19 && $PY plan_a.py run --set campbell27      # 案A
$PY plan_b.py qwen --step boxes && $PY plan_b.py sam --step boxes          # 案B（枠だけ）
$PY plan_b.py qwen --step phrase && $PY plan_b.py reask && $PY plan_b.py sam --step phrase \
  && $PY plan_b.py match --variant a && $PY plan_b.py match --variant b    # 案B（＋照合）
$PY collage.py run --refs clean         # 参照を並べる
$PY dino.py run && $PY phrases.py run   # DINOv2、SAM 3.1 に渡す言葉
$PY report.py --summary                 # 全写真のレポートと index.html・img/
```

結果は `results/*.json` に上書きし、マスクや全写真のレポートは `.cache-sam31/sam31_dinov2/` に出す。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `dino.py` | 検出（`detect`）と27枚での評価（`run`） |
| `phrases.py` | SAM 3.1 に渡す言葉の比較 |
| `plan_a.py`・`plan_b.py`・`collage.py` | 比べたほかの方式 |
| `common.py`・`remote/` | 共通処理と、DGX のコンテナで動かすスクリプト |
| `report.py`・`summary_template.html` | レポートと `index.html` の作成 |
| `data/`・`results/`・`img/` | 正解、各方式の結果、代表画像 |
