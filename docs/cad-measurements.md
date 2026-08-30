# CAD からの寸法抽出メモ

対象: `upstream/Astra_Hardwares/Astra/Astra.STEP` (22 MB, STEP AP214, 280k lines)

## 発見された部品

STEP の `PRODUCT` エントリを走査した結果、車輪関連は 3 つのみ:

| PRODUCT ID | 名前 | 役割 |
| --- | --- | --- |
| #48084 | `HB-139` | 駆動輪 (中国製ハブモーター型番と思われる) |
| #62817 | `AuxWheel` | 補助輪 (前 or 後キャスター) |
| #230535 | `HB-139_Mount` | HB-139 の取付部品 |

STL/ フォルダには 3D プリント品しか無く、車輪 STL は存在しない → 市販品を購入して使う想定。

## 抽出できた寸法 (確定)

**駆動輪半径 = 42.00 mm (直径 84 mm)**

根拠: `CYLINDRICAL_SURFACE ( '...', #ref, 42.00 )` エントリが STEP 中に **正確に 4 面**存在。差動二輪 (左右で 2 個) の外周と内周にちょうど一致。他の同型半径は全て 20 mm 未満 (ネジ・軸類) か 100 mm 超 (該当なし)。

## 抽出できなかった寸法 (要 FreeCAD)

- **wheel_separation** (左右車輪間距離): STEP の `CARTESIAN_POINT` は各部品のローカル座標系で記録されており、トップアセンブリでの位置を得るには `AXIS2_PLACEMENT_3D` の変換チェーンを再帰的に合成する必要がある。純 Python の regex 走査では不可能。
- **AuxWheel (キャスター) の寸法**: 同上。
- **HB-139 の車輪幅**: 同上 (シリンダの高さは変換後にしか出ない)。

## 抽出方法 (今後 FreeCAD 導入時)

Debian の APT リポジトリには FreeCAD が無いため、以下いずれかで導入:

**方法 A: FreeCAD AppImage をコンテナ or ホストへ**

```bash
wget https://github.com/FreeCAD/FreeCAD/releases/download/1.0.0/FreeCAD_1.0.0-conda-Linux-x86_64-py311.AppImage
chmod +x FreeCAD_*.AppImage
./FreeCAD_*.AppImage --console
>>> import Import
>>> Import.open("upstream/Astra_Hardwares/Astra/Astra.STEP")
>>> # doc.Objects で全パーツを列挙、Placement で world 座標が取れる
```

**方法 B: macOS 側で FreeCAD.app を開く (最短)**

1. `brew install --cask freecad` → FreeCAD.app 起動
2. Astra.STEP を開く
3. Model tree で `HB-139` (2 個ある想定) を選択 → Placement.Base で左右 world 座標を確認
4. `AuxWheel` の位置と直径も同様に測定
5. 結果を `mobile_base.xacro` の xacro:property と `controllers.yaml` の `wheel_separation` に反映

**方法 C: pythonocc-core (Python から OpenCascade)**

```bash
pip install pythonocc-core  # conda-forge 版が必要な場合あり
```

いずれも重量物 (500 MB〜1 GB) なので、実機準備時にまとめて実施を推奨。

## 現時点の xacro / yaml 値

| パラメータ | 現在値 | 出典 |
| --- | --- | --- |
| `wheel_radius` | **0.042 m** | ✅ STEP CYLINDRICAL_SURFACE (4 面) |
| `wheel_separation` | 0.40 m | ⚠️ 推定 (paper のベース幅 500 mm から) |
| `wheel_width` | 0.04 m | ⚠️ 推定 |
| `caster_radius` | 0.03 m | ⚠️ 推定 |
| `caster_xoffset` | -0.18 m | ⚠️ 推定 |
| `wheel_mass` | 0.5 kg | ⚠️ 推定 |
