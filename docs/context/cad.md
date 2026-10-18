# CAD寸法の調査記録

実機測定を伴わないSTEPデータの調査。以下は当時の判断と未確認事項をまとめたもの。

## 2026-08-30の調査記録

対象は [Astra.STEP](../../upstream/Astra_Hardwares/Astra/Astra.STEP)。
STEPのPRODUCTエントリから、駆動輪候補 `HB-139`（#48084）、キャスター候補
`AuxWheel`（#62817）、取付部品 `HB-139_Mount`（#230535）を識別した。
半径42 mmの `CYLINDRICAL_SURFACE` が4面あり、左右駆動輪の内外周に対応すると判断した。
当初想定していた半径75 mmと異なるため、半径42 mmを寸法の根拠とした。
ただし、部品と面の対応をCADビューアで再確認した記録はない。

車輪間距離・車輪幅・キャスター位置は取得できなかった。
`CARTESIAN_POINT` の走査で得られるのは各部品のローカル座標であり、
アセンブリの配置変換を合成しないと機体上の位置・距離にならないため。
この調査ではその変換を扱っておらず、CAD上の配置や実機寸法を測定した結果ではない。

| 当時のパラメータ | 値 | 根拠・確認範囲 |
| --- | --- | --- |
| `wheel_radius` | 0.042 m | STEP走査による推定。部品との対応はビューアで未確認 |
| `wheel_separation` | 0.40 m | 推定。アセンブリ上の距離は未測定 |
| `wheel_width` | 0.04 m | 推定 |
| `caster_radius` | 0.03 m | 推定 |
| `caster_xoffset` | −0.18 m | 推定 |
| `wheel_mass` | 0.5 kg | 推定。実機質量は未測定 |

現在の設定先は [mobile_base.xacro](../../overlay_ws/src/aha_description/urdf/mobile_base.xacro)・
[controllers.yaml](../../overlay_ws/src/aha_description/config/controllers.yaml)。
上表は調査当時の値と根拠であり、現在値の一覧ではない。

