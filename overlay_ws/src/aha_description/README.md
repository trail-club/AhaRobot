# aha_description

AhaRobotのURDF。引数は [aha_robot.urdf.xacro](urdf/aha_robot.urdf.xacro) を参照。

## タッチディスプレイ

xacro引数 `display:=true` のとき、頭の下・2本の昇降レールの前に、ケース付きのWaveshareのディスプレイと取付部品を加える（[display.xacro](urdf/display.xacro)）。
既定は `sim` と同じで、実機のモデルには入れない。

取付部品は [meshes/display_mount](meshes/display_mount) のSTL（mm）で、印刷とURDFの両方に使う。どれもBambu Lab A1 mini（180 mm角）で印刷できる。

| 部品 | 取付 |
| --- | --- |
| `05_spine` | 背骨。左右の昇降レールの前面のT溝にM4 × 20を2本ずつ（溝にM4のナット）。背面の六角穴にM5のナットを入れる |
| `02_lower_left` / `03_lower_right` | 背骨の前面にそれぞれM5 × 2。ケースの下の角を受ける |
| `04_top_keeper` | 背骨の前面にM5 × 2。ケースの上面を押さえる |

受けと押さえはM5 × 25で留める。穴は上下±2 mmの長穴。

ディスプレイは昇降する腕の高さにあり、腕を体の前へ回したまま昇降0.9 m以上へ上げると当たる。
MoveItの計画は避けるが、操作パネルからの昇降などMoveItを通らない指令では防げない。
