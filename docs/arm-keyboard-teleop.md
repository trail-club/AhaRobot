# 両腕を ROS 経由でキーボード教示する

`tools/servo/keyboard_teleop.py` はサーボバスを直接叩く。腕が `arm_node` の下で
動いているときは、同じキーの意味で `/left/arm/joint_command` と
`/right/arm/joint_command` に出す。MoveIt や rosbag2 と同じ口である。

インターフェースは足していない。`upstream/astra_controller` の `arm_node` が
もともと購読しているトピックを使っている。

## 何がシリアル教示と同じで、何が違うか

同じもの:

- 1 回叩くと 8 step (約 0.7°)。押しっぱなしは 250 step/秒。`[` `]` で増減する。
- 目標は実位置から 80 step 以上先行しない。追従しないまま押し続けると、先行分を捨ててその向きを止める。
- 可動域は [保存済み可動域データ](context/motor.md#保存済み可動域データと解釈の限界) の片側 step (joint0 ±920、joint1 ±990、wrist12 は 1950 で頭打ち、wrist13 ±1360、wrist14 ±1650、gripper ±1310)。
- エンコーダ原点 (raw 0 / 4095) から 150 step 以内の向きへは出さない。
- 1 回の観測で 250 step を超えて飛んだ関節は、その関節だけ指令を止める。`0` のあと方向キーで入れ直す。
- `space` でその場停止。`0` でトルク OFF。`q` でトルク OFF して終了。

違うもの:

- 左腕が小文字、右腕が大文字。`w` と `W` は同時に別の腕へ行く。
- 指令の単位は `arm_controller.to_si_unit` のラジアンと、グリッパだけメートル。
  シリアル教示の +方向は +1 サーボの raw を増やす向きで、ROS の角度はそのとき減る。
  キーを押したときの物理方向がシリアル教示と揃うように、符号はそこで反転している。
- グリッパ指令の `name` は `joint_l7r` / `joint_r7r` だけ。`arm_node` が反対側の指を反転して付ける。
- サーボの負荷レジスタは ROS 側に出てこない。止める判定は「目標が先行しているのに実位置が動かない」だけ。
- このノードは基板を焼かない。Waveshare のシリアル転送デモのままでは `arm_node` は無応答である。その間は今までどおり `tools/servo/keyboard_teleop.py`。

## 起動

開発コンテナの中で、`astra_controller_interfaces` と `astra_controller` と overlay をビルドして source する。
コンテナに入ったシェルでは先に `export ROS_DOMAIN_ID=0` する（未設定だと `ros2 topic` などが落ちる）。
```bash
cd /app/overlay_ws

colcon build --symlink-install \
  --paths src/aha_arm_teleop src/aha_bringup \
          ../upstream/astra_controller_interfaces ../upstream/astra_controller

source /app/overlay_ws/install/setup.bash
```

`astra_controller` は `simple_cap` に依存している。腕ノードだけ欲しいホストで
そのパッケージが取れなければ、そこは別途入れる。teleop ノード自体は
`astra_controller_interfaces` の `JointCommand` だけで足りる。

腕ノードを上げる。デバイス名は上流の udev (`tty_puppet_left` / `tty_puppet_right`) が既定。

```bash
ros2 launch aha_bringup arms.launch.py
```
`ros2 run aha_bringup teleop_arms.sh` は、現状 `teleop_arms.sh` がインストールされないため使えない。（Tong: 1018）
このマシンのように片腕だけ、または `/dev/ttyUSB0` のときは名前を渡す。

```bash
ros2 launch aha_bringup arms.launch.py \
  use_right:=false \
  left_device:=/dev/ttyUSB0
```

同じlaunchで、軌道を受けたときだけ腕へ指令するブリッジ（[aha_manipulation](../overlay_ws/src/aha_manipulation/README.md#実機の腕の軌道実行)）も起動する。
教示はブリッジが軌道を実行していないときに行う。

別の対話シェルで教示を起動する。`ros2 launch` はキーをこのプロセスに渡さないので、`ros2 run` にする。

```bash
source /app/install/setup.bash
ros2 run aha_arm_teleop keyboard_teleop
```

`aha_bringup` を入れたあとなら `ros2 run aha_bringup teleop_arms.sh` でも同じである。

片腕だけにするときはノード側も揃える。

```bash
ros2 run aha_arm_teleop keyboard_teleop --ros-args -p sides:="['left']"
```

ステップを変えるとき:

```bash
ros2 run aha_arm_teleop keyboard_teleop --ros-args -p step:=20 -p rate:=400 -p lead:=120
```

## キー

| キー | 左腕 | 右腕 |
| --- | --- | --- |
| `w` / `s` | `joint_l2` (joint0) | `W` / `S` が `joint_r2` |
| `e` / `d` | `joint_l3` (joint1) | `E` / `D` |
| `r` / `f` | `joint_l4` (wrist12) | `R` / `F` |
| `t` / `g` | `joint_l5` (wrist13) | `T` / `G` |
| `u` / `j` | `joint_l6` (wrist14) | `U` / `J` |
| `y` / `h` | `joint_l7r` (gripper) | `Y` / `H` |

`joint_states` にその腕の 5 関節と `joint_*7r` が揃うまで、方向キーは無視する。

## 動いたことの確認

ロジック (step、符号、名前、lead、可動域、原点、stall、飛び) はコンテナの外でも回る。
`make test` の `colcon test` にも含まれる。

```bash
python3 -m pytest overlay_ws/src/aha_arm_teleop/test -q
```

実機では腕ノードを上げたまま、別シェルで次を見る。`/joint_states` は左右とグリッパが同じトピックに載るので、`hz` は合計になる。名前に両腕が入っていることと、キーでその値が変わることを見る。

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --field name --once
```

左の `w` で `joint_l2` の値が減り、右の `W` でも `joint_r2` の値が減る。
URDF の正方向は両腕で同じ向きに定義されているので、物理的には鏡映にならない
（`w` で左腕は内側、`W` で右腕は外側へ動く）。

## 実機

2026-10-11、コンテナ `aharobot_aha_project_1` の中から右腕が動いた。`arms.launch.py` を `use_left:=false right_device:=/dev/ttyUSB0` で上げ、別の `docker exec -it` で `keyboard_teleop`（`sides:=['right']`）を打った。左腕は同じ日に、コンテナの外の WSL 上のキーボードで各関節が動いた。左右を同時には上げていない。本番機で udev 名 `/dev/tty_puppet_left` と `/dev/tty_puppet_right` が出るなら、launch の初期値のままでよい。イメージを新しく作るときは `python3-serial` が要る。`arm_node` はそれを import する。

## 2026-10-18 の検証

### 確認したこと

Windows + WSL2（Docker Desktop）の開発コンテナで、符号パッチ（`patches/astra_controller-trail-club-ver1004.patch`）を当てた `arm_node` を使い、左腕だけ・右腕だけ・両腕同時の順に確認した。

- 両腕とも全 6 関節で、キーを押したときの `/joint_states` の増減と URDF の正方向が一致した。
  `w e r t u`（右腕は大文字）はどれも値を減らす向き、`h` / `H` はグリッパを開く向きに動く。
- 両腕同時では左右のキーが互いに干渉せず、`space` / `0` / `q` が両腕に効いた。
- 2 つのキーを同時に押し続ける操作は確認できない。端末のオートリピートは最後に押したキーだけで、`w` と `W` は Shift を共有する。

### 手順で本文と違ったところ

| 現象 | 原因 | 今回の対処 |
| --- | --- | --- |
| `colcon build` で 2 パッケージしかビルドされない | `--paths` を 2 回書くと最後の 1 つだけが効く | `--paths` を 1 つにまとめた |
| 教示側のシェルで setup が見つからない | `source /app/install/setup.bash` はパスが違う | `/app/overlay_ws/install/setup.bash` を使った |
| `ros2 run aha_bringup teleop_arms.sh` が使えない | `teleop_arms.sh` がインストールされていない | `ros2 run aha_arm_teleop keyboard_teleop` を使った |
| `arm_node` がシリアルを開けない（`Permission denied`） | コンテナの `trail` が `dialout` に入っていない | `sudo usermod -aG dialout trail` のあと `newgrp dialout` |
| `ros2 topic` などの CLI が `ValueError` で落ちる | `ROS_DOMAIN_ID` 未設定で compose が空文字を渡す | 各シェルで `export ROS_DOMAIN_ID=0` |

### グリッパの押しっぱなしが滑らかでない問題

`h` / `y` を押し続けると、グリッパが進んでは止まり、約 2 秒ごとに 1.6 mm ほど逆向きに戻った。教示ノードの表示が間欠的に「この向きは停止中」になった。左右とも同じだった。

原因は教示ノード側にあった。`arm_node` とファームは関係しない。

- 教示ノードはメインループ 1 周で `/joint_states` を 1 本しか処理していなかった。`/joint_states` には 1 腕あたり関節とグリッパの 2 本の発行元があり、届く本数がループ周期とほぼ同じだったため、片方の発行元のメッセージが取りこぼされ、グリッパの実位置が数百 ms 古いままになっていた。
- 古い実位置を基準に目標が lead（80 step）で止まり、実位置が動かないので「動かない」と判定され、目標が古い実位置へ戻された。これが逆向きの戻りだった。
- 別の問題として、押しっぱなしの増分 `rate * dt` を毎回整数に切り捨てていたため、速度が `rate` ではなくループ頻度で決まり、開く向きと閉じる向きで約 2 倍違っていた。

`aha_arm_teleop` を次のように直した。

- メインループ 1 周で、たまっているコールバックを最大 10 本まで処理する。
- 実位置が 0.1 秒より古いときは、押しっぱなしでも目標を進めず、「動かない」判定もしない。
- 押しっぱなしの増分は端数を残して積み上げ、指令するときに丸める。速度は向きによらず `rate`（既定 250 step/秒）になる。以前より速く感じることがある。

ロジックの確認は `overlay_ws/src/aha_arm_teleop/test/test_feedback_timing.py` にある。
修正後、両腕でグリッパと joint0 の押しっぱなしを確認し、止まりや逆向きの戻りは出なかった。