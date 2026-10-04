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
- 可動域は [`docs/servo-bringup.md`](servo-bringup.md) の片側 step (joint0 ±920、joint1 ±990、wrist12 は 1950 で頭打ち、wrist13 ±1360、wrist14 ±1650、gripper ±1310)。
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

```bash
cd /app/overlay_ws
colcon build --symlink-install \
  --paths src/aha_arm_teleop src/aha_bringup \
  --paths ../upstream/astra_controller_interfaces ../upstream/astra_controller
source install/setup.bash
```

`astra_controller` は `simple_cap` に依存している。腕ノードだけ欲しいホストで
そのパッケージが取れなければ、そこは別途入れる。teleop ノード自体は
`astra_controller_interfaces` の `JointCommand` だけで足りる。

腕ノードを上げる。デバイス名は上流の udev (`tty_puppet_left` / `tty_puppet_right`) が既定。

```bash
ros2 launch aha_bringup arms.launch.py
```

このマシンのように片腕だけ、または `/dev/ttyUSB0` のときは名前を渡す。

```bash
ros2 launch aha_bringup arms.launch.py \
  use_right:=false \
  left_device:=/dev/ttyUSB0
```

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

```bash
make test
```

実機では腕ノードを上げたまま、別シェルで次を見る。`/joint_states` は左右とグリッパが同じトピックに載るので、`hz` は合計になる。名前に両腕が入っていることと、キーでその値が変わることを見る。

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --field name --once
```

左の `w` で `joint_l2` が減る方向へ動き、右の `W` で `joint_r2` が同じ向きに動く。`q` で両腕の `/arm/torque_enable` に 0 が行く。

## まだ実機で見ていないこと

この変更を入れた時点では、制御基板は Astra ファームではなくシリアル転送デモのままで、`arm_node` の 921600 には応答しない。両腕をこのノードで動かした確認は、ファームを焼いて `arms.launch.py` が `/joint_states` を出したあとに取る。焼く作業はこの手順に含めない。
