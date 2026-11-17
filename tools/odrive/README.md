# ODriveツール

台車駆動（HB-139 ×2、Makerbase ODrive v3.6-56V、fw 0.5.6）の確認・ホール信号の記録・
解析・ファームウェア書き込み用。機体固有の観測結果は
[台車駆動の測定・検証記録](../../docs/context/odrive.md)に日付順でまとめる。

## 接続

ODriveを接続したPCで、`odrive==0.5.4` を入れたPythonを使う。
lockinで回すツールは、車輪を浮かせ、電源をすぐ切れる状態で使う。
設定は保存しない。終了時やエラー時はIDLEに戻す。

```bash
# 例: ODriveを接続したPCのvenv
~/odrive-venv/bin/python tools/odrive/lockin_teleop.py 0
```

fw 0.5.6のlockinでは `general_lockin.accel` を正の値にする。負の値では磁界が正方向へ
際限なく加速する。回転方向は `vel` の符号だけで決まり、`ramp_distance` は使われない。

## ツール

| スクリプト | 用途 |
| --- | --- |
| `lockin_teleop.py` | キーボードでlockinを操作し、ホールの乱れを毎秒集計する |
| `kt_measure.py` | 逆起電力からトルク定数Ktを測る |
| `gain_test.py` | `vel_gain` を段階的に変え、保持・ステップ・速度追従を同じ手順で測る |
| `closed_loop_teleop.py` | キーボードで閉ループを操作し、追従・電流・フィルタが捨てた遷移を表示する |
| `ground_teleop.py` | 接地した台車を両輪の閉ループでキーボード走行させる（aha_baseの制御部を使う） |
| `pll_bw_test.py` | lockinの一定速度で `encoder.config.bandwidth` ごとの `vel_estimate` のばらつきを比べる |
| `hall_record.py` | 条件ごとにホールの生データをCSVへ記録する。`--plan baseline / filter / idle` |
| `hall_analyze.py` | 記録を条件ごとに集計する（ノイズ件数・追従・`vel_estimate`・フィルタのカウンタ） |
| `hall_filter_eval.py` | フィルタ候補を記録へ当て、偽エッジ・遅れ・カウントずれ・速度推定を比べる |
| `flash_fw.sh` | 設定を退避してからdfu-utilで書き込み、再接続後に版数と主要設定を照合する |

解析はPythonの標準ライブラリだけで動くので、記録を手元へ移して実行できる。

```bash
python3 tools/odrive/hall_record.py 0 logs/hall_YYYYMMDD --plan filter   # ODrive接続PC
python3 tools/odrive/hall_analyze.py logs/hall_YYYYMMDD
python3 tools/odrive/hall_filter_eval.py logs/hall_YYYYMMDD
```

ホスト側の読み取りは2.7–4.6 kHzで、ファームウェアの8 kHzより粗い。
50サンプルごとの補助データ読み取りで約2 msの空白ができるため、ノイズは
「元の値へ戻る短い食い違い、または0/7」として数え、切り替わり位置のずれは数えない。

## 新しいODriveの立ち上げ（配線後から校正まで）

基板・モーター・配線がすべて新しい状態から、両軸を校正済みにするまでの手順。
極対数・抵抗・Ktなどは、既存の機体の値を流用せずこの手順で求める。
終わったら[ゲインの決め方](#ゲインの決め方)へ進む。

- 車輪を浮かせ、電源をすぐ切れるようにしておく。
- USBにはODriveを1台だけつなぐ。`flash_fw.sh` と各ツールは最初に見つかった1台を使う。
- コマンドはODriveを接続したPC（trail-pc）で実行する。ツールは `~/odrive-fw/tools/` にこのディレクトリを写しておく。
- 各段階の結果（シリアル、モーターの型番、配線の色と端子の対応、極対数、校正値、エラー）を
  [台車駆動の測定・検証記録](../../docs/context/odrive.md)へ残す。

### 1. 接続を確かめる

```bash
lsusb -d 1209:0d32          # ODrive（DFU中は 0483:df11）
~/odrive-venv/bin/odrivetool shell --no-ipython
```

odrive 0.5.4のシェルはIPython 9では起動しないため、`--no-ipython` を付ける。接続まで数秒かかる。
MKS基板は純正と判定されず、`Not a genuine ODrive!` と出て `dev0` として接続される。
以下の例は `odrv0` で書いているので、シェルに入ったら最初に `odrv0 = dev0` とする（再起動のたびに必要）。

`odrv0.serial_number`（16進で記録する）、`odrv0.fw_version_*`、`odrv0.vbus_voltage` を見て、
`dump_errors(odrv0)` でエラーがないことを確かめる。

### 2. ファームウェアを書き込む

ホールのフィルタ付きファームウェアを書き込む。版数とフィルタの属性の有無が表示される。

```bash
~/odrive-fw/tools/flash_fw.sh ~/odrive-fw/artifacts/ODriveFirmware_v3.6-56V_halldebounce.bin
```

`hall filter attrs: True` を確かめる。設定の照合の `DIFF` は、次の段階で初期化するので無視してよい。

### 3. 設定を初期化して基板の設定を入れる

出荷時や前の利用者の設定を消し、基板とエンコーダの種類を設定する。

```python
# odrivetool
odrv0.erase_configuration()          # 再起動する。再接続後に odrv0 = dev0
odrv0.config.enable_brake_resistor = True
odrv0.config.brake_resistance = 2.0  # 付けたブレーキ抵抗の値 [Ω]
odrv0.can.config.baud_rate = 250000  # CAN経由（base_controller.py）で使う場合
for i, ax in enumerate((odrv0.axis0, odrv0.axis1)):
    ax.config.can.node_id = i        # axis0 = 左、axis1 = 右
    ax.config.can.encoder_rate_ms = 10
    ax.encoder.config.mode = ENCODER_MODE_HALL
    ax.encoder.config.bandwidth = 50
odrv0.save_configuration()           # 再起動しなければ odrv0.reboot()
```

`axisN.config.startup_*` がすべてFalseのままであることを確かめる（立ち上げ中に勝手に動かさない）。
電流制限 `motor.config.current_lim` は既定10 A。モーターの定格がこれより小さければ下げる。

### 4. ホールと極対数を手回しで確かめる（無通電）

IDLEのまま車輪をゆっくり1回転させ、状態と位置カウントを見る。軸ごとに行う。

```python
# odrivetool。axis0 の例
import time
enc = odrv0.axis0.encoder; c0 = enc.shadow_count
for _ in range(400): print(enc.hall_state, enc.shadow_count - c0); time.sleep(0.05)
```

- `hall_state` は6つの値を一定の順に巡り、変わるたびに1ビットだけ変わる。0・7や飛びがない。
  逆に回すと逆順になる。巡る順は配線によるので、記録しておく。
- 1回転での `shadow_count` の変化 ÷ 6 が極対数。整数にならない場合は回転が1回転からずれているので測り直す。

外れる場合はホールの電源・GND・信号線を直す。決まった極対数を入れて保存する。

```python
for ax in (odrv0.axis0, odrv0.axis1):
    ax.motor.config.pole_pairs = 10            # 測った値
    ax.encoder.config.cpr = 6 * ax.motor.config.pole_pairs
odrv0.save_configuration()
```

### 5. 通電してホールを確かめる

lockinで回し、通電中に状態が化けないか、フィルタが働くかを見る。

```bash
~/odrive-venv/bin/python ~/odrive-fw/tools/lockin_teleop.py 0 --current 3
```

`trans` が 6 × |vel| / 2π 回/秒に合い、`illegal`・`skip` が少ないことを確かめる。
条件ごとに記録して比べるときは `hall_record.py`（電流2–8 A。モーターの定格を超える条件は外す）を使う。

```bash
~/odrive-venv/bin/python ~/odrive-fw/tools/hall_record.py 0 ~/odrive-fw/logs/hall_YYYYMMDD --plan filter
python3 tools/odrive/hall_analyze.py logs/hall_YYYYMMDD   # 記録を手元へ移して集計
```

`hall_debounce_cycles` 4で各条件の追従が1.00、無通電で捨てた遷移が0であることを確かめる。

### 6. モーター校正

相抵抗Rとインダクタンスを測る。抵抗の測定は、`calibration_current` × R が
`motor.config.resistance_calib_max_voltage` を超えると `PHASE_RESISTANCE_OUT_OF_RANGE` で失敗する。
先に電源を切ってテスターで相間の抵抗を測り、その半分をRの目安にして、
`calibration_current` × R が `resistance_calib_max_voltage` の半分程度になる電流を選ぶ
（`current_lim` 以下、モーターの定格以下）。

軸ごとに、保存せず3回行って値の再現を確かめる。odrivetoolで次の関数を定義して使う。

```python
# odrivetool
import time
def run(ax, state):
    ax.requested_state = state
    time.sleep(0.5)
    while ax.current_state != AXIS_STATE_IDLE: time.sleep(0.1)
    dump_errors(odrv0)

ax = odrv0.axis0
ax.motor.config.calibration_current = 5      # 上の目安で決めた値
for _ in range(3):
    run(ax, AXIS_STATE_MOTOR_CALIBRATION)
    print(ax.motor.config.phase_resistance, ax.motor.config.phase_inductance)
ax.motor.config.pre_calibrated = True
```

Rが回ごとにばらつく、またはテスターの目安より大きく外れる場合は、相線の接触を疑う。

### 7. ホール極性校正

```python
run(ax, AXIS_STATE_ENCODER_HALL_POLARITY_CALIBRATION)
print(ax.encoder.config.hall_polarity, ax.encoder.config.hall_polarity_calibrated)
```

エラーが出る場合は、5で通電中の化けを確かめる。

### 8. エンコーダオフセット校正

モーターをゆっくり正逆に回して、ホールと電気角の対応を求める。
保存せず3回行い、`direction` と `phase_offset` が一致することを確かめる。

```python
for _ in range(3):
    run(ax, AXIS_STATE_ENCODER_OFFSET_CALIBRATION)
    c = ax.encoder.config
    print(c.direction, c.phase_offset, c.phase_offset_float, ax.encoder.hall_rejected_count)
ax.encoder.config.pre_calibrated = True
```

`CPR_POLEPAIRS_MISMATCH` が出たら、4の極対数と `cpr` を見直す。
ホールまたは相線をつなぎ直したら、7と8をやり直す。

### 9. 保存と再起動後の確認

6–8をもう一方の軸でも行い、保存する。

```python
odrv0.save_configuration()
# 再接続後（odrv0 = dev0）
for ax in (odrv0.axis0, odrv0.axis1):
    print(ax.motor.is_calibrated, ax.encoder.is_ready, ax.encoder.config.direction, ax.encoder.config.phase_offset)
```

両軸で `is_calibrated` と `is_ready` がTrueで、校正値が残っていれば校正は完了。
設定はODriveの中にしか残らないので、退避しておく。

```bash
~/odrive-venv/bin/odrivetool backup-config ~/odrive-fw/backups/config_calibrated_YYYYMMDD.json
```

### 10. トルク定数Kt

軸ごとに逆起電力からKtを測り、保存する。既定値0.04のままだと、速度ループのゲインが実際とずれる。
`kt_measure.py` は `current_lim` を5 A、中断を4.5 Aにしている。モーターの定格がこれより小さい場合は使わない。

```bash
~/odrive-venv/bin/python ~/odrive-fw/tools/kt_measure.py 0
```

```python
odrv0.axis0.motor.config.torque_constant = 0.374   # 表示された値
odrv0.save_configuration()
```

モーターのKV値 [rpm/V] がわかる場合は、Kt ≈ 8.27 ÷ KVと大きく違わないことを確かめる。
ROS 2側で基板を指定する場合は、`overlay_ws/src/aha_base/config/base.yaml` の `serial_number` を新しい基板の値にする。

## ゲインの決め方

速度ループの電流指令は `vel_gain` × 速度誤差 ÷ Ktなので、Ktが実際と違うとゲインが同じ比でずれる。
先に各軸のKtを測り、ゲインはNm単位（`vel_gain` [Nm/(turn/s)]）で決める。
浮かせた車輪は慣性が小さく、接地より低いゲインで発振するので、ゲインは接地して決める。

1. **Kt**：[立ち上げの10](#10-トルク定数kt)で軸ごとに測って保存しておく。

2. **動作確認**：浮かせたまま、低いゲインだけで `gain_test.py` を流し、監視と記録が働くことを確かめる。

   ```bash
   ~/odrive-venv/bin/python ~/odrive-fw/tools/gain_test.py 0 logs/gain_YYYYMMDD_lifted --vel-gains 0.1,0.2
   ```

3. **`vel_gain`**：両輪を接地し、前後に必要な距離（最大速度 × `--stream-secs` turn、1 turn = 0.437 m）を空ける。
   `vel_gain` を小さい順に上げ、保持で発振したら自動で止まる。発振しなかった最大値の半分程度を候補にする。

   ```bash
   ~/odrive-venv/bin/python ~/odrive-fw/tools/gain_test.py 0,1 logs/gain_YYYYMMDD_ground --plan ground \
       --vel-gains 0.1,0.15,0.2,0.3,0.45,0.6,0.9
   ```

4. **`pos_gain`**：候補の `vel_gain` 1つで `--pos-gain` を変えて流し、ステップの行き過ぎと整定時間を比べる。
5. **`vel_integrator_gain`**：`--vel-int` を小さい値から上げ、低速（0.1・0.3 turn/s）の速度のばらつきと
   止まった後の残差が減り、保持の振れ幅が増えない値を選ぶ。保持で揺れる場合は `--vel-int-limit`、
   目標近くでゲインを下げる `--gain-scheduling` を試す。
6. 決めた値と記録を[台車駆動の測定・検証記録](../../docs/context/odrive.md)へ残し、
   ODriveに保存する。[aha_base](../../overlay_ws/src/aha_base/README.md) の `base_node.py` は保存値を使う。

`gain_test.py` の主な指標：

| 指標 | 内容 |
| --- | --- |
| `A/(turn/s)`・上限 | `vel_gain` ÷ Kt と、積分なしで出せる電流 `vel_gain` × `vel_limit` ÷ Kt |
| 保持 Iq sd・振れ幅 | 発振の判定に使う。カウント変化回数は、止まっていてもホールの境目で増えるので参考値 |
| 行き過ぎ・整定・残差 | `--step` turnのステップ応答。整定は目標±1.5カウントに収まった時刻 |
| 速度 sd/平均 | 速度追従のばらつき。低速ではホールの量子化も含むので、ゲイン間の比較に使う |
| 遅れ | 指令位置と実位置の差。台形軌道（加速度 `--accel`）による遅れを含む |

`vel_limit` は位置誤差から作る速度指令の上限でもあり、積分なしでは出せる電流を頭打ちにする。
`--vel-limit`（既定4 turn/s）を小さくすると、止まった車輪が回り始めないことがある。

## 閉ループの手動操作

`closed_loop_teleop.py` は既定で `base_controller.py` と同じ方式（位置制御＋台形軌道、
0.1秒ごとに目標位置を進め、現在位置から±2 turnで頭打ち）で動かす。`--mode vel` は速度制御。
ゲインとKtはデバイスの値を使い、`--pos-gain` などで指定したものだけRAMで上書きする。
速度上限は `--vel-max`（既定1 turn/s）、電流制限は `--current-lim`（既定5 A）で、
Iqが電流制限の90%を0.3秒超えるとIDLEにする。キー入力を読むため、端末から `ssh -t` で実行する。

```bash
ssh -t trail-pc '~/odrive-venv/bin/python ~/odrive-fw/tools/closed_loop_teleop.py 0'
```

## 接地走行の手動操作

`ground_teleop.py` は [aha_base](../../overlay_ws/src/aha_base/README.md) の `odrive_usb.py` / `diff_drive.py` を使い、
`base_node.py` と同じ方式・同じ寸法と回転方向（`base.yaml` の既定）で両輪を動かす。ROSは使わない。
走行キーは押している間だけ有効で、最後の入力から `--hold`（既定0.6秒）で速度0になる。
端末のキーリピートが始まるまでの遅れより長くする。

- 速度は1–4キーで最大（`--max-lin` 0.4 m/s・`--max-ang` 1.0 rad/s）の25/50/75/100%。開始時は50%。
- ゲインとKtはデバイスの保存値を使い、`current_lim` はRAMで `--current-lim`（既定5 A）にする。
- ODriveのwatchdog（0.5秒）を使う。Iqが `--i-abort`（既定4 A）を0.3秒超えるか、
  軸が閉ループから外れたら両軸IDLEにする。`g` で再開できる。
- `--csv PATH` で20 Hzの指令・位置・速度・目標位置・Iq・odomを記録する。
- 終了時は両軸IDLEにしてRAMの値を戻す。設定は保存しない。

| キー | 動作 |
| --- | --- |
| `g` / `b` | 閉ループ開始（その場で保持）/ IDLE |
| `w` `s` `a` `d`・矢印 | 前進・後退・左旋回・右旋回（押している間） |
| space | 速度0（保持） |
| `1`–`4` | 速度段 |
| `r` | odomリセット |
| `q`・Ctrl-C | 終了 |

リポジトリ外で使う場合は、`overlay_ws/src/aha_base/aha_base/` をスクリプトの隣に置くか、
`--aha-base` で `aha_base/` を含むディレクトリを指定する。周囲の人と障害物を確かめ、電源をすぐ切れる状態で使う。

```bash
ssh -t trail-pc '~/odrive-venv/bin/python ~/odrive-fw/tools/ground_teleop.py --csv ~/odrive-fw/logs/ground_YYYYMMDD.csv'
```

## ホールのグリッチ除去フィルタ

fw-v0.5.6の `encoder.cpp` に、隣接状態への遷移を一定周期続いたときだけ受け付ける
フィルタを加えたファームウェアを使う。追加した属性は保存されない実行時の値で、
保存済み設定の構造は標準版と同じ。

| 属性 | 内容 |
| --- | --- |
| `encoder.hall_debounce_cycles` | 隣接遷移を受け付けるまでの連続周期数（8 kHz）。既定4（0.5 ms）、0で標準と同じ動作 |
| `encoder.hall_recover_cycles` | 隣接しない遷移を受け付けるまでの周期数。既定8（1 ms） |
| `encoder.hall_filtered_cnt` | フィルタ後の状態番号（0–5） |
| `encoder.hall_rejected_count` | 受け付ける前に捨てた遷移の累計 |
| `encoder.hall_illegal_count` | 0/7を読んだ周期の累計 |

ビルドはfw-v0.5.6のソースに差分を当て、`ubuntu:22.04` の `gcc-arm-none-eabi` と `tup` で
`CONFIG_BOARD_VERSION=v3.6-56V` を指定して行う。同じ環境で差分なしにビルドした版は、
配布バイナリと大きさが一致し、版数情報の63バイトだけが異なった。

書き込みは `flash_fw.sh` を使う。`odrivetool dfu` はMKS基板のOTPにハードウェア版数がなく使えない。

```bash
tools/odrive/flash_fw.sh ODriveFirmware_v3.6-56V_halldebounce.bin
```
