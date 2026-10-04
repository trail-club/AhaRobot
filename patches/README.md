# AstraArmController用パッチ

[AstraArmController-trail-club-hardware.patch](AstraArmController-trail-club-hardware.patch)は、
対向取付の符号と単発サーボIDを表にした実機用の変更。
測定根拠と適用上の制約は [モーターの測定・検証記録](../docs/context/motor.md)を参照。

| 設定 | 内容 |
| --- | --- |
| `JOINT_SERVO_SIGN` | サーボごとの取付符号。読み取りとPWM出力で使用 |
| `NONE_JOINT_ID` | 単発サーボのID。−1のスロットは読み書きを省略し、6スロットの通信形式を維持 |

機体固有の符号・IDを確認してから、入れ子submoduleへ適用する。

```bash
cd upstream/AstraFirmwares/AstraArmController
git apply ../../../patches/AstraArmController-trail-club-hardware.patch
pio run
```

submoduleの変更を共有する場合は [fork側のPRとSHA更新](../docs/upstream-workflow.md)を行う。


## 2026-10-04更新

### 左腕の初期化が ID10 で失敗した問題と、ファーム初期化処理の修正
#### 現象

joint0 / joint1 を URDF ゼロ姿勢 (旧零点の座標系で j0 ≈ +1°, j1 ≈ +11°) に置き、
`init_arm.py` (`setupTorque(128)`) を実行したところ、同じ姿勢で 2 回続けて次で停止した。

```text
Maybe cause wrong init_pos0, check power status, or reinstall the servo #6
E (...) task_wdt: Task watchdog got triggered ... Rebooting...
```

`#6` はループ添字で ID10 (= 4 + 6)。停止は予圧より前のため PWM 出力はなく、
タイマタスク内の `while (1)` がタスク WDT に検出されて約 5 秒後に ESP32 が再起動し、
`setupTorque(0)` で全サーボ脱力した。`/config.txt` は書き換わっていない (旧零点のまま)。
ID15 の `doInitJoint` と ID4–11 の PWM モード書き込みは失敗前に実行済みだった。

#### 原因

ブリッジファームで同じ姿勢のまま読んだ値 (mode=2 = PWM モード):

| ID | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 読み値 | 992 | 340 | 182 | 199 | 3904 | 151 | **4088** | 393 |

ID10 = 4088 がファームの範囲チェック `20 < pos < 4076` を外れていた。

さらに、ID4–11 を mode=0 (位置モード) に切り替えて読むと 2034–2109 となり、
**PWM モードの読み値 = 位置モードの読み値 + offset** が全数で一致した
(例: ID4 は 2034 + (−1042) = 992、ID10 は 2048 + 2040 = 4088 (mod 4096))。
つまり **PWM モードでは offset が適用されず、エンコーダ生値が返る**。
このため `rezero.py` で ID4–11 の offset を変えても、初期化 (PWM モード) で読む値は変わらない。

- PWM モードの ID10 に `rezero.py` (レジスタ 40 に 128) を実行すると「★失敗」と表示されたが、
  offset 自体は 2047 → 2040 に書き換わっていた (読み値が変わらないため失敗と判定された)
- ID9 (151) と ID10 (4088) はどちらも符号 −1 で、エンコーダ境界が joint1 の URDF ゼロ付近にある。
  ID10 を境界から離す向きに joint1 を動かすと ID9 が 0 側の境界に近づくため、
  両方が `20 < pos < 4076` を満たす joint1 の姿勢は約 10° の幅しかない。
  9 月 27 日の可動域中心での初期化は、推算で ID9 ≈ 26 とぎりぎり通っていた
- 範囲チェックに加えて、予圧 2 回の差 `abs(pos1 - pos2)` と平均 `(pos1 + pos2) / 2` も
  境界をまたぐ場合を扱っていない (例: 4090 と 10 → 差 4080 で誤判定、平均 2050 で零点が誤る)。
  運転中の `read_pos()` は `init_pos` を基準に mod 4096 で計算しているため、境界をまたいでも問題ない。
  **問題は初期化処理だけにある**

#### 対応

ファームの初期化処理を、エンコーダ境界をまたいでも正しく動くよう修正した
(パッチ: [`patches/AstraArmController-trail-club-ver1004.patch`](../patches/))。

1. `init_pos0` の範囲チェックを、読み取り失敗 (`ReadPos` が負) の検出のみに変更
2. 予圧 2 回の差を円周差 (±2048 に折り返し) で判定
3. `init_pos` を円周平均で計算 (4090 と 10 → 2)
4. 異常時の `while (1)` を、予圧を止めて ID4–11 を脱力し、`/config.txt` を書かずに戻る処理に変更。
   RAM 上の `init_pos` が途中まで書き換わっている可能性があるため、
   再起動か再初期化まで制御ループを止める (`init_pos_inited = false`)

### AstraArmController-trail-club-ver1004.patch

初期化（`setupTorque(128)`）を、エンコーダの0/4095境界をまたいでも正しく動くようにする変更。
PWMモードではサーボのoffsetが効かず生値が返るため、組み付けによっては正常な姿勢でも
境界付近になり、範囲チェックで初期化が止まっていた（左腕ID10 = 4088）。
経緯は [モーターの測定・検証記録](../docs/context/motor.md) の2026-10-04を参照。

| 変更 | 内容 |
| --- | --- |
| 範囲チェック | `20 < pos < 4076` → 読み取り失敗（負値）のみ検出 |
| 予圧の差 | 円周差（±2048に折り返し）で判定 |
| `init_pos` | 円周平均で計算 |
| 異常時 | `while (1)` → 予圧を止めて脱力し、configを書かずに戻る |

submodule `c5b2af4`（符号表・ID表を含む）に対して作成。

## 適用方法

パッチは入れ子submodule `upstream/AstraFirmwares/AstraArmController` の作業ツリーへ当てる。
`AstraArmController-trail-club-ver1004.patch` は `src/dualMotor.cpp` の初期化処理だけを変更し、
`AstraArmController-trail-club-hardware.patch` の符号表・ID表には触れない。

```bash
cd upstream/AstraFirmwares/AstraArmController
git status --short   # 想定外の未コミット変更がないこと

# 適用済みかを確認（逆適用が通れば適用済み、何も出力しなければ未適用）
for p in trail-club-hardware trail-club-ver1004; do
  git apply --reverse --check ../../../patches/AstraArmController-$p.patch 2>/dev/null && echo "$p: applied"
done

# 未適用のものだけ当てる（--check が通ってから実行）
git apply --check ../../../patches/AstraArmController-trail-club-ver1004.patch
git apply ../../../patches/AstraArmController-trail-club-ver1004.patch
git --no-pager diff --stat

pio run
```

書き込みは腕ごとにポートを確認してから行う。`pio run -t upload` はLittleFS上の `/config.txt`
（`init_pos`）を消さないため、初期化処理を差し替えるだけなら再初期化は不要。
零点を取り直す場合は書き込み後に [init_arm.py](../tools/servo/init_arm.py) を実行する。

```bash
pio run -t upload --upload-port /dev/ttyUSB0
```

- 取り消しは `git apply -R <パッチ>`
- submoduleを切り替え・更新した後は、上の確認ループで適用状態を見直す
- fork側に取り込まれ、AhaRobotのsubmodule SHAが更新された後は、このパッチは不要になる
# astra_controller用パッチ

[astra_controller-trail-club-ver1004.patch](astra_controller-trail-club-ver1004.patch)は、
上位機の `ArmController` に腕ごとの関節符号と可動域（`ARM_PROFILES`）を追加する変更。
`ArmController(name, side=...)` で `left` / `right` を選び、省略時はデバイス名から推定する。
`arm_node.py` は `side` パラメータ（空なら `joint_names` の `joint_l*` / `joint_r*` から推定）を渡す。
値の根拠は [モーターの測定・検証記録](../docs/context/motor.md) の2026-10-04を参照。

```bash
cd upstream/astra_controller
git apply --reverse --check ../../patches/astra_controller-trail-club-ver1004.patch 2>/dev/null && echo applied
git apply --check ../../patches/astra_controller-trail-club-ver1004.patch
git apply ../../patches/astra_controller-trail-club-ver1004.patch
```

AstraArmController側の零点（URDFゼロ姿勢での初期化）とセットで使う。
旧来の可動域中心で初期化した機体に当てると、符号・可動域が実機と合わない。
