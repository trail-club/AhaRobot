# パッチ

`upstream/` のsubmoduleへ当てる実機用の変更。測定根拠と適用上の制約は
[モーターの測定・検証記録](../docs/context/motor.md)を参照。

| パッチ | 対象 | 状態 |
| --- | --- | --- |
| [AstraArmController-trail-club-hardware.patch](AstraArmController-trail-club-hardware.patch) | `upstream/AstraFirmwares/AstraArmController` | submodule `c5b2af4` に取り込み済み。適用不要 |
| [AstraArmController-trail-club-ver1004.patch](AstraArmController-trail-club-ver1004.patch) | `upstream/AstraFirmwares/AstraArmController` | `c5b2af4` に対して適用する |
| [astra_controller-trail-club-ver1004.patch](astra_controller-trail-club-ver1004.patch) | `upstream/astra_controller` | `9c1b96e` に対して適用する |

fork側に取り込まれ、AhaRobotのsubmodule SHAが更新された後は、そのパッチは不要になる。
submoduleの変更を共有する場合は [fork側のPRとSHA更新](../docs/upstream-workflow.md)を行う。

## AstraArmController用パッチ

### trail-club-hardware

対向取付の符号と単発サーボIDを表にした変更。取り込み済みのsubmoduleには、正方向にも逆方向にも当たらない。

| 設定 | 内容 |
| --- | --- |
| `JOINT_SERVO_SIGN` | サーボごとの取付符号。読み取りとPWM出力で使用 |
| `NONE_JOINT_ID` | 単発サーボのID。−1のスロットは読み書きを省略し、6スロットの通信形式を維持 |

### trail-club-ver1004

初期化（`setupTorque(128)`）を、エンコーダの0/4095境界をまたいでも正しく動くようにする変更。
`src/dualMotor.cpp` の初期化処理だけを変更し、符号表・ID表には触れない。
経緯は [左腕の初期化がID10で止まった問題](../docs/context/motor.md#左腕の初期化がid10で止まった問題)を参照。

| 変更 | 内容 |
| --- | --- |
| 範囲チェック | `20 < pos < 4076` → 読み取り失敗（負値）のみ検出 |
| 予圧の差 | 円周差（±2048に折り返し）で判定 |
| `init_pos` | 円周平均で計算（4090と10 → 2） |
| 異常時 | `while (1)` → 予圧を止めてID4–11を脱力し、`/config.txt` を書かずに戻る。再起動か再初期化まで制御ループを止める |

### 適用方法

入れ子submodule `upstream/AstraFirmwares/AstraArmController` の作業ツリーへ当てる。

```bash
cd upstream/AstraFirmwares/AstraArmController
git status --short   # 想定外の未コミット変更がないこと

# 適用済みかを確認（逆適用が通れば適用済み、何も出力しなければ未適用）
git apply --reverse --check ../../../patches/AstraArmController-trail-club-ver1004.patch 2>/dev/null && echo applied

# 未適用なら当てる（--check が通ってから実行）
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

取り消しは `git apply -R <パッチ>`。submoduleを切り替え・更新した後は、適用状態を確認し直す。

## astra_controller用パッチ

### trail-club-ver1004

上位機の `ArmController` に腕ごとの関節符号と可動域（`ARM_PROFILES`）を追加する変更。
`ArmController(name, side=...)` で `left` / `right` を選び、省略時はデバイス名から推定する。
`arm_node.py` は `side` パラメータ（空なら `joint_names` の `joint_l*` / `joint_r*` から推定）を渡す。
判別できない場合は警告を出し、upstream既定の符号・可動域で動く。
値の根拠は [上位機の符号と可動域](../docs/context/motor.md#上位機の符号と可動域)を参照。

```bash
cd upstream/astra_controller
git apply --reverse --check ../../patches/astra_controller-trail-club-ver1004.patch 2>/dev/null && echo applied
git apply --check ../../patches/astra_controller-trail-club-ver1004.patch
git apply ../../patches/astra_controller-trail-club-ver1004.patch
```

AstraArmController側の零点（URDFゼロ姿勢での初期化）とセットで使う。
旧来の可動域中心で初期化した機体に当てると、符号・可動域が実機と合わない。
