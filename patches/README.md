# upstream 向けパッチ (暫定置き場)

`upstream/*` は CONTRIBUTING の方針どおり super repo からは触らない。
本来は fork 側に feature branch を切って PR を出す
([`docs/upstream-workflow.md`](../docs/upstream-workflow.md))。

ただし `AstraArmController` は `upstream/AstraFirmwares` の**入れ子 submodule**で、
現状 `hilookas/AstraArmController` を直接指しており、**`trail-club/AstraArmController`
の fork がまだ存在しない**。fork が用意できるまでの暫定として、実機に必要な差分を
パッチとしてここに置く。

| パッチ | 対象 | 内容 |
| --- | --- | --- |
| `AstraArmController-trail-club-hardware.patch` | `upstream/AstraFirmwares/AstraArmController` | 対向符号の表化 + ID14 欠番対応 |

## 適用

```bash
cd upstream/AstraFirmwares/AstraArmController
git apply ../../../patches/AstraArmController-trail-club-hardware.patch
pio run                 # ビルド確認
pio run -t upload       # 書き込み (upload_port を実機に合わせること)
```

## 内容

実機の実測にもとづく 2 点。根拠は [`docs/servo-bringup.md`](../docs/servo-bringup.md)。

1. **対向取付の符号を表に外出し** (`JOINT_SERVO_SIGN[]`)
   upstream は全関節に `{+1,-1,+1,-1}` を決め打ちしているが、実機は両関節とも
   `{+1,-1,-1,+1}`。読み取りの平均と PWM 出力の 2 箇所がこの表を参照する。
   符号が違うとサーボ同士が押し合って機構を壊すため、実機ごとに
   `tools/servo/teach_calibrate.py --verify` で実測して差し替える。

2. **単発関節の ID を表に外出し** (`NONE_JOINT_ID[]`)
   upstream は `12,13,14,15` の連番決め打ち。実機に ID14 は無いので `-1` を
   入れて読み書きをスキップする。スロット数は 6 のまま残すのでホスト側は無改造。

## fork 後にやること

`trail-club/AstraArmController` を作ったら、

1. `upstream/AstraFirmwares` の `.gitmodules` の URL を差し替え
2. fork 側に `feature/trail-club-hardware` を切ってこのパッチをコミット
3. このディレクトリのパッチを削除

1 の符号表化は upstream にも有用なので `hilookas/AstraArmController` への PR も検討する。
2 の ID14 欠番は本機体固有なので upstream には戻さない。
