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
