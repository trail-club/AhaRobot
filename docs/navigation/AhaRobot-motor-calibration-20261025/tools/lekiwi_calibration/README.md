# LeKiwi機体1の追加評価ツール

操作すると実機が動く。対応するROS 2環境と `lekiwi_base_bringup` が必要。
AhaRobotのアーム・頭部サーボへは使用しない。

[評価の定義](../../docs/motor-calibration.md)と
[測定記録](../../docs/context/motor.md#2026-10-25--lekiwi機体1のオドメトリと横移動の校正)を参照。
上流コードの基準は `lekiwi_challenge_2026` の `567f1b9`。

`near_wall_capture.py` は近距離壁の静止点群取得とエンコーダ目標による停止、
`near_wall_matching.py` は双方向ICP検査、
`final_motion_evaluation.py` は最終設定での追加評価を行う。

## 配置・実行

機体1のコンテナ内 `/maps` に3ファイルを置く。これは当日の測定用配置であり、
AhaRobotの通常launchへ自動で組み込まれる機能ではない。
LiDAR・TF・エンコーダドライバを機体1の校正設定で起動してから、次を実行する。

```bash
docker exec -it robot-1 /entrypoint.sh python3 /maps/final_motion_evaluation.py
```

既定の動作は、前後0.03 m/sを各1秒、前後3 cmの目標移動、
左右360°の目標回転、左右0.2 rad/sを各2秒。
各区間で停止・静止スキャン取得を行う。周囲の余裕と操作者の見守りが必要。
前後進は観測した±30°方向の点が12個以上、車体中心から壁まで0.275 m以上、
回転は点群の最小半径0.235 m以上を要求する。
これらのしきい値だけでケーブル・人・動く障害物の安全は確認できない。

結果は `/maps/odom_final_evaluation_<Unix秒>/` の点群と `report.json` に保存する。
指令履歴、エンコーダ軌跡、各区間の時間・終了条件、照合値または不成立理由を残す。
一定時間の試験はオドメトリの距離到達で終了しない。
速度の指令積分量は要求時間から計算し、送信開始からゼロ指令の反復終了までの実時間・最初から最後の非ゼロ指令の配信時間も併記する。
開始前のLiDAR欠測でもreportを残し、走行成功と区別する。

終了後はドライバのlaunchをCtrl+Cで正常終了し、トルクOFFと目標速度0を確認する。
設定値を自動的に変更するツールではない。

## オフライン確認

実機を接続せず、ROSをimportせずに一定時間制御の終了条件と異常時の停止を確認する。

```bash
python3 -m unittest discover -s tools/lekiwi_calibration -p 'test_*.py'
```

2026-10-25に3件通過。実機での確認結果は検証記録を参照。
