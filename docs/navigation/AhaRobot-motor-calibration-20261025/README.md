# LeKiwiキャリブレーション・追加評価資料

AhaRobot作業ツリーへ作成した資料・測定データ・評価ツール。
本ZIPの docs/ と tools/ はAhaRobotの同名パスに対応する。

- [評価方法](docs/motor-calibration.md)
- [測定記録の2026-10-25項](docs/context/motor.md#2026-10-25--lekiwi機体1のオドメトリと横移動の校正)
- [追加評価の一覧](docs/context/data/lekiwi-20261025/final-evaluation.csv)
- [追加評価ツール](tools/lekiwi_calibration/README.md)

元リポジトリの既存モーター記録を保持しているため、以前の日付の節にはこのZIPに含まれない資料へのリンクがある。
追加したコードの構文、付属オフラインテスト3件、今回追加したMarkdownのローカルリンク、git diff --checkを確認。既存記録にある未取得submoduleへのリンク2件は未解決。
AhaRobot全体のmake testは未実施。コミット・push・PR作成はしていない。

最終設定で前後短距離、左右一周、一定時間走行を追加評価した。
左一周で約13.2°の残差があり、精度改善がすべて完了したという判定ではない。
物理動画は操作者が撮影すると回答したが、こちらではファイル未受領のため含まない。
