# 開発ルール

AhaRobotはRoboCup@Home向けのROS 2スタック。
独自コードは `overlay_ws/`、外部由来コード・資源は `upstream/` のsubmoduleで管理する。

## 環境と実装方針

- [開発コンテナ](docs/docker.md)を使う。
- `upstream/*` の改変はfork側でPRを作り、このリポジトリではsubmoduleのSHAを更新する。[手順](docs/upstream-workflow.md)
- AhaRobot固有のlaunch・設定・機能は `overlay_ws/` に実装する。
- topic・frame・msgを変更する場合は利用側への影響を確認し、[ROSインターフェース](docs/interfaces.md)も更新する。

## 資料の参照と記載

変更対象に応じて参照する。

| 作業 | 参照先 |
| --- | --- |
| 環境・コンテナ操作 | [docs/docker.md](docs/docker.md) |
| ROSパッケージ・制限 | [overlay_ws/README.md](overlay_ws/README.md)、対象パッケージのREADME |
| シミュレーション起動 | [docs/sobits-rcjo2026.md](docs/sobits-rcjo2026.md) |
| 実機の測定・検証記録 | [docs/context/README.md](docs/context/README.md) |

- READMEと `docs/` には、現在の実装・設定・操作手順を簡潔に書く。同じ情報を複数の文書へ重複させない。
- 実機の測定・検証は `docs/context/` に対象別に各資料に日付順で記録する。背景・条件・過程・問題・決定事項と根拠・確認範囲を残し、不明点を明記する。現在の仕様や動作保証と区別し、シミュレーション・実装履歴・通常のテスト結果は含めない。
- ツールの操作手順は各 `tools/*/README.md` に置き、機体固有の観測結果は補助資料へリンクする。
- 暫定の設計方針、未実装の計画、チーム間の分担・連携ルールは開発ドキュメントに追加しない。Issue / PRテンプレートには、変更ごとの影響チームと動作確認の項目を残す。

## 作業の進め方

- 実装・必要な検証・変更に起因する不具合の修正まで進める。ローカルの編集、ビルド、テスト、シミュレーションは各段階で確認を求めずに行う。
- 実機への動作指令、校正値・ファームウェアの書き込みは依頼に含まれる場合に行う。
- 依頼の対象外の未コミット変更を保持する。不足情報があっても進められる部分は進め、結果を左右する場合に確認する。
- 変更内容を既存のドキュメントへ反映し、完了時は変更・検証結果・未確認事項を簡潔に報告する。

## 動作確認

作業中の検証は変更の影響範囲に合わせる。
PR前に、起動済み開発コンテナ、pre-commit、Python 3.11を用意し、ホストで実行する。

```bash
make test
```

lint / format、perception単体テスト、ROSビルドとAhaRobot側のテストを実行する。
