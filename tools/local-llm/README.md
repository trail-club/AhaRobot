# ローカルLLM

DGX Spark上のQwen3.8-27Bを、各自のPCからAPIで使う。ClineなどのエージェントはPCで動き、推論だけをDGX Sparkが行う。

| 項目 | 値 |
| --- | --- |
| Base URL | `http://10.99.0.1:8080/v1`（OpenAI互換） |
| モデルID | `qwen3.8-27b` |
| 認証 | 個人のAPIキー（`Authorization: Bearer <キー>`） |
| 接続経路 | Cloudflare One（WARP）。trail-club/directory-access の名簿に載っている人だけが届く |

## 1. APIキーの発行

Cloudflare OneをConnectedにし、このリポジトリ直下で実行する。DGX Sparkに自分用のキーを作って表示する。

```bash
ssh <Unixユーザー名>@10.99.0.1 'bash -s' < tools/local-llm/issue-key.sh
```

- 10秒ほどで使えるようになる。再実行すると同じキーを表示する。
- キーはDGX Sparkの `~/.config/dgx-qwen/api_key`（本人だけが読める）に保存される。共有やコミットはしない。
- 作り直す場合は `'bash -s -- --rotate'`、無効にする場合は `'bash -s -- --revoke'` を指定する。

## 2. 接続確認

```bash
curl -H "Authorization: Bearer <キー>" http://10.99.0.1:8080/v1/models
```

`qwen3.8-27b` が返れば接続できている。

## 3. エージェントの設定

Cline（VS Code拡張 `saoudrizwan.claude-dev`）では、API Providerを **OpenAI Compatible** にして次を入力する。

| 項目 | 値 |
| --- | --- |
| Base URL | `http://10.99.0.1:8080/v1` |
| API Key | 発行したキー |
| Model ID | `qwen3.8-27b` |
| Context Window | `131072` |

他のOpenAI互換クライアントも、Base URL・APIキー・モデルIDの3つで接続する。

## 注意

- 同時に処理するリクエストは4つまで。それ以上は順番待ちになる。
- 生成速度の目安（2026-10-04）: コード約100 tok/s、英語の文章約35〜50 tok/s、日本語の文章約20〜27 tok/s。
- キャッシュの効かない長い入力は読み込みに時間がかかる（約6万トークンで約50秒）。会話の続きはキャッシュが効く。
- `seed` を指定しない場合、同じ入力には同じ出力が返る。

## サーバーの管理

`server/` にTensorFoldのイメージと起動スクリプトがある。データは `~/local-llm`（`LOCAL_LLM_DATA` で変更）に置く。

```bash
tools/local-llm/server/server.sh build     # イメージ local/tensorfold:0.6.5 を作る
tools/local-llm/server/server.sh start     # サーバーとキー収集を起動（再起動後も自動で起動）
tools/local-llm/server/server.sh status
```

- `keysync.sh` が各ユーザーの `~/.config/dgx-qwen/api_key` を10秒ごとに集め、ユーザー名をラベルにして `keys/keys.txt` を作る。本人だけが読めるファイル以外とリンクは無視する。
- ユーザーのキーを止める場合は `~/local-llm/keys/revoked.txt` にユーザー名を書く。
- 速度とツール呼び出しは `OPENAI_API_KEY=<キー> python3 tools/local-llm/bench_decode.py http://10.99.0.1:8080/v1` で確認する。
