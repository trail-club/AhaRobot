# ローカルLLM

DGX Spark上のQwen3.8-27Bを、各自のPCからAPIで使う。ClineなどのエージェントはPCで動き、推論だけをDGX Sparkが行う。

| 項目 | 値 |
| --- | --- |
| Base URL | `http://10.99.0.1:8080/v1`（OpenAI互換） |
| モデルID | `qwen3.8-27b` |
| 認証 | なし。Cloudflare One（WARP）経由で、trail-club/directory-access の名簿に載っている人だけが届く |

## 1. 接続確認

Cloudflare OneをConnectedにして（directory-access のREADMEを参照）、PCで実行する。

```bash
curl http://10.99.0.1:8080/v1/models
```

`qwen3.8-27b` が返れば接続できている。

## 2. エージェントの設定

Cline（VS Code拡張 `saoudrizwan.claude-dev`）では、API Providerを **OpenAI Compatible** にして次を入力する。

| 項目 | 値 |
| --- | --- |
| Base URL | `http://10.99.0.1:8080/v1` |
| API Key | 任意の文字列（例: `local`）。空欄にできないため |
| Model ID | `qwen3.8-27b` |
| Context Window | `131072` |

他のOpenAI互換クライアントも、Base URL・モデルID・任意のAPIキーで接続する。

## 注意

- 同時に処理するリクエストは4つまで。それ以上は順番待ちになる。
- 生成速度の目安（2026-10-04）: コード約100 tok/s、英語の文章約35〜50 tok/s、日本語の文章約20〜27 tok/s。
- キャッシュの効かない長い入力は読み込みに時間がかかる（約6万トークンで約50秒）。会話の続きはキャッシュが効く。
- `seed` を指定しない場合、同じ入力には同じ出力が返る。

## サーバーの管理

`server/` にTensorFoldのイメージと起動スクリプトがある。dockerグループのメンバーなら誰でも操作できる。

```bash
tools/local-llm/server/server.sh build     # イメージ local/tensorfold:0.6.5 を作る
tools/local-llm/server/server.sh start     # 起動（DGX Sparkの再起動後も自動で起動）
tools/local-llm/server/server.sh status
tools/local-llm/server/server.sh restart   # 設定を変えた後
```

| データ | 置き場所 |
| --- | --- |
| モデル | `/srv/shared/models/huggingface`（Hugging Faceのキャッシュ形式。`HF_DIR` で変更）。無い場合は初回起動時に取得する |
| カーネルのビルド結果 | Dockerボリューム `local-llm-cache` |
| APIキー（`AUTH=keys` のとき） | Dockerボリューム `local-llm-keys` |

- 待ち受けは `10.99.0.1:8080` だけで、DGX SparkのLANやlocalhostからは届かない。
- 速度とツール呼び出しは `python3 tools/local-llm/bench_decode.py` で確認する。
- モデル・同時処理数・待ち受けアドレスなどは環境変数で変える（`server.sh` の冒頭を参照）。

### 個人ごとのAPIキーを使う場合

`AUTH=keys` で起動し直すと、各ユーザーが発行したキーのないリクエストを拒否する。`server.sh restart` でキーなしに戻る。

```bash
AUTH=keys tools/local-llm/server/server.sh restart
# 各ユーザーが手元のPCで実行し、表示されたキーをAPI Keyに入れる（--rotate で作り直し、--revoke で無効化）
ssh <Unixユーザー名>@10.99.0.1 'bash -s' < tools/local-llm/issue-key.sh
tools/local-llm/server/server.sh keys                # キーが有効なユーザー
tools/local-llm/server/server.sh revoke <ユーザー名>  # 停止（unrevoke で戻す）
```

キーは各ユーザーの `~/.config/dgx-qwen/api_key` に置かれ、コンテナ `tensorfold-keysync`（`server/keysync.sh`）が
10秒ごとに集めてユーザー名をラベルにする。
