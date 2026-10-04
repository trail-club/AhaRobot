#!/usr/bin/env bash
# サーバーを AUTH=keys で動かすとき、DGX Spark上に自分用のAPIキーを発行して表示する。既にあれば同じキーを表示する。
#   手元のPCから: ssh <ユーザー名>@10.99.0.1 'bash -s' < tools/local-llm/issue-key.sh
#   作り直す:     ssh <ユーザー名>@10.99.0.1 'bash -s -- --rotate' < tools/local-llm/issue-key.sh
#   無効にする:   ssh <ユーザー名>@10.99.0.1 'bash -s -- --revoke' < tools/local-llm/issue-key.sh
set -euo pipefail
key_file="$HOME/.config/dgx-qwen/api_key"

case "${1:-}" in
  "") ;;
  --rotate) rm -f "$key_file" ;;
  --revoke) rm -f "$key_file"; echo "revoked. 10秒ほどで使えなくなる" >&2; exit 0 ;;
  *) echo "usage: issue-key.sh [--rotate|--revoke]" >&2; exit 1 ;;
esac

if [[ ! -s $key_file ]]; then
  (umask 077 && mkdir -p "$(dirname "$key_file")" && echo "tf-$(openssl rand -hex 24)" > "$key_file")
  echo "issued. 10秒ほどで使えるようになる" >&2
fi
chmod 600 "$key_file"  # keysync.sh は本人だけが読めるファイルしか使わない
cat "$key_file"
