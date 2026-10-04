#!/bin/bash
# DGX Spark上で自分用のQwen APIキーを発行して表示する。既にあれば同じキーを表示する。
#   手元のPCから: ssh <ユーザー名>@10.99.0.1 'bash -s' < tools/local-llm/issue-key.sh
#   作り直す:     ssh <ユーザー名>@10.99.0.1 'bash -s -- --rotate' < tools/local-llm/issue-key.sh
#   無効にする:   ssh <ユーザー名>@10.99.0.1 'bash -s -- --revoke' < tools/local-llm/issue-key.sh
set -euo pipefail
f="$HOME/.config/dgx-qwen/api_key"

case "${1:-}" in
  --revoke) rm -f "$f"; echo "revoked. 10秒ほどで使えなくなる" >&2; exit 0 ;;
  --rotate) rm -f "$f" ;;
  "") ;;
  *) echo "usage: issue-key.sh [--rotate|--revoke]" >&2; exit 1 ;;
esac

if [ ! -s "$f" ]; then
  mkdir -p "$(dirname "$f")"
  (umask 077 && echo "tf-$(openssl rand -hex 24)" > "$f")
  echo "issued. 10秒ほどで使えるようになる" >&2
fi
chmod 600 "$f"
cat "$f"
