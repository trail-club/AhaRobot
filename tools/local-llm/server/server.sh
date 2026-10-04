#!/bin/bash
# TensorFold（Qwen3.8-27B MLX 4-bit + DFlash2）を常時起動のサーバーとして動かす。dockerグループの誰でも操作できる。
#   server.sh build | start | stop | restart | status | logs | keys | revoke <user> | unrevoke <user>
# 10.99.0.1:8080 で待ち受ける。届くのはCloudflare Gatewayが通した名簿（directory-access の roster.yml）のメンバーだけ。
# 既定はAPIキーなし。AUTH=keys で起動すると、各ユーザーが issue-key.sh で発行したキーを必須にする。
# モデルは共有のHugging Faceキャッシュ（$HF_DIR）、キーとビルド結果はDockerボリュームに置き、個人のホームを使わない。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NAME=tensorfold-server
SYNC=tensorfold-keysync
IMAGE=local/tensorfold:0.6.5
HF_DIR="${HF_DIR:-/srv/shared/models/huggingface}"
MODEL="${MODEL:-TensorFold/Qwen3.8-27B-MLX-4bit}"
PARALLEL="${PARALLEL:-4}"
MEM_GB="${MEM_GB:-70}"
HOST="${HOST:-10.99.0.1}"
AUTH="${AUTH:-none}"

exists() { docker ps -a --format '{{.Names}}' | grep -qx "$1"; }
user_arg() { [[ "${1:-}" =~ ^[a-z0-9._-]+$ ]] || { echo "usage: $0 revoke|unrevoke <unix user>" >&2; exit 1; }; }
need_sync() { exists "$SYNC" || { echo "APIキーは無効（AUTH=keys で restart すると有効になる）" >&2; exit 1; }; }

case "${1:-status}" in
  build)
    docker build -t "$IMAGE" "$DIR" ;;
  start)
    key_args=()
    serve_args=()
    if [ "$AUTH" = keys ]; then
      if exists "$SYNC"; then docker start "$SYNC" >/dev/null; else
        docker run -d --name "$SYNC" --restart unless-stopped --network none \
          -v /home:/home:ro -v local-llm-keys:/out --entrypoint bash "$IMAGE" /usr/local/bin/keysync.sh >/dev/null
      fi
      until docker exec "$SYNC" test -s /out/keys.txt; do sleep 1; done
      key_args=(-v local-llm-keys:/keys:ro)
      serve_args=(--api-key-file /keys/keys.txt)
    fi
    if exists "$NAME"; then docker start "$NAME"; exit 0; fi
    docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc=host --network host \
      -v "$HF_DIR:/root/.cache/huggingface" \
      -v local-llm-cache:/root/.cache/torch_extensions \
      "${key_args[@]}" \
      -e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$MEM_GB" -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1 \
      "$IMAGE" tensorfold serve "$MODEL" --name qwen3.8-27b --host "$HOST" --port 8080 --parallel "$PARALLEL" \
        "${serve_args[@]}" ;;
  stop)
    for c in "$NAME" "$SYNC"; do exists "$c" && docker stop "$c" >/dev/null; done; true ;;
  restart)
    docker rm -f "$NAME" "$SYNC" >/dev/null 2>&1 || true; "$0" start ;;
  status)
    docker ps -a --filter "name=^${NAME}\$" --filter "name=^${SYNC}\$" --format '{{.Names}}: {{.Status}}'
    exists "$SYNC" && echo "auth: keys" || echo "auth: none"
    curl -sf -m 3 "http://$HOST:8080/health" && echo || echo "not ready" ;;
  logs) docker logs -f "$NAME" ;;
  keys)  # キーが有効なユーザー名と、無効にしたユーザー名（AUTH=keys のとき）
    need_sync
    docker exec "$SYNC" sh -c 'grep -v "^#" /out/keys.txt | cut -d: -f1; sed "s/^/revoked: /" /out/revoked.txt' ;;
  revoke)
    need_sync; user_arg "${2:-}"
    docker exec "$SYNC" sh -c "grep -qxF '$2' /out/revoked.txt || echo '$2' >> /out/revoked.txt" ;;
  unrevoke)
    need_sync; user_arg "${2:-}"
    docker exec "$SYNC" sh -c "grep -vxF '$2' /out/revoked.txt > /out/.r; mv /out/.r /out/revoked.txt" ;;
  *) echo "usage: $0 build|start|stop|restart|status|logs|keys|revoke <user>|unrevoke <user>"; exit 1 ;;
esac
