#!/bin/bash
# TensorFold（Qwen3.8-27B MLX 4-bit + DFlash2）を常時起動のサーバーとして動かす。
#   server.sh [build|start|stop|restart|status|logs]
# このノードのWARP用アドレス 10.99.0.1:8080 で待ち受ける（Cloudflare Gatewayで名簿のメンバーだけが届く）。
# APIキーは各ユーザーが issue-key.sh で ~/.config/dgx-qwen/api_key に発行し、keysync.sh が集める。
# データは $LOCAL_LLM_DATA（既定 ~/local-llm）: keys/keys.txt（生成）、keys/keys.static（手動のキー）、
# keys/revoked.txt（無効にするユーザー名）、torch_extensions/（初回起動時のビルド結果）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="${LOCAL_LLM_DATA:-$HOME/local-llm}"
NAME=tensorfold-server
SYNC=tensorfold-keysync
IMAGE=local/tensorfold:0.6.5
MODEL="${MODEL:-TensorFold/Qwen3.8-27B-MLX-4bit}"
PARALLEL="${PARALLEL:-4}"
MEM_GB="${MEM_GB:-70}"
HOST="${HOST:-10.99.0.1}"

start_keysync() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$SYNC"; then docker start "$SYNC" >/dev/null; return; fi
  mkdir -p "$DATA/keys" && chmod 700 "$DATA/keys"
  docker run -d --name "$SYNC" --restart unless-stopped --network none \
    -v /home:/home:ro -v "$DATA/keys:/out" -v "$DIR/keysync.sh:/keysync.sh:ro" \
    -e OUT_UID="$(id -u)" -e OUT_GID="$(id -g)" \
    --entrypoint bash "$IMAGE" /keysync.sh >/dev/null
}

case "${1:-status}" in
  build)
    docker build -t "$IMAGE" "$DIR" ;;
  start)
    start_keysync
    until [ -s "$DATA/keys/keys.txt" ]; do sleep 1; done
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then docker start "$NAME"; exit 0; fi
    mkdir -p "$DATA/torch_extensions"
    docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc=host --network host \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$DATA/torch_extensions:/root/.cache/torch_extensions" \
      -v "$DATA/keys:/keys:ro" \
      -e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$MEM_GB" -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1 \
      "$IMAGE" tensorfold serve "$MODEL" --name qwen3.8-27b --host "$HOST" --port 8080 \
        --parallel "$PARALLEL" --api-key-file /keys/keys.txt ;;
  stop) docker stop "$NAME" "$SYNC" ;;
  restart) docker rm -f "$NAME" "$SYNC" >/dev/null 2>&1 || true; "$0" start ;;
  status)
    docker ps -a --filter "name=^${NAME}\$" --filter "name=^${SYNC}\$" --format '{{.Names}}: {{.Status}}'
    echo "keys: $(grep -vc '^#' "$DATA/keys/keys.txt" 2>/dev/null || echo 0)"
    curl -sf -m 3 "http://$HOST:8080/health" && echo || echo "not ready" ;;
  logs) docker logs -f "$NAME" ;;
  *) echo "usage: $0 [build|start|stop|restart|status|logs]"; exit 1 ;;
esac
