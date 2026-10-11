#!/usr/bin/env bash
# TensorFold（Qwen3.8-27B MLX 4-bit + DFlash2）のAPIサーバーを常時起動にする。使い方は ../README.md。
#   server.sh build | start | stop | restart | status | logs
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image=local/tensorfold:0.6.5
name=tensorfold-server

hf_dir="${HF_DIR:-/srv/shared/models/huggingface}"  # Hugging Faceのキャッシュ（共有）
model="${MODEL:-TensorFold/Qwen3.8-27B-MLX-4bit}"
parallel="${PARALLEL:-4}"                           # 同時に処理するリクエスト数
mem_gb="${MEM_GB:-70}"                              # GPUメモリの上限
addr="${ADDR:-10.99.0.1}"                           # WARP用アドレス。名簿のメンバーだけが届くのでAPIキーは使わない

exists() { docker ps -a --format '{{.Names}}' | grep -qx "$name"; }

case "${1:-status}" in
  build)
    docker build -t "$image" "$dir" ;;
  start)
    # 既存のコンテナは作成時の設定のまま起動する。設定を変えた場合は restart
    if exists; then
      docker start "$name"
    else
      docker run -d --name "$name" --restart unless-stopped --gpus all --ipc=host --network host \
        -v "$hf_dir:/root/.cache/huggingface" \
        -v local-llm-cache:/root/.cache/torch_extensions \
        -e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$mem_gb" -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1 \
        "$image" tensorfold serve "$model" --name qwen3.8-27b --host "$addr" --port 8080 --parallel "$parallel"
    fi ;;
  stop)
    if exists; then docker stop "$name" >/dev/null; fi ;;
  restart)
    docker rm -f "$name" >/dev/null 2>&1 || true
    "$0" start ;;
  status)
    docker ps -a --filter "name=^$name\$" --format '{{.Names}}: {{.Status}}'
    curl -sf -m 3 "http://$addr:8080/health" && echo || echo "not ready" ;;
  logs)
    docker logs -f "$name" ;;
  *)
    echo "usage: $0 build|start|stop|restart|status|logs" >&2
    exit 1 ;;
esac
