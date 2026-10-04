#!/usr/bin/env bash
# TensorFold（Qwen3.8-27B MLX 4-bit + DFlash2）のAPIサーバーを常時起動にする。使い方は ../README.md。
#   server.sh build | start | stop | restart | status | logs | keys | revoke <user> | unrevoke <user>
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image=local/tensorfold:0.6.5
server=tensorfold-server
keysync=tensorfold-keysync

hf_dir="${HF_DIR:-/srv/shared/models/huggingface}"  # Hugging Faceのキャッシュ（共有）
model="${MODEL:-TensorFold/Qwen3.8-27B-MLX-4bit}"
parallel="${PARALLEL:-4}"                           # 同時に処理するリクエスト数
mem_gb="${MEM_GB:-70}"                              # GPUメモリの上限
addr="${ADDR:-10.99.0.1}"                           # WARP用アドレス。名簿のメンバーだけが届く
auth="${AUTH:-none}"                                # none | keys（issue-key.sh のキーを必須にする）

usage() {
  echo "usage: $0 build|start|stop|restart|status|logs|keys|revoke <user>|unrevoke <user>" >&2
  exit 1
}

exists() { docker ps -a --format '{{.Names}}' | grep -qx "$1"; }

start_keysync() {
  if exists "$keysync"; then
    docker start "$keysync" >/dev/null
  else
    docker run -d --name "$keysync" --restart unless-stopped --network none \
      -v /home:/home:ro -v local-llm-keys:/out \
      --entrypoint bash "$image" /usr/local/bin/keysync.sh >/dev/null
  fi
  until docker exec "$keysync" test -s /out/keys.txt; do sleep 1; done
}

run_server() {
  local key_mount=() key_args=()
  if [[ $auth == keys ]]; then
    key_mount=(-v local-llm-keys:/keys:ro)
    key_args=(--api-key-file /keys/keys.txt)
  fi
  docker run -d --name "$server" --restart unless-stopped --gpus all --ipc=host --network host \
    -v "$hf_dir:/root/.cache/huggingface" \
    -v local-llm-cache:/root/.cache/torch_extensions \
    "${key_mount[@]}" \
    -e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$mem_gb" -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1 \
    "$image" tensorfold serve "$model" --name qwen3.8-27b --host "$addr" --port 8080 \
    --parallel "$parallel" "${key_args[@]}"
}

# keysyncコンテナの中でスクリプトを実行する。引数は $1, $2, ... として渡す
in_keysync() {
  exists "$keysync" || { echo "APIキーは無効（AUTH=keys で restart すると有効になる）" >&2; exit 1; }
  docker exec "$keysync" sh -c "$1" sh "${@:2}"
}

[[ $auth == none || $auth == keys ]] || { echo "AUTH must be none or keys" >&2; exit 1; }

case "${1:-status}" in
  build)
    docker build -t "$image" "$dir" ;;
  start)
    # 既存のコンテナは作成時の設定のまま起動する。設定を変えた場合は restart
    if exists "$server"; then
      if exists "$keysync"; then docker start "$keysync" >/dev/null; fi
      docker start "$server"
    else
      if [[ $auth == keys ]]; then start_keysync; fi
      run_server
    fi ;;
  stop)
    for c in "$server" "$keysync"; do
      if exists "$c"; then docker stop "$c" >/dev/null; fi
    done ;;
  restart)
    docker rm -f "$server" "$keysync" >/dev/null 2>&1 || true
    "$0" start ;;
  status)
    docker ps -a --filter "name=^$server\$" --filter "name=^$keysync\$" --format '{{.Names}}: {{.Status}}'
    if exists "$server"; then
      if docker inspect -f '{{json .Args}}' "$server" | grep -q -- --api-key-file; then
        echo "auth: keys"
      else
        echo "auth: none"
      fi
    fi
    curl -sf -m 3 "http://$addr:8080/health" && echo || echo "not ready" ;;
  logs)
    docker logs -f "$server" ;;
  keys)  # キーが有効なユーザーと、revoke したユーザー
    in_keysync 'grep -v "^#" /out/keys.txt | cut -d: -f1; sed "s/^/revoked: /" /out/revoked.txt' ;;
  revoke | unrevoke)
    [[ ${2:-} =~ ^[a-z0-9._-]+$ ]] || usage
    # shellcheck disable=SC2016  # $1 はコンテナの中で展開する
    if [[ $1 == revoke ]]; then
      in_keysync 'grep -qxF "$1" /out/revoked.txt || echo "$1" >> /out/revoked.txt' "$2"
    else
      in_keysync 'grep -vxF "$1" /out/revoked.txt > /out/.revoked; mv /out/.revoked /out/revoked.txt' "$2"
    fi ;;
  *)
    usage ;;
esac
