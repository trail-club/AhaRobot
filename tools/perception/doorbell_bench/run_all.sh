#!/usr/bin/env bash
# Score every model and window of the comparison. GPU workers take jobs from one
# queue in order (a job is claimed by creating claims/<job>); finished and claimed
# jobs are skipped, so the script can be restarted. YAMNet runs on the CPU.
# Logs: $DOORBELL_RESULTS/logs.
set -uo pipefail
cd "$(dirname "$0")"
results="${DOORBELL_RESULTS:-/data/results}"
mkdir -p "$results/logs" "$results/claims"
gpu_workers="${GPU_WORKERS:-2}"

# "<model> <window s | x for 10 s frame models> [score.py options]". SSLAM and EAT
# pad every window to 10 s, so only 2 s windows are scored; EAT-large (slowest)
# leaves out the hard negatives.
JOBS=(
  "ced_mini 2" "ced_base 2" "sslam 2" "beats 2" "effat_mn40 2" "sed_mn10 x"
  "clap_fused 2" "clap_larger 2" "sed_atstf x" "sed_beats x"
  "ced_mini 1" "ced_base 1" "beats 1" "effat_mn40 1" "clap_fused 1" "clap_larger 1"
  "eat_large 2 --skip-hard"
)

run_job() {
  local model=$1 window=$2 out
  shift 2
  case "$model" in
    sed_*) out="$results/scores/$model/frames.npz" ;;
    yamnet) out="$results/scores/$model/patch.npz" ;;
    *) out="$results/scores/$model/w${window}.npz" ;;
  esac
  [ -f "$out" ] && return
  mkdir "$results/claims/${model}_w${window}" 2>/dev/null || return
  [ "$window" = x ] && window=2
  python score.py --model "$model" --window "$window" --batch 128 --workers 6 "$@" \
    > "$results/logs/${model}_w${window}.log" 2>&1 || echo "FAILED $model $window"
}

worker() {
  for job in "${JOBS[@]}"; do
    # shellcheck disable=SC2086
    run_job $job
  done
}

for _ in $(seq "$gpu_workers"); do
  worker &
  sleep 1
done
run_job yamnet x &
wait
echo DONE
