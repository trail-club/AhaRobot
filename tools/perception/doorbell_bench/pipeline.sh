#!/usr/bin/env bash
# The whole comparison without supervision: scoring (run_all.sh), evaluation with
# ensembles and CLAP gates (evaluate.py --auto), then inference times with the GPU
# otherwise idle. Start it detached so that a dropped SSH session does not stop it:
#   docker exec -d <container> bash /app/tools/perception/doorbell_bench/pipeline.sh
# Progress: $DOORBELL_RESULTS/pipeline.log (one line per stage, then the outputs).
set -uo pipefail
cd "$(dirname "$0")"
results="${DOORBELL_RESULTS:-/data/results}"
mkdir -p "$results"
log() { echo "== $(date '+%F %T') $*" >> "$results/pipeline.log"; }

log "scoring"
bash run_all.sh >> "$results/pipeline.log" 2>&1
while pgrep -f "python score.py" > /dev/null; do sleep 30; done  # jobs of another run_all.sh
log "evaluating"
python evaluate.py --auto --jobs 8 >> "$results/pipeline.log" 2>&1 || log "evaluate FAILED"
log "timing"
rm -f "$results/timing.jsonl"
for model in yamnet ced_mini ced_base effat_mn40 beats sslam eat_large \
  sed_mn10 sed_atstf sed_beats clap_fused clap_larger; do
  devices="cuda cpu"
  [ "$model" = yamnet ] && devices=cpu
  python timing.py --model "$model" --devices $devices --runs 30 \
    >> "$results/pipeline.log" 2>&1 || log "timing $model FAILED"
done
log "done"
