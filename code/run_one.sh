#!/usr/bin/env bash
# Run one RAVE job within dale's GPU quota (at most 10 GPU processes per account; over it, the newest are terminated).
# Usage: run_one.sh <gpu> <dir> <fold> <model>
cd "$(dirname "$0")"
PY=${PY:-/home/pemaase/miniconda3/envs/RoboTwin/bin/python}
MAX_ACCOUNT_GPU_PROCS=${MAX_ACCOUNT_GPU_PROCS:-8}     # stay 2 under the quota of 10
gpu=$1; dir=$2; fold=$3; model=$4; name="${dir}_${fold}_${model}"

account_gpu_procs() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read -r pid; do ps -o user= -p "$pid" 2>/dev/null; done | grep -c "^$USER$"
}

for attempt in 1 2 3; do
  sleep $((RANDOM % 30))                               # stagger launches so parallel checks do not race
  while [ "$(account_gpu_procs)" -ge "$MAX_ACCOUNT_GPU_PROCS" ]; do sleep 30; done
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY rave.py --train "$dir" --fold "$fold" --model "$model" > "logs/$name.log" 2>&1
  rc=$?
  [ $rc -eq 143 ] && [ $(( $(date +%s) - start )) -lt 600 ] || exit $rc
  echo "$(date +%T) $name stopped by quota (exit 143), retry $attempt" >> logs/quota_retries.log
  sleep 60
done
