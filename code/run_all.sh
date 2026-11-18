#!/usr/bin/env bash
# 5-fold cross-validation, both directions, all models and seeds. JOBS at a time (default 3), each quota-aware (run_one.sh).
# Revision 1: three Koopman ablations and several seeds. Seed 0 of the five original models already exists in results/
# and is skipped, so only the new runs are launched.  MODELS / SEEDS can be overridden from the environment.
cd "$(dirname "$0")"
GPUS=(${GPUS:-4 5})
JOBS=${JOBS:-3}
MODELS=(${MODELS:-lam signrev rave_noanchor rave oracle rave_sr rave_norev rave_nodrift})
SEEDS=(${SEEDS:-0 1 2})
mkdir -p logs results
i=0
for seed in "${SEEDS[@]}"; do
  tag=""; [ "$seed" != 0 ] && tag="_s${seed}"
  for dir in I R; do
    for fold in 0 1 2 3 4; do
      for model in "${MODELS[@]}"; do
        [ -f "results/${dir}_${fold}_${model}${tag}.json" ] && continue
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
        echo "bash run_one.sh $gpu $dir $fold $model $seed"
      done
    done
  done
done | xargs -P "$JOBS" -I{} bash -c "{}"
