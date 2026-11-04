#!/usr/bin/env bash
# 5-fold cross-validation, both directions, all models. JOBS at a time (default 3), each quota-aware (run_one.sh).
cd "$(dirname "$0")"
GPUS=(${GPUS:-4 5})
JOBS=${JOBS:-3}
mkdir -p logs results
i=0
for dir in I R; do
  for fold in 0 1 2 3 4; do
    for model in lam signrev rave_noanchor rave oracle; do
      [ -f "results/${dir}_${fold}_${model}.json" ] && continue
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
      echo "bash run_one.sh $gpu $dir $fold $model"
    done
  done
done | xargs -P "$JOBS" -I{} bash -c "{}"
