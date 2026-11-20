#!/usr/bin/env bash
# Revision 2: 10-fold cross-validation, one seed per fold, all eight models (results/k10_<dir>_<fold>_<model>.json).
cd "$(dirname "$0")"
GPUS=(${GPUS:-2 5}); JOBS=${JOBS:-3}
MODELS=(${MODELS:-lam signrev rave_noanchor rave oracle rave_sr rave_norev rave_nodrift})
export EXTRA_ARGS="--nfold 10" LOGTAG="k10_"
mkdir -p logs results
i=0
for dir in I R; do for fold in $(seq 0 9); do for model in "${MODELS[@]}"; do
  [ -f "results/k10_${dir}_${fold}_${model}.json" ] && continue
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  echo "bash run_one.sh $gpu $dir $fold $model 0"
done; done; done | xargs -P "$JOBS" -I{} bash -c "{}"
