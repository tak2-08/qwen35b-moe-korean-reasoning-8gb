#!/bin/bash
# c2 training: 5 chunks of ~460k tokens, each: build cache -> capture activations (with the previous chunk's adapter
# applied) -> base-model targets for background rows -> 1 epoch -> delete the capture. Then real-engine checks.
# Environment: PY (python), HM_DATA, EXAM_DATA, RESULTS, MODEL, ENGINE (see README).
set -e -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKB=2400
PY=${PY:-python}; S=$(cd "$(dirname "$0")" && pwd)
HM_DATA=${HM_DATA:-data/hm}; EXAM_DATA=${EXAM_DATA:-data/exam}; RESULTS=${RESULTS:-results}
export HM_DATA EXAM_DATA RESULTS
W=surr_c2                          # capture folder (relative to HM_DATA)
log() { echo "[$(date +%H:%M:%S)] $*"; }
TRAIN="--ans_only --bs 4 --eval_train 96 --rank 16 --wd_bg 12 --bg_len 128 --out $RESULTS/exam_runs.jsonl"
NCH=${NCH:-5}; FROM=${FROM:-0}
cd "$S"
for k in $(seq $FROM $((NCH-1))); do
  log "chunk $k: build cache"
  CHUNK=$k BUDGET=460000 MAXLEN=2000 $PY build_c2_cache.py | tail -2
  rm -rf "$HM_DATA/$W"; mkdir -p "$HM_DATA/$W"
  if [ $k -eq 0 ]; then TAG=base; LORA=""; INIT=""; LR=3e-4
  else TAG=c2_$((k-1)); LORA=$EXAM_DATA/c2_$((k-1)).gguf; INIT="--init $EXAM_DATA/c2_$((k-1)).pt"; LR=2.5e-4; fi
  log "chunk $k: capture (tag $TAG)"
  CACHE=$EXAM_DATA/cache_c2_$k.npz SURR=$W TAG=$TAG LORA=$LORA NBG=90 $PY extract_surr.py | tail -1
  if [ $k -gt 0 ]; then   # KL targets: the unedited model on the background rows
    CACHE=$EXAM_DATA/cache_c2_$k.npz SURR=$W $PY make_hbase.py | tail -1
  fi
  log "chunk $k: train 1 epoch lr $LR"
  CACHE=$EXAM_DATA/cache_c2_$k.npz SURR=$W $PY train_surr.py $TRAIN --epochs 1 --lr $LR --tag $TAG $INIT \
      --save $EXAM_DATA/c2_$k.pt --export $EXAM_DATA/c2_$k.gguf | grep -E '"stage"' | tail -2
  rm -rf "$HM_DATA/$W"
done
log "real-engine background agreement (gate >= 0.92) for every chunk's adapter"
for k in $(seq 0 $((NCH-1))); do $PY eval_mid.py $EXAM_DATA/c2_$k.gguf 1 exam_c2_$k | tail -1; done
log "exam evaluation (direct answer + step-by-step with 2000-token answers)"
for k in 2 $((NCH-1)); do COTMAX=2000 MODES=direct,cot $PY eval_exam.py $EXAM_DATA/c2_$k.gguf c2_$k 300; done
log "done"
