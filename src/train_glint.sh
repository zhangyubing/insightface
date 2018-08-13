#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=../datasets/glint

NETWORK=r100
# JOB=combined_margin_glint
# JOB=arcface_glint
JOB=amsoftmax_glint
MODELDIR="../models/model-$NETWORK-$JOB"
LR_STEPS='100000,140000,160000,200000'
TEST_TARGET='cvte_baby9000,lfw'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

# # combined margin
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network $NETWORK --loss-type 5 \
# --margin-a 0.9 --margin-m 0.4 --margin-b 0.15  --prefix "$PREFIX" \
# --per-batch-size 50 --lr-step "$LR_STEPS"  --data-dir $DATA_DIR \
# --target "$TEST_TARGET"  > "$LOGFILE" 2>&1 &

# # arcface
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network $NETWORK --loss-type 4 \
# --margin-m 0.5  --prefix "$PREFIX" \
# --per-batch-size 50 --lr-step "$LR_STEPS"  --data-dir $DATA_DIR \
# --target "$TEST_TARGET"  > "$LOGFILE" 2>&1 &

# amsoftmax
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network $NETWORK --loss-type 2 \
--margin-m 0.35  --prefix "$PREFIX" \
--per-batch-size 50 --lr-step "$LR_STEPS"  --data-dir $DATA_DIR \
--target "$TEST_TARGET"  > "$LOGFILE" 2>&1 &

tail -f "$LOGFILE"
