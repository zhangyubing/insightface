#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

#DATA_DIR=/opt/jiaguo/faces_vgg_112x112
DATA_DIR=../datasets/faces_emore

#NETWORK=r50
NETWORK=r100
#JOB=softmax1e3
JOB=arcface_faces_emore
MODELDIR="../models/model-$NETWORK-$JOB"
LR_STEPS='100000,140000,160000,200000'
TEST_TARGET='lfw,cvte_baby9000'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 128 > "$LOGFILE" 2>&1 &

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network $NETWORK --loss-type 4 \
--margin-m 0.5  --prefix "$PREFIX" \
--per-batch-size 64 --lr-step "$LR_STEPS"  --data-dir $DATA_DIR \
--target "$TEST_TARGET"  > "$LOGFILE" 2>&1 &

tail -f "$LOGFILE"
