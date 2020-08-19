#!/usr/bin/env bash
# test
set -x

# commonly used opts:
# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
CKPT=$3
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=$2 python tools/main.py --config-file $CFG --num-gpus $NGPU --eval-only \
    MODEL.WEIGHTS $CKPT \
    ${@:4}
