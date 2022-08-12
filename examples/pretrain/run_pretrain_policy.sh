#!/bin/sh

set -eux

TOP_DIR="$(cd $(dirname "$0")/../..; pwd)"
# export TOP_DIR=${1:-~/Garlic}
echo "TOP_DIR: $TOP_DIR"

PYTHON=${PYTHON:-'python3.8 -u'}

OUTDIR=results
DATASET_DIR=results
DATASET=$DATASET_DIR/worker
VOCAB=$DATASET_DIR/vocab.csv

mkdir -p $OUTDIR

$PYTHON $TOP_DIR/scripts/rl/pretrain_policy.py \
    --gpu 0 \
    -v $VOCAB \
    --input_pkl $DATASET \
    --epoch 2 \
    --snap_freq 500 \
    --snap_name "final_snap.pt" \
    --resume "$OUTDIR/final_snap.pt" \
    --save_best_snapshot \
    --log_interval 100 \
    --log_interval_unit iteration \
    --batch_size 1 \
    --hidden_size 8 \
    --num_workers=1 \
    --n_total=22 \
    -o $OUTDIR \
