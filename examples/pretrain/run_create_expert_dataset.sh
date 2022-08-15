#!/bin/sh

TOP_DIR="$(cd $(dirname "$0")/../..; pwd)"
PYTHON=${PYTHON:-'python3.8 -u'}
DATASET_DIR=results
VOCAB=$DATASET_DIR/vocab.csv
OUTPUT_DIR=results

mkdir -p $OUTPUT_DIR
$PYTHON $TOP_DIR/scripts/rl/create_expert_dataset.py \
    -v $VOCAB \
    -i $DATASET_DIR/train \
    --out_dir $OUTPUT_DIR \
    --prefix worker \
    --batch_size 128 \
    --num_workers 16
