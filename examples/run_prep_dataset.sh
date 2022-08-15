#!/bin/sh

TOP_DIR="$(cd $(dirname "$0")/..; pwd)"
PYTHON=${PYTHON:-'python3.8 -u'}
DATASET_CSV=data/zinc250k.csv
OUTPUT_DIR=results
VOCAB=$OUTPUT_DIR/vocab.csv

mkdir -p $OUTPUT_DIR
$PYTHON $TOP_DIR/scripts/prep_dataset.py \
    --split 16 \
    -i $DATASET_CSV \
    --column smiles \
    -o $OUTPUT_DIR/train \
    -v $VOCAB
