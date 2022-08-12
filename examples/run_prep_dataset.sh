#!/bin/sh

TOP_DIR="$(cd $(dirname "$0")/..; pwd)"
echo "TOP_DIR: $TOP_DIR"

PYTHON=${PYTHON:-'python3.8 -u'}

ASSETS_DIR=$TOP_DIR/tests/test_assets

DATASET_CSV=data/zinc250k.csv

OUTPUT_DIR=results
VOCAB=$OUTPUT_DIR/vocab.csv

mkdir -p $OUTPUT_DIR
$PYTHON $TOP_DIR/scripts/prep_dataset.py \
    --split 1 \
    -i $DATASET_CSV \
    --column smiles \
    -o $OUTPUT_DIR/train \
    -v $VOCAB
