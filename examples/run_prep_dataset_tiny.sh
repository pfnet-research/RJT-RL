#!/bin/sh

TOP_DIR="$(cd $(dirname "$0")/..; pwd)"
echo "TOP_DIR: $TOP_DIR"

PYTHON=${PYTHON:-'python3.8 -u'}

ASSETS_DIR=$TOP_DIR/tests/test_assets

DATASET_CSV=$ASSETS_DIR/tiny_dataset.csv

OUTPUT_DIR=results
VOCAB=$OUTPUT_DIR/vocab.csv

mkdir -p $OUTPUT_DIR
$PYTHON $TOP_DIR/scripts/prep_dataset.py \
    --split 1 \
    -i $DATASET_CSV \
    --column SMILES \
    -o $OUTPUT_DIR/train \
    -v $VOCAB
