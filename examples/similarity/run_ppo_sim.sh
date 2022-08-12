#!/bin/sh

set -eux

PYTHON=${PYTHON:-'python3.8 -u'}
TOP_DIR="$(cd $(dirname "$0")/../../; pwd)"
echo "TOP_DIR: $TOP_DIR"

export SCRIPT_DIR=$TOP_DIR/examples/similarity/
CONFIG_YAML=$SCRIPT_DIR/config_sim_tiny.yaml

export DATASET=$TOP_DIR/results/worker
export PT_MODEL=$TOP_DIR/results/best_model.pt
export VOCAB=$TOP_DIR/results/vocab.csv

export OUTDIR=results

mkdir -p $OUTDIR

$PYTHON $TOP_DIR/scripts/rl/train_ppo.py \
    yaml=$CONFIG_YAML \
