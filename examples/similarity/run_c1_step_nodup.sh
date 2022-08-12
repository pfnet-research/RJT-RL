#!/bin/sh
# C1: Step reward/no dup penalty

set -eux

PYTHON=${PYTHON:-'python3.8 -u'}
export SCRIPT_DIR="$(cd $(dirname "$0"); pwd)"
TOP_DIR=$SCRIPT_DIR/../../
echo "TOP_DIR: $TOP_DIR"
CONFIG_YAML=$SCRIPT_DIR/config_c1.yaml
DATASET_DIR=$TOP_DIR/data
OUTDIR=$TOP_DIR/results/c1/

mkdir -p $OUTDIR
export OUTDIR
export DATASET_DIR

$PYTHON $TOP_DIR/scripts/rl/train_ppo.py \
    yaml=$CONFIG_YAML \
    trainer.reward.use_final_reward=false \
    > $OUTDIR/run.log 2>&1
