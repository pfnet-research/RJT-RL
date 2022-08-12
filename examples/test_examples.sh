#!/bin/sh

set -eux

TOP_DIR="$(cd $(dirname "$0")/..; pwd)"
echo "TOP_DIR: $TOP_DIR"

EXAMPLES_DIR=$TOP_DIR/examples
OUTDIR=results

# Prepare mols (csv --> mol pkl)
function prep_mols () {
    rm -f $OUTDIR/train_000.pkl
    rm -f $OUTDIR/vocab.csv
    bash $EXAMPLES_DIR/run_prep_dataset_tiny.sh
    if [ ! -f $OUTDIR/train_000.pkl ] || [ ! -f $OUTDIR/vocab.csv ]; then
        echo "output is not generated."
        ls -la $OUTDIR
        exit 1
    fi
}

# Prepare RL expert dataset (mol pkl --> worker pkl)
function prep_expert_dataset () {
    rm -f $OUTDIR/worker_0.pkl
    bash $EXAMPLES_DIR/pretrain/run_create_expert_dataset.sh
    if [ ! -f $OUTDIR/worker_0.pkl ]; then
        echo "output is not generated."
        ls -la $OUTDIR
        exit 1
    fi
}

# Run RL pretraining (--> results/best_model.pt)
function pretrain_policy () {
    rm -f $OUTDIR/best_model.pt
    bash $EXAMPLES_DIR/pretrain/run_pretrain_policy.sh
    if [ ! -f $OUTDIR/best_model.pt ]; then
        echo "output is not generated."
        ls -la $OUTDIR
        exit 1
    fi
}

# Run RL training (LogP reward)
function rl_plogp () {
    rm -f $OUTDIR/run_config.yaml
    rm -f $OUTDIR/300_finish/model.pt
    bash $EXAMPLES_DIR/penalized_logp/run_ppo_logp.sh
    if [ ! -f $OUTDIR/run_config.yaml ] || [ ! -f $OUTDIR/300_finish/model.pt ]; then
        echo "output is not generated."
        ls -la $OUTDIR
        exit 1
    fi
}

# Run RL training (Similarity reward)
function rl_similarity () {
    rm -f $OUTDIR/run_config.yaml
    rm -f $OUTDIR/300_finish/model.pt
    bash $EXAMPLES_DIR/similarity/run_ppo_sim.sh
    if [ ! -f $OUTDIR/run_config.yaml ] || [ ! -f $OUTDIR/300_finish/model.pt ]; then
        echo "output is not generated."
        ls -la $OUTDIR
        exit 1
    fi
}

prep_mols
prep_expert_dataset
pretrain_policy
rl_plogp
rl_similarity
