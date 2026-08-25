#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Clustering baseline experiments." "$@"

PRIOR="standard_normal"
SEED="1,2,3,4,5"

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params=$PRIOR \
    loggers.wandb.name='${model.prior_params.name}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v2"] \
    seed=$SEED
