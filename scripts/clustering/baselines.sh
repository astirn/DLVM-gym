#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Clustering baseline experiments." "$@"

SEED="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# Standard normal prior
PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params=standard_normal \
    loggers.wandb.name='${model.prior_params.name}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v3"] \
    seed=$SEED

# VMM (Stirn & Knowles, 2025)
NUM_DATA="64,256,1024,4096,16384,???"  # Stirn & Knowles (2025) artificially set this to scale influence of prior
U_INITIAL_STRATEGY="data,stratified_data,random"
PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    model.prior_params.num_data=$NUM_DATA \
    model.u_initial_strategy=$U_INITIAL_STRATEGY \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params=crypt/vmm \
    loggers.wandb.name='${model.prior_params.name}-${model.u_initial_strategy}-${model.prior_params.num_data}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v3"] \
    seed=$SEED
