#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Sweep DP concentration hyperparameters for a clustering dataset." "$@"

PRIOR="crypt/dp_vampprior,crypt/dp_vampprior_unchained"
CONCENTRATION="0.1,0.5,1.0,1.5,2.0,3.0,4.0,5.0,7.5,10.0"
U_INITIAL_STRATEGY="data,stratified_data,random"
SEED="1,2,3,4,5"

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params=$PRIOR \
    model.prior_params.concentration=$CONCENTRATION \
    model.u_initial_strategy=$U_INITIAL_STRATEGY \
    loggers.wandb.name='${model.prior_params.name}-${model.u_initial_strategy}-${model.prior_params.concentration}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=sweep \
    loggers.wandb.tags=["v2"] \
    seed=$SEED
