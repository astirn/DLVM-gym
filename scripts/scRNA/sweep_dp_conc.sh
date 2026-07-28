#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Sweep DP concentration hyperparameters for a scRNA dataset." "$@"

PRIOR="crypt/dp_vampprior_unchained"
CONCENTRATION="0.5,1.0,2.0"
KL_WARMUP="0,50,100,200,400"
SEED="1,2,3"

PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=true \
    priors@model.prior_params=$PRIOR \
    model.prior_params.concentration=$CONCENTRATION \
    model.kl_warmup=$KL_WARMUP \
    model.u_initial_strategy="stratified_data" \
    loggers.wandb.name='${model.prior_params.name}-${model.kl_warmup}-${model.prior_params.concentration}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=sweep \
    loggers.wandb.tags=["v1"] \
    seed=$SEED
