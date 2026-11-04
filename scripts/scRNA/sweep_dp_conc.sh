#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Sweep DP concentration hyperparameters for a scRNA dataset." "$@"

PRIOR="crypt/dp_vampprior_unleashed"
CONCENTRATION="0.5,1.0,2.0"
KL_WARMUP="0,50,100,200,400"
SEED="1,2,3"

PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/open_problems --multirun \
    data/scRNA@data=${DATASET} \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=true \
    model.kl_warmup_epochs=$KL_WARMUP \
    model.prior_params.concentration=$CONCENTRATION \
    model.u_initial_strategy="stratified_data" \
    priors@model.prior_params=$PRIOR \
    loggers.wandb.name='${model.prior_params.name}-${model.kl_warmup_epochs}-${model.prior_params.concentration}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=sweep \
    loggers.wandb.tags=["v1"] \
    seed=$SEED
