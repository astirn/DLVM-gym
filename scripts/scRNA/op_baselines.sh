#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "OpenProblems baseline experiments." "$@"

ENCODE_COVARIATES="false,true"
PRIOR="standard_normal"
SEED="1,2,3"

PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=$ENCODE_COVARIATES \
    priors@model.prior_params=$PRIOR \
    loggers.wandb.name='${model.prior_params.name}-${model.encode_covariates}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v2"] \
    seed=$SEED
